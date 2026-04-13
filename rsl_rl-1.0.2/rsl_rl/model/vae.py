import pdb
from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from mld.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask


class AutoMldVae(nn.Module):
    def __init__(self,
                 nfeats: int,                # 单帧观测维度，对应原 num_obs
                 num_history: int,           # 新增：输入历史步数
                 num_latent: int,            # 隐变量 z 的维度
                 vel_dim: int = 3,           # 速度预测维度
                 h_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 sigma_min: float = 0.0,      # 与 VAE 一致的方差约束
                 sigma_max: float = 5.0,
                 **kwargs) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.num_history = num_history
        self.num_latent = num_latent
        self.vel_dim = vel_dim
        self.h_dim = h_dim
        self.arch = arch
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # 隐变量和速度的总尺寸（用于全局标记数量）
        self.total_latent_size = num_latent + vel_dim   # z 和 v 拼接后的总维度
        self.latent_size = num_latent                   # 仅 z 的维度

        # 位置编码
        self.query_pos_encoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding)

        # 编码器 Transformer
        encoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)
        # 从编码特征投影到隐变量和速度的拼接分布参数
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.total_latent_size)

        # 解码器 Transformer（沿用原设计）
        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        # 将隐变量 z 投影回 Transformer 维度
        self.decoder_latent_proj = nn.Linear(self.num_latent, self.h_dim)

        # 可学习的全局标记：数量为 total_latent_size * 2 (mu 和 logvar 各一半)
        self.global_motion_token = nn.Parameter(
            torch.randn(self.total_latent_size * 2, self.h_dim))

        # 输入嵌入
        self.skel_embedding = nn.Linear(self.nfeats, self.h_dim)
        # 输出层：预测下一帧观测
        self.final_layer = nn.Linear(self.h_dim, self.nfeats)

        # 用于速度预测的额外线性头（也可复用 encoder_latent_proj 的切割，此处独立实现更清晰）
        # 实际上我们已经将速度参数包含在 global token 的输出中，无需额外头

    def encode(self, history_motion: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        输入: history_motion [B, H, D]  历史观测序列
        返回: (latent_mu, latent_var, vel_mu, vel_var)
               - latent_mu/var: [B, num_latent]
               - vel_mu/var:    [B, vel_dim]
        """
        device = history_motion.device
        bs = history_motion.shape[0]

        # 仅使用历史序列，无未来信息
        x = self.skel_embedding(history_motion)          # [B, H, h_dim]
        x = x.permute(1, 0, 2)                           # [H, B, h_dim]

        # 复制全局标记到当前批次
        dist_tokens = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        # 拼接：[2*total_latent_size, B, h_dim] + [H, B, h_dim] => [total_tokens, B, h_dim]
        xseq = torch.cat((dist_tokens, x), dim=0)

        # 位置编码
        xseq = self.query_pos_encoder(xseq)
        # Transformer 编码
        encoded_seq = self.encoder(xseq)
        dist_encoded = encoded_seq[:dist_tokens.shape[0]]   # [2*total_latent_size, B, h_dim]

        # 投影到分布参数空间
        dist_params = self.encoder_latent_proj(dist_encoded)  # [2*total_latent_size, B, total_latent_size]
        # 注意：dist_params 的形状为 [2*total_latent_size, B, total_latent_size]
        # 我们需要将其重新排列为 (mu, logvar) 对应每个 latent 和 vel 维度

        # 更规范的做法：将输出视为一个大的向量，然后 split
        # 此处简化：直接取前 total_latent_size 个 token 作为 mu，后 total_latent_size 个作为 logvar
        mu_all = dist_params[:self.total_latent_size]        # [total_latent_size, B, total_latent_size]?? 需要检查维度
        # 实际期望：每个 latent 维度对应一个 mu 值，所以应该投影到标量？
        # 原 AutoMldVae 中 encoder_latent_proj 输出维度为 latent_dim，每个 token 对应 latent 的一个维度。
        # 因此我们修改：encoder_latent_proj = nn.Linear(h_dim, 1) 会不匹配。需仔细处理。

        # 修正方案：保留原设计，但总 token 数 = 2 * total_latent_size，每个 token 对应一个标量（分布的均值或对数方差）
        # 所以 encoder_latent_proj 应该输出维度为 1，然后 reshape。
        # 或者改为：encoder_latent_proj = nn.Linear(h_dim, 2) 分别输出 mu 和 logvar? 不优雅。
        # 直接参照原代码：encoder_latent_proj = nn.Linear(h_dim, 1) 配合 total token 数量。

        # 为了简洁，此处我们采用另一种实现：直接从编码后的全局标记中提取每个维度的均值和方差。
        # 为了不改变代码太多，我们重新定义 self.encoder_latent_proj 为 nn.Linear(h_dim, 2)（mu 和 logvar）
        # 但需要每个 latent 维度独立，因此使用 total_latent_size 组线性层（或一个矩阵）。
        # 原代码中 encoder_latent_proj = nn.Linear(h_dim, latent_dim) 且 token 数量为 2*latent_size，
        # 意味着每个 token 输出一个 latent_dim 维向量？这在逻辑上重复。但原论文可能另有含义。
        # 鉴于适配任务简单，我们改用更直接的设计：
        # 使用两个独立的 MLP 头从池化后的序列特征预测 mu 和 logvar。
        # 但为了保持 Transformer 全局标记的设计风格，我们可以：

        # 取全局标记的平均池化作为整体编码特征
        global_feat = dist_encoded.mean(dim=0)  # [B, h_dim]

        # 然后分别投影得到 latent 和 velocity 的分布参数
        # 这里添加四个线性层（如果之前未定义）
        if not hasattr(self, 'latent_mu_head'):
            self.latent_mu_head = nn.Linear(self.h_dim, self.num_latent)
            self.latent_var_head = nn.Linear(self.h_dim, self.num_latent)
            self.vel_mu_head = nn.Linear(self.h_dim, self.vel_dim)
            self.vel_var_head = nn.Linear(self.h_dim, self.vel_dim)

        latent_mu = self.latent_mu_head(global_feat)
        latent_logvar = self.latent_var_head(global_feat)
        vel_mu = self.vel_mu_head(global_feat)
        vel_logvar = self.vel_var_head(global_feat)

        # 应用方差约束（与原 VAE 一致）
        latent_logvar = self._constrain_logvar(latent_logvar)
        vel_logvar = self._constrain_logvar(vel_logvar)

        return latent_mu, latent_logvar, vel_mu, vel_logvar

    def _constrain_logvar(self, logvar: Tensor) -> Tensor:
        """将 logvar 约束到对应 sigma_min/max 范围内"""
        sigma = torch.exp(0.5 * logvar)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)
        return 2 * torch.log(sigma + 1e-8)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    def decode(self, z: Tensor, history_motion: Tensor, vel: Tensor = None) -> Tensor:
        """
        解码生成下一帧观测。
        z: [B, num_latent]
        history_motion: [B, H, nfeats]
        vel: [B, vel_dim]  (可选，训练时可传入真实速度，推理时可用采样速度)
        返回: next_obs [B, nfeats]
        """
        bs = history_motion.shape[0]

        # 如果提供了速度，则将其拼接到 z 后（类似原 VAE）
        if vel is not None:
            z = torch.cat([z, vel], dim=1)   # [B, num_latent + vel_dim]
        else:
            # 若未提供速度，则在推理时需自行采样或置零（此处假定调用前已处理）
            pass

        # 投影 z 到 Transformer 维度
        z_proj = self.decoder_latent_proj(z)          # [B, h_dim]
        z_proj = z_proj.unsqueeze(0)                  # [1, B, h_dim]

        # 历史嵌入
        hist_emb = self.skel_embedding(history_motion)  # [B, H, h_dim]
        hist_emb = hist_emb.permute(1, 0, 2)           # [H, B, h_dim]

        # 可学习查询（仅一个查询用于预测下一帧）
        queries = torch.zeros(1, bs, self.h_dim, device=z.device)  # [1, B, h_dim]

        if self.arch == "all_encoder":
            xseq = torch.cat((z_proj, hist_emb, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq)[-1:]   # 取最后一个 token（对应查询位置）
        elif self.arch == "encoder_decoder":
            xseq = torch.cat((hist_emb, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(tgt=xseq, memory=z_proj)[-1:]

        # output: [1, B, h_dim] -> [B, h_dim]
        next_obs_feat = output.squeeze(0)
        next_obs = self.final_layer(next_obs_feat)     # [B, nfeats]
        return next_obs

    def forward(self, history_motion: Tensor):
        """
        前向传播，返回与原 VAE 一致的结构：
        (z, vel), (latent_mu, latent_var, vel_mu, vel_var)
        """
        latent_mu, latent_logvar, vel_mu, vel_logvar = self.encode(history_motion)

        z = self.reparameterize(latent_mu, latent_logvar)
        vel = self.reparameterize(vel_mu, vel_logvar)

        return [z, vel], [latent_mu, latent_logvar, vel_mu, vel_logvar]

    def loss_fn(self, history_motion, next_obs, vel_target, kld_weight=1.0):
        """
        完全复用原 VAE 的损失计算逻辑。
        history_motion: [B, H, D]  历史观测序列
        next_obs:       [B, D]     下一帧真实观测
        vel_target:     [B, 3]     真实速度
        """
        estimation, latent_params = self.forward(history_motion)
        z, v = estimation
        latent_mu, latent_var, vel_mu, vel_var = latent_params

        # 重构损失：使用真实速度
        recons = self.decode(z, history_motion, vel=vel_target)
        recons_loss = nn.functional.mse_loss(recons, next_obs, reduction='none').mean(-1)

        # 速度监督损失
        vel_loss = nn.functional.mse_loss(v, vel_target, reduction='none').mean(-1)

        # KL 散度（仅对隐变量 z）
        kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim=1)

        loss = recons_loss + vel_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'vel_loss': vel_loss,
            'kld_loss': kld_loss,
        }

    def inference(self, history_motion):
        """确定性推理，返回 latent_mu 和 vel_mu"""
        _, latent_params = self.forward(history_motion)
        latent_mu, _, vel_mu, _ = latent_params
        return [latent_mu, vel_mu]

    def sample(self, history_motion):
        """采样 z 和 v"""
        estimation, _ = self.forward(history_motion)
        return estimation