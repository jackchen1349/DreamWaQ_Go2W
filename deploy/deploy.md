# GO2W DreamWaQ 环境配置指南

# Python 部署

## 依赖列表
| 依赖 | 用途 |
|------|------|
| torch (ARM64) | 神经网络推理 |
| numpy | 数值计算 |
| pyyaml | YAML 配置解析 |
| unitree_sdk2_python | 机器人 DDS 通信 |

## 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n deploy python=3.8 -y
conda activate deploy

# 2. 安装 PyTorch (Jetson ARM64 版)
# 从 NVIDIA 论坛下载对应 JetPack 版本的 wheel 文件
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/
pip install torch-2.1.0-cp38-cp38-linux_aarch64.whl

# 3. 安装 Python 依赖
pip install numpy pyyaml

# 4. 安装 Unitree SDK2 Python
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

## 运行命令

```bash
cd DreamWaQ_Go2W/deploy/deploy_real
python3 deploy_real_go2w_DWAQ.py eth0
```

---

# C++ 部署

## 依赖列表
| 依赖 | 用途 |
|------|------|
| cmake, g++ | 编译工具 |
| libtorch (ARM64) | PyTorch C++ 推理 |
| yaml-cpp | YAML 配置解析 |
| unitree_sdk2 | 机器人 DDS 通信 |
| Cyclone DDS | DDS 中间件 (随 SDK 安装) |

## 安装步骤

```bash
# 1. 系统依赖
sudo apt update
sudo apt install -y cmake build-essential libyaml-cpp-dev

# 2. 安装 Cyclone DDS
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install

# 3. 安装 Unitree SDK2 C++
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2 && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install

# 4. 下载 LibTorch (ARM64 版)
# 从 PyTorch 官网下载 cxx11 ABI 版本
# https://pytorch.org/get-started/locally/
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-*.zip -d DreamWaQ_Go2W/deploy/deploy_real/cpp_go2w/

# 5. 编译项目
cd DreamWaQ_Go2W/deploy/deploy_real/cpp_go2w
mkdir build && cd build
cmake .. && make -j$(nproc)
```

## 运行命令

```bash
cd DreamWaQ_Go2W/deploy/deploy_real/cpp_go2w/build
./go2w_deploy_real eth0
```

---

# 模型文件

确保以下文件存在于 `deploy/pre_train/g2wDWAQ/` 目录：
- actor_dwaq.pt
- encoder_dwaq.pt
- latent_mu_dwaq.pt
- latent_var_dwaq.pt
- vel_mu_dwaq.pt
- vel_var_dwaq.pt

---


