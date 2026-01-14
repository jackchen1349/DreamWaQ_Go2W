#!/usr/bin/env python3
"""
GO2W Python 部署 - 带性能测量版本
基于 deploy_real_go2w_DWAQ.py，集成了性能监控功能
"""

from typing import Union
import numpy as np
import time
import os
import sys
import argparse
import torch

# 添加 benchmark 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from python.benchmark_wrapper import PerformanceMonitor, get_monitor

# region ########### Unitree通信相关导入 ###########
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# 使用绝对路径确保能找到 common 模块
DEPLOY_REAL_DIR = "/home/jackie/DreamWaQ_Go2W/deploy/deploy_real"
sys.path.insert(0, DEPLOY_REAL_DIR)

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


def trans_r2s(qj):
    tmp = qj.copy()
    tmp[0] = qj[3];  tmp[1] = qj[4];  tmp[2] = qj[5];  tmp[3] = qj[13]
    tmp[4] = qj[0];  tmp[5] = qj[1];  tmp[6] = qj[2];  tmp[7] = qj[12]
    tmp[8] = qj[9];  tmp[9] = qj[10]; tmp[10] = qj[11]; tmp[11] = qj[15]
    tmp[12] = qj[6]; tmp[13] = qj[7]; tmp[14] = qj[8];  tmp[15] = qj[14]
    return tmp


def trans_s2r(qj):
    tmp = qj.copy()
    tmp[3] = qj[0];  tmp[4] = qj[1];  tmp[5] = qj[2];  tmp[13] = qj[3]
    tmp[0] = qj[4];  tmp[1] = qj[5];  tmp[2] = qj[6];  tmp[12] = qj[7]
    tmp[9] = qj[8];  tmp[10] = qj[9]; tmp[11] = qj[10]; tmp[15] = qj[11]
    tmp[6] = qj[12]; tmp[7] = qj[13]; tmp[8] = qj[14];  tmp[14] = qj[15]
    return tmp


class BenchmarkController:
    """带性能监控的控制器"""
    
    def __init__(self, config: Config, monitor: PerformanceMonitor) -> None:
        self.config = config
        self.monitor = monitor
        self.remote_controller = RemoteController()
        
        # 加载模型文件
        model_path = '/home/jackie/DreamWaQ_Go2W/deploy/pre_train/g2wDWAQ/'
        self.actor = torch.jit.load(model_path + 'actor_dwaq.pt')
        self.encoder = torch.jit.load(model_path + 'encoder_dwaq.pt')
        self.latent_mu = torch.jit.load(model_path + 'latent_mu_dwaq.pt')
        self.latent_var = torch.jit.load(model_path + 'latent_var_dwaq.pt')
        self.vel_mu = torch.jit.load(model_path + 'vel_mu_dwaq.pt')
        self.vel_var = torch.jit.load(model_path + 'vel_var_dwaq.pt')

        # 状态变量初始化
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_real_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.obs_hist_buf = np.zeros(config.num_obs * 5, dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0])
        self.counter = 0

        # DDS通信初始化
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        self.wait_for_low_state()
        init_cmd_go(self.low_cmd)
        
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()  

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandUp()
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        self.monitor.start_comm_send()
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)
        self.monitor.end_comm_send()

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 1
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.joint2motor_idx
        default_pos = self.config.default_real_angles
        dof_size = len(dof_idx)
        
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        for i in range(num_step):
            alpha = i / num_step
            for j in range(12):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = 80
                self.low_cmd.motor_cmd[motor_idx].kd = 5
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(12):
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_real_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = 70.0
                self.low_cmd.motor_cmd[motor_idx].kd = 5.0
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def reparameterise(self, mean, logvar):
        var = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(var)
        return mean + var * code_temp

    def run(self):
        """主控制循环 - 带性能测量"""
        self.monitor.start_loop()
        self.counter += 1
          
        # 获得机器人电机数据
        for i in range(len(self.config.joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq

        self.qj = trans_r2s(self.qj)
        self.dqj = trans_r2s(self.dqj)
        self.action = trans_r2s(self.action)

        # 角速度与重力向量
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        ang_vel_obs = ang_vel * self.config.ang_vel_scale
        quat = self.low_state.imu_state.quaternion
        gravity_orientation = get_gravity_orientation(quat)
        
        # 外部控制命令
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        # 关节误差
        err_obs = self.qj - self.config.default_sim_angles
        err_obs[self.config.wheel_sim_indices] = 0
        err_obs = err_obs * self.config.dof_err_scale
        
        # 关节速度与位置
        dqj_obs = self.dqj * self.config.dof_vel_scale
        qj_obs = self.qj.copy()
        qj_obs[self.config.wheel_sim_indices] = 0

        # 观测向量构建
        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel_obs
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale
        self.obs[9:9+num_actions] = err_obs
        self.obs[9+num_actions:9+num_actions*2] = dqj_obs
        self.obs[9+num_actions*2:9+num_actions*3] = qj_obs
        self.obs[9+num_actions*3:9+num_actions*4] = self.action

        # 拼接观测数据
        self.obs_hist_buf = self.obs_hist_buf[73:]
        self.obs_hist_buf = np.concatenate((self.obs_hist_buf, self.obs), axis=-1)

        # ========== 神经网络推理（带计时） ==========
        self.monitor.start_inference()
        
        self.monitor.start_encoder()
        tmp = torch.from_numpy(self.obs_hist_buf)
        h = self.encoder(tmp)
        self.monitor.end_encoder()
        
        vel_mu = self.vel_mu(h)
        vel_var = self.vel_var(h)
        latent_mu = self.latent_mu(h)
        latent_var = self.latent_var(h)
        vel = self.reparameterise(vel_mu, vel_var)
        latent = self.reparameterise(latent_mu, latent_var)

        code = torch.cat((vel, latent), dim=-1)
        tmpp = torch.from_numpy(self.obs)
        obs_all = torch.cat((code, tmpp), dim=-1)

        self.monitor.start_actor()
        self.action = self.actor(obs_all).detach().numpy().squeeze()
        self.monitor.end_actor()
        
        self.monitor.end_inference()
        # ========== 推理结束 ==========

        self.qj = trans_s2r(self.qj)
        self.action = trans_s2r(self.action)
        self.dqj = trans_s2r(self.dqj)

        # 构建目标位置（用于控制质量测量）
        target_positions = np.zeros(12, dtype=np.float32)
        actual_positions = np.zeros(12, dtype=np.float32)

        for i in range(len(self.config.joint2motor_idx)):
            if i >= 12:
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = 0.0
                self.low_cmd.motor_cmd[motor_idx].dq = self.action[i] * 10.0
                self.low_cmd.motor_cmd[motor_idx].kp = 0.0
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            else:
                motor_idx = self.config.joint2motor_idx[i]
                target_pos = self.config.default_real_angles[i] + self.action[i] * self.config.action_scale
                self.low_cmd.motor_cmd[motor_idx].q = target_pos
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
                
                # 记录用于控制质量分析
                target_positions[i] = target_pos
                actual_positions[i] = self.qj[i]

        # 记录控制质量
        self.monitor.record_control_quality(target_positions, actual_positions)

        self.send_cmd(self.low_cmd)
        
        self.monitor.end_loop()
        
        time.sleep(self.config.control_dt)


def main():
    parser = argparse.ArgumentParser(description='GO2W Python Benchmark Deployment')
    parser.add_argument("net", type=str, help="network interface (e.g., eth0)")
    parser.add_argument("config", type=str, help="config file name", default="g2w.yaml")
    parser.add_argument("--duration", type=int, default=60, help="benchmark duration in seconds")
    parser.add_argument("--output", type=str, default=None, help="output JSON file path")
    args = parser.parse_args()

    # 加载配置
    config_path = f"/home/jackie/DreamWaQ_Go2W/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # 初始化 DDS
    ChannelFactoryInitialize(0, args.net)

    # 创建性能监控器
    monitor = PerformanceMonitor()

    # 创建控制器
    controller = BenchmarkController(config, monitor)

    # 状态机执行
    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    print(f'RL Benchmark Started - Running for {args.duration}s...')
    
    start_time = time.time()
    
    try:
        while True:
            controller.run()
            
            # 检查时间限制
            if time.time() - start_time >= args.duration:
                print(f"\nBenchmark duration ({args.duration}s) completed.")
                break
            
            # 检查退出按钮
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("\nSelect button pressed, exiting.")
                break
                
    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")

    # 保存和显示结果
    monitor.print_summary()
    
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'results', 'python_benchmark.json'
        )
    
    monitor.save_results(output_path)

    # 进入阻尼模式
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")


if __name__ == "__main__":
    main()
