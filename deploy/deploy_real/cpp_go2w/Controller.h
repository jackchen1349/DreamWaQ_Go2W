/**
 * @file Controller.h
 * @brief GO2W 机器人 RL 部署控制器 (DreamWaQ)
 */
#include <mutex>

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

#include "torch/script.h"
#include <eigen3/Eigen/Eigen>

#include "RemoteController.h"
#include "DataBuffer.h"
#include <string>
#include <array>

class Controller
{
public:
    Controller(const std::string &net_interface);
    
    // State machine functions
    void zero_torque_state();
    void move_to_default_pos();
    void default_pos_state();
    void run();
    void damp();
    
    bool isSelectPressed() const { return remote_controller.button[KeyMap::select] == 1; }

private:
    // ==================== 回调与通信 ====================
    
    /**
     * @brief LowState 消息回调处理
     * @param message 接收到的 LowState 消息指针
     * 
     * 由 DDS 订阅器在接收到消息时调用。
     * 将消息存入线程安全缓冲区，并解析遥控器数据。
     */
    void low_state_message_handler(const void *message);
    
    /**
     * @brief LowCmd 发送处理（后台线程）
     * 
     * 每2ms执行一次，计算CRC并发送当前指令。
     * 这种异步发送模式确保指令发送频率稳定。
     */
    void low_cmd_write_handler();
    
    // ==================== 指令生成 ====================
    
    /**
     * @brief 初始化电机指令结构
     * 
     * 设置 LowCmd 的头部、标志位，
     * 将所有电机设为初始安全状态（PosStopF/VelStopF）。
     */
    void init_cmd_go();
    
    /**
     * @brief 创建零力矩指令
     * 
     * 将所有电机的 kp、kd、tau 设为0。
     */
    void create_zero_cmd();
    
    /**
     * @brief 创建阻尼指令
     * 
     * 将所有电机的 kd 设为8，其他为0。
     * 机器人会缓慢下降而不是自由落体。
     */
    void create_damping_cmd();
    
    // ==================== 坐标变换 ====================
    
    /**
     * @brief 实机顺序转仿真顺序
     * @param qj 实机关节顺序的数组
     * @return 仿真关节顺序的数组
     */
    std::array<float, 16> trans_r2s(const std::array<float, 16>& qj);
    
    /**
     * @brief 仿真顺序转实机顺序
     * @param qj 仿真关节顺序的数组
     * @return 实机关节顺序的数组
     */
    std::array<float, 16> trans_s2r(const std::array<float, 16>& qj);
    
    /**
     * @brief 从四元数计算重力方向
     * @param quat 机体姿态四元数 [w, x, y, z]
     * @return 机体坐标系下的重力方向向量 [gx, gy, gz]
     */
    std::array<float, 3> get_gravity_orientation(const std::array<float, 4>& quat);
    
    /**
     * @brief VAE 重参数化采样
     * @param mean 均值张量
     * @param logvar 对数方差张量
     * @return 采样结果张量
     */
    torch::Tensor reparameterise(torch::Tensor mean, torch::Tensor logvar);
    
    /**
     * @brief 计算 CRC32 校验值
     * @param ptr 数据指针
     * @param len 数据长度（32位字）
     * @return CRC32 校验值
     */
    uint32_t crc32_core(uint32_t* ptr, uint32_t len);

    // ==================== 通信成员 ====================
    unitree::common::ThreadPtr low_cmd_write_thread_ptr;  ///< 指令发送线程
    DataBuffer<unitree_go::msg::dds_::LowState_> mLowStateBuf;  ///< LowState 线程安全缓冲区
    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;  ///< LowCmd 发布器
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;  ///< LowState 订阅器
    unitree_go::msg::dds_::LowCmd_ low_cmd;  ///< 当前电机指令
    RemoteController remote_controller;  ///< 遥控器状态

    // ==================== 配置参数（来自 g2w.yaml）====================
    float control_dt;  ///< 控制周期（秒）
    std::array<int, 16> joint2motor_idx;  ///< 关节到电机索引映射
    std::array<float, 16> kps;  ///< 位置增益
    std::array<float, 16> kds;  ///< 速度增益
    std::array<float, 16> default_real_angles;  ///< 默认站立角度（实机顺序）
    std::array<float, 16> default_sim_angles;   ///< 默认站立角度（仿真顺序）
    std::array<int, 4> wheel_sim_indices;  ///< 轮子在仿真顺序中的索引
    float ang_vel_scale, dof_err_scale, dof_vel_scale, action_scale;  ///< 观测/动作缩放因子
    std::array<float, 3> cmd_scale;  ///< 速度指令缩放因子
    int num_actions, num_obs;  ///< 动作维度(16)和观测维度(73)

    // ==================== 状态变量 ====================
    std::array<float, 16> qj, dqj, action;  ///< 关节位置、速度、动作
    Eigen::VectorXf obs, obs_hist_buf;  ///< 当前观测和历史观测缓冲区
    std::array<float, 3> cmd;  ///< 速度指令 [vx, vy, omega]
    int counter;  ///< 控制周期计数器

    // ==================== DreamWaQ 神经网络模型 ====================
    torch::jit::script::Module actor;      ///< 策略网络
    torch::jit::script::Module encoder;    ///< 历史编码器
    torch::jit::script::Module latent_mu;  ///< 隐变量均值网络
    torch::jit::script::Module latent_var; ///< 隐变量方差网络
    torch::jit::script::Module vel_mu;     ///< 速度估计均值网络
    torch::jit::script::Module vel_var;    ///< 速度估计方差网络

    // 新增互斥锁，保护 low_cmd
    std::mutex cmd_mutex;
};

#endif
