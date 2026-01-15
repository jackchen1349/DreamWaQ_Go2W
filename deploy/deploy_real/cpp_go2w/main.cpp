/**
 * @file main.cpp
 * @brief GO2W DreamWaQ 强化学习部署主程序
 */

#include <iostream>
#include <csignal>
#include "Controller.h"

/// 全局控制器指针，用于信号处理函数访问
Controller* g_controller = nullptr;

/**
 * @brief 信号处理函数 - 处理 Ctrl+C 确保安全退出
 */
void signalHandler(int signum)
{
    std::cout << "\nInterrupt - entering damp mode" << std::endl;
    if (g_controller) g_controller->damp();
    exit(signum);
}

int main(int argc, char **argv)
{
    // 检查命令行参数
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <network_interface>" << std::endl;
        return 1;
    }

    // 注册信号处理（Ctrl+C 安全退出）
    signal(SIGINT, signalHandler);

    // 初始化控制器
    Controller controller(argv[1]);
    g_controller = &controller;

    // 状态机流程
    controller.zero_torque_state();    // 1. 零力矩等待 Start 键
    controller.move_to_default_pos();  // 2. 移动到默认位置
    controller.default_pos_state();    // 3. 等待 A 键

    std::cout << "RL Started - Press Select to exit" << std::endl;
    
    // 主控制循环
    while (!controller.isSelectPressed())
    {
        controller.run();
    }
    
    controller.damp();
    return 0;
}
