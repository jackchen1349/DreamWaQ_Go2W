/**
 * @file main.cpp
 * @brief GO2W DreamWaQ RL Deployment
 */

#include <iostream>
#include <csignal>
#include "Controller.h"

Controller* g_controller = nullptr;

void signalHandler(int signum)
{
    std::cout << "\nInterrupt - entering damp mode" << std::endl;
    if (g_controller) g_controller->damp();
    exit(signum);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <network_interface>" << std::endl;
        return 1;
    }

    std::cout << "GO2W DreamWaQ Deployment\n";
    std::cout << "Make sure robot is hung up or on ground. Press Enter..." << std::endl;
    std::cin.ignore();

    signal(SIGINT, signalHandler);

    Controller controller(argv[1]);
    g_controller = &controller;

    controller.zero_torque_state();
    controller.move_to_default_pos();
    controller.default_pos_state();

    std::cout << "RL Started - Press Select to exit" << std::endl;
    
    while (!controller.isSelectPressed())
    {
        controller.run();
    }
    
    controller.damp();
    return 0;
}
