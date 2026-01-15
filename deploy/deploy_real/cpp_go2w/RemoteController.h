/**
 * @file RemoteController.h
 * @brief 无线遥控器数据解析器
 */

#ifndef REMOTE_CONTROLLER_H
#define REMOTE_CONTROLLER_H

#include <cstdint>
#include <cstring>
#include <vector>

//定义遥控器各按键在 button 数组中的索引位置。
namespace KeyMap {
    constexpr int R1 = 0, L1 = 1, start = 2, select = 3;
    constexpr int R2 = 4, L2 = 5, F1 = 6, F2 = 7;
    constexpr int A = 8, B = 9, X = 10, Y = 11;
    constexpr int up = 12, right = 13, down = 14, left = 15;
}

/**
 * 存储遥控器的当前状态，包括摇杆位置和按键状态。
 * 
 * 摇杆值范围：-1.0 到 1.0
 * - lx: 左摇杆X轴（左右）
 * - ly: 左摇杆Y轴（前后）- 用于控制前进/后退速度
 * - rx: 右摇杆X轴（左右）- 用于控制旋转速度
 * - ry: 右摇杆Y轴（前后）
 * 
 * 按键状态：0=未按下，1=按下
 */
struct RemoteController {
    float lx = 0;       ///< 左摇杆X轴（左右移动）
    float ly = 0;       ///< 左摇杆Y轴（前后移动）- 控制前进速度
    float rx = 0;       ///< 右摇杆X轴（左右旋转）- 控制转向速度
    float ry = 0;       ///< 右摇杆Y轴（未使用）
    int button[16] = {0};  ///< 16个按键的状态数组

    /**
     * @brief 解析遥控器原始数据
     * @param data 来自 LowState.wireless_remote 的原始字节数据
     */
    void set(const std::vector<uint8_t>& data) {
        if (data.size() >= 24) {
            // 解析按键状态（16位位图）
            uint16_t keys;
            std::memcpy(&keys, &data[2], sizeof(uint16_t));
            for (int i = 0; i < 16; i++)
                button[i] = (keys & (1 << i)) >> i;
            
            // 解析摇杆值（4个float）
            std::memcpy(&lx, &data[4], sizeof(float));   // 左摇杆X
            std::memcpy(&rx, &data[8], sizeof(float));   // 右摇杆X
            std::memcpy(&ry, &data[12], sizeof(float));  // 右摇杆Y
            std::memcpy(&ly, &data[20], sizeof(float));  // 左摇杆Y
        }
    }
};

#endif
