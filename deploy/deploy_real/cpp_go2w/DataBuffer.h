/**
 * @file DataBuffer.h
 * @brief 线程安全的数据缓冲区模板类
 * 
 * 本文件提供了一个泛型的线程安全数据缓冲区，用于在多线程环境中
 * 安全地共享数据。在本项目中主要用于存储和访问机器人的 LowState 数据。
 * 
 * 工作原理：
 * - 使用 AFLock 保护所有数据访问
 * - 通过 shared_ptr 管理数据生命周期
 * - 订阅线程写入数据，控制线程读取数据
 */

#ifndef DATA_BUFFER_H
#define DATA_BUFFER_H

#include <memory>
#include "AtomicLock.h"

/**
 * @class DataBuffer
 * @brief 线程安全的数据缓冲区
 * @tparam T 缓冲区存储的数据类型
 * 
 * 主要用于：
 * - 在 DDS 订阅回调中存储接收到的 LowState 消息
 * - 在控制循环中读取最新的机器人状态
 */
template<typename T>
class DataBuffer
{
	public:
		explicit DataBuffer() = default;
		~DataBuffer() = default;

		/**
		 * @brief 设置数据指针
		 * @param dataPtr 要设置的 shared_ptr
		 * 
		 * 直接替换内部的数据指针。
		 */
		void SetDataPtr(const std::shared_ptr<T>& dataPtr)
		{
			ScopedLock<AFLock> lock(mLock);
			mDataPtr = dataPtr;
		}

		/**
		 * @brief 获取数据指针
		 * @param clear 如果为 true，获取后清空内部指针
		 * @return 数据的 shared_ptr，如果缓冲区为空则返回 nullptr
		 * 
		 * 这是推荐的数据访问方式，避免不必要的数据拷贝。
		 * 在控制循环中使用：auto state = mLowStateBuf.GetDataPtr();
		 */
		std::shared_ptr<T> GetDataPtr(bool clear = false)
		{
			ScopedLock<AFLock> lock(mLock);
			if (clear)
			{
				std::shared_ptr<T> dataPtr = mDataPtr;
				mDataPtr.reset();
				return dataPtr;
			}
			else
			{
				return mDataPtr;
			}
		}

		/**
		 * @brief 交换数据指针
		 * @param dataPtr 要交换的 shared_ptr（会被修改）
		 */
		void SwapDataPtr(std::shared_ptr<T>& dataPtr)
		{
			ScopedLock<AFLock> lock(mLock);
			mDataPtr.swap(dataPtr);
		}

		/**
		 * @brief 存储数据（拷贝方式）
		 * @param data 要存储的数据
		 * 
		 * 创建数据的副本并存储。在 DDS 回调中使用此方法，
		 * 因为回调参数在函数返回后可能失效。
		 */
		void SetData(const T& data)
		{
			ScopedLock<AFLock> lock(mLock);
			mDataPtr = std::shared_ptr<T>(new T(data));
		}

		/**
		 * @brief 获取数据（拷贝方式）
		 * @param data 用于接收数据的引用
		 * @param clear 如果为 true，获取后清空内部指针
		 * @return true 如果成功获取数据，false 如果缓冲区为空
		 */
		bool GetData(T& data, bool clear = false)
		{
			ScopedLock<AFLock> lock(mLock);
			if (mDataPtr == NULL)
			{
				return false;
			}

			data = *mDataPtr;
			if (clear)
			{
				mDataPtr.reset();
			}

			return true;
		}

		/**
		 * @brief 清空缓冲区
		 */
		void Clear()
		{
			ScopedLock<AFLock> lock(mLock);
			mDataPtr.reset();
		}

	private:
		std::shared_ptr<T> mDataPtr;  ///< 数据指针
		AFLock mLock;                  ///< 保护数据访问的锁
};
#endif
