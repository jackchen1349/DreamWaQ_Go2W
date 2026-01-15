/**
 * @file AtomicLock.h
 * @brief 原子锁实现 - 用于多线程数据同步
 * 
 * 本文件提供了两个类：
 * 1. AFLock: 基于 std::atomic_flag 实现的自旋锁
 * 2. ScopedLock: RAII 风格的锁管理类，自动获取和释放锁
 */

#ifndef ATOMIC_LOCK_H
#define ATOMIC_LOCK_H

#include <atomic>

/**
 * @class AFLock
 * @brief 原子标志锁（Atomic Flag Lock）
 * 
 * 使用 C++11 的 std::atomic_flag 实现的轻量级自旋锁。
 * 特点：
 * - 无锁竞争时开销极小
 * - 适用于临界区很短的场景
 * - 用于保护 LowState 数据的线程安全访问
 */
class AFLock
{
	public:
		AFLock() = default;
		~AFLock() = default;

		/**
		 * @brief 获取锁（阻塞式）
		 * 
		 * 使用自旋等待直到成功获取锁。
		 * test_and_set 会原子地将标志设为 true 并返回之前的值。
		 * memory_order_acquire 确保后续读操作不会被重排到锁获取之前。
		 */
		void Lock()
		{
			while (mAFL.test_and_set(std::memory_order_acquire));
		}

		/**
		 * @brief 尝试获取锁（非阻塞式）
		 * @return true 如果成功获取锁，false 如果锁已被占用
		 */
		bool TryLock()
		{
			return mAFL.test_and_set(std::memory_order_acquire) ? false : true;
		}

		/**
		 * @brief 释放锁
		 * 
		 * memory_order_release 确保之前的写操作不会被重排到锁释放之后。
		 */
		void Unlock()
		{
			mAFL.clear(std::memory_order_release);
		}

	private:
		std::atomic_flag mAFL = ATOMIC_FLAG_INIT;  ///< 原子标志，初始化为未设置状态
};

/**
 * @class ScopedLock
 * @brief RAII 风格的作用域锁
 * @tparam L 锁类型（需要有 Lock() 和 Unlock() 方法）
 * 
 * 在构造时自动获取锁，在析构时自动释放锁。
 * 确保即使发生异常也能正确释放锁，避免死锁。
 * 
 * 使用示例：
 * @code
 * AFLock lock;
 * {
 *     ScopedLock<AFLock> scopedLock(lock);
 *     // 临界区代码...
 * } // 自动释放锁
 * @endcode
 */
template<typename L>
class ScopedLock
{
	public:
		/**
		 * @brief 构造函数 - 获取锁
		 * @param lock 要获取的锁的引用
		 */
		explicit ScopedLock(L& lock) :
			mLock(lock)
	{
		mLock.Lock();
	}

		/**
		 * @brief 析构函数 - 释放锁
		 */
		~ScopedLock()
		{
			mLock.Unlock();
		}

	private:
		L& mLock;  ///< 锁的引用
};
#endif
