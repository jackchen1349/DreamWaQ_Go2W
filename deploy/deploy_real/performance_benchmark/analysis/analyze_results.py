#!/usr/bin/env python3
"""
GO2W Python vs C++ 性能对比分析脚本
"""

import json
import argparse
import os

def load_results(filepath):
    """加载 JSON 结果文件"""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_results(python_data, cpp_data):
    """对比性能数据"""
    py_timing = python_data.get("timing", {})
    cpp_timing = cpp_data.get("timing", {})
    
    results = {
        "timing_comparison": {},
        "control_quality_comparison": {},
        "summary": {}
    }
    
    # 时序对比
    metrics = [
        ("loop_time_ms", "循环时间"),
        ("inference_time_ms", "推理时间"),
        ("encoder_time_ms", "Encoder时间"),
        ("actor_time_ms", "Actor时间"),
    ]
    
    for key, name in metrics:
        py_val = py_timing.get(key, {}).get("mean", 0)
        cpp_val = cpp_timing.get(key, {}).get("mean", 0)
        
        if py_val > 0 and cpp_val > 0:
            speedup = py_val / cpp_val
            reduction = (py_val - cpp_val) / py_val * 100
        else:
            speedup = 0
            reduction = 0
            
        results["timing_comparison"][key] = {
            "name": name,
            "python_ms": py_val,
            "cpp_ms": cpp_val,
            "speedup": speedup,
            "reduction_percent": reduction
        }
    
    # 控制质量对比
    py_cq = python_data.get("control_quality", {})
    cpp_cq = cpp_data.get("control_quality", {})
    
    results["control_quality_comparison"] = {
        "python_mean_error": py_cq.get("mean_error", 0),
        "cpp_mean_error": cpp_cq.get("mean_error", 0),
        "python_max_error": py_cq.get("max_error", 0),
        "cpp_max_error": cpp_cq.get("max_error", 0),
    }
    
    # 样本数
    py_samples = python_data.get("metadata", {}).get("total_samples", 0)
    cpp_samples = cpp_data.get("metadata", {}).get("total_samples", 0)
    
    # 汇总结论
    loop_speedup = results["timing_comparison"]["loop_time_ms"]["speedup"]
    inference_speedup = results["timing_comparison"]["inference_time_ms"]["speedup"]
    
    results["summary"] = {
        "python_samples": py_samples,
        "cpp_samples": cpp_samples,
        "loop_speedup": loop_speedup,
        "inference_speedup": inference_speedup,
        "cpp_is_faster": loop_speedup > 1.0
    }
    
    return results

def print_report(comparison):
    """打印对比报告"""
    print("\n" + "=" * 70)
    print("        GO2W Python vs C++ 性能对比报告")
    print("=" * 70)
    
    summary = comparison["summary"]
    print(f"\n样本数量: Python={summary['python_samples']}, C++={summary['cpp_samples']}")
    
    # 时序对比表
    print("\n" + "-" * 70)
    print("【时序性能对比】")
    print("-" * 70)
    print(f"{'指标':<20} {'Python (ms)':<15} {'C++ (ms)':<15} {'加速比':<12} {'减少%':<10}")
    print("-" * 70)
    
    for key, data in comparison["timing_comparison"].items():
        name = data["name"]
        py = data["python_ms"]
        cpp = data["cpp_ms"]
        speedup = data["speedup"]
        reduction = data["reduction_percent"]
        
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "-"
        reduction_str = f"{reduction:.1f}%" if reduction != 0 else "-"
        
        print(f"{name:<20} {py:<15.3f} {cpp:<15.3f} {speedup_str:<12} {reduction_str:<10}")
    
    # 控制质量对比
    cq = comparison["control_quality_comparison"]
    print("\n" + "-" * 70)
    print("【控制质量对比】")
    print("-" * 70)
    print(f"{'指标':<25} {'Python':<20} {'C++':<20}")
    print("-" * 70)
    print(f"{'平均跟踪误差 (rad)':<25} {cq['python_mean_error']:<20.6f} {cq['cpp_mean_error']:<20.6f}")
    print(f"{'最大跟踪误差 (rad)':<25} {cq['python_max_error']:<20.6f} {cq['cpp_max_error']:<20.6f}")
    
    # 结论
    print("\n" + "=" * 70)
    print("【结论】")
    print("=" * 70)
    
    if summary["cpp_is_faster"]:
        print(f"✓ C++ 控制循环速度更快 ({summary['loop_speedup']:.2f}x)")
        print(f"✓ C++ 推理速度更快 ({summary['inference_speedup']:.2f}x)")
    else:
        print(f"✗ Python 控制循环速度更快 ({1/summary['loop_speedup']:.2f}x)")
    
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='GO2W 性能对比分析')
    parser.add_argument('python_result', type=str, help='Python benchmark JSON 文件路径')
    parser.add_argument('cpp_result', type=str, help='C++ benchmark JSON 文件路径')
    parser.add_argument('--output', type=str, help='输出对比结果 JSON 文件')
    args = parser.parse_args()
    
    # 加载数据
    python_data = load_results(args.python_result)
    cpp_data = load_results(args.cpp_result)
    
    # 对比分析
    comparison = compare_results(python_data, cpp_data)
    
    # 打印报告
    print_report(comparison)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n对比结果已保存到: {args.output}")

if __name__ == "__main__":
    main()
