# GO2W Python vs C++ 性能对比框架

## 目录结构

```
performance_benchmark/
├── python/
│   ├── benchmark_wrapper.py   # Python 性能监控器
│   └── benchmark_deploy.py    # Python benchmark 部署脚本
├── cpp/
│   ├── Controller.cpp/h       # 带计时的控制器（独立于原始代码）
│   ├── benchmark_main.cpp     # C++ benchmark 入口
│   ├── benchmark_timer.h      # C++ 计时工具
│   └── build/                 # 编译输出
├── analysis/
│   └── analyze_results.py     # 结果对比分析
└── results/                   # 测试结果 JSON
```

## 快速开始

### 1. Python 测试

```bash
cd performance_benchmark/python
python3 benchmark_deploy.py eth1 g2w.yaml --duration 60
```

### 2. C++ 测试

```bash
cd performance_benchmark/cpp/build
./go2w_benchmark eth1 60
```

### 3. 对比分析

```bash
cd performance_benchmark
python3 analysis/analyze_results.py results/python_benchmark.json results/cpp_benchmark.json
```

## 测量指标

| 指标 | 说明 |
|------|------|
| Loop Time | 控制循环执行时间（不含 sleep） |
| Inference Time | 神经网络总推理时间 |
| Encoder Time | Encoder 模块时间 |
| Actor Time | Actor 网络时间 |
| Tracking Error | 位置跟踪误差 |
