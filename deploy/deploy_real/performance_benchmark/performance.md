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
