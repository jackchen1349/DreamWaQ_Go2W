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
