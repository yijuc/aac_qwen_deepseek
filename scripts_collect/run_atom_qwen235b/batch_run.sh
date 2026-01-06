#!/bin/bash
# 1. 正常模式
bash run.sh

# 2. 開啟 GEMM Log (產生 kernel_...log 並存入 benchmark_output_gemm_results.csv)
# ENABLE_GEMM=1 bash run.sh

# 3. 開啟 Profiler (錄製 profiler 並存入 benchmark_results_profiler.csv)
ENABLE_PROFILER=1 bash run.sh

# 4. 開啟 Aiter Log More 2 (存入 benchmark_output_aiter_results.csv)
ENABLE_AITER_LOG=1 bash run.sh

tail -f /dev/null