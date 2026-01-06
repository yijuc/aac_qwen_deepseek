#!/bin/bash
# --- 基礎路徑 ---
export MODEL="/dev/shm/Qwen3-235B-A22B-Instruct-2507-FP8/"
export WORK_DIR="/workdir/atom_qwen235b"
# 確保每次 source 都會檢查並建立當天的資料夾
export LOG_DIR="${WORK_DIR}/logs_batch_$(date +%m%d)"
mkdir -p "$LOG_DIR"
export PORT=8000

# --- 基礎優化變數 ---
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1