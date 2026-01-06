#!/bin/bash
# --- 基礎路徑 ---
export MODEL="/dev/shm/DeepSeek-R1-0528/"
export WORK_DIR="/workdir/atom_deepseekr1"
# 確保每次 source 都會檢查並建立當天的資料夾
export LOG_DIR="${WORK_DIR}/logs_t2prof_batch_$(date +%m%d)"
mkdir -p "$LOG_DIR"
export PORT=8000