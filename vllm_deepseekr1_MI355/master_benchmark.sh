#!/bin/bash
export enable_profiler=0
export kunlun_is_random=1
export kunlun_use_sla=True

export TP=8
export MODEL="/dev/shm/DeepSeek-R1-0528"
export PORT=8000
export SERVER_LOG_DIR="/dockerx/eveline/vllm_deepseekr1/logs_0206_night0205_kunlun_sla_random_newenv/"
export CLIENT_LOG_DIR="$SERVER_LOG_DIR"

TEST_CASES=(
  "800 1000 1600 2000 100"
  "3000 3600 300 500 100"
  "3600 4400 1800 2200 100"
  "11000 15000 2500 2900 100"
  "16000 20000 300 500 100"
)

mkdir -p "$SERVER_LOG_DIR"

# --- 偵測 GPU 顯存是否清空的函式 ---
wait_for_gpu_clear() {
    echo "Checking GPU VRAM status..."
    MAX_RETRIES=12
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if command -v nvidia-smi > /dev/null; then
            VRAM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
        elif command -v rocm-smi > /dev/null; then
            # 使用更強健的 Python 解析，遍歷 JSON 尋找包含 "Used" 且包含 "VRAM" 的數值
            VRAM_USAGE=$(rocm-smi --showmeminfo vram --json | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    total_used = 0
    # 遍歷所有 GPU (如 card0, card1...)
    for card, metrics in data.items():
        if isinstance(metrics, dict):
            # 尋找 Key 包含 'VRAM Total Used' 或 'VRAM Used' 的值
            used = metrics.get('VRAM Total Used') or metrics.get('VRAM Used') or 0
            total_used += int(used)
    print(total_used // 1024 // 1024) # 轉換為 MiB
except Exception:
    print(0)
")
        else
            VRAM_USAGE=0
        fi

        if [ "$VRAM_USAGE" -lt 500 ]; then
            echo "GPU VRAM is clear (Current: ${VRAM_USAGE} MiB)."
            return 0
        fi
        
        echo "GPU VRAM still occupied (${VRAM_USAGE} MiB). Waiting 10s... ($((RETRY_COUNT+1))/$MAX_RETRIES)"
        sleep 10
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done
}

# --- main ---
for CASE in "${TEST_CASES[@]}"; do
    
    pkill -9 -f vllm
    pkill -9 -f "vllm.entrypoints.openai.api_server"
    sleep 5

    wait_for_gpu_clear

    read -r MIN_IN MAX_IN MIN_OUT MAX_OUT CONC <<< "$CASE"
    export MIN_IN MAX_IN MIN_OUT MAX_OUT CONC

    # 啟動 Server
    echo ">>> Starting Test Case: $CASE"
    bash run_server.sh &
    SERVER_PID=$!
    
    # 等待 Server (Health Check)...
    until curl -s http://localhost:8000/health; do sleep 5; done

    # 執行 Client
    bash run_kunlun_client.sh "$MIN_IN" "$MAX_IN" "$MIN_OUT" "$MAX_OUT" "$CONC"

    # 結束後清理
    echo "Round finished. Starting deep cleanup..."
    kill $SERVER_PID 2>/dev/null
    pkill -9 -f vllm
    pkill -9 -f "vllm.entrypoints.openai.api_server"
    pkill -9 -f "multiprocessing.spawn"    
    sleep 10
done