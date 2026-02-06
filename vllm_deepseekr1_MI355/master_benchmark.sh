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
#   "800 1000 1600 2000 100"
  "3000 3600 300 500 100"
  "3600 4400 1800 2200 100"
  "11000 15000 2500 2900 100"
  "16000 20000 300 500 100"
)

mkdir -p "$SERVER_LOG_DIR"

# --- 偵測 GPU 顯存是否清空的函式 ---
wait_for_gpu_clear() {
    echo "Checking GPU VRAM status..."
    MAX_RETRIES=20
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if command -v nvidia-smi > /dev/null; then
            VRAM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
        elif command -v rocm-smi > /dev/null; then
            # 強化版解析：直接抓取所有包含 "Used" 的數字並加總
            VRAM_USAGE=$(rocm-smi --showmeminfo vram --json | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    total = 0
    def find_used(d):
        s = 0
        if isinstance(d, dict):
            for k, v in d.items():
                if 'Used' in k and isinstance(v, (int, str)):
                    try: s += int(v)
                    except: pass
                else: s += find_used(v)
        return s
    total = find_used(data)
    print(total // 1024 // 1024)
except:
    print(999999) # 發生錯誤時回傳大數字，強迫等待
")
        else
            VRAM_USAGE=0
        fi

        # 這裡門檻設為 1000MB，因為 DeepSeek-R1 很大，留一點餘裕
        if [ "$VRAM_USAGE" -lt 3000 ]; then
            echo "GPU VRAM is clear (Current: ${VRAM_USAGE} MiB)."
            return 0
        fi
        
        echo "GPU VRAM still occupied (${VRAM_USAGE} MiB). Waiting 15s... ($((RETRY_COUNT+1))/$MAX_RETRIES)"
        sleep 15
        RETRY_COUNT=$((RETRY_COUNT + 1))
        # 額外補刀：每等一次就殺一次進程
        pkill -9 -f vllm
    done
    return 1
}

# --- main ---
for CASE in "${TEST_CASES[@]}"; do
    
    pkill -9 -f vllm
    pkill -9 -f "vllm.entrypoints.openai.api_server"
    pkill -9 -f "api_server"
    pkill -9 -f "EngineCore"
    pkill -9 -f "WorkerProc"
    pkill -9 -f "multiprocessing.spawn"    
    sleep 5

    wait_for_gpu_clear

    read -r MIN_IN MAX_IN MIN_OUT MAX_OUT CONC <<< "$CASE"
    export MIN_IN MAX_IN MIN_OUT MAX_OUT CONC

    # 啟動 Server
    echo ">>> Starting Test Case: $CASE"
    bash run_server.sh &
    SERVER_PID=$!
    
    # Wait for health
    echo "Waiting for Server Health..."
    MAX_WAIT=60
    WAIT_SEC=0
    while ! curl -s http://localhost:8000/health > /dev/null; do
        sleep 5
        WAIT_SEC=$((WAIT_SEC + 5))
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server died unexpectedly. Check logs."
            break
        fi
        if [ $WAIT_SEC -gt 300 ]; then
            echo "Server start timeout."
            break
        fi
    done

    # 執行 Client
    bash run_kunlun_client.sh "$MIN_IN" "$MAX_IN" "$MIN_OUT" "$MAX_OUT" "$CONC"

    # 結束後清理
    echo "Round finished. Starting deep cleanup..."
    kill $SERVER_PID 2>/dev/null
    pkill -9 -f vllm
    pkill -9 -f "vllm.entrypoints.openai.api_server"
    pkill -9 -f "api_server"
    pkill -9 -f "EngineCore"
    pkill -9 -f "WorkerProc"
    pkill -9 -f "multiprocessing.spawn"
    sleep 20
done