#!/bin/bash
export enable_profiler=0
export kunlun_is_random=1
export kunlun_use_sla=True

export TP=8
export PORT=8000
export MODEL="/dev/shm/DeepSeek-R1-0528"
export SERVER_LOG_DIR="/dockerx/eveline/atom_deepseekr1/logs_0209_02060323_kunlun_sla_random_mix"
export CLIENT_LOG_DIR="$SERVER_LOG_DIR"

# 修改後的 TEST_CASES，增加第五個參數以後的腳本名稱
TEST_CASES=(
  "800 1000 1600 2000 100 run_server_1.sh"
  "3000 3600 300 500 100 run_server_2.sh"
  "3600 4400 1800 2200 100 run_server_2.sh"
  "11000 15000 2500 2900 100 run_server_2.sh"
  "16000 20000 300 500 100 run_server_3.sh"
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
    print(999999)
")
        else
            VRAM_USAGE=0
        fi

        if [ "$VRAM_USAGE" -lt 3000 ]; then
            echo "GPU VRAM is clear (Current: ${VRAM_USAGE} MiB)."
            return 0
        fi
        
        echo "GPU VRAM still occupied (${VRAM_USAGE} MiB). Waiting 15s... ($((RETRY_COUNT+1))/$MAX_RETRIES)"
        sleep 15
        RETRY_COUNT=$((RETRY_COUNT + 1))
        pkill -9 -f atom
    done
    return 1
}

# --- main ---
for CASE_LINE in "${TEST_CASES[@]}"; do
    # 解析參數：前五個是測試設定，第六個是腳本名稱
    read -r MIN_IN MAX_IN MIN_OUT MAX_OUT CONC TARGET_SERVER_SCRIPT <<< "$CASE_LINE"
    
    echo "Deep cleaning before case: $CASE_LINE"
    pkill -9 -f atom
    pkill -9 -f "atom.entrypoints.openai_server"
    pkill -9 -f "multiprocessing.spawn" 
    sleep 5

    wait_for_gpu_clear

    # Export 參數給子腳本
    export MIN_IN MAX_IN MIN_OUT MAX_OUT CONC

    # 啟動指定的 Server 腳本
    echo ">>> Starting Test Case with Script: $TARGET_SERVER_SCRIPT"
    echo ">>> Parameters: In[$MIN_IN-$MAX_IN], Out[$MIN_OUT-$MAX_OUT], Conc:$CONC"
    
    # 執行指定的腳本
    bash "$TARGET_SERVER_SCRIPT" &
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
        if [ $WAIT_SEC -gt 420 ]; then # 考慮到有些腳本啟動較慢，放寬到 7 分鐘
            echo "Server start timeout."
            break
        fi
    done

    # 執行 Client
    if curl -s http://localhost:8000/health > /dev/null; then
        bash run_kunlun_client.sh "$MIN_IN" "$MAX_IN" "$MIN_OUT" "$MAX_OUT" "$CONC"
    else
        echo "Skipping client benchmark due to server startup failure."
    fi

    # 結束後清理
    echo "Round finished. Starting deep cleanup..."
    kill $SERVER_PID 2>/dev/null
    pkill -9 -f atom
    sleep 20
done

echo "All Atom benchmarks completed. Logs at $SERVER_LOG_DIR"