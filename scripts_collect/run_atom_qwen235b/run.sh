#!/bin/bash
source ./env_config.sh

# ====== 全域功能開關 ======
ENABLE_GEMM=${ENABLE_GEMM:-0}
ENABLE_PROFILER=${ENABLE_PROFILER:-0}
ENABLE_AITER_LOG=${ENABLE_AITER_LOG:-0}

if [ "$ENABLE_GEMM" = "1" ]; then
    TAG="gemm"; export FINAL_CSV="${LOG_DIR}/benchmark_output_gemm_results.csv"
elif [ "$ENABLE_PROFILER" = "1" ]; then
    TAG="profiler"; export FINAL_CSV="${LOG_DIR}/benchmark_results_profiler.csv"
elif [ "$ENABLE_AITER_LOG" = "1" ]; then
    TAG="aiter"; export FINAL_CSV="${LOG_DIR}/benchmark_output_aiter_results.csv"
else
    TAG="normal"; export FINAL_CSV="${LOG_DIR}/benchmark_results.csv"
fi

# 強力清理函數：確保 GPU 顯存回歸
clean_gpu_completely() {
    echo "--- [Action] 執行深度顯存清理與進程回收 ---"
    # 1. 嘗試優雅關閉
    pkill -2 -f "atom"
    pkill -2 -f "openai_server"
    sleep 5
    
    # 2. 強制殺掉所有相關 Python 與 Atom 進程
    pkill -9 -f "atom"
    pkill -9 -f "openai_server"
    ps -ef | grep python | grep -v grep | grep -v "batch_run.sh" | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    
    echo "正在偵測 GPU 狀態..."
    local retry=0
    while [ $retry -lt 12 ]; do
        # 嘗試抓取第一個可用的數字，並過濾掉所有非數字字元 (處理警告訊息)
        local free_mem_raw=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Free" | awk '{print $4}' | tr -cd '0-9' | head -n 1)
        
        # 如果抓不到數字，嘗試另一種指令檢查是否有進程佔用 GPU
        local gpu_users=$(fuser /dev/dri/renderD* 2>/dev/null)

        # 判斷邏輯：如果有數字且大於 200GB，或者已經沒有任何進程佔用 GPU 裝置
        if [[ -n "$free_mem_raw" && "$free_mem_raw" -gt 200000000000 ]] || [[ -z "$gpu_users" ]]; then
            echo "✅ GPU 已就緒！ (剩餘空間: ${free_mem_raw:-"Unknown"}, 佔用進程: None)"
            return 0
        fi
        
        echo "⏳ GPU 資源仍被佔用，等待中... ($((retry*5))s)"
        sleep 5
        ((retry++))
    done
    echo "⚠️ 警告: 到達等待上限，強制繼續執行..."
}
# 定義測試組合 (TP ISL OSL CONC)
configs=(
    "4 1000 1000 256"
    "4 1000 1000 128"
    "4 4000 1000 128"
    "4 4000 1000 64"
    "4 10000 1000 64"
    "4 10000 1000 32"
    "8 1000 1000 256"
    "8 1000 1000 128"
    "8 4000 1000 128"
    "8 4000 1000 64"
    "8 10000 1000 64"
    "8 10000 1000 32"
)

for config in "${configs[@]}"; do
    read TP ISL OSL CONC <<< "$config"
    [ -z "$CONC" ] && continue

    # 1. 啟動前深度清理 (解決 OOM 的核心)
    clean_gpu_completely

    # 2. 準備路徑
    SUFFIX="${TAG}_tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}"
    SERVER_LOG="${LOG_DIR}/server_${SUFFIX}.log"
    touch "${SERVER_LOG}"

    EXTRA_ARGS=""
    if [ "$ENABLE_PROFILER" = "1" ]; then
        export VLLM_TORCH_PROFILER_WITH_STACK=1
        export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
        PROF_DIR="${LOG_DIR}/prof_${SUFFIX}"
        mkdir -p "$PROF_DIR"
        EXTRA_ARGS=" --torch-profiler-dir ${PROF_DIR}"
    fi

    if [ "$ENABLE_GEMM" = "1" ]; then
        export HIPBLASLT_LOG_FILE="${LOG_DIR}/kernel_${SUFFIX}.log"
        export HIPBLASLT_LOG_MASK=32
    fi

    if [ "$ENABLE_AITER_LOG" = "1" ]; then
        export AITER_LOG_MORE=2
    fi

    # 3. 啟動 Server
    echo ">>> [Mode: $TAG] Starting Server: TP:$TP ISL:$ISL OSL:$OSL CONC:$CONC"
    nohup bash run_server.sh $TP "$EXTRA_ARGS" > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    # 4. 等待就緒
    until grep -q "Uvicorn running" "$SERVER_LOG"; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ Server Failed. 最後 10 行日誌:"
            tail -n 10 "$SERVER_LOG"
            exit 1
        fi
        sleep 5
    done
    echo "🚀 Server READY."

    # 5. 執行壓測
    bash run_client.sh $TP $ISL $OSL $CONC

    # 6. 結束與解析
    echo "正在優雅關閉 Server 並等待數據寫入 (20s)..."
    kill -2 $SERVER_PID
    sleep 20 # 給予 Qwen-235B 足夠時間寫入 Profiler .gz 檔案並回收子進程

    # --- 後處理解析 ---
    if [ "$ENABLE_PROFILER" = "1" ]; then
        GZ_FILE=$(ls ${LOG_DIR}/prof_${SUFFIX}/rank_0/*.gz 2>/dev/null | head -n 1)
        if [ -n "$GZ_FILE" ]; then
            echo "📊 解析 Profiler: $GZ_FILE"
            python3 /workdir/Tools/profiler_json_analysis/parse_torchprofile_trace_to_detailed_op_stats.py -i "$GZ_FILE"
        else
            echo "❌ 警告: 找不到 Profiler .gz 檔案"
        fi
    fi

    if [ "$ENABLE_AITER_LOG" = "1" ]; then
        echo "📜 解析 Aiter Log: $SERVER_LOG"
        python /workdir/atom_qwen235b/parse_aiter.py -i "$SERVER_LOG"
    fi

    # 最後強制清理殘留進程
    kill -9 $SERVER_PID 2>/dev/null
done

echo "--- $TAG 階段測試完成！ ---"