#!/bin/bash
# ====== Setting TP, Profiler switch =====
TP=8 # only used for filename, will follow the TP server setting
enable_profiler=0
# ========================
KUNLUN_DIR="/workdir/eveline/aac_qwen_deepseek/kunlun-benchmark"

unset HIP_VISIBLE_DEVICES
if [ "$TP" = "8" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ "$TP" = "4" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3
else
    echo "Unsupported TP value: $TP"
    exit 1
fi

client_log_dir="/workdir/eveline/atom_qwen235b/logs_vultr_kunlun_benchmark_isSLA"
mkdir -p ${client_log_dir}

log_file=${1:-"benchmark_tp${TP}_results.log"}
csv_file=${2:-"benchmark_tp${TP}_results.csv"}

if [ "$enable_profiler" = "1" ]; then
    log_file="profiler_${log_file}"
    csv_file="profiler_${csv_file}"
fi

log_file="${client_log_dir}/${log_file}"
csv_file="${client_log_dir}/${csv_file}"

if [ ! -f "${csv_file}" ]; then
    echo "TimeStamp,MIN_Input,MAX_Input,MIN_Output,MAX_Output,Max_Concurrency,Num_Prompts,Request_throughput_req_s,Mean_TTFT_ms,Mean_TPOT_ms,Token_Throughput" > "${csv_file}"
fi

# MODEL="/shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969"
MODEL="/mnt/nfs/RAID/shared/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969/"

PORT=8000
KUNLUN_DIR="/opt/kunlun-benchmark"

export HF_OFFLINE=1

# ================= 測試參數組合 =================
# 格式: MIN_INPUT MAX_INPUT MIN_OUTPUT MAX_OUTPUT CONC
INPUT_OUTPUT_COMBOS=(
#    "800 1000 1600 2000 512"
#    "3000 3600 300 500 256"
#    "3600 4400 1800 2200 256"
   "11000 15000 2500 2900 1" # tp4, tp8 failed
   "16000 20000 300 500 1" # tp4, tp8 failed
)

# ================= 主迴圈 =================
for COMBO in "${INPUT_OUTPUT_COMBOS[@]}"; do
    # 1. 解析參數
    read -r MIN_INPUT MAX_INPUT MIN_OUTPUT MAX_OUTPUT CONC <<< "$COMBO"
    
    # 計算 Prompt 數量
    num_prompts=$((CONC * 4))
    
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    LOG_FILE="${client_log_dir}/benchmark_${timestamp// /_}.log"
    temp_output=$(mktemp) # 建立暫存檔來分析數據

    # 2. 顯示開始訊息
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running benchmark at: ${timestamp}" | tee -a "$LOG_FILE"
    echo "  Config: In[${MIN_INPUT}-${MAX_INPUT}], Out[${MIN_OUTPUT}-${MAX_OUTPUT}]" | tee -a "$LOG_FILE"
    echo "  Concurrency: ${CONC}, Prompts: ${num_prompts}" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # 3. 執行 Kunlun Benchmark (並同時輸出到螢幕、Log檔、與暫存檔)
    # 注意：這裡使用了 2>&1 把錯誤訊息也導向 stdout，方便 grep 抓取
    ${KUNLUN_DIR}/kunlun-benchmark vllm server \
        --port $PORT \
        --work_mode manual \
        --max_input_len $MAX_INPUT \
        --min_input_len $MIN_INPUT \
        --max_output_len $MAX_OUTPUT \
        --min_output_len $MIN_OUTPUT \
        --concurrency ${CONC} \
        --query_num ${num_prompts} \
        --result_dir $client_log_dir \
        --model_path $MODEL \
        --is_sla True \
        --sla_decode 50 \
        --sla_prefill 3000 \
        --tp $TP \
        --extra_server_args "--enable-expert-parallel " \
        2>&1 | tee -a "$LOG_FILE" | tee "$temp_output"

    # 4. 數據擷取 (Data Extraction)
    LATEST_JSON=$(ls -t ${client_log_dir}/*.json 2>/dev/null | head -n 1)

    if [ -z "$LATEST_JSON" ]; then
        echo "Warning: No JSON result found in ${client_log_dir}"
        # 設定預設值以免 CSV 錯位
        request_throughput="N/A"
        mean_ttft="N/A"
        mean_tpot="N/A"
        token_throughput="N/A"
    else
        echo "Parsing result from: $LATEST_JSON"
        
        # 使用 Python 讀取 Kunlun 特有的 JSON 結構
        METRICS=$(python3 -c "
import sys, json

try:
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    # 進入 perf_result 層級
    res = data.get('perf_result', {})
    
    # 1. Request Throughput (RPS)
    rps = res.get('queries_per_second', 'N/A')
    
    # 2. Mean TTFT (Prefill Time)
    ttft = res.get('average_prefill_time', 'N/A')
    
    # 3. Mean TPOT (Decode Time)
    tpot = res.get('average_decode_time', 'N/A')
    
    # 4. Total Token Throughput
    # 注意：JSON key 包含括號，必須完整匹配
    tps = res.get('total_tokens_per_second(tps)', 'N/A')

    print(f'{rps},{ttft},{tpot},{tps}')

except Exception as e:
    # 如果 JSON 壞掉或找不到 key，印出錯誤以便除錯 (寫在 stderr)，並回傳 N/A
    sys.stderr.write(f'Python Parse Error: {e}\n')
    print('N/A,N/A,N/A,N/A')
" "$LATEST_JSON")

        # 將 Python 吐回來的 CSV 字串拆解成變數
        IFS=',' read -r request_throughput mean_ttft mean_tpot token_throughput <<< "$METRICS"
    fi

    # 5. 寫入 CSV 
    echo "${timestamp},${MIN_INPUT},${MAX_INPUT},${MIN_OUTPUT},${MAX_OUTPUT},${CONC},${num_prompts},${request_throughput},${mean_ttft},${mean_tpot},${token_throughput}" >> "$csv_file"

    echo "----------------------------------------------------------------"
    echo "Run Complete." 
    echo "File: $LATEST_JSON"
    echo "RPS: $request_throughput | TTFT: $mean_ttft | TPOT: $mean_tpot"
    echo "----------------------------------------------------------------"

    # 6. 清理與冷卻
    rm -f "$temp_output"
    sleep 5
done