#!/bin/bash
# ====== Setting TP, Profiler switch =====
TP=8  # or TP=4
enable_profiler=0
# ========================

unset HIP_VISIBLE_DEVICES
if [ "$TP" = "8" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ "$TP" = "4" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3
else
    echo "Unsupported TP value: $TP"
    exit 1
fi

client_log_dir="/workdir/atom_deepseekr1/logs_vultr_kunlun_benchmark_isSLA"
mkdir -p ${client_log_dir}
log_tag="results"

log_file=${1:-"benchmark_tp${TP}_${log_tag}.log"}
csv_file=${2:-"benchmark_tp${TP}_${log_tag}.csv"}

if [ "$enable_profiler" = "1" ]; then
    log_file="profiler_${log_file}"
    csv_file="profiler_${csv_file}"
fi

log_file="${client_log_dir}/${log_file}"
csv_file="${client_log_dir}/${csv_file}"

if [ ! -f "${csv_file}" ]; then
    echo "TimeStamp,Input_Tokens,Output_Tokens,Max_Concurrency,Num_Prompts,Request_throughput_req_s,Mean_TTFT_ms,Mean_TPOT_ms,Token_Throughput" > "${csv_file}"
fi

# MODEL="/shared/amdgpu/home/share/deepseek/DeepSeek-R1-0528"
MODEL="/mnt/nfs/RAID/shared/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52/"

echo "Input_Tokens,Output_Tokens,Max_Concurrency,Num_Prompts,Request_throughput_req_s,Mean_TTFT_ms,Mean_TPOT_ms,Token_Throughput" > ${csv_file}


PORT=8000
KUNLUN_DIR="/opt/kunlun-benchmark"

export HF_OFFLINE=1
# ================= 測試參數組合 =================
# 格式: MIN_INPUT MAX_INPUT MIN_OUTPUT MAX_OUTPUT MAXCONC
INPUT_OUTPUT_COMBOS=(
#   "800 1000 1600 2000 400"
#   "3000 3600 300 500 100"
#   "3600 4400 1800 2200 200"
#   "11000 15000 2500 2900 80"
   "16000 20000 300 500 1"
)

# ================= 主迴圈 =================
for COMBO in "${INPUT_OUTPUT_COMBOS[@]}"; do
    read -r MIN_INPUT MAX_INPUT MIN_OUTPUT MAX_OUTPUT CONC <<< "$COMBO"
    
    num_prompts=$((CONC * 4))
    
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    LOG_FILE="${client_log_dir}/benchmark_${timestamp// /_}.log"
    temp_output=$(mktemp) 

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running benchmark at: ${timestamp}" | tee -a "$LOG_FILE"
    echo "  Config: In[${MIN_INPUT}-${MAX_INPUT}], Out[${MIN_OUTPUT}-${MAX_OUTPUT}]" | tee -a "$LOG_FILE"
    echo "  Concurrency: ${CONC}, Prompts: ${num_prompts}" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

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
        #--is_sla True \
        #--sla_decode 50 \
        #--sla_prefill 3000 \
        2>&1 | tee -a "$LOG_FILE" | tee "$temp_output"

    LATEST_JSON=$(ls -t ${client_log_dir}/*.json 2>/dev/null | head -n 1)

    if [ -z "$LATEST_JSON" ]; then
        echo "Warning: No JSON result found in ${client_log_dir}"
        request_throughput="N/A"
        mean_ttft="N/A"
        mean_tpot="N/A"
        token_throughput="N/A"
    else
        echo "Parsing result from: $LATEST_JSON"
        
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
    tps = res.get('total_tokens_per_second(tps)', 'N/A')

    print(f'{rps},{ttft},{tpot},{tps}')

except Exception as e:
    sys.stderr.write(f'Python Parse Error: {e}\n')
    print('N/A,N/A,N/A,N/A')
" "$LATEST_JSON")

        IFS=',' read -r request_throughput mean_ttft mean_tpot token_throughput <<< "$METRICS"
    fi

    echo "${timestamp},${MIN_INPUT},${MAX_INPUT},${MIN_OUTPUT},${MAX_OUTPUT},${CONC},${num_prompts},${request_throughput},${mean_ttft},${mean_tpot},${token_throughput}" >> "$csv_file"

    echo "----------------------------------------------------------------"
    echo "Run Complete." 
    echo "File: $LATEST_JSON"
    echo "RPS: $request_throughput | TTFT: $mean_ttft | TPOT: $mean_tpot"
    echo "----------------------------------------------------------------"

    rm -f "$temp_output"
    sleep 5
done
