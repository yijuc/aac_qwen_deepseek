#!/bin/bash
# ====== Setting Profiler switch =====
enable_profiler=0

# ====== Setting TP, Model path, Kunlun-benchmark directory, Port, log directory =====
# TP: only used for metadata/filename, will follow the actual server setting
TP=8
KUNLUN_DIR="/opt/kunlun-benchmark"
# Model path configuration
MODEL="/mnt/md0/models/DeepSeek-R1-0528"
# Port
PORT=8000
# Directory for storing client logs
client_log_dir="/workdir/vllm_qwen235b/logs_dsr1_kunlun_0114"
log_tag="results"
# ========================================

# Inspect GPU
BACKEND="CPU"
if command -v nvidia-smi > /dev/null; then
    echo "NVIDIA environment detected"
    BACKEND="NVIDIA"
elif command -v rocminfo > /dev/null; then
    echo "ROCm environment detected"
    BACKEND="ROCM"
else
    echo "No supported GPU environment detected"
fi

if [ "$BACKEND" == "NVIDIA" ]; then
    # Reset GPU visibility settings
    unset CUDA_VISIBLE_DEVICES
    if [ "$TP" = "8" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    elif [ "$TP" = "4" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [ "$TP" = "2" ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    else
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    fi
elif [ "$BACKEND" == "ROCM" ]; then
    # Reset GPU visibility settings
    unset HIP_VISIBLE_DEVICES
    if [ "$TP" = "8" ]; then
        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    elif [ "$TP" = "4" ]; then
        export HIP_VISIBLE_DEVICES=0,1,2,3
    elif [ "$TP" = "2" ]; then
        export HIP_VISIBLE_DEVICES=0,1
    else
        export HIP_VISIBLE_DEVICES=0,1,2,3
    fi
fi

mkdir -p ${client_log_dir}
# Define result file names (Defaults to benchmark_tpX_results.log/csv if not provided as arguments)
log_file=${1:-"benchmark_tp${TP}_${log_tag}.log"}
csv_file=${2:-"benchmark_tp${TP}_${log_tag}.csv"}

# Prefix filenames if profiler is enabled
if [ "$enable_profiler" = "1" ]; then
    log_file="profiler_${log_file}"
    csv_file="profiler_${csv_file}"
fi

log_file="${client_log_dir}/${log_file}"
csv_file="${client_log_dir}/${csv_file}"

# Initialize CSV header if file does not exist
if [ ! -f "${csv_file}" ]; then
    echo "TimeStamp,MIN_Input,MAX_Input,MIN_Output,MAX_Output,Num_Prompts,Generate_Token_Throughput,Total_Token_Throughput,Request_throughput_req_s,Concurrency,Mean_TTFT_ms,Mean_TPOT_ms" > "${csv_file}"
fi

export HF_OFFLINE=1
# ================= Test Parameter Combinations =================
# Format: "MIN_INPUT MAX_INPUT MIN_OUTPUT MAX_OUTPUT CONC"
INPUT_OUTPUT_COMBOS=(
#   "800 1000 1600 2000 400"
#   "3000 3600 300 500 100"
#   "3600 4400 1800 2200 200"
#   "11000 15000 2500 2900 80"
   "16000 20000 300 500 1"
)

# ================= Main Loop =================
for COMBO in "${INPUT_OUTPUT_COMBOS[@]}"; do
    # 1. Parse parameters from current combo
    read -r MIN_INPUT MAX_INPUT MIN_OUTPUT MAX_OUTPUT CONC <<< "$COMBO"

    # Calculate number of total prompts (Set to 4x concurrency)
    num_prompts=$((CONC * 4))

    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    LOG_FILE="${client_log_dir}/benchmark_${timestamp// /_}.log"
    temp_output=$(mktemp) # Create a temporary file for data extraction

    # 2. Display start message
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running benchmark at: ${timestamp}" | tee -a "$LOG_FILE"
    echo "  Config: In[${MIN_INPUT}-${MAX_INPUT}], Out[${MIN_OUTPUT}-${MAX_OUTPUT}]" | tee -a "$LOG_FILE"
    echo "  Concurrency: ${CONC}, Prompts: ${num_prompts}" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # 3. Execute Kunlun Benchmark
    # Redirect stderr to stdout to ensure all output is captured for logging
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
        --is_sla False \
        --sla_decode 50 \
        --sla_prefill 3000 \
        --tp $TP \
        --extra_server_args "--enable-expert-parallel " \
        2>&1 | tee -a "$LOG_FILE" | tee "$temp_output"

    # 4. Data Extraction
    # Find the most recently generated JSON result file
    LATEST_JSON=$(ls -t ${client_log_dir}/*.json 2>/dev/null | head -n 1)

    if [ -z "$LATEST_JSON" ]; then
        echo "Warning: No JSON result found in ${client_log_dir}"
        # Set default values to prevent CSV misalignment
        request_throughput="N/A"
        mean_ttft="N/A"
        mean_tpot="N/A"
        token_throughput="N/A"
        generate_token_throughput="N/A"
    else
        echo "Parsing result from: $LATEST_JSON"

        # Use Python to read the specific Kunlun JSON structure
        METRICS=$(python3 -c "
import sys, json
try:
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)

    # Navigate to perf_result level
    res = data.get('perf_result', {})

    # 1. Request Throughput (RPS)
    rps = res.get('queries_per_second', 'N/A')

    # 2. Mean TTFT (Average Prefill Time)
    ttft = res.get('average_prefill_time', 'N/A')

    # 3. Mean TPOT (Average Decode Time)
    tpot = res.get('average_decode_time', 'N/A')

    # 4. Total Token Throughput (TPPS)
    tps = res.get('total_tokens_per_second(tps)', 'N/A')

    # 5. Generate Token Throughput (GTPS)
    gtps = res.get('generate_tokens_per_second(tps)', 'N/A')

    print(f'{rps},{ttft},{tpot},{tps},{gtps}')

except Exception as e:
    # Print parse error to stderr and return N/A placeholders to stdout
    sys.stderr.write(f'Python Parse Error: {e}\n')
    print('N/A,N/A,N/A,N/A,N/A')
" "$LATEST_JSON")

        # Split the CSV string returned by Python into bash variables
        IFS=',' read -r request_throughput mean_ttft mean_tpot token_throughput generate_token_throughput <<< "$METRICS"
    fi

    # 5. Write results to CSV
    echo "${timestamp},${MIN_INPUT},${MAX_INPUT},${MIN_OUTPUT},${MAX_OUTPUT},${num_prompts},${generate_token_throughput},${token_throughput},${request_throughput},${CONC},${mean_ttft},${mean_tpot}" >> "$csv_file"

    echo "----------------------------------------------------------------"
    echo "Run Complete."
    echo "File: $LATEST_JSON"
    echo "RPS: $request_throughput | TTFT: $mean_ttft | TPOT: $mean_tpot"
    echo "----------------------------------------------------------------"

    # 6. Cleanup and cooldown period
    rm -f "$temp_output"
    sleep 5
done