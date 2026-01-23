#!/bin/bash
# ====== Setting Profiler switch =====
enable_profiler=0

# ====== Setting TP, Model path, Kunlun-benchmark directory, Port, log directory =====
# TP: only used for metadata/filename, will follow the actual server setting
TP=4
KUNLUN_DIR="/opt/kunlun-benchmark"
# Model path configuration
# MODEL="/mnt/nfs/RAID/shared/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969/"
MODEL="/dev/shm/Qwen3-235B-A22B-Instruct-2507-FP8/"
# Port
PORT=8000
# Directory for storing client logs
client_log_dir="/workdir/eveline/atom_qwen235b/logs_kunlun_qwen_202601200319_max-model-len_16384_max-num-batched-tokens_20k"
# ========================================

mkdir -p ${client_log_dir}
# Define result file names (Defaults to benchmark_tpX_results.log/csv if not provided as arguments)
log_file=${1:-"benchmark_tp${TP}_results.log"}
csv_file=${2:-"benchmark_tp${TP}_results.csv"}

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

unset HF_OFFLINE KUNLUN_RANDOM
export HF_OFFLINE=1
export KUNLUN_RANDOM=0
# ================= Test Parameter Combinations =================
# Format: "MIN_INPUT MAX_INPUT MIN_OUTPUT MAX_OUTPUT CONC"
INPUT_OUTPUT_COMBOS=(
#   "800 1000 1600 2000 512"
#   "3000 3600 300 500 125"
#   "3600 4400 1800 2200 257"
  "11000 15000 2500 2900 1"
#    "16000 20000 300 500 26" # tp4, tp8 failed
)

# ================= Main Loop =================
for COMBO in "${INPUT_OUTPUT_COMBOS[@]}"; do
    # 1. Parse parameters from current combo
    read -r MIN_INPUT MAX_INPUT MIN_OUTPUT MAX_OUTPUT CONC <<< "$COMBO"

    # Calculate number of total prompts (Set to 4x concurrency)
    num_prompts=$((CONC * 4))

    timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
    LOG_FILE="${client_log_dir}/benchmark_${timestamp// /_}.log"
    temp_output=$(mktemp) # Create a temporary file for data extraction

    # 2. Display start message
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Host: $(hostname)" | tee -a "$LOG_FILE"
    echo "Envs: " | tee -a "$LOG_FILE"
    printenv | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running benchmark at: ${timestamp}" | tee -a "$LOG_FILE"
    echo "  Config: In[${MIN_INPUT}-${MAX_INPUT}], Out[${MIN_OUTPUT}-${MAX_OUTPUT}]" | tee -a "$LOG_FILE"
    echo "  Concurrency: ${CONC}, Prompts: ${num_prompts}" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # 3. Execute Kunlun Benchmark
    # Redirect stderr to stdout to ensure all output is captured for logging
    CMD="
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
    "
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running Command: $CMD" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    eval $CMD 2>&1 | tee -a "$LOG_FILE" | tee "$temp_output"

    # 4. Data Extraction
    # PATTERN FOR SEARCH: TP, ISL, OSL, CONC, num_prompts
    # Attention: Kunlun filename, input is $MAX_INPUT&$MIN_INPUT, output is $MAX_OUTPUT&$MIN_OUTPUT
    FILE_PATTERN="*_normal_distribution_unknown_server_vllm_tp-${TP}_${MAX_INPUT}\&${MIN_INPUT}_${MAX_OUTPUT}\&${MIN_OUTPUT}_${CONC}_${num_prompts}_*_ai_perf_benchmark.json"

    echo "Searching for pattern: ${FILE_PATTERN}"

    # Search the latest json can match pattern
    LATEST_JSON=$(ls -t ${client_log_dir}/${FILE_PATTERN} 2>/dev/null | head -n 1)

    if [ -z "$LATEST_JSON" ] || [ ! -f "$LATEST_JSON" ]; then
        echo "Error: Could not find JSON file matching current parameters."
        echo "Expected Pattern: ${FILE_PATTERN}"
        # Fallback: If cannot match match, use latest json file.
        LATEST_JSON=$(ls -t ${client_log_dir}/*.json 2>/dev/null | head -n 1)
        echo "Using fallback (latest overall JSON): $LATEST_JSON"
    else
        echo "Successfully matched JSON: $(basename "$LATEST_JSON")"
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

unset HF_OFFLINE KUNLUN_RANDOM