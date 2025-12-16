#!/bin/bash
# ====== Setting TP =====
TP=8  # or TP=4
# ========================

if [ "$TP" = "8" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ "$TP" = "4" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3
else
    echo "Unsupported TP value: $TP"
    exit 1
fi

log_dir="logs"
mkdir -p ${log_dir}
log_file=${1:-"benchmark_results.log"}
csv_file=${2:-"benchmark_tp${TP}_results.csv"}

log_file="${log_dir}/${log_file}"
csv_file="${log_dir}/${csv_file}"

if [ -f ${log_file} ]; then
    echo "Log file ${log_file} already exists. Please remove it before running the benchmark."
    exit 1
fi
if [ -f ${csv_file} ]; then
    echo "CSV file ${csv_file} already exists. Please remove it before running the benchmark."
    exit 1
fi

MODEL="/shared/amdgpu/home/share/deepseek/DeepSeek-R1"

echo "Input_Tokens,Output_Tokens,Max_Concurrency,Num_Prompts,Mean_TTFT_ms,Mean_TPOT_ms,Token_Throughput" > ${csv_file}

PORT=8000
configs=(
    "1000 1000 4"
    "1000 1000 8"
    "1000 1000 16"
    "1000 1000 32"
    "1000 1000 64"
    "1000 1000 128"
)

for config in "${configs[@]}"; do
    read ISL OSL CONC <<< "$config"
    num_prompts=$((CONC * 10))
    RESULT_FILENAME="deepseek_r1_FP8_tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}_infrrate"
    
    echo "" | tee -a ${log_file}
    echo "========================================" | tee -a ${log_file}
    echo "Running benchmark:" | tee -a ${log_file}
    echo "  Input tokens: ${ISL}" | tee -a ${log_file}
    echo "  Output tokens: ${OSL}" | tee -a ${log_file}
    echo "  Max concurrency: ${CONC}" | tee -a ${log_file}
    echo "  Num prompts: ${num_prompts}" | tee -a ${log_file}
    echo "  Resquest rate: inf" | tee -a ${log_file}
    echo "  Started at: $(date)" | tee -a ${log_file}
    echo "========================================" | tee -a ${log_file}
    
    temp_output=$(mktemp)
    python /root/bench_serving/benchmark_serving.py \
    --model=$MODEL --backend=vllm --base-url=http://localhost:$PORT \
    --dataset-name=random \
    --random-input-len=${ISL} --random-output-len=${OSL} \
    --random-range-ratio 1 \
    --num-prompts=$num_prompts \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --result-dir=${log_dir} --result-filename=$RESULT_FILENAME.json \
    --percentile-metrics="ttft,tpot,itl,e2el" 2>&1 | tee -a ${log_file} | tee ${temp_output}

    mean_ttft=$(grep "Mean TTFT (ms):" ${temp_output} | tail -1 | awk '{print $4}')
    mean_tpot=$(grep "Mean TPOT (ms):" ${temp_output} | tail -1 | awk '{print $4}')
    token_throughput=$(grep "Total Token throughput (tok/s):" ${temp_output} | tail -1 | awk '{print $5}')
    
    echo "${ISL},${OSL},${CONC},${num_prompts},${mean_ttft},${mean_tpot},${token_throughput}" >> ${csv_file}
    
    rm -f ${temp_output}
    
    echo "" | tee -a ${log_file}
    echo "Completed at: $(date)" | tee -a ${log_file}
    echo "========================================" | tee -a ${log_file}
    
    sleep 3
    # done
done
