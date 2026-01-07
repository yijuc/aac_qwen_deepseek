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

client_log_dir="/workdir/atom_deepseekr1/logs_dsr1_atom_12241340"
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
MODEL="/dev/shm/DeepSeek-R1-0528"

echo "Input_Tokens,Output_Tokens,Max_Concurrency,Num_Prompts,Request_throughput_req_s,Mean_TTFT_ms,Mean_TPOT_ms,Token_Throughput" > ${csv_file}

PORT=8000
configs=(
    # "1024 1024 4"
    # "1024 1024 8"
    # "1024 1024 16"
    # "1024 1024 32"
    # "1024 1024 64"
    # "1024 1024 128"
    "4096 1024 64"
    "4096 1024 128"
    "10240 1024 32"
    "10240 1024 64"
    # "10240 1024 128"
)

for config in "${configs[@]}"; do
    read ISL OSL CONC <<< "$config"
    num_prompts=$((CONC * 10))
    RESULT_FILENAME="${client_log_dir}/deepseek_r1_FP8_tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}_infrrate"
    profiler_args=""
    if [ "$enable_profiler" = "1" ]; then
        RESULT_FILENAME="${RESULT_FILENAME}_profiler"
        profiler_args=" --profile"
    fi

    timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    echo "" | tee -a ${log_file}
    echo "========================================" | tee -a ${log_file}
    echo "Running benchmark at: ${timestamp}" | tee -a ${log_file}
    echo "  Input tokens: ${ISL}" | tee -a ${log_file}
    echo "  Output tokens: ${OSL}" | tee -a ${log_file}
    echo "  Max concurrency: ${CONC}" | tee -a ${log_file}
    echo "  Num prompts: ${num_prompts}" | tee -a ${log_file}
    echo "  Resquest rate: inf" | tee -a ${log_file}
    echo "========================================" | tee -a ${log_file}
    
    temp_output=$(mktemp)
    # python /root/bench_serving/benchmark_serving.py \
    PYTHONPATH=/ATOM/atom python -m benchmarks.benchmark_serving \
    --model=$MODEL --backend=vllm --base-url=http://localhost:$PORT \
    --dataset-name=random \
    --random-input-len=${ISL} --random-output-len=${OSL} \
    --random-range-ratio 0.8 \
    --num-prompts=$num_prompts \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --result-dir=${client_log_dir} --result-filename=$RESULT_FILENAME.json \
    --percentile-metrics="ttft,tpot,itl,e2el" $profiler_args 2>&1 | tee -a ${log_file} | tee ${temp_output}

    request_throughput=$(grep -i "Request throughput (req/s):" ${temp_output} | tail -1 | awk '{print $NF}')
    mean_ttft=$(grep "Mean TTFT (ms):" ${temp_output} | tail -1 | awk '{print $4}')
    mean_tpot=$(grep "Mean TPOT (ms):" ${temp_output} | tail -1 | awk '{print $4}')
    token_throughput=$(grep "Total Token throughput (tok/s):" ${temp_output} | tail -1 | awk '{print $5}')
    
    echo "${timestamp},${ISL},${OSL},${CONC},${num_prompts},${request_throughput},${mean_ttft},${mean_tpot},${token_throughput}" >> ${csv_file}
    
    rm -f ${temp_output}
    
    echo "" | tee -a ${log_file}
    echo "Completed at: $(date)" | tee -a ${log_file}
    echo "========================================" | tee -a ${log_file}
    
    sleep 3
done
