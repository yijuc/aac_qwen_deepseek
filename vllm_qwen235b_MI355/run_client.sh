#!/bin/bash
# ====== Setting TP, Profiler switch =====
TP=8
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

client_log_dir="/workdir/vllm_qwen235b/logs_0114_env"
log_tag="results"
mkdir -p ${client_log_dir}
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

# MODEL="/shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969"
MODEL="/dev/shm/Qwen3-235B-A22B-Instruct-2507-FP8/"

PORT=8000
configs=(
    # "1000 1000 256"
    "1024 1024 128"
    # "4000 1000 128"
    # "4000 1000 64"
    # "10000 1000 64"
    # "10000 1000 32"
    # 3.0~3.6k/0.3~0.5k
    # "3300 400 256"
    # 16~20k/0.3~0.5k
    # "18000 400 2"
    # 0.8~1k/1.6~2k
    # "900 1800 32"
    # 3.6~4.4k/1.8~2.2k
    # "4000 2000 128"
    # 11~15k/2.5~2.9k
    # "13000 2700 64"
)

for config in "${configs[@]}"; do
    read ISL OSL CONC <<< "$config"
    num_prompts=$((CONC * 4))
    RESULT_FILENAME="${client_log_dir}/qwen3_235b_a22b_instrct_2507_FP8_TP${TP}_isl${ISL}_osl${OSL}_conc${CONC}_infrrate"
    
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
    
    python -m vllm.entrypoints.cli.main bench serve \
        --host localhost \
        --port 8000 \
        --model ${MODEL} \
        --dataset-name random \
        --random-input-len $ISL \
        --random-output-len $OSL \
        --max-concurrency $CONC \
        --num-prompts $num_prompts \
        --seed 123 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --ignore-eos \
        2>&1 | tee -a ${log_file} | tee ${temp_output}

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
