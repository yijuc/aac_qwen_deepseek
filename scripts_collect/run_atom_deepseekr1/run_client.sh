#!/bin/bash
source ./env_config.sh

# 接收參數
TP=$1; ISL=$2; OSL=$3; CONC=$4
num_prompts=$((CONC * 10))

# 處理 Profiler 相關參數
PROFILER_CLIENT_ARGS=""
RESULT_FILENAME="tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}"

if [ "$ENABLE_PROFILER" = "1" ]; then
    RESULT_FILENAME="${RESULT_FILENAME}_profiler"
    PROFILER_CLIENT_ARGS=" --profile"
fi

timestamp=$(date "+%Y-%m-%d %H:%M:%S")

# 輸出測試資訊 (對應你的 log 格式)
echo "========================================"
echo "Running benchmark at: ${timestamp}"
echo "  Input tokens: ${ISL}"
echo "  Output tokens: ${OSL}"
echo "  Max concurrency: ${CONC}"
echo "  Num prompts: ${num_prompts}"
echo "  Profiler: ${ENABLE_PROFILER}"
echo "========================================"

temp_output=$(mktemp)

# 執行指令：加入 $PROFILER_CLIENT_ARGS
python -m atom.benchmarks.benchmark_serving \
    --model=$MODEL --backend=vllm --base-url=http://localhost:$PORT \
    --dataset-name=random \
    --random-input-len=${ISL} --random-output-len=${OSL} \
    --random-range-ratio 0.8 \
    --num-prompts=$num_prompts \
    --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos \
    --save-result --result-dir="${LOG_DIR}" --result-filename="${RESULT_FILENAME}.json" \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    ${PROFILER_CLIENT_ARGS} 2>&1 | tee ${temp_output}

# --- 解析與寫入 CSV (邏輯同前) ---
REQ_TPS=$(grep -i "Request throughput (req/s):" ${temp_output} | tail -1 | awk '{print $NF}')
TTFT=$(grep "Mean TTFT (ms):" ${temp_output} | tail -1 | awk '{print $4}')
TPOT=$(grep "Mean TPOT (ms):" ${temp_output} | tail -1 | awk '{print $4}')
TOK_TPS=$(grep "Total Token throughput (tok/s):" ${temp_output} | tail -1 | awk '{print $5}')

[[ ! -f "$FINAL_CSV" ]] && echo "TimeStamp,TP,ISL,OSL,CONC,Num_Prompts,Req_TPS,TTFT,TPOT,Token_TPS" > "$FINAL_CSV"
echo "${timestamp},${TP},${ISL},${OSL},${CONC},${num_prompts},${REQ_TPS},${TTFT},${TPOT},${TOK_TPS}" >> "$FINAL_CSV"

rm -f ${temp_output}