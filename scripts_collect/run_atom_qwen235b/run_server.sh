#!/bin/bash
source ./env_config.sh

# 接收主腳本傳來的 TP
TP=$1
EXTRA_ARGS=$2

# 動態設定 GPU
unset HIP_VISIBLE_DEVICES
if [ "$TP" = "8" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    export HIP_VISIBLE_DEVICES=0,1,2,3
fi

echo "Starting Server with TP=${TP} on GPUs: ${HIP_VISIBLE_DEVICES}"

rm -rf /root/.cache/atom/

CMD="python -m atom.entrypoints.openai_server \
    --model ${MODEL} \
    -tp ${TP} \
    --port ${PORT} \
    --enable-expert-parallel \
    --kv_cache_dtype fp8 \
    --max-num-batched-tokens 20000 \
    --max-model-len 16384 \
    ${EXTRA_ARGS}"

echo "------------------------------------------------"
echo "準備執行 Server 指令:"
echo "$CMD"
echo "------------------------------------------------"

$CMD