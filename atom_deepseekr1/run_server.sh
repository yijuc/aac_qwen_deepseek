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

MODEL="/shared/amdgpu/home/share/deepseek/DeepSeek-R1"

python -m atom.entrypoints.openai_server --model $MODEL -tp $TP --block-size 16 --kv_cache_dtype fp8