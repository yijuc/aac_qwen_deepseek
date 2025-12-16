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

MODEL="/shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969"

rm -rf /root/.cache/atom/

python -m atom.entrypoints.openai_server --model ${MODEL} -tp 4 --enable-expert-parallel --kv_cache_dtype fp8