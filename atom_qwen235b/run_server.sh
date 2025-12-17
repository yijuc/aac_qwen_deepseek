#!/bin/bash
# ====== Setting TP, Profiler swith =====
TP=8  # or TP=4
enable_profiler=1
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

profiler_args=""
if [ "$enable_profiler" = "1" ]; then
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=1  
    profiler_dir="/workdir/atom_qwen235b/qwen3_235b_a22b_instrct_2507_FP8_TP${TP}_isl1000_osl1000_conc256_infrrate"  
    mkdir -p $profiler_dir
    profiler_args=" --torch-profiler-dir ${profiler_dir}"
fi

MODEL="/shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969"

rm -rf /root/.cache/atom/

python -m atom.entrypoints.openai_server --model ${MODEL} -tp $TP --enable-expert-parallel --kv_cache_dtype fp8 $profiler_args