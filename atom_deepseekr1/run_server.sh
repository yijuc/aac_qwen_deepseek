#!/bin/bash
# ====== Setting TP, Profiler swith =====
TP=8  # or TP=4 (Currently deepseek-r1 cannot run with TP4)
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

profiler_args=""
if [ "$enable_profiler" = "1" ]; then
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=1  
    profiler_dir="/workdir/atom_deepseekr1/torch_args_atom_fp8_tp${TP}_deepseek_r1_in1k_out100_conc128"  
    mkdir -p $profiler_dir
    profiler_args=" --torch-profiler-dir ${profiler_dir}"
fi

MODEL="/shared/amdgpu/home/share/deepseek/DeepSeek-R1"
rm -rf /root/.cache/atom/
python -m atom.entrypoints.openai_server --model $MODEL -tp $TP --block-size 16 --kv_cache_dtype fp8 $profiler_args --enable-chunked-prefill