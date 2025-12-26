#!/bin/bash
# ====== Setting TP, Profiler switch, output gemm switch =====
TP=8  # TP=8 or 4
enable_profiler=0
enable_output_gemm=0

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

# ===== Log file ======
# unset AMD_LOG_LEVEL
# export AMD_LOG_LEVEL=3
# export AMD_LOG_LEVEL_FILE="/workdir/atom_qwen235b/logs_amd_level3/qwen3_235b_a22b_instrct_FP8_TP${TP}_isl1000_osl1000_conc256_infrrate_loglv3.log"

unset AITER_LOG_MORE
export AITER_LOG_MORE=2

server_log_dir="/workdir/atom_qwen235b/logs_cpu_setting_untuned"
mkdir -p ${server_log_dir}
server_log_file="${server_log_dir}/server_running_qwen3_235b_a22b_instrct_FP8_TP${TP}_isl1000_osl1000_conc256_infrrate_aiter_logv2.log"

# =====================


unset HIPBLASLT_LOG_FILE HIPBLASLT_LOG_MASK
if [ "$enable_output_gemm" = "1" ]; then
    export HIPBLASLT_LOG_FILE="${server_log_dir}/qwen3_235b_a22b_instrct_FP8_TP${TP}_isl1000_osl1000_conc256_infrrate_kernel.log"
    hipblaslt_log_dir=$(dirname "$HIPBLASLT_LOG_FILE")
    mkdir -p "$hipblaslt_log_dir"
    export HIPBLASLT_LOG_MASK=32
fi

profiler_args=""
if [ "$enable_profiler" = "1" ]; then
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=1  
    profiler_dir="${server_log_dir}/qwen3_235b_a22b_instrct_2507_FP8_TP${TP}_isl1000_osl1000_conc256_infrrate"  
    mkdir -p $profiler_dir
    profiler_args=" --torch-profiler-dir ${profiler_dir}"
fi

MODEL="/shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969"

rm -rf /root/.cache/atom/

CMD="
python -m atom.entrypoints.openai_server --model ${MODEL} -tp $TP --enable-expert-parallel --kv_cache_dtype fp8 $profiler_args
"

{
    echo "Running command: $CMD"
    $CMD

# } 2>&1 | tee "$server_log_file"
} >> "$server_log_file" 2>&1