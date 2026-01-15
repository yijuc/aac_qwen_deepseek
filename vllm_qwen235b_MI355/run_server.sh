#!/bin/bash
# ====== Setting TP, Profiler switch, output gemm switch =====
TP=8  # TP=8 or 4
enable_profiler=0
enable_output_gemm=0

# MODEL="/shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969"
MODEL="/dev/shm/Qwen3-235B-A22B-Instruct-2507-FP8/"
server_log_dir="/workdir/vllm_qwen235b/logs_0114_env"

## case
input_len=1000
output_len=1000
concurrency=256
# ========================

# Inspect GPU
BACKEND="CPU"
if command -v nvidia-smi > /dev/null; then
    echo "NVIDIA environment detected"
    BACKEND="NVIDIA"
elif command -v rocminfo > /dev/null; then
    echo "ROCm environment detected"
    BACKEND="ROCM"
else
    echo "No supported GPU environment detected"
fi

if [ "$BACKEND" == "NVIDIA" ]; then
    # Reset GPU visibility settings
    unset CUDA_VISIBLE_DEVICES
    if [ "$TP" = "8" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    elif [ "$TP" = "4" ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [ "$TP" = "2" ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    else
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    fi
elif [ "$BACKEND" == "ROCM" ]; then
    # Reset GPU visibility settings
    unset HIP_VISIBLE_DEVICES
    if [ "$TP" = "8" ]; then
        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    elif [ "$TP" = "4" ]; then
        export HIP_VISIBLE_DEVICES=0,1,2,3
    elif [ "$TP" = "2" ]; then
        export HIP_VISIBLE_DEVICES=0,1
    else
        export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    fi
fi

unset VLLM_ROCM_QUICK_REDUCE_QUANTIZATION VLLM_V1_USE_PREFILL_DECODE_ATTENTION VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_MOE VLLM_USE_TRITON_FLASH_ATTN SAFETENSORS_FAST_GPU
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export SAFETENSORS_FAST_GPU=1

# ===== Log file ======
# unset AMD_LOG_LEVEL
# export AMD_LOG_LEVEL=3
# export AMD_LOG_LEVEL_FILE="/workdir/vllm_qwen235b/logs_amd_level3/qwen3_235b_a22b_instrct_FP8_TP${TP}_isl${input_len}_osl${output_len}_conc${concurrency}_infrrate_loglv3.log"

unset AITER_LOG_MORE
# export AITER_LOG_MORE=2

mkdir -p ${server_log_dir}
# server_log_file="${server_log_dir}/server_running_qwen3_235b_a22b_instrct_FP8_TP${TP}_isl${input_len}_osl${output_len}_conc${concurrency}_infrrate.log"
server_log_file="${server_log_dir}/server_running_qwen3_235b_a22b_instrct_FP8_TP${TP}.log"

# =====================


unset HIPBLASLT_LOG_FILE HIPBLASLT_LOG_MASK
if [ "$enable_output_gemm" = "1" ]; then
    export HIPBLASLT_LOG_FILE="${server_log_dir}/qwen3_235b_a22b_instrct_FP8_TP${TP}_isl${input_len}_osl${output_len}_conc${concurrency}_infrrate_kernel.log"
    hipblaslt_log_dir=$(dirname "$HIPBLASLT_LOG_FILE")
    mkdir -p "$hipblaslt_log_dir"
    export HIPBLASLT_LOG_MASK=32
fi

profiler_args=""
if [ "$enable_profiler" = "1" ]; then
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=1  
    profiler_dir="${server_log_dir}/qwen3_235b_a22b_instrct_2507_FP8_TP${TP}_isl${input_len}_osl${output_len}_conc${concurrency}_infrrate"
    mkdir -p $profiler_dir
    profiler_args=" --torch-profiler-dir ${profiler_dir}"
fi


# rm -rf /root/.cache/vllm/torch_compile_cache/

CMD="
vllm serve ${MODEL} \
    --tensor-parallel-size $TP \
    --enable-expert-parallel \
    --kv-cache-dtype fp8
    --max-num-seqs 512 \
    --max-model-len 32768 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --compilation-config '{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\", \"custom_ops\": [\"-rms_norm\", \"-quant_fp8\", \"-silu_and_mul\"]}' \
    --no-enable-prefix-caching
"

{
    echo "Running command: $CMD"
    eval $CMD

} 2>&1 | tee "$server_log_file"
# } >> "$server_log_file" 2>&1
