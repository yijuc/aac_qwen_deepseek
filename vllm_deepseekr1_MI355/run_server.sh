#!/bin/bash
# ====== Setting TP, Profiler switch, output gemm switch =====
TP=${TP:-8}  #(Currently deepseek-r1 cannot run with TP4)
enable_profiler=${enable_profiler:-0}
enable_output_gemm=${enable_output_gemm:-0}
MODEL=${MODEL:-"/dev/shm/DeepSeek-R1-0528"}
SERVER_LOG_DIR=${SERVER_LOG_DIR:-"/dockerx/eveline/vllm_deepseekr1/test"}
CASE_TAG=${CASE_TAG:-""}
# ========================

mkdir -p ${SERVER_LOG_DIR}
if [ -n "$MIN_IN" ]; then
    # 格式：vllm_tp8_in800-1000_out1600-2000_c186
    log_tag="vllm_tp${TP}_in${MIN_IN}-${MAX_IN}_out${MIN_OUT}-${MAX_OUT}_c${CONC}"
else
    # default tag
    log_tag="vllm_fp8_tp${TP}_deepseek_r1"
fi
server_log_file="${SERVER_LOG_DIR}/${log_tag}_server_running.log"

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

unset USE_FASTSAFETENSOR SAFETENSORS_FAST_GPU VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_MHA VLLM_ROCM_USE_AITER_PAGED_ATTN VLLM_USE_AITER_TRITON_ROPE VLLM_ROCM_USE_AITER_RMSNORM VLLM_V1_USE_PREFILL_DECODE_ATTENTION AMDGCN_USE_BUFFER_OPS VLLM_ROCM_USE_AITER_TRITON_LINEAR VLLM_USE_AITER_TRITON_SILU_MUL VLLM_USE_V1 VLLM_USE_TRITON_FLASH_ATTN HF_HOME TRANSFORMERS_CACHE
# VLLM_ROCM_QUICK_REDUCE_QUANTIZATION
export USE_FASTSAFETENSOR=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
export VLLM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export AMDGCN_USE_BUFFER_OPS=1
export VLLM_ROCM_USE_AITER_TRITON_LINEAR=1
export VLLM_USE_AITER_TRITON_SILU_MUL=0
export VLLM_USE_V1=1
export VLLM_USE_TRITON_FLASH_ATTN=1
# export TRANSFORMERS_CACHE="/dev/shm/cache"
# export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="INT4"

## New env frop tp4 (20260205)
unset VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT VLLM_ROCM_QUICK_REDUCE_QUANTIZATION VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
# export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=2048


unset HIPBLASLT_LOG_FILE HIPBLASLT_LOG_MASK
if [ "$enable_output_gemm" = "1" ]; then
    export HIPBLASLT_LOG_FILE="${SERVER_LOG_DIR}/${log_tag}_gemm_output.log"
    hipblaslt_log_dir=$(dirname "$HIPBLASLT_LOG_FILE")
    mkdir -p "$hipblaslt_log_dir"
    export HIPBLASLT_LOG_MASK=32
fi

profiler_args=""
if [ "$enable_profiler" = "1" ]; then
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
    profiler_dir="${SERVER_LOG_DIR}/${log_tag}"
    mkdir -p $profiler_dir
    profiler_args=" --torch-profiler-dir ${profiler_dir}"
fi


rm -rf /root/.cache/vllm/torch_compile_cache/
#     --quantization fp8 \ --enable-chunked-prefill     --max-seq-len-to-capture 16384 \


## smaller
CMD="
vllm serve ${MODEL} \
    --port 8000 \
    --swap-space 64 \
    --tensor-parallel-size $TP \
    --kv_cache_dtype fp8 \
    --max-num-seqs 256 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 65536 \
    --max-model-len 16384 \
    --block-size 1 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --dtype auto \
    --disable_log_requests \
    --trust_remote_code \
    --distributed_executor_backend mp \
    $profiler_args 
"

# ## no max args (for kunlun 11k, 16k)
# CMD="
# vllm serve ${MODEL} \
#     --port 8000 \
#     --swap-space 64 \
#     --tensor-parallel-size $TP \
#     --kv_cache_dtype fp8 \
#     --no-enable-prefix-caching \
#     --block-size 1 \
#     --gpu-memory-utilization 0.95 \
#     --async-scheduling \
#     --dtype auto \
#     --disable_log_requests \
#     --trust_remote_code \
#     --distributed_executor_backend mp \
#     $profiler_args 
# "

{
    echo "Running command: $CMD"
    eval $CMD

} 2>&1 | tee -a "$server_log_file"

unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset VLLM_ROCM_QUICK_REDUCE_QUANTIZATION VLLM_V1_USE_PREFILL_DECODE_ATTENTION VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_MOE VLLM_USE_TRITON_FLASH_ATTN SAFETENSORS_FAST_GPU
unset AMD_LOG_LEVEL AITER_LOG_MORE HIPBLASLT_LOG_FILE HIPBLASLT_LOG_MASK