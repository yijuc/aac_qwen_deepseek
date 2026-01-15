#!/bin/bash
# ====== Setting TP, Profiler switch, output gemm switch =====
TP=8  # or TP=4 (Currently deepseek-r1 cannot run with TP4)
enable_profiler=0
enable_output_gemm=0
MODEL="/dev/shm/DeepSeek-R1-0528"
# ========================

server_log_dir="/workdir/vllm_deepseekr1/logs_kunlun_0114"
mkdir -p ${server_log_dir}
log_tag="vllm_fp8_tp${TP}_deepseek_r1"
# log_tag="atom_fp8_tp${TP}_deepseek_r1_in1k_out1k_conc64_kernel"
server_log_file="${server_log_dir}/${log_tag}_server_running.log"

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

unset VLLM_ROCM_USE_AITER VLLM_USE_AITER_TRITON_ROPE VLLM_ROCM_USE_AITER_RMSNORM VLLM_ROCM_USE_AITER_TRITON_LINEAR VLLM_ROCM_QUICK_REDUCE_QUANTIZATION
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_TRITON_LINEAR=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="INT4"

unset HIPBLASLT_LOG_FILE HIPBLASLT_LOG_MASK
if [ "$enable_output_gemm" = "1" ]; then
    export HIPBLASLT_LOG_FILE="${server_log_dir}/${log_tag}_gemm_output.log"
    hipblaslt_log_dir=$(dirname "$HIPBLASLT_LOG_FILE")
    mkdir -p "$hipblaslt_log_dir"
    export HIPBLASLT_LOG_MASK=32
fi

profiler_args=""
if [ "$enable_profiler" = "1" ]; then
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
    profiler_dir="${server_log_dir}/${log_tag}"
    mkdir -p $profiler_dir
    profiler_args=" --torch-profiler-dir ${profiler_dir}"
fi


# rm -rf /root/.cache/vllm/torch_compile_cache/


CMD="
vllm serve ${MODEL} \
    --port 8000 \
    --swap-space 16 \
    --tensor-parallel-size $TP \
    --max-num-batched-tokens 131072 \
    --max-model-len 32768 \
    --async-scheduling \
    --kv-cache-dtype auto \
    --quantization fp8 \
    --block-size 1 \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --max-num-seqs 4096 \
    --enable-chunked-prefill $profiler_args 
"

{
    echo "Running command: $CMD"
    eval $CMD

} 2>&1 | tee "$server_log_file"
