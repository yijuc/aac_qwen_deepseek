#!/bin/bash
# ====== Setting TP, Profiler switch, output gemm switch =====
TP=8  # or TP=4 (Currently deepseek-r1 cannot run with TP4)
enable_profiler=0
enable_output_gemm=0
server_log_dir="/workdir/eveline/atom_deepseekr1/logs_kunlun_dsr1_atom_12241340"
MODEL="/dev/shm/DeepSeek-R1-0528"
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
        export CUDA_VISIBLE_DEVICES=0,1,2,3
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
        export HIP_VISIBLE_DEVICES=0,1,2,3
    fi
fi

mkdir -p ${server_log_dir}
log_tag="atom_fp8_tp${TP}_deepseek_r1"
# log_tag="atom_fp8_tp${TP}_deepseek_r1_in1k_out1k_conc64_kernel"
server_log_file="${server_log_dir}/${log_tag}_server_running.log"

unset HIP_VISIBLE_DEVICES
if [ "$TP" = "8" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ "$TP" = "4" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3
else
    echo "Unsupported TP value: $TP"
    exit 1
fi

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


rm -rf /root/.cache/atom/

## 1k => add --max-model-len 2048
# CMD="
# python -m atom.entrypoints.openai_server --model $MODEL -tp $TP --block-size 16 --kv_cache_dtype fp8 --max-model-len 2048 $profiler_args 
# "

## 4k/10k
# CMD="
# python -m atom.entrypoints.openai_server --model $MODEL -tp $TP --block-size 16 --kv_cache_dtype fp8 $profiler_args 
# "

## 16k~20k --max-num-batched-tokens 20000
CMD="
python -m atom.entrypoints.openai_server --model $MODEL -tp $TP --block-size 16 --kv_cache_dtype fp8 --max-num-batched-tokens 20000  $profiler_args 
"


{
    echo "Running command: $CMD"
    $CMD

} 2>&1 | tee -a "$server_log_file"

unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset VLLM_ROCM_QUICK_REDUCE_QUANTIZATION VLLM_V1_USE_PREFILL_DECODE_ATTENTION VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_MOE VLLM_USE_TRITON_FLASH_ATTN SAFETENSORS_FAST_GPU
unset AMD_LOG_LEVEL AITER_LOG_MORE HIPBLASLT_LOG_FILE HIPBLASLT_LOG_MASK