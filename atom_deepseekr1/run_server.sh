#!/bin/bash
# ====== Setting TP, Profiler switch, output gemm switch =====
TP=${TP:-4}
enable_profiler=${enable_profiler:-0}
enable_output_gemm=${enable_output_gemm:-0}
PORT=${PORT:-8000}
MODEL=${MODEL:-"/dev/shm/DeepSeek-R1-0528"}
SERVER_LOG_DIR=${SERVER_LOG_DIR:-"/dockerx/eveline/atom_qwen235b/logs_0209_2060323_max-model-len_16384_max-num-batched-tokens_20k"}

if [ -n "$MIN_IN" ]; then
    # Format: atom_tp4_in1000-1000_out1000-1000_c256
    log_tag="atom_tp${TP}_in${MIN_IN}-${MAX_IN}_out${MIN_OUT}-${MAX_OUT}_c${CONC}"
else
    log_tag="atom_fp8_tp${TP}_qwen235b_default"
fi

mkdir -p ${SERVER_LOG_DIR}
server_log_file="${SERVER_LOG_DIR}/${log_tag}_server.log"
# =====================

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

unset AITER_QUICK_REDUCE_QUANTIZATION ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

# ===== Log file ======
# unset AMD_LOG_LEVEL
# export AMD_LOG_LEVEL=3
# export AMD_LOG_LEVEL_FILE="/workdir/atom_qwen235b/logs_amd_level3/qwen3_235b_a22b_instrct_FP8_TP${TP}_isl${input_len}_osl${output_len}_conc${concurrency}_infrrate_loglv3.log"

unset AITER_LOG_MORE
# export AITER_LOG_MORE=2


unset HIPBLASLT_LOG_FILE HIPBLASLT_LOG_MASK
if [ "$enable_output_gemm" = "1" ]; then
    export HIPBLASLT_LOG_FILE="${SERVER_LOG_DIR}/${log_tag}_kernel.log"
    export HIPBLASLT_LOG_MASK=32
fi

profiler_args=""
if [ "$enable_profiler" = "1" ]; then
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=1  
    profiler_dir="${SERVER_LOG_DIR}/${log_tag}_profiler"
    export VLLM_TORCH_PROFILER_DIR="${profiler_dir}"
    mkdir -p $profiler_dir
    # profiler_args=" --torch-profiler-dir ${profiler_dir}"
fi

unset VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT VLLM_ROCM_USE_AITER_MHA VLLM_ROCM_QUICK_REDUCE_QUANTIZATION VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=2048

rm -rf /root/.cache/atom/

## 1k => add --max-model-len 2048
CMD="
python -m atom.entrypoints.openai_server --model $MODEL -tp $TP --block-size 16 --kv_cache_dtype fp8 --max-model-len 2048 $profiler_args 
"

## 4k/10k
# CMD="
# python -m atom.entrypoints.openai_server --model $MODEL -tp $TP --block-size 16 --kv_cache_dtype fp8 $profiler_args 
# "

## 16k~20k --max-num-batched-tokens 20000
# CMD="
# python -m atom.entrypoints.openai_server --model $MODEL -tp $TP --block-size 16 --kv_cache_dtype fp8 --max-num-batched-tokens 20000  $profiler_args 
# "


{
    echo "Running command: $CMD"
    $CMD

} 2>&1 | tee -a "$server_log_file"

unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset VLLM_ROCM_QUICK_REDUCE_QUANTIZATION VLLM_V1_USE_PREFILL_DECODE_ATTENTION VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_MOE VLLM_USE_TRITON_FLASH_ATTN SAFETENSORS_FAST_GPU
unset AMD_LOG_LEVEL AITER_LOG_MORE HIPBLASLT_LOG_FILE HIPBLASLT_LOG_MASK