#!/bin/bash
# ====== Setting TP, Profiler switch, output gemm switch =====
TP=4  # TP=8 or 4
enable_profiler=0
enable_output_gemm=0
server_log_dir="/workdir/eveline/atom_qwen235b/logs_kunlun_qwen_202601200319_max-model-len_16384_max-num-batched-tokens_20k"
PORT=8000
# MODEL="/mnt/nfs/RAID/shared/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969/"
MODEL="/dev/shm/Qwen3-235B-A22B-Instruct-2507-FP8/"

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

mkdir -p ${server_log_dir}
# server_log_file="${server_log_dir}/server_running_qwen3_235b_a22b_instrct_FP8_TP${TP}_isl${input_len}_osl${output_len}_conc${concurrency}_infrrate_PORT${PORT}.log"
server_log_file="${server_log_dir}/server_running_qwen3_235b_a22b_instrct_FP8_TP${TP}_PORT${PORT}.log"

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
    export VLLM_TORCH_PROFILER_DIR="${profiler_dir}"
    mkdir -p $profiler_dir
    # profiler_args=" --torch-profiler-dir ${profiler_dir}"
fi


rm -rf /root/.cache/atom/

# CMD="
# python -m atom.entrypoints.openai_server --model ${MODEL} -tp $TP --enable-expert-parallel --kv_cache_dtype fp8 $profiler_args
# "

CMD="
python -m atom.entrypoints.openai_server --model ${MODEL} -tp $TP --enable-expert-parallel --kv_cache_dtype fp8 --max-num-batched-tokens 20000 --max-model-len 16384 --server-port $PORT $profiler_args
"


# CMD="
# python -m atom.entrypoints.openai_server --model ${MODEL} -tp $TP --enable-expert-parallel --kv_cache_dtype fp8 --max-num-batched-tokens 20000 --server-port $PORT $profiler_args
# "

{
    echo "Running command: $CMD"
    $CMD

} 2>&1 | tee -a "$server_log_file"
# } >> "$server_log_file" 2>&1

unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset VLLM_ROCM_QUICK_REDUCE_QUANTIZATION VLLM_V1_USE_PREFILL_DECODE_ATTENTION VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_MOE VLLM_USE_TRITON_FLASH_ATTN SAFETENSORS_FAST_GPU
unset AMD_LOG_LEVEL AITER_LOG_MORE HIPBLASLT_LOG_FILE HIPBLASLT_LOG_MASK
