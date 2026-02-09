#!/bin/bash
# ====== Setting TP, Profiler switch, output gemm switch =====
TP=8  # or TP=4 (Currently deepseek-r1 cannot run with TP4)
enable_profiler=0
enable_output_gemm=0
MODEL="/data/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52/"
server_log_dir="/workdir/data/vllm_deepseekr1/logs_0202_kunlun_random_noep_no-enable-prefix-caching_fp8"
# ========================

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
        export CUDA_VISIBLE_DEVICES=2,3
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

unset VLLM_ATTENTION_BACKEND VLLM_USE_FLASHINFER_MOE_FP8
export VLLM_ATTENTION_BACKEND=CUTLASS_MLA
export VLLM_USE_FLASHINFER_MOE_FP8=1

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
# --block-size 16 

# CMD="
# python -m vllm.entrypoints.openai.api_server  --port 8000 --model $MODEL -tp $TP --kv_cache_dtype fp8 $profiler_args 
# "
# CMD="
# vllm serve $MODEL \
#   --trust-remote-code \
#   --tensor-parallel-size $TP
# "
CMD="
vllm serve $MODEL \
  --trust-remote-code \
  --tensor-parallel-size $TP \
  --no-enable-prefix-caching \
  --kv_cache_dtype fp8
"

{
    echo "Running command: $CMD"
    eval $CMD

} 2>&1 | tee -a "$server_log_file"
