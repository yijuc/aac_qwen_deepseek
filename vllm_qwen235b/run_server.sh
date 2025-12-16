# ====== Setting TP =====
TP=8  # or TP=4
# ========================

if [ "$TP" = "8" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ "$TP" = "4" ]; then
    export HIP_VISIBLE_DEVICES=0,1,2,3
else
    echo "Unsupported TP value: $TP"
    exit 1
fi

export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_DEBUG=WARN
export VLLM_RPC_TIMEOUT=1800000
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_TRITON_ROPE=1
export VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP8_QUANT=1
export VLLM_ROCM_USE_AITER_FAKE_BALANCED_EXPERTS=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1

# for profiling
export VLLM_TORCH_PROFILER_DIR="tp${TP}_qwen_in1k_out100"
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
model_path="/shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969"

# export AITER_REBUILD=1
# rm ~/.cache/vllm/torch_compile_cache/ -r
vllm serve $model_path \
	--tensor-parallel-size $TP \
	--max-num-batched-tokens 32768 \
	--trust-remote-code \
	--no-enable-prefix-caching \
	--disable-log-requests \
	--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["-rms_norm", "-quant_fp8", "-silu_and_mul"]}' \
	--gpu_memory_utilization 0.85 \
	--enable-expert-parallel \
	--port 8899 \
	--async-scheduling \
	--kv-cache-dtype fp8 \

