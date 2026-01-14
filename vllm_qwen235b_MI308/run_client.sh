## Refered to https://amd.atlassian.net/wiki/spaces/MLSE/pages/1312564741/Qwen235B+scripts#ATOM-script%3A

# curl -X POST "http://localhost:8000/v1/completions" \
#      -H "Content-Type: application/json" \
#      -d '{
#          "prompt": "The capital of China", "max_tokens": 100, "temperature": 0, "top_p": 1, "top_k": 0, "repetition_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "stream": false, "ignore_eos": false, "n": 1, "seed": 123 
#  }'

# ====== Setting TP =====
TP=8  # or TP=4
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

model_path="/shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969"
# python -m vllm.benchmarks.serve
# vllm bench serve \
# python -m vllm.entrypoints.cli.main bench serve \
#     --host localhost \
#     --port 8899 \
#     --model ${model_path} \
#     --dataset-name random \
#     --random-input-len 1000 \
#     --random-output-len 100 \
#     --max-concurrency 128 \
#     --num-prompts 256 \
#     --seed 123 \
#     --percentile-metrics ttft,tpot,itl,e2el \
#     --ignore-eos \
#     --profile
#     # --seed 123 \
#     # --request-rate 2 \

# ==== inLen 1000 outLen 1000 maxConcurrency 256 ====
inLen=1000; outLen=1000; maxConcurrency=256
python -m vllm.entrypoints.cli.main bench serve \
    --host localhost \
    --port 8899 \
    --model ${model_path} \
    --dataset-name random \
    --random-input-len $inLen \
    --random-output-len $outLen \
    --max-concurrency $maxConcurrency \
    --num-prompts $((maxConcurrency * 5)) \
    --seed 123 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
	2>&1 | tee in-$inLen-out-$outLen-max-concurrency-$maxConcurrency-tp$TP.log

# ==== inLen 1000 outLen 1000 maxConcurrency 128 ====
inLen=1000; outLen=1000; maxConcurrency=128
python -m vllm.entrypoints.cli.main bench serve \
    --host localhost \
    --port 8899 \
    --model ${model_path} \
    --dataset-name random \
    --random-input-len $inLen \
    --random-output-len $outLen \
    --max-concurrency $maxConcurrency \
    --num-prompts $((maxConcurrency * 5)) \
    --seed 123 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
	2>&1 | tee in-$inLen-out-$outLen-max-concurrency-$maxConcurrency-tp$TP.log

# ==== inLen 4000 outLen 1000 maxConcurrency 128 ====
inLen=4000; outLen=1000; maxConcurrency=128
python -m vllm.entrypoints.cli.main bench serve \
    --host localhost \
    --port 8899 \
    --model ${model_path} \
    --dataset-name random \
    --random-input-len $inLen \
    --random-output-len $outLen \
    --max-concurrency $maxConcurrency \
    --num-prompts $((maxConcurrency * 5)) \
    --seed 123 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
	2>&1 | tee in-$inLen-out-$outLen-max-concurrency-$maxConcurrency-tp$TP.log

# ==== inLen 4000 outLen 1000 maxConcurrency 64 ====
inLen=4000; outLen=1000; maxConcurrency=64
python -m vllm.entrypoints.cli.main bench serve \
    --host localhost \
    --port 8899 \
    --model ${model_path} \
    --dataset-name random \
    --random-input-len $inLen \
    --random-output-len $outLen \
    --max-concurrency $maxConcurrency \
    --num-prompts $((maxConcurrency * 5)) \
    --seed 123 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
	2>&1 | tee in-$inLen-out-$outLen-max-concurrency-$maxConcurrency-tp$TP.log

# ==== inLen 10000 outLen 1000 maxConcurrency 64 ====
inLen=10000; outLen=1000; maxConcurrency=64
python -m vllm.entrypoints.cli.main bench serve \
    --host localhost \
    --port 8899 \
    --model ${model_path} \
    --dataset-name random \
    --random-input-len $inLen \
    --random-output-len $outLen \
    --max-concurrency $maxConcurrency \
    --num-prompts $((maxConcurrency * 5)) \
    --seed 123 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
	2>&1 | tee in-$inLen-out-$outLen-max-concurrency-$maxConcurrency-tp$TP.log

# ==== inLen 10000 outLen 1000 maxConcurrency 32 ====
inLen=10000; outLen=1000; maxConcurrency=32
python -m vllm.entrypoints.cli.main bench serve \
    --host localhost \
    --port 8899 \
    --model ${model_path} \
    --dataset-name random \
    --random-input-len $inLen \
    --random-output-len $outLen \
    --max-concurrency $maxConcurrency \
    --num-prompts $((maxConcurrency * 5)) \
    --seed 123 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
	2>&1 | tee in-$inLen-out-$outLen-max-concurrency-$maxConcurrency-tp$TP.log