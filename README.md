# aac_qwen_deepseek

## Run ATOM framework for Qwen-235B
docker: docker.io/rocm/atom:ROCm_7.2_atom_qwen3_1215

Run docker:
```bash
podman run -it --name qwen3_atom --privileged --network=host --ipc=host -v $HOME:/workdir -v /shareddata:/shareddata -v /shared:/shared --workdir /workdir -e HF_HOME=/shared/amdgpu/home/share/huggingface/ docker://docker.io/rocm/atom:ROCm_7.2_atom_qwen3_1215 bash
```
The Qwen-235B model path on mi355-gpu-20: /shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969

Run benchmark command:
```bash
cd atom_qwen235b
# modify the tp=4 or 8 (both on run_server.sh / run_client.sh)
# replace the model_path to your own Qwen-235B model storage place.
# run server
./run_server.sh
# run client
./run_client.sh
```
## Run ATOM framework for DeepSeek-R1
docker: docker.io/rocm/atom:ROCm_7.2_atom_1214

Run docker:
```bash
podman run -it --name dsr1_atom --privileged --network=host --ipc=host -v $HOME:/workdir -v /shareddata:/shareddata -v /shared:/shared --workdir /workdir -e HF_HOME=/shared/amdgpu/home/share/huggingface/ docker://docker.io/rocm/atom:ROCm_7.2_atom_1214 bash
```
The DeepSeek-R1 model path on mi355-gpu-20: /shared/amdgpu/home/share/deepseek/DeepSeek-R1/

Run benchmark command:
```bash
cd atom_deepseekr1
# modify the tp=4 or 8 (both on run_server.sh / run_client.sh)
# currently Deepseek-R1 cannot run with tp=4
# replace the model_path to your own DeepSeek-R1 model storage place.

# run server
./run_server.sh
# run client
./run_client.sh
```
## Run vLLM framework for Qwen-235B
docker: docker.io/rocm/vllm-private:355_wip_508_b927d00ec_1204

Run docker:
```bash
podman run -it --name qwen3_vllm --privileged --network=host --ipc=host -v $HOME:/workdir -v /shareddata:/shareddata -v /shared:/shared --workdir /workdir -e HF_HOME=/shared/amdgpu/home/share/huggingface/ docker://docker.io/rocm/vllm-private:355_wip_508_b927d00ec_1204 bash
```
The Qwen-235B model path on mi355-gpu-20: /shared/amdgpu/home/share/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969

Run benchmark command:
```bash
cd vllm_qwen235b
# modify the tp=4 or 8 (both on run_server.sh / run_client.sh)
# replace the model_path to your own Qwen-235B model storage place.

# run server
./run_server.sh
# run client
./run_client.sh
```

# Run kunlun-benchmark
## Download kunlun-benchmark pacakge
### ubuntu 22.02
for py3.12
wget https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/ai_perf/pack/kunlun_benchmark/main/ubuntu22.04/py3.12.8/kunlun-benchmark.tar.gz

for py3.10
wget https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/ai_perf/pack/kunlun_benchmark/main/ubuntu22.04/py3.10.12/kunlun-benchmark.tar.gz
### ubuntu 24.04
wget https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/ai_perf/pack/kunlun_benchmark/main/ubuntu24.04/py3.12.8/kunlun-benchmark.tar.gz

tar -zxvf kunlun-benchmark.tar.gz

### error on 24.04
替換kunlun-benchmark  
https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/home/shizhiguo/scripts/kunlun-benchmark  
chmod +x /opt/kunlun-benchmark/kunlun-benchmark  

## Install requirements
pip install loguru jsonlines prettytable oss2

## Fix cpp_builder.py if you are not using windows os
```bash
cp /opt/venv/lib/python3.12/site-packages/torch/_inductor/cpp_builder.py /opt/venv/lib/python3.12/site-packages/torch/_inductor/cpp_builder.py.bk
vim /opt/venv/lib/python3.12/site-packages/torch/_inductor/cpp_builder.py
```
In Line 22, 
```python
from ctypes import cdll, wintypes
```
Remove wintypes
```python
from ctypes import cdll
```

## Set KUNLUN_DIR
KUNLUN_DIR="/opt/kunlun-benchmark" # where you put kunlun-benchmark
```bash
# Exmpale in client
${KUNLUN_DIR}/kunlun-benchmark vllm server
  --port $PORT \
  --work_mode manual \
  --max_input_len 1000 \
  --min_input_len 800 \
  --max_output_len 500 \
  --min_output_len 400 \
  --concurrency ${max_concurrency} \
  --query_num ${num_prompts} \
  --result_dir $client_log_dir \
  --model_path $MODEL \
  --is_sla False \
  --sla_decode 50 \
  --sla_prefill 3000 
```
The --sla_decode and --sla_prefill is only used when is_sla True (Will automatically find a solution; costs lots of time)

Server command is same with the one above.

# Referece
https://amd.atlassian.net/wiki/spaces/MLSE/pages/1312564741/Qwen235B+scripts#ATOM-script%3A
