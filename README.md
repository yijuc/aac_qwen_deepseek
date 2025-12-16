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

# Referece
https://amd.atlassian.net/wiki/spaces/MLSE/pages/1312564741/Qwen235B+scripts#ATOM-script%3A
