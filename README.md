# aac_qwen_deepseek

## Run ATOM framework for Qwen-235B
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
```bash
cd vllm_qwen235b
# modify the tp=4 or 8 (both on run_server.sh / run_client.sh)
# replace the model_path to your own Qwen-235B model storage place.

# run server
./run_server.sh
# run client
./run_client.sh
```
