# !/bin/bash
export CUDA_VISIBLE_DEVICES="1"
tensor_parallel_size=1
port=8002

model_base_path=/code/models  # TODO: change to your own path
model_name_or_path=$model_base_path/Mixtral-8x7B-Instruct-v0.1-AWQ

echo "Starting vllm engine for $model_name_or_path as student simulator..."
python -m vllm.entrypoints.openai.api_server \
    --model $model_name_or_path \
    --port $port \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.95 \
    --max-model-len 10380 \
    --enforce-eager \
    --api-key "EMPTY"
