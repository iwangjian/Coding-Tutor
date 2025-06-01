# !/bin/bash
export CUDA_VISIBLE_DEVICES="0"
tensor_parallel_size=1
port=8001

model_base_path=/code/models  # TODO: change to your own path
model_name_or_path=$model_base_path/Meta-Llama-3.1-8B-Instruct

echo "Starting vllm engine for $model_name_or_path as tutor agent..."
python -m vllm.entrypoints.openai.api_server \
    --model $model_name_or_path \
    --port $port \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --api-key "EMPTY"
