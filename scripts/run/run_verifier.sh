# !/bin/bash
model_base_path=/code/models  # TODO: change to your own path
data_dir=output/verifier_data
output_dir=output/verifier_model
pretrained_model_name_or_path=$model_base_path/Mistral-7B-v0.1
eval_parts=(part0 part1 part2 part3 part4)

for part in ${eval_parts[@]}; do
    echo "Running training on part-$part ..."
    deepspeed --master_port=29400 --include="localhost:0" traver/train_verifier.py --data_dir $data_dir  \
        --eval_part $part \
        --pretrained_model_name_or_path $pretrained_model_name_or_path \
        --output_dir $output_dir/$part \
        --max_length 2200 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --fp16 true \
        --bf16 false \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --logging_steps 10 \
        --eval_steps 500 \
        --save_steps 100 \
        --deepspeed config/deepspeed_config_s2.json
done