# !/bin/bash
export CUDA_VISIBLE_DEVICES=1
tutor_setting="traver"
tutor_model="gpt-4o"

namespace_file=prompt/namespaces.json
prompt_element_file=prompt/prompt_elements_final.jsonl
output_dir=output/dialogue

tutor_vllm_endpoint="http://localhost:8001/v1"
api_key_file="xxx"  # TODO: change to your own Azure API key
azure_endpoint="xxx"  # TODO: change to your own Azure endpoint
model_base_path=/code/models  # TODO: change to your own path

if [ "$tutor_model" == "gpt-3.5" ]; then
    tutor_model_name_or_path="gpt-3.5"
elif [ "$tutor_model" == "gpt-4o" ]; then
    tutor_model_name_or_path="gpt-4o"
elif [ "$tutor_model" == "Meta-Llama-3.1-8B-Instruct" ]; then
    tutor_model_name_or_path=$model_base_path/Meta-Llama-3.1-8B-Instruct
elif [ "$tutor_model" == "Meta-Llama-3.1-70B-Instruct" ]; then
    tutor_model_name_or_path=$model_base_path/Meta-Llama-3.1-70B-Instruct
elif [ "$tutor_model" == "Qwen2-7B-Instruct" ]; then
    tutor_model_name_or_path=$model_base_path/Qwen2-7B-Instruct
elif [ "$tutor_model" == "Qwen2-72B-Instruct" ]; then
    tutor_model_name_or_path=$model_base_path/Qwen2-72B-Instruct
else
    echo "Not supported models!"
fi

verifier_base_model_path=$model_base_path/Mistral-7B-v0.1
verifier_model_dir=output/verifier_model
tutor_num_responses=10

student_model_name_or_path=$model_base_path/Mixtral-8x7B-Instruct-v0.1-AWQ
student_vllm_endpoint="http://localhost:8002/v1"
student_levels=(low_level med_level high_level)


for level in ${student_levels[@]}; do
    echo "Running interactive simulation between tutor and $level student ..."
    python src/run_traver.py --tutor_setting $tutor_setting \
        --namespace_file $namespace_file \
        --prompt_element_file $prompt_element_file \
        --output_dir $output_dir \
        --verifier_base_model_path $verifier_base_model_path \
        --verifier_model_dir $verifier_model_dir \
        --tutor_model_name_or_path $tutor_model_name_or_path \
        --tutor_num_responses $tutor_num_responses \
        --student_model_name_or_path $student_model_name_or_path \
        --student_setting $level \
        --api_key_file $api_key_file \
        --azure_endpoint $azure_endpoint \
        --vllm_endpoint_tutor $tutor_vllm_endpoint \
        --vllm_endpoint_student $student_vllm_endpoint \
        --show_description false \
        --show_message false
done