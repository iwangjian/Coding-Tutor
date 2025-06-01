# !/bin/bash
gpu_id=1
model_base_path=/code/models  # TODO: change to your own path
model_name_or_path=$model_base_path/Mixtral-8x7B-Instruct-v0.1-AWQ

tutor_setting="vanilla"  # TODO: change to your own setting
tutor_model="gpt-4o"     # TODO: change to your own model
student_levels=(low_level mid_level high_level)

output_dir=output/dialogue
namespace_file=prompt/namespaces.json
prompt_element_file=prompt/prompt_elements_final.jsonl
prompt_base_dir=prompt/student_posttest
output_base_dir=output/student_posttest

rounds=(1 2 3 4 5 6 7 8)
max_interaction_round=8
max_cognitive_load=60
k="1,3,5,10"
n=10

# Step 1: Create prompts
for level in ${student_levels[@]}; do
    echo "Running making prompt for posttest: tutor_setting: $tutor_setting tutor_model: $tutor_model student_level: $level"
    python traver/utils/make_prompt.py --student_posttest \
        --prompt_element_file $prompt_element_file \
        --simulated_file output/simulation/$tutor_setting/$tutor_model/$level/simulated_dialogs.jsonl \
        --output_dir $prompt_base_dir/$tutor_setting/$tutor_model/$level \
        --student_level $level \
        --max_interaction_round $max_interaction_round \
        --max_cognitive_load $max_cognitive_load
done


# Step 2: Run code generation
for level in ${student_levels[@]}; do
    for rdx in ${rounds[@]}; do
        echo "Running code generation for posttest: tutor_setting: $tutor_setting tutor_model: $tutor_model student_level: $level round: $rdx"

        CUDA_VISIBLE_DEVICES=$gpu_id python traver/utils/LM_inference.py --model_name_or_path $model_name_or_path \
            --prompt_file $prompt_base_dir/$tutor_setting/$tutor_model/$level/prompt_round_$rdx.jsonl \
            --output_dir $output_base_dir/$tutor_setting/$tutor_model/$level/round_$rdx \
            --decoding "sampling" \
            --N $n
    done

done

# Step 3: Process completions
for level in ${student_levels[@]}; do
    for rdx in ${rounds[@]}; do
        echo "Processing completions for: tutor_setting: $tutor_setting tutor_model: $tutor_model student_level: $level round: $rdx"
        python traver/utils/process_completion.py \
            --completion_file $output_base_dir/$tutor_setting/$tutor_model/$level/round_$rdx/completion_lm.jsonl \
            --output_file $output_base_dir/$tutor_setting/$tutor_model/$level/round_$rdx/completion.jsonl
    done
done