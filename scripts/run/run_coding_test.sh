# !/bin/bash
ROOT=/data/EvoCodeBench/EvoCodeBench-2403  # TODO: change to your own path
Source_Code_Root=$ROOT/Source_Code
Dependency_Root=$ROOT/Dependency_Data

metadata_file=EvoCodeBench-2403/metadata.jsonl
prompt_element_file=prompt/prompt_elements_final.jsonl
prompt_base_dir=prompt/student_posttest
output_base_dir=output/student_posttest

tutor_settings=(vanilla)    # TODO: change to your own setting
models=(Meta-Llama-3.1-70B-Instruct) # TODO: change to your own model
student_levels=(low_level med_level high_level)
k="1,3,5,10"
n=10

for setting in ${tutor_settings[@]}; do
    for model in ${models[@]}; do
        for level in ${student_levels[@]}; do
            echo "Running recall@$k for tutor_setting: $setting model: $model student_level: $level"
            python traver/utils/check_source_code.py $Source_Code_Root
            python traver/parser/recall_k.py \
                --output_file $output_base_dir/$setting/$model/$level/completion.jsonl \
                --log_file $output_base_dir/$setting/$model/$level/dependency_results.jsonl \
                --data_file $metadata_file \
                --source_code_root $Source_Code_Root \
                --dependency_data_root $Dependency_Root \
                --k $k
            
            echo "Running pass@$k for tutor_setting: $setting model: $model student_level: $level"
            python traver/utils/check_source_code.py $Source_Code_Root
            python traver/parser/pass_k.py \
                --output_file $output_base_dir/$setting/$model/$level/completion.jsonl \
                --log_file $output_base_dir/$setting/$model/$level/test_results.jsonl \
                --data_file $metadata_file \
                --source_code_root $Source_Code_Root \
                --k $k \
                --n $n
        done
    done
done