# !/bin/bash
ROOT=/data/EvoCodeBench/EvoCodeBench-2403 # TODO: change to your own path
Source_Code_Root=$ROOT/Source_Code
Dependency_Root=$ROOT/Dependency_Data

model_base_path=/code/models  # TODO: change to your own path
model_name_or_path=$model_base_path/Mixtral-8x7B-Instruct-v0.1-AWQ
prompt_element_file=prompt/prompt_elements_final.jsonl
prompt_base_dir=prompt/student_pretest
output_base_dir=output/student_pretest/Mixtral-8x7B-Instruct

student_levels=(low_level mid_level high_level oracle)
k="1,3,5,10"
n=10

for level in ${student_levels[@]}; do
    echo "Running recall@$k for $level student pre-test"
    python traver/utils/check_source_code.py $Source_Code_Root
    python traver/parser/recall_k.py \
        --output_file $output_base_dir/$level/completion.jsonl \
        --log_file $output_base_dir/$level/dependency_results.jsonl \
        --data_file $ROOT/data_final.jsonl \
        --source_code_root $Source_Code_Root \
        --dependency_data_root $Dependency_Root \
        --k $k
    
    echo "Running pass@$k for $level student pre-test"
    python traver/utils/check_source_code.py $Source_Code_Root
    python traver/parser/pass_k.py \
        --output_file $output_base_dir/$level/completion.jsonl \
        --log_file $output_base_dir/$level/test_results.jsonl \
        --data_file $ROOT/data_final.jsonl \
        --source_code_root $Source_Code_Root \
        --k $k \
        --n $n
done