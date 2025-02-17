import json
import os
import random
import time
from argparse import ArgumentParser
from openai import AzureOpenAI
from utils import adjust_indent, load_api


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--source_code_root', type=str)
    parser.add_argument('--metadata_file', type=str, default='EvoCodeBench-2403/data_final.jsonl')
    parser.add_argument("--prompt_elements_file", type=str, default='prompt/prompt_elements.jsonl')
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--dependency_ratio", type=float, default=0.5)
    parser.add_argument('--model', type=str, default='gpt-4', choices=['gpt-3.5', 'gpt-4'])
    parser.add_argument('--api_key_file', type=str, default='prompt/openai_api_key.txt')
    parser.add_argument("--azure_endpoint", type=str, default='')
    parser.add_argument('--T', type=float, default=0.4)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def search_signature(file_to_parse, signature_name):
    with open(file_to_parse, 'r', encoding='utf-8') as f:
        file_lines = f.readlines()
    
    signature = None
    start_idx, end_idx = 0, 0
    for i, line in enumerate(file_lines):
        if line.strip().startswith(signature_name):
            start_idx = i
            break
    for i in range(start_idx, len(file_lines)):
        if file_lines[i].strip().endswith(':'):
            end_idx = i
            break
    if start_idx >0 and end_idx > 0 and end_idx >= start_idx:
        signature = "".join(file_lines[start_idx:end_idx+1])

        # check docstring exists or not
        if file_lines[end_idx + 1].strip().startswith('\"\"\"'):
            docstring_start = end_idx + 1
            if file_lines[docstring_start].strip() == '\"\"\"':
                for i in range(docstring_start + 1, len(file_lines)):
                    if file_lines[i].strip().endswith('\"\"\"'):
                        docstring_end = i
                        break
            elif file_lines[docstring_start].strip().endswith('\"\"\"'):
                docstring_end = docstring_start
            else:
                for i in range(docstring_start + 1, len(file_lines)):
                    if file_lines[i].strip().endswith('\"\"\"'):
                        docstring_end = i
                        break
            signature = "".join(file_lines[start_idx:docstring_end+1])
    
    return signature


def parse_dependency_docstring(source_code_root, completion_path, dep_name):
    project_name = completion_path.split('/')[0]
    project_root = os.path.join(source_code_root, project_name)
    dep_name_list = dep_name.split('.')
    file_to_parse = None
    dep_docstring = ""
    for i, name in enumerate(dep_name_list):
        tmp_file = os.path.join(project_root, '/'.join(dep_name_list[:i+1]) + '.py')
        if os.path.exists(tmp_file):
            file_to_parse = tmp_file
            break
    if file_to_parse is None:
        #print(f"Dependenecy not found: {dep_name}")
        return dep_docstring
    
    # search function signature
    signature_name = f"def {dep_name.split('.')[-1]}("
    signature = search_signature(file_to_parse, signature_name)
    # search class signature
    if signature is None:
        signature_name = f"class {dep_name.split('.')[-1]}("
        signature = search_signature(file_to_parse, signature_name)
    if signature is None:
        signature_name = f"class {dep_name.split('.')[-1]}:"
        signature = search_signature(file_to_parse, signature_name)
    
    dep_docstring = adjust_indent(signature, 0) if signature is not None else ""
    return dep_docstring

def parse_reference_code(source_code_root, meta_data):
    # parse reference code
    file_to_parse = os.path.join(source_code_root, meta_data["completion_path"])
    sos, eos = meta_data['signature_position'][0]-1, meta_data['body_position'][1]
    with open(file_to_parse, 'r') as f:
        file_lines = f.readlines()
    ref_code = ''.join(file_lines[sos:eos])
    ref_code = adjust_indent(ref_code, 0)
    return ref_code


def setup(args):
    api_key = load_api(args.api_key_file)
    client = AzureOpenAI(
        azure_endpoint = args.azure_endpoint, 
        api_key=api_key,
        api_version="2024-02-15-preview"
    )
    template = open(f'prompt/template/preprocess_code.txt', 'r').read()
    
    # build key solution steps for each sample
    reference_code, reference_steps = {}, {}
    with open(args.metadata_file, "r") as f:
        for idx, line in enumerate(f):
            meta_data = json.loads(line)
            ref_code = parse_reference_code(args.source_code_root, meta_data)
            reference_code[meta_data["namespace"]] = ref_code
            try:
                prompt = template.format(reference_code=ref_code)
                completion = client.chat.completions.create(
                    model=args.model, 
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=args.T,
                    top_p=args.top_p
                )
                response = completion.choices[0].message.content
                response = response.strip()
                reference_steps[meta_data["namespace"]] = response
                print(f"Processed {idx+1} samples")
            except Exception as e:
                raise ValueError(f"Error in processing {meta_data['namespace']}: {e}")
    assert len(reference_code) == len(reference_steps)
    print(f"Number of reference code and steps: {len(reference_code)}")

    # build dependency data
    dependency_all, dependency_sampled = {}, {}
    with open(args.metadata_file, "r", encoding='utf-8') as f:
        for line in f:
            meta_data = json.loads(line)
            
            # form dependency with all dependency paths
            intra_class_deps = meta_data['dependency']['intra_class']
            intra_file_deps = meta_data['dependency']['intra_file']
            cross_file_deps = meta_data['dependency']['cross_file']

            # parse dependency docstring
            dep_docstrings = {}
            all_deps = intra_class_deps + intra_file_deps + cross_file_deps
            for dep in all_deps:
                dep_docstrings[dep] = parse_dependency_docstring(args.source_code_root, meta_data['completion_path'], dep)
           
            # concat dependency paths and docstrings
            for dep in dep_docstrings:
                if dep in intra_class_deps:
                    intra_class_deps[intra_class_deps.index(dep)] = '{}\n{}'.format(dep, dep_docstrings[dep])
                if dep in intra_file_deps:
                    intra_file_deps[intra_file_deps.index(dep)] = '{}\n{}'.format(dep, dep_docstrings[dep])
                if dep in cross_file_deps:
                    cross_file_deps[cross_file_deps.index(dep)] = '{}\n{}'.format(dep, dep_docstrings[dep])

            dependency_path_all = ""
            if len(intra_class_deps) > 0:
                dependency_path_all += "# Intra-class Dependency:\n{}\n".format("\n".join(intra_class_deps))
            if len(intra_file_deps) > 0:
                dependency_path_all += "# Intra-file Dependency:\n{}\n".format("\n".join(intra_file_deps))
            if len(cross_file_deps) > 0:
                dependency_path_all += "# Cross-file Dependency:\n{}\n".format("\n".join(cross_file_deps))

            dependency_all[meta_data["namespace"]] = dependency_path_all

            # samling dependency path list for student simulation
            dependencies = []
            for dep in intra_class_deps:
                dependencies.append(('intra_class', dep))
            for dep in intra_file_deps:
                dependencies.append(('intra_file', dep))
            for dep in cross_file_deps:
                dependencies.append(('cross_file', dep))       
            total_num = len(dependencies)
            if total_num * args.dependency_ratio < 1:
                sampled_num = 1
            else:
                sampled_num = int(total_num * args.dependency_ratio)
            sampled_dependency = random.sample(dependencies, sampled_num)
            sampled_intra_class_deps = [dep[1] for dep in sampled_dependency if dep[0] == 'intra_class']
            sampled_intra_file_deps = [dep[1] for dep in sampled_dependency if dep[0] == 'intra_file']
            sampled_cross_file_deps = [dep[1] for dep in sampled_dependency if dep[0] == 'cross_file']
    
            dependency_path = ""
            if len(sampled_intra_class_deps) > 0:
                dependency_path += "# Intra-class Dependency:\n{}\n".format("\n".join(sampled_intra_class_deps))
            if len(sampled_intra_file_deps) > 0:
                dependency_path += "# Intra-file Dependency:\n{}\n".format("\n".join(sampled_intra_file_deps))
            if len(sampled_cross_file_deps) > 0:
                dependency_path += "# Cross-file Dependency:\n{}\n".format("\n".join(sampled_cross_file_deps))
            
            dependency_sampled[meta_data["namespace"]] = dependency_path
    
    assert len(dependency_all) == len(dependency_sampled)
    print(f"Number of denpendency data: {len(dependency_all)}")

    # Load prompt elements
    prompt_elements = []
    with open(args.prompt_elements_file, "r", encoding='utf-8') as f:
        for line in f:
            prompt = json.loads(line)
            if prompt["namespace"] in dependency_all:
                new_prompt = {
                    "namespace": prompt["namespace"],
                    "type": prompt["type"],
                    "class_name": prompt["class_name"],
                    "function_name": prompt["function_name"],
                    "dependency_all": dependency_all[prompt["namespace"]],
                    "dependency_sampled": dependency_sampled[prompt["namespace"]],
                    "contexts_above": prompt["contexts_above"],
                    "contexts_below": prompt["contexts_below"],
                    "input_code": prompt["input_code"],
                    "reference_steps": reference_steps[prompt["namespace"]],
                    "reference_code": reference_code[prompt["namespace"]]
                }
                prompt_elements.append(new_prompt)
    print(f"Number of prompt elements: {len(prompt_elements)}")

    # Save prompt elements
    with open(args.output_file, "w", encoding='utf-8') as f:
        for prompt in prompt_elements:
            f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    print(f"Saved to {args.output_file}")


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)

    if args.model == 'gpt-3.5':
        args.model = 'gpt35-1106'  # Azure OpenAI deploy name for 'gpt-3.5-turbo-1106'
    elif args.model == 'gpt-4':
        args.model = 'GPT4-1106-preview' # Azure OpenAI deploy name for 'gpt-4-1106-preview'
    else:
        raise ValueError('Invalid model name')

    setup(args)