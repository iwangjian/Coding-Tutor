import os, json
import textwrap

def load_json_dict(input_file: str):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def load_json_data(input_file: str):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            data.append(js)
    return data

def load_api(path: str, is_pool: bool = False):
    api_keys = []
    with open(path, 'r') as f:
        for line in f:
            key = line.strip()
            api_keys.append(key)
    if is_pool:
        return api_keys
    else:
        return api_keys[0]

def load_finished_data(log_file):
    finished_ids = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                js = json.loads(line)
                finished_ids.append(js['namespace']) 
    return finished_ids

def convert_to_json(data_fp, save_fp):
    all_data = []
    with open(data_fp, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    
    with open(save_fp, 'w') as f:
        json.dump(all_data, f, indent=4)
    print(f"Saved {len(all_data)} samples to {save_fp} in json format.")

def count_indent(args, data):
    code_file_path = os.path.join(args.source_code_root, data['completion_path'])
    with open(code_file_path, 'r') as f:
        lines = f.readlines()
    body_first_line = lines[data['body_position'][0]-1]
    indent = len(body_first_line) - len(textwrap.dedent(body_first_line))
    return indent

def adjust_indent(code, new_indent):
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code