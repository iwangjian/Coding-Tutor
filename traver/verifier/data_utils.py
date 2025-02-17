import json
import os
from torch.utils.data import Dataset, DataLoader


class RewardDataset(Dataset):

    def __init__(self, data_js, tokenizer, max_length=2048):
        self.data_js = data_js
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        prompt_response = self.data_js[idx]['prompt_response']
        label = self.data_js[idx]['label']

        encoded_pair = self.tokenizer.encode_plus(
            prompt_response,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_pair['input_ids'].squeeze(),
            'attention_mask': encoded_pair['attention_mask'].squeeze(),
            'labels': label
        }


class OnlineDataBuilder:

    def __init__(self, elements, data_template, tokenizer, max_length=2048):
        self.elements = elements
        self.data_template = data_template
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.namespace = None
    
    def set_namespace(self, namespace):
        self.namespace = namespace
    
    def get_element(self):
        return_d = None
        for element_d in self.elements:
            if element_d['namespace'] == self.namespace:
                return_d = element_d
                break
        return return_d
    
    def build_data(self, conversation_tx, response_list):
        element_d = self.get_element()

        if element_d['class_name']:
            input_code = f"class {element_d['class_name']}:\n" + element_d['input_code']
        else:
            input_code = element_d['input_code']
        
        data_samples = []
        for idx, response in enumerate(response_list):
            sample = {
                "prompt_response": self.data_template.format(
                    function_name=element_d['function_name'],
                    input_code=input_code,
                    dependency_path=element_d['dependency_all'].strip(),
                    reference_steps=element_d['reference_steps'].strip(),
                    conversation=conversation_tx,
                    response=response
                ),
                "label": 0
            }
            data_samples.append(sample)
        
        online_dataset = RewardDataset(data_samples, tokenizer=self.tokenizer, max_length=self.max_length)
        online_dataloader = DataLoader(online_dataset, batch_size=1, shuffle=False)
        
        return online_dataloader


def load_json_data(input_file: str):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            data.append(js)
    return data

def load_data(data_file):
    data_dict = {}
    with open(data_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            if namespace not in data_dict:
                data_dict[namespace] = [js]
            else:
                data_dict[namespace].append(js)
    return data_dict

def save_data(data_dict, data_file):
    with open(data_file, 'w') as f:
        for namespace, data in data_dict.items():
            for js in data:
                f.write(json.dumps(js) + '\n')

def compute_process_reward(total_turn, current_turn, outcome_label):
    process_reward = 0
    t = 1
    while t <= current_turn:
        leading_dist = total_turn - t
        weight = (1 - process_reward) * (2*outcome_label - 1) / (leading_dist + 1)
        process_reward = max(process_reward + weight, 0)
        t += 1
    # keep at most 4 decimal places
    process_reward = round(process_reward, 4)
    return process_reward

def build_model_data(elements, dialog, student_level, data_template):
    namespace = dialog['namespace']
    d = None
    for element in elements:
        if element['namespace'] == namespace:
            d = element
            break

    if d['class_name']:
        input_code = f"class {d['class_name']}:\n" + d['input_code']
    else:
        input_code = d['input_code']
    
    if student_level == "low_level":
        student_knowledge = "None"
    elif student_level == "med_level":
        student_knowledge = d["dependency_sampled"].strip()
    else:
        ref_steps = d['reference_steps'].split('2.')[0].strip()
        student_knowledge = "{}\n\n{}".format(d["dependency_sampled"].strip(), ref_steps)
    
    data_samples = []
    conversation = dialog['conversation']
    outcome_label = dialog['outcome_label']
    total_turn = len(conversation) // 2
    idx = 0
    while idx < len(conversation):
        if "tutor" in conversation[idx]:
            current_turn = idx // 2 + 1
            if idx == 0:
                conv_ctx = []
            else:
                conv_ctx = conversation[:idx]
            response = conversation[idx]["tutor"]

            process_reward = compute_process_reward(total_turn, current_turn, outcome_label)
            sample = {
                "namespace": namespace,
                "prompt_response": data_template.format(
                    function_name=d['function_name'],
                    input_code=input_code,
                    dependency_path=d['dependency_all'].strip(),
                    reference_steps=d['reference_steps'].strip(),
                    conversation=conv_ctx,
                    response=response
                ),
                "label": process_reward
            }
        
            idx += 1
            data_samples.append(sample)
        idx += 1
    
    return data_samples


def check_adjust_posttest(posttest_dir):
    # since some examples may not have completions starting from a certain round
    # we need to get completions and eval results from the last round that has completions
    round_dirs = os.listdir(posttest_dir)
    max_round = max([int(round_dir.split('round_')[1]) for round_dir in round_dirs])

    completion_file = os.path.join(posttest_dir, "round_1/completion.jsonl")
    test_file = os.path.join(posttest_dir, "round_1/test_results.jsonl")
    dep_file = os.path.join(posttest_dir, "round_1/dependency_results.jsonl")
    prev_completions = load_data(completion_file)
    prev_tests = load_data(test_file)
    prev_deps = load_data(dep_file)

    for rdx in range(2, max_round + 1):
        completion_file = os.path.join(posttest_dir, f"round_{rdx}/completion.jsonl")
        test_file = os.path.join(posttest_dir, f"round_{rdx}/test_results.jsonl")
        dep_file = os.path.join(posttest_dir, f"round_{rdx}/dependency_results.jsonl")
        cur_completions = load_data(completion_file)
        cur_tests = load_data(test_file)
        cur_deps = load_data(dep_file)
        for namespace, completions in prev_completions.items():
            if namespace not in cur_completions:
                cur_completions[namespace] = completions
                cur_tests[namespace] = prev_tests[namespace]
                cur_deps[namespace] = prev_deps[namespace]
        assert len(cur_completions) == len(prev_completions)
        assert len(cur_tests) == len(prev_tests)
        assert len(cur_deps) == len(prev_deps)
        # save to files
        save_data(cur_completions, completion_file)
        save_data(cur_tests, test_file)
        save_data(cur_deps, dep_file)
        # update
        del prev_completions
        del prev_tests
        del prev_deps
        prev_completions = cur_completions
        prev_tests = cur_tests
        prev_deps = cur_deps
    
    return max_round