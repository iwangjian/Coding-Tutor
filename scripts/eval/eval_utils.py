import json
import os
import copy
import numpy as np


def check_adjust_posttest(posttest_dir):
    # since some examples may not have completions starting from a certain round
    # we need to get completions and eval results from the latest round that has completions
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
                cur_completions[namespace] = copy.deepcopy(completions)
                cur_tests[namespace] = copy.deepcopy(prev_tests[namespace])
                cur_deps[namespace] = copy.deepcopy(prev_deps[namespace])

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

def check_adjust_pass(pass_list):
    # check if the current pass is less than the previous pass
    for i in range(1, len(pass_list)):
        if pass_list[i] < pass_list[i-1]:
            pass_list[i] = pass_list[i-1]
    return pass_list

def report_pass_k(completion_file, log_file, args_k, args_n, eval_namespaces=None):
    # Collect passed completions for each namespace
    passed_completion = {}
    with open(log_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            if js['Result'] == 'Pass':
                namespace, completion = js['namespace'], js['completion']
                if namespace not in passed_completion:
                    passed_completion[namespace] = set()
                passed_completion[namespace].add(completion)

    # Iterate through all completions and count the number of passed completions for each namespace
    results = {}
    with open(completion_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            if eval_namespaces is None:
                if namespace not in results:
                    results[namespace] = 0
                if namespace in passed_completion and completion in passed_completion[namespace]:
                    results[namespace] += 1
            else:
                if namespace in eval_namespaces:
                    if namespace not in results:
                        results[namespace] = 0
                    if namespace in passed_completion and completion in passed_completion[namespace]:
                        results[namespace] += 1
            
    # Compute Pass@k
    pass_at_k_list = []
    k_list = [int(k) for k in args_k.split(',')]
    for k in k_list:
        if k > args_n:
            continue
        pass_at_k = np.mean([compute_pass_at_k(args_n, pass_num, k) for namespace, pass_num in results.items()])
        #print(f'Pass@{k}: {pass_at_k*100}%')
        pass_at_k_list.append(pass_at_k)
    
    return pass_at_k_list

def report_recall_k(completion_file, log_file, data_file, args_k, eval_namespaces=None):
    # Parse the k values
    k_list = []
    for _k in args_k.split(','):
        k_list.append(int(_k))
    max_k = max(k_list)

    # Load the completion data
    output_data = dict()
    with open(completion_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            if eval_namespaces is None:
                if js['namespace'] not in output_data:
                    output_data[namespace] = []
                if len(output_data[namespace]) < max_k: # only consider max_k completions
                    output_data[namespace].append(js)
            else:
                if namespace in eval_namespaces:
                    if js['namespace'] not in output_data:
                        output_data[namespace] = []
                    if len(output_data[namespace]) < max_k: # only consider max_k completions
                        output_data[namespace].append(js)
    
    # Load the benchmark data
    benchmark_data = {}
    with open(data_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            benchmark_data[namespace] = js
    
    parse_results = dict()
    with open(log_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            if namespace not in parse_results:
                parse_results[namespace] = dict()
            parse_results[namespace][completion] = js['generated_dependency']

    results = {}
    for namespace, outputs in output_data.items():
        for output in outputs:
            completion = output['completion']
            if namespace in parse_results:
                generated_dependency = parse_results[namespace][completion]
                data = benchmark_data[namespace]
                reference_dependency = data['dependency']
                recall = compute_recall(generated_dependency, reference_dependency)
                if namespace not in results:
                    results[namespace] = []
                results[namespace].append(recall)
    
    # Compute Recall@k
    recall_at_k_list = []
    for k in k_list:
        recall = 0
        for namespace, recall_list in results.items():
            recall += max(recall_list[:k])
        recall /= len(results) # average the accuracy of samples
        #print(f"Recall@{k}: {recall*100}%")
        recall_at_k_list.append(recall)
    
    return recall_at_k_list

def compute_pass_at_k(n, c, k):
    """
    n: total number of completions per task
    c: number of completions that pass all tests
    k: k in pass_at_k
    """
    if n - c < k:
        return 1
    else:
        return 1.0 - np.prod(1.0 - k / np.arange(n-c+1, n+1))
    
def compute_recall(generated_dependency, reference_dependency):
    reference = []
    for _type, _list in reference_dependency.items():
        reference.extend(_list)
    if generated_dependency is None:
        return 0
    prediction = []
    for _type, _list in generated_dependency.items():
        prediction.extend(_list)
    reference = set(reference)
    prediction = set(prediction)
    recall = len(reference.intersection(prediction)) / len(reference)
    return recall

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