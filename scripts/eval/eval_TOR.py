import os
import json
import numpy as np
from argparse import ArgumentParser
from eval_utils import report_pass_k, report_recall_k
from eval_utils import check_adjust_posttest


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--eval_mode', type=str, default='post-test', choices=['pre-test', 'post-test', 'oracle'])
    parser.add_argument('--pretest_dir', type=str, default='output/student_pretest/Mixtral-8x7B-Instruct')
    parser.add_argument('--posttest_dir', type=str, default='output/student_posttest/vanilla/gpt-4o')
    parser.add_argument('--namespace_file', type=str, default='prompt/namespaces.json')
    parser.add_argument('--data_file', type=str, default='EvoCodeBench-2403/metadata.jsonl')
    parser.add_argument('--k', type=str, default='1,3,5,10')
    parser.add_argument('--n', type=int, default=10)
    return parser.parse_args()

def compute_pretest_recall(args, namespace_list, all_levels=['low_level', 'med_level', 'high_level']):
    # compute pre-test recall@k for each level
    recalls = []
    level_recall = {}
    for level in all_levels:
        completion_file = os.path.join(args.pretest_dir, level, "completion.jsonl")
        dep_file = os.path.join(args.pretest_dir, level, "dependency_results.jsonl")
        recall_at_k_list = report_recall_k(completion_file, dep_file, args.data_file, args.k, eval_namespaces=namespace_list)
        recall = np.mean(recall_at_k_list)
        level_recall[level] = recall
        recalls.append(recall)
    avg_recall = np.mean(recalls)
    return level_recall, avg_recall

def compute_posttest_recall(args, namespace_list, all_levels=['low_level', 'med_level', 'high_level']):
    # call compute_recall to get pre-test recall
    pretest_recall, avg_pretest_recall = compute_pretest_recall(args, namespace_list, all_levels)
    
    posttest_recalls = []
    posttest_level_recall = {}
    for level in all_levels:
        if os.path.exists(os.path.join(args.posttest_dir, level, "completion.jsonl")):
            # posttest data with all rounds
            posttest_completion_file = os.path.join(args.posttest_dir, level, "completion.jsonl")
            posttest_dep_file = os.path.join(args.posttest_dir, level, "dependency_results.jsonl")
        else:
            # check and adjust posttest data
            max_round = check_adjust_posttest(os.path.join(args.posttest_dir, level))
            posttest_completion_file = os.path.join(args.posttest_dir, level, f"round_{max_round}/completion.jsonl")
            posttest_dep_file = os.path.join(args.posttest_dir, level, f"round_{max_round}/dependency_results.jsonl")
        
        # compute recall@k using the last round of posttest data
        posttest_recall_at_k_list = report_recall_k(posttest_completion_file, posttest_dep_file, 
                                                    args.data_file, args.k, eval_namespaces=namespace_list)
        posttest_recall = np.mean(posttest_recall_at_k_list)
        posttest_level_recall[level] = posttest_recall
        posttest_recalls.append(posttest_recall)
    
    # containers for metrics
    avg_posttest_recall = np.mean(posttest_recalls)
    metrics = {
        'pretest_recall': {
            'avg': avg_pretest_recall, 
            'low': pretest_recall['low_level'], 
            'med': pretest_recall['med_level'], 
            'high': pretest_recall['high_level']
        },
        'posttest_recall': {
            'avg': avg_posttest_recall, 
            'low': posttest_level_recall['low_level'], 
            'med': posttest_level_recall['med_level'], 
            'high': posttest_level_recall['high_level']
        }
    }
    return metrics


def compute_pretest_pass(args, namespace_list, all_levels=['low_level', 'med_level', 'high_level']):
    # compute pre-test pass@k for each level
    passes = []
    level_pass = {}
    for level in all_levels:
        completion_file = os.path.join(args.pretest_dir, level, "completion.jsonl")
        test_file = os.path.join(args.pretest_dir, level, "test_results.jsonl")
        pass_at_k_list = report_pass_k(completion_file, test_file, args.k, args.n, eval_namespaces=namespace_list)
        pass_at_k = np.mean(pass_at_k_list)
        level_pass[level] = pass_at_k
        passes.append(pass_at_k)
    avg_pass = np.mean(passes)
    return level_pass, avg_pass


def compute_posttest_pass(args, namespace_list, all_levels=['low_level', 'med_level', 'high_level']):
    # call compute_pass to get pre-test pass
    pretest_pass, avg_pretest_pass = compute_pretest_pass(args, namespace_list, all_levels)

    posttest_passes = []
    posttest_level_pass = {}
    for level in all_levels:
        if os.path.exists(os.path.join(args.posttest_dir, level, "completion.jsonl")):
            # posttest data with all rounds
            posttest_completion_file = os.path.join(args.posttest_dir, level, "completion.jsonl")
            posttest_test_file = os.path.join(args.posttest_dir, level, "test_results.jsonl")
        else:
            # check and adjust posttest data
            max_round = check_adjust_posttest(os.path.join(args.posttest_dir, level))
            posttest_completion_file = os.path.join(args.posttest_dir, level, f"round_{max_round}/completion.jsonl")
            posttest_test_file = os.path.join(args.posttest_dir, level, f"round_{max_round}/test_results.jsonl")

        # compute pass@k using the last round of posttest data
        posttest_pass_at_k_list = report_pass_k(posttest_completion_file, posttest_test_file, args.k, args.n, eval_namespaces=namespace_list)
        posttest_pass = np.mean(posttest_pass_at_k_list)
        posttest_level_pass[level] = posttest_pass
        posttest_passes.append(posttest_pass)
    
    avg_posttest_pass = np.mean(posttest_passes)
    metrics = {
        'pretest_pass': {
            'avg': avg_pretest_pass, 
            'low': pretest_pass['low_level'], 
            'med': pretest_pass['med_level'], 
            'high': pretest_pass['high_level']
        },
        'posttest_pass': {
            'avg': avg_posttest_pass, 
            'low': posttest_level_pass['low_level'], 
            'med': posttest_level_pass['med_level'], 
            'high': posttest_level_pass['high_level']
        }
    }
    return metrics


def compute_stats(values):
    """
    Helper function to compute mean and std
    """
    return np.mean(values), np.std(values)


if __name__ == '__main__':
    args = parse_args()

    with open(args.namespace_file, 'r') as f:
        namespaces_split = json.load(f)
    num_parts = namespaces_split["num_parts"]
    namespaces_all = namespaces_split["namespaces_all"]
    part_lists = namespaces_split["part_lists"]

    if args.eval_mode == 'pre-test':
        print("Pre-test Evaluation:")
        # Initialize containers for metrics
        metrics = {
            'recall': {'avg': [], 'low': [], 'med': [], 'high': []},
            'pass': {'avg': [], 'low': [], 'med': [], 'high': []}
        }
        
        # Collect metrics for each namespace
        for namespace_list in part_lists:
            level_recall, avg_recall = compute_pretest_recall(args, namespace_list)
            level_pass, avg_pass = compute_pretest_pass(args, namespace_list)
            
            metrics['recall']['avg'].append(avg_recall)
            metrics['pass']['avg'].append(avg_pass)
            
            for level in ['low', 'med', 'high']:
                level_key = f'{level}_level'
                metrics['recall'][level].append(level_recall[level_key])
                metrics['pass'][level].append(level_pass[level_key])
        
        # Compute statistics
        results = {}
        for metric_type in ['recall', 'pass']:
            results[metric_type] = {}
            for level in ['avg', 'low', 'med', 'high']:
                results[metric_type][level] = compute_stats(metrics[metric_type][level])
        
        # Print results
        for level in ['low', 'med', 'high', 'avg']:
            mean, std = results['recall'][level]
            print(f"{level.capitalize()} Recall: {mean*100:.3f}% ± {std*100:.3f}%")
        print()
        for level in ['low', 'med', 'high', 'avg']:
            mean, std = results['pass'][level]
            print(f"{level.capitalize()} Pass: {mean*100:.3f}% ± {std*100:.3f}%")

    elif args.eval_mode == 'post-test':
        print(f"Post-test Evaluation for {args.posttest_dir} ...")
        # Initialize containers for metrics
        metrics = {
            'pretest_recall': {'avg': [], 'low': [], 'med': [], 'high': []},
            'posttest_recall': {'avg': [], 'low': [], 'med': [], 'high': []},
            'pretest_pass': {'avg': [], 'low': [], 'med': [], 'high': []},
            'posttest_pass': {'avg': [], 'low': [], 'med': [], 'high': []},
        }
        for namespace_list in part_lists:
            recall_metrics = compute_posttest_recall(args, namespace_list)
            pass_metrics = compute_posttest_pass(args, namespace_list)
            for metric_type in ['pretest_recall', 'posttest_recall']:
                for level in ['avg', 'low', 'med', 'high']:
                    metrics[metric_type][level].append(recall_metrics[metric_type][level])
            for metric_type in ['pretest_pass', 'posttest_pass']:
                for level in ['avg', 'low', 'med', 'high']:
                    metrics[metric_type][level].append(pass_metrics[metric_type][level])
        
        # Compute statistics
        results = {}
        for metric_type in ['pretest_recall', 'posttest_recall', 'pretest_pass', 'posttest_pass']:
            results[metric_type] = {}
            for level in ['avg', 'low', 'med', 'high']:
                results[metric_type][level] = compute_stats(metrics[metric_type][level])
        for metric_type in ['TOR_Recall', 'TOR_Pass']:
            results[metric_type] = {}
            for level in ['avg', 'low', 'med', 'high']:
                m = metric_type.split('_')[1].lower()
                posttest_mean, _ = results[f'posttest_{m}'][level]
                pretest_mean, _ = results[f'pretest_{m}'][level]
                results[metric_type][level] = (posttest_mean - pretest_mean) / pretest_mean
        
        # Print results
        for level in ['low', 'med', 'high', 'avg']:
            mean_recall, std_recall = results['posttest_recall'][level]
            mean_tor_recall = results['TOR_Recall'][level]
            print(f"{level.capitalize()} Recall: {mean_recall*100:.3f}% ± {std_recall*100:.3f}%\t{level.capitalize()} TOR-Recall: {mean_tor_recall*100:.3f}%")
        print()
        for level in ['low', 'med', 'high', 'avg']: 
            mean_pass, std_pass = results['posttest_pass'][level]
            mean_tor_pass = results['TOR_Pass'][level]
            print(f"{level.capitalize()} Pass: {mean_pass*100:.3f}% ± {std_pass*100:.3f}%\t{level.capitalize()} TOR-Pass: {mean_tor_pass*100:.3f}%")

    elif args.eval_mode == 'oracle':
        print("Oracle Evaluation:")
        # Initialize containers for metrics
        metrics = {
            'recall': {'avg': [], 'low': [], 'med': [], 'high': []},
            'pass': {'avg': [], 'low': [], 'med': [], 'high': []}
        }
        for namespace_list in part_lists:
            oracle_recall, oracle_avg_recall = compute_pretest_recall(args, namespace_list, all_levels=['oracle'])
            oracle_pass, oracle_avg_pass = compute_pretest_pass(args, namespace_list, all_levels=['oracle'])
            metrics['recall']['avg'].append(oracle_avg_recall)
            metrics['pass']['avg'].append(oracle_avg_pass)
            
            for level in ['low', 'med', 'high']:
                metrics['recall'][level].append(oracle_avg_recall)
                metrics['pass'][level].append(oracle_avg_pass)
            
        # Compute statistics
        results = {}
        for metric_type in ['recall', 'pass']:
            results[metric_type] = {}
            for level in ['low', 'med', 'high', 'avg']:
                results[metric_type][level] = compute_stats(metrics[metric_type][level])
        
        # Print results
        for metric_type in ['recall', 'pass']:
            for level in ['low', 'med', 'high', 'avg']:
                mean, std = results[metric_type][level]
                print(f"Oracle {level.capitalize()} {metric_type.capitalize()}: {mean*100:.3f}% ± {std*100:.3f}%")
            print()
