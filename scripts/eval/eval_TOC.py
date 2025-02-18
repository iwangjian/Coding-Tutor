import json
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt
from eval_utils import report_pass_k, report_recall_k
from eval_utils import check_adjust_posttest


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--tutor_settings', type=str, default='vanilla')
    parser.add_argument('--tutor_models', type=str, default='Qwen2-72B-Instruct,Meta-Llama-3.1-70B-Instruct,gpt-3.5,gpt-4o')
    parser.add_argument('--posttest_base_dir', type=str, default='output/student_posttest')
    parser.add_argument('--posttest_levels', type=str, default='low_level,med_level,high_level')
    parser.add_argument('--pretest_base_dir', type=str, default='output/student_pretest/Mixtral-8x7B-Instruct')
    parser.add_argument('--data_file', type=str, default='EvoCodeBench-2403/metadata.jsonl')
    parser.add_argument('--k', type=str, default='1,3,5,10')
    parser.add_argument('--n', type=int, default=10)
    return parser.parse_args()

def compute_TO(args):
    # check and adjust posttest data
    max_round = check_adjust_posttest(args.posttest_dir)

    # compute per-round pass@k and recall@k
    pass_by_round, recall_by_round = [], []
    for rdx in range(1, max_round + 1):
        completion_file = os.path.join(args.posttest_dir, f"round_{rdx}/completion.jsonl")
        test_file = os.path.join(args.posttest_dir, f"round_{rdx}/test_results.jsonl")
        dep_file = os.path.join(args.posttest_dir, f"round_{rdx}/dependency_results.jsonl")
        
        pass_at_k_list = report_pass_k(completion_file, test_file, args.k, args.n)
        avg_pass = np.mean(pass_at_k_list)
        pass_by_round.append(avg_pass)

        recall_at_k_list = report_recall_k(completion_file, dep_file, args.data_file, args.k)
        avg_recall = np.mean(recall_at_k_list)
        recall_by_round.append(avg_recall)
    
    return pass_by_round, recall_by_round


def plot_TOC_compare(compared_data):
    sns.set_theme(style="white")
    # Create a figure with two subplots side by side
    axs = []
    if len(compared_data) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
        axs.append(ax1)
        axs.append(ax2)
    elif len(compared_data) == 4:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 3.8))
        axs.append(ax1)
        axs.append(ax2)
        axs.append(ax3)
        axs.append(ax4)
    else:
        raise ValueError("Not implemented!")

    # Plot data for each method in separate subplots
    for idx, data in enumerate(compared_data):
        data_name = data['data_name']
        TO_pass_data = data['TO_pass']
        ax = axs[idx]

        # Plot each level's data
        for level, pass_list in TO_pass_data.items():
            turns = list(range(len(pass_list)))
            value_list = [p * 100 for p in pass_list]
            
            if level == 'oracle':
                sns.lineplot(x=turns, y=value_list, ax=ax, label='Oracle', linestyle='--', color='black')
            elif level == 'low_level':
                sns.lineplot(x=turns, y=value_list, ax=ax, label='Low-level', marker='o')
            elif level == 'med_level':
                sns.lineplot(x=turns, y=value_list, ax=ax, label='Med-level', marker='o')
            elif level == 'high_level':
                sns.lineplot(x=turns, y=value_list, ax=ax, label='High-level', marker='o')

        # Configure each subplot
        ax.set_xlabel('Dialogue Turn', fontsize=12)
        ax.set_ylabel('Pass Rate (%)', fontsize=12)
        ax.set_title(data_name, fontsize=12)
        ax.set_xticks(turns)

        ax.grid(True, linestyle='--', color='0.9')
        ax.tick_params(axis='both', which='major', direction='out', length=4, width=1)

        # Add a legend and set legend location
        if data_name == "Llama-3-8B-Instruct" or data_name == "GPT-3.5-Turbo":
            ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1, 0.95))
        else:
            ax.legend(fontsize=12)

    # Adjust layout to minimize white space
    plt.tight_layout()
    plt.show()
    

def main():
    args = parse_args()
    setting_names = args.tutor_settings.split(',')
    model_names = args.tutor_models.split(',')
    
    compared_data = []
    for setting in setting_names:
        for model in model_names:
            model_k = model.replace("gpt-3.5", "GPT-3.5-Turbo")
            model_k = model_k.replace("gpt-4o", "GPT-4o")
            model_k = model_k.replace("Meta-", "")
            if "traver" in setting:
                data_name = f"Ours ({model_k})"
            elif "tree_instruct" in setting:
                data_name = f"TreeInstruct ({model_k})"
            elif "self_refine" in setting:
                data_name = f"Self-Refine ({model_k})"
            else:
                data_name = model_k

            TO_pass_data, TO_recall_data = {}, {}
            posttest_levels = args.posttest_levels.split(',')
            for level in posttest_levels:
                # get pretest pass and recall as the 0-th round's TOs
                pretest_completion_file = os.path.join(args.pretest_base_dir, f"{level}/completion.jsonl")
                pretest_test_file = os.path.join(args.pretest_base_dir, f"{level}/test_results.jsonl")
                pretest_dep_file = os.path.join(args.pretest_base_dir, f"{level}/dependency_results.jsonl")

                pass_at_k_list = report_pass_k(pretest_completion_file, pretest_test_file, args.k, args.n)
                pretest_avg_pass = np.mean(pass_at_k_list)

                recall_at_k_list = report_recall_k(pretest_completion_file, pretest_dep_file, args.data_file, args.k)
                pretest_avg_recall = np.mean(recall_at_k_list)

                # compute TO for each round
                args.posttest_dir = os.path.join(args.posttest_base_dir, f"{setting}/{model}/{level}")
                pass_by_round, recall_by_round = compute_TO(args)
                TO_pass_data[level] = [pretest_avg_pass] + pass_by_round
                TO_recall_data[level] = [pretest_avg_recall] + recall_by_round
            
            # get oracle pass and recall as the upper bound
            oracle_completion_file = os.path.join(args.pretest_base_dir, "oracle/completion.jsonl")
            oracle_test_file = os.path.join(args.pretest_base_dir, "oracle/test_results.jsonl")
            oracle_dep_file = os.path.join(args.pretest_base_dir, "oracle/dependency_results.jsonl")
            pass_at_k_list = report_pass_k(oracle_completion_file, oracle_test_file, args.k, args.n)
            oracle_avg_pass = np.mean(pass_at_k_list)
            recall_at_k_list = report_recall_k(oracle_completion_file, oracle_dep_file, args.data_file, args.k)
            oracle_avg_recall = np.mean(recall_at_k_list)

            TO_pass_data['oracle'] = [oracle_avg_pass] * (len(pass_by_round) + 1)
            TO_recall_data['oracle'] = [oracle_avg_recall] * (len(recall_by_round) + 1)

            data = {
                "data_name": data_name,
                "TO_pass": TO_pass_data,
                "TO_recall": TO_recall_data
            }
            compared_data.append(data)

    plot_TOC_compare(compared_data)


if __name__ == '__main__':
    main()
