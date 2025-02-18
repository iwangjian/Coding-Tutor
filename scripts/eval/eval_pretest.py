import json
import os
import numpy as np
from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from eval_utils import report_pass_k, report_recall_k


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pretest_dir', type=str, default='output/student_pretest/Mixtral-8x7B-Instruct')
    parser.add_argument('--namespace_file', type=str, default='prompt/namespaces.json')
    parser.add_argument('--data_file', type=str, default='EvoCodeBench-2403/metadata.jsonl')
    parser.add_argument('--k', type=str, default='1,3,5,10')
    parser.add_argument('--n', type=int, default=10)
    return parser.parse_args()

def plot_recall(recall_kmetrics):
    data = []
    for metric in ['R@1', 'R@3', 'R@5', 'R@10']:
        for level, metrics in recall_kmetrics.items():
            mean, std = metrics[metric]
            level_name = {
                'low_level': 'Low-level',
                'med_level': 'Med-level',
                'high_level': 'High-level',
                'oracle': 'Oracle'
            }[level]
            data.append({
                'Student': metric,
                'Level': level_name,
                'Score': mean * 100,
                'Std': std * 100
            })
    
    df = pd.DataFrame(data)
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(5, 4.2))
    
    sns.set_theme(style='white')
    ax = sns.barplot(x='Student', y='Score', hue='Level', data=df)
    
    # compute bar width
    n_groups = len(df['Student'].unique())
    n_items = len(df['Level'].unique())
    total_width = 0.8
    bar_width = total_width / n_items
    
    # add error bars
    for i, student in enumerate(df['Student'].unique()):
        for j, level in enumerate(df['Level'].unique()):
            # get current data point
            mask = (df['Student'] == student) & (df['Level'] == level)
            row = df[mask].iloc[0]
            
            # compute bar x position
            x = i + (j - n_items/2 + 0.5) * bar_width
            y = row['Score']
            yerr = row['Std']
            
            # add error bars
            plt.errorbar(x, y, yerr=yerr, color='black', capsize=3, capthick=1,
                        fmt='none', zorder=10, elinewidth=0.5)
    
    ax.set_title('', fontsize=12)
    ax.set_xlabel('' ,fontsize=12)
    ax.set_ylabel('Recall Rate (%)', fontsize=12)

    # Adjust tick font sizes
    ax.tick_params(axis='x', labelsize=12) 
    ax.tick_params(axis='y', labelsize=12)

    # Add a legend and set legend location
    ax.legend(fontsize=12)
    legend = ax.get_legend()
    legend.set_zorder(20) 

    # set margin
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.08)

    # Adjust layout to minimize white space
    plt.tight_layout()

    plt.show()

def plot_pass(pass_kmetrics):
    # Convert metrics data to DataFrame format
    data = []
    for metric in ['P@1', 'P@3', 'P@5', 'P@10']:
        for level, metrics in pass_kmetrics.items():
            mean, std = metrics[metric]
            # Format level name for display
            level_name = {
                'low_level': 'Low-level',
                'med_level': 'Med-level',
                'high_level': 'High-level',
                'oracle': 'Oracle'
            }[level]
            data.append({
                'Student': metric,
                'Level': level_name,
                'Score': mean * 100,  # Convert to percentage
                'Std': std * 100      # Convert to percentage
            })
    
    df = pd.DataFrame(data)
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(5, 4.2))
    
    sns.set_theme(style='white')
    # Add error bars using the 'ci' parameter
    ax = sns.barplot(x='Student', y='Score', hue='Level', data=df)
    
    # compute bar width
    n_groups = len(df['Student'].unique())
    n_items = len(df['Level'].unique())
    total_width = 0.8
    bar_width = total_width / n_items
    
    # add error bars
    for i, student in enumerate(df['Student'].unique()):
        for j, level in enumerate(df['Level'].unique()):
            # get current data point
            mask = (df['Student'] == student) & (df['Level'] == level)
            row = df[mask].iloc[0]
            
            # compute bar x position
            x = i + (j - n_items/2 + 0.5) * bar_width
            y = row['Score']
            yerr = row['Std']
            
            # add error bars
            plt.errorbar(x, y, yerr=yerr, color='black', capsize=3, capthick=1,
                        fmt='none', zorder=10, elinewidth=0.5)
    
    ax.set_title('', fontsize=12)
    ax.set_xlabel('',fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)

    # Adjust tick font sizes
    ax.tick_params(axis='x', labelsize=12) 
    ax.tick_params(axis='y', labelsize=12)

    # Add a legend and set legend location
    ax.legend(fontsize=12)
    legend = ax.get_legend()
    legend.set_zorder(20) 

    # set margin
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.08)

    # Adjust layout to minimize white space
    plt.tight_layout()

    plt.show()

def compute_stats(values):
    """
    Helper function to compute mean and std
    """
    return (np.mean(values), np.std(values))

def plot_metrics(recall_kmetrics, pass_kmetrics):
    data = []
    for metric in ['R@1', 'R@3', 'R@5', 'R@10']:
        for level, metrics in recall_kmetrics.items():
            mean, std = metrics[metric]
            level_name = {
                'low_level': 'Low-level',
                'med_level': 'Med-level',
                'high_level': 'High-level',
                'oracle': 'Oracle'
            }[level]
            data.append({
                'Metric Type': 'Recall',
                'K': metric,
                'Level': level_name,
                'Score': mean * 100,
                'Std': std * 100
            })
    
    for metric in ['P@1', 'P@3', 'P@5', 'P@10']:
        for level, metrics in pass_kmetrics.items():
            mean, std = metrics[metric]
            level_name = {
                'low_level': 'Low-level',
                'med_level': 'Med-level',
                'high_level': 'High-level',
                'oracle': 'Oracle'
            }[level]
            data.append({
                'Metric Type': 'Pass',
                'K': metric,
                'Level': level_name,
                'Score': mean * 100,
                'Std': std * 100
            })
    
    df = pd.DataFrame(data)
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    sns.set_theme(style='white')
    
    # Plot recall
    sns.barplot(x='K', y='Score', hue='Level', data=df[df['Metric Type'] == 'Recall'], ax=ax1)
    
    # Plot Pass
    sns.barplot(x='K', y='Score', hue='Level', data=df[df['Metric Type'] == 'Pass'], ax=ax2)
    
    for ax, metric_type in [(ax1, 'Recall'), (ax2, 'Pass')]:
        df_filtered = df[df['Metric Type'] == metric_type]
        n_groups = len(df_filtered['K'].unique())
        n_items = len(df_filtered['Level'].unique())
        total_width = 0.8
        bar_width = total_width / n_items
        
        for i, k in enumerate(df_filtered['K'].unique()):
            for j, level in enumerate(df_filtered['Level'].unique()):
                mask = (df_filtered['K'] == k) & (df_filtered['Level'] == level)
                row = df_filtered[mask].iloc[0]
                
                x = i + (j - n_items/2 + 0.5) * bar_width
                y = row['Score']
                yerr = row['Std']
                
                ax.errorbar(x, y, yerr=yerr, color='black', capsize=3, capthick=1,
                          fmt='none', zorder=10, elinewidth=0.5)
    
    ax1.set_title('')
    ax2.set_title('')
    ax1.set_ylabel('Recall Rate (%)', fontsize=12)
    ax2.set_ylabel('Pass Rate (%)', fontsize=12)
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    legend1 = ax1.legend(fontsize=12)
    legend1.set_zorder(20)
    legend2 = ax2.legend(fontsize=12)
    legend2.set_zorder(20)
    
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout(h_pad=0.2)
    plt.show()


def main():
    args = parse_args()

     # load namespaces
    with open(args.namespace_file, 'r') as f:
        namespaces_split = json.load(f)
    part_lists = namespaces_split["part_lists"]

    # Initialize containers for metrics
    recall_kmetrics = {'low_level': {'R@1': [], 'R@3': [], 'R@5': [], 'R@10': []},
                       'med_level': {'R@1': [], 'R@3': [], 'R@5': [], 'R@10': []},
                       'high_level': {'R@1': [], 'R@3': [], 'R@5': [], 'R@10': []},
                       'oracle': {'R@1': [], 'R@3': [], 'R@5': [], 'R@10': []}}
    pass_kmetrics = {'low_level': {'P@1': [], 'P@3': [], 'P@5': [], 'P@10': []},
                       'med_level': {'P@1': [], 'P@3': [], 'P@5': [], 'P@10': []},
                       'high_level': {'P@1': [], 'P@3': [], 'P@5': [], 'P@10': []},
                       'oracle': {'P@1': [], 'P@3': [], 'P@5': [], 'P@10': []}}

    for level in ['low_level', 'med_level', 'high_level', 'oracle']:
        for namespace_list in part_lists:
            # compute recall@k
            completion_file = os.path.join(args.pretest_dir, level, "completion.jsonl")
            dep_file = os.path.join(args.pretest_dir, level, "dependency_results.jsonl")
            recall_at_k_list = report_recall_k(completion_file, dep_file, args.data_file, args.k, eval_namespaces=namespace_list)
            recall_kmetrics[level]['R@1'].append(recall_at_k_list[0])
            recall_kmetrics[level]['R@3'].append(recall_at_k_list[1])
            recall_kmetrics[level]['R@5'].append(recall_at_k_list[2])
            recall_kmetrics[level]['R@10'].append(recall_at_k_list[3])
            # compute pass@k
            completion_file = os.path.join(args.pretest_dir, level, "completion.jsonl")
            test_file = os.path.join(args.pretest_dir, level, "test_results.jsonl")
            pass_at_k_list = report_pass_k(completion_file, test_file, args.k, args.n, eval_namespaces=namespace_list)
            pass_kmetrics[level]['P@1'].append(pass_at_k_list[0])
            pass_kmetrics[level]['P@3'].append(pass_at_k_list[1])
            pass_kmetrics[level]['P@5'].append(pass_at_k_list[2])
            pass_kmetrics[level]['P@10'].append(pass_at_k_list[3])
    # compute statistics
    for level in ['low_level', 'med_level', 'high_level', 'oracle']:
        for k in ['R@1', 'R@3', 'R@5', 'R@10']:
            recall_kmetrics[level][k] = compute_stats(recall_kmetrics[level][k])
        for k in ['P@1', 'P@3', 'P@5', 'P@10']:
            pass_kmetrics[level][k] = compute_stats(pass_kmetrics[level][k])
    
    plot_metrics(recall_kmetrics, pass_kmetrics)


if __name__ == '__main__':
    main()