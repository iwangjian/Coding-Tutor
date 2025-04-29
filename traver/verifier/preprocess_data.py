import os
import json
import random
from argparse import ArgumentParser
from data_utils import load_json_data, build_model_data, check_adjust_posttest

random.seed(42)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt_element_file", type=str, default="prompt/prompt_elements_final.jsonl")
    parser.add_argument("--chosen_models", type=str, default="all")
    parser.add_argument("--simulation_dir", type=str, default="output/dialogue/vanilla")
    parser.add_argument("--posttest_dir", type=str, default="output/student_posttest/vanilla")
    parser.add_argument("--output_dir", type=str, default="")
    
    return parser.parse_args()

def collect_dialog_data(dialogue_fp, posttest_fp):
    # find which namespaces correctly passed the posttest
    outcome_label = {}
    with open(posttest_fp, 'r', encoding='utf-8') as f:
        for line in f:
            js = json.loads(line)
            if js["namespace"] not in outcome_label:
                outcome_label[js["namespace"]] = 0
            if js['Result'] == 'Pass':
                outcome_label[js["namespace"]] = 1
    # annotate outcome reward for each namespace
    dialogues = []
    if dialogue_fp.endswith('.jsonl'):
        with open(dialogue_fp, 'r', encoding='utf-8') as f:
            for line in f:
                conv = json.loads(line)
                conv["outcome_label"] = outcome_label[conv["namespace"]]
                dialogues.append(conv)
    elif dialogue_fp.endswith('.json'):
        with open(dialogue_fp, 'r', encoding='utf-8') as f:
            conv_list = json.load(f)
            for conv in conv_list:
                conv["outcome_label"] = outcome_label[conv["namespace"]]
                dialogues.append(conv)
    else:
        raise ValueError("Unsupported file format for dialogue_fp. Only .jsonl and .json are supported.")

    return dialogues


def main(args):
    # load template
    template = open(f'prompt/template/verifier.txt', 'r').read()

    prompt_elements = load_json_data(args.prompt_element_file)

    if args.chosen_models == "all":
        chosen_models = os.listdir(args.simulation_dir)
    else:
        chosen_models = args.chosen_models.split(',')
    print(chosen_models)

    all_data = []
    for model in chosen_models:
        model_dir = os.path.join(args.simulation_dir, model)
        for level in os.listdir(model_dir):
            print(f"Processing vanilla {model} - {level} student dialogue data")
            posttest_dir = os.path.join(args.posttest_dir, model, level)
            # check and adjust posttest data
            max_round = check_adjust_posttest(posttest_dir)

            dialogues = collect_dialog_data(os.path.join(model_dir, level, "simulated_dialogs.json"),
                                            os.path.join(posttest_dir, f"round_{max_round}/test_results.jsonl"))
            for dialog in dialogues:
                data_samples = build_model_data(prompt_elements, dialog, level, template)
                all_data.extend(data_samples)
    print("Total number of raw data samples:", len(all_data))

    # stat reward label distribution
    positive_data = []
    negative_data = []
    for data in all_data:
        if data["label"] > 0:
            positive_data.append(data)
        else:
            negative_data.append(data)

    sampled_negative_data = random.sample(negative_data, len(positive_data))
    balanced_data = positive_data + sampled_negative_data
    random.shuffle(balanced_data)
    print("Total number of data samples after balance sampling:", len(balanced_data))


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    with open(os.path.join(args.output_dir, "verifier_data.jsonl"), 'w') as f:
        for data in balanced_data:
            f.write(json.dumps(data) + '\n')
    print("All data saved to", os.path.join(args.output_dir, "verifier_data.jsonl"))
    
    # Load all namespaces and split them into k folds
    with open("prompt/namespaces.json", 'r') as f:
        namespaces = json.load(f)
    num_parts = namespaces["num_parts"]
    part_lists = namespaces["part_lists"]
    data_parts = [[] for _ in range(num_parts)]
    data_parts_label = [[] for _ in range(num_parts)]
    for data in balanced_data:
        namespace = data["namespace"]
        for i in range(num_parts):
            if namespace in part_lists[i]:
                data_parts[i].append(data)
                data_parts_label[i].append(data["label"])
                break
    # stat label distribution
    for i in range(num_parts):
        positive_count = 0
        for label in data_parts_label[i]:
            if label > 0:
                positive_count += 1
        #print(f"Part {i} positive ratio = {positive_count / len(data_parts_label[i])}")
    
    for i in range(num_parts):
        with open(os.path.join(args.output_dir, f"verifier_data_part{i}.jsonl"), 'w') as f:
            for data in data_parts[i]:
                f.write(json.dumps(data) + '\n')
        print(f"Data part {i} saved to", os.path.join(args.output_dir, f"verifier_data_part{i}.jsonl"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
