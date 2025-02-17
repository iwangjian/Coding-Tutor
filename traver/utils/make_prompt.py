import json
import os
import re
from utils import load_json_data
from tqdm import tqdm
from argparse import ArgumentParser
import tiktoken


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--student_pretest", action='store_true')
    parser.add_argument("--student_posttest", action='store_true')
    parser.add_argument("--prompt_element_file", type=str, default='prompt/prompt_elements_final.jsonl',
                        help='The prompt element file containing the prompt elements.')
    parser.add_argument("--simulated_file", type=str,
                        help='The simulated file containing the simulated dialogues.')
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--student_level", type=str, choices=['low_level', 'med_level', 'high_level', 'oracle'])
    parser.add_argument("--max_interaction_round", type=int,default=8, 
                        help="The max number of interaction rounds.")
    parser.add_argument("--max_cognitive_load", type=int, default=60,
                        help="The max number of words at a turn, denoting the cognitive load for the student.")
    parser.add_argument("--max_code_context", type=int, default=1024,
                        help="The max number of tokens for context above the target code.")
    return parser.parse_args()


def prompt_student(d, tokenizer, level, max_code_context=1024, is_pretest=False):
    # load template
    assert level in ['low_level', 'med_level', 'high_level', 'oracle']
    setting = f'pretest_{level}' if is_pretest else level
    template = open(f'prompt/template/student_{setting}.txt', 'r').read()
    
    if d['class_name']:
        input_code = f"class {d['class_name']}:\n" + d['input_code']
    else:
        input_code = d['input_code']

    if setting == 'low_level' or setting == 'pretest_low_level':
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code
        )
    elif setting == 'med_level' or setting == 'pretest_med_level':
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code,
            dependency_path=d['dependency_sampled']
        )
    elif setting == 'high_level' or setting == 'pretest_high_level':
        contexts_above_ids = tokenizer.encode(d['contexts_above'])
        if len(contexts_above_ids) > max_code_context:
            contexts_above_ids = contexts_above_ids[-max_code_context:]
            contexts_above = tokenizer.decode(contexts_above_ids)
        else:
            contexts_above = d['contexts_above']
        
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code,
            dependency_path=d['dependency_sampled'],
            contexts_above=contexts_above
        )
    else:
        # pretest oracle student
        contexts_above_ids = tokenizer.encode(d['contexts_above'])
        if len(contexts_above_ids) > max_code_context:
            contexts_above_ids = contexts_above_ids[-max_code_context:]
            contexts_above = tokenizer.decode(contexts_above_ids)
        else:
            contexts_above = d['contexts_above']
        
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code,
            dependency_path=d['dependency_all'],
            contexts_above=contexts_above,
            reference_steps=d['reference_steps']
        )
    return prompt

def prompt_student_posttest(conversation, d, tokenizer, level, max_cognitive_load=60, max_code_context=1024):
    # load template
    assert level in ['low_level', 'med_level', 'high_level']
    template = open(f'prompt/template/student_posttest_{level}.txt', 'r').read()

    # get dialogue context
    dialogue_context = ""
    for turn in conversation:
        # set max cognitive load if the tutor's content is overloaded
        if "tutor" in turn:
            if len(turn["tutor"].split(' ')) > max_cognitive_load:
                turn["tutor"] = " ".join(turn["tutor"].split(' ')[-max_cognitive_load:])
        dialogue_context += f"{str(turn)}\n"

    if d['class_name']:
        input_code = f"class {d['class_name']}:\n" + d['input_code']
    else:
        input_code = d['input_code']

    if level == 'low_level':
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code,
            dialogue_context=dialogue_context
        )
    elif level == 'med_level':
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code,
            dependency_path=d['dependency_sampled'],
            dialogue_context=dialogue_context
        )
    else:
        # high level
        contexts_above_ids = tokenizer.encode(d['contexts_above'])
        if len(contexts_above_ids) > max_code_context:
            contexts_above_ids = contexts_above_ids[-max_code_context:]
            contexts_above = tokenizer.decode(contexts_above_ids)
        else:
            contexts_above = d['contexts_above']
        
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code,
            dependency_path=d['dependency_sampled'],
            contexts_above=contexts_above,
            dialogue_context=dialogue_context
        )
    return prompt
    

def prompt_tutor(d, tokenizer, setting="base", max_code_context=1024):
    # load template
    if setting == "base":
        template = open(f'prompt/template/tutor_base.txt', 'r').read()
    elif setting == "RG":
        template = open(f'prompt/template/tutor_RG.txt', 'r').read()
    elif setting == "KT":
        template = open(f'prompt/template/tutor_KT.txt', 'r').read()
    else:
        if os.path.exists(f'prompt/template/tutor_{setting}.txt'):
            template = open(f'prompt/template/tutor_{setting}.txt', 'r').read()
        else:
            raise FileNotFoundError(f"Template file for tutor setting {setting} is not found.")
    
    if d['class_name']:
        input_code = f"class {d['class_name']}:\n" + d['input_code']
    else:
        input_code = d['input_code']
    dependency_paths = d['dependency_all'].strip().split("\n\n")
    reference_steps = extract_steps(d['reference_steps'])
    kc_dependency, kc_reference = [], []
    idx = 1
    for dp in dependency_paths:
        dp = dp.replace("{", "[").replace("}", "]")
        kc_dependency.append("KC-{}: {}".format(idx, dp))
        idx += 1
    for rs in reference_steps:
        rs = rs.replace("{", "[").replace("}", "]")
        kc_reference.append("KC-{}: {}".format(idx, rs))
        idx += 1

    if setting == "RG":
        prompt = template
    elif setting == "KT":
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code,
            dependency_path="\n".join(kc_dependency),
            reference_steps="\n".join(kc_reference),
            conversation="{conversation}",
            previous_estimation="{previous_estimation}"
        )
    else:
        contexts_above_ids = tokenizer.encode(d['contexts_above'])
        if len(contexts_above_ids) > max_code_context:
            contexts_above_ids = contexts_above_ids[-max_code_context:]
            contexts_above = tokenizer.decode(contexts_above_ids)
        else:
            contexts_above = d['contexts_above']
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code,
            contexts_above=contexts_above,
            dependency_path="\n".join(kc_dependency),
            reference_steps="\n".join(kc_reference),
        )
    return prompt


def prompt_moderator(d, tokenizer, max_code_context=1024):
    # load template
    template = open(f'prompt/template/moderator.txt', 'r').read()
    
    if d['class_name']:
        input_code = f"class {d['class_name']}:\n" + d['input_code']
    else:
        input_code = d['input_code']

    # get context above the target code
    contexts_above_ids = tokenizer.encode(d['contexts_above'])
    if len(contexts_above_ids) > max_code_context:
        contexts_above_ids = contexts_above_ids[-max_code_context:]
        contexts_above = tokenizer.decode(contexts_above_ids)
    else:
        contexts_above = d['contexts_above']
    
    prompt = template.format(
        function_name=d['function_name'],
        input_code=input_code,
        dependency_path=d['dependency_all'],
        contexts_above=contexts_above,
        reference_steps=d['reference_steps']
    )
    return prompt

def get_element(prompt_elements, namespace):
    for d in prompt_elements:
        if d['namespace'] == namespace:
            return d
    return None

def extract_steps(reference_steps):
    steps = re.split(r'\n(?=\d+\.)', reference_steps.strip())
    return [step.strip() for step in steps if step.strip()]  

def main():
    args = parse_args()
    print(args)

    if args.student_pretest:
        prompt_elements = load_json_data(args.prompt_element_file)
        tokenizer = tiktoken.encoding_for_model("gpt-4")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_file = os.path.join(args.output_dir, f'prompt_pretest_{args.student_level}.jsonl')
        with open(output_file, 'w') as f:
            for d in tqdm(prompt_elements):
                prompt = prompt_student(d, tokenizer,
                                        level=args.student_level,
                                        max_code_context=args.max_code_context,
                                        is_pretest=True)
                f.write(json.dumps({'namespace': d['namespace'], 'prompt': prompt}) + '\n')
        print(f"Prompt file is saved to {output_file}")
    elif args.student_posttest:
        if args.simulated_file is None:
            raise ValueError("Please specify the simulated file for student posttest!")
        simulated_dialogs = load_json_data(args.simulated_file)
        prompt_elements = load_json_data(args.prompt_element_file)
        assert len(simulated_dialogs) == len(prompt_elements)
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # generate prompts for all rounds
        output_file = os.path.join(args.output_dir, f'prompt_all_rounds.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for dial in tqdm(simulated_dialogs):
                pel = get_element(prompt_elements, dial['namespace'])
                prompt = prompt_student_posttest(
                    dial['conversation'], pel, tokenizer,
                    level=args.student_level,
                    max_cognitive_load=args.max_cognitive_load,
                    max_code_context=args.max_code_context
                )
                f.write(json.dumps({'namespace': pel['namespace'], 'prompt': prompt}) + '\n')
        print(f"Prompt file is saved to {output_file}")
        
        # generate prompts for each round
        for rdx in range(1, args.max_interaction_round + 1):
            output_file = os.path.join(args.output_dir, f'prompt_round_{rdx}.jsonl')
            with open(output_file, 'w', encoding='utf-8') as f:
                for dial in tqdm(simulated_dialogs):
                    convs = dial['conversation']
                    pel = get_element(prompt_elements, dial['namespace'])
                    if rdx * 2 <= len(convs):
                        conv = convs[:rdx * 2]
                        prompt = prompt_student_posttest(
                            conv, pel, tokenizer,
                            level=args.student_level,
                            max_cognitive_load=args.max_cognitive_load,
                            max_code_context=args.max_code_context
                        )
                        f.write(json.dumps({'namespace': pel['namespace'], 'round': rdx, 'prompt': prompt}) + '\n')
            print(f"Prompt file is saved to {output_file}")


if __name__ == '__main__':
    main()