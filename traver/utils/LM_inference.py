from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
from argparse import ArgumentParser
import os
import re
from utils import load_finished_data

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument('--decoding', type=str, default='sampling', choices=['greedy', 'sampling'])
    parser.add_argument("--context_window", type=int, default=20000)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument('--T', type=float, default=0.4)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--yes_or_no_required", action='store_true')
    
    return parser.parse_args()

def load_model(model_name_or_path: str, context_window: int, gpu_memory_utilization: float = 0.95, tensor_parallel_size: int = 1):
    if os.path.exists(model_name_or_path):
        print(f"Loading model from {model_name_or_path}")
    else:
        raise ValueError(f"Model {model_name_or_path} not found.")
    
    model = LLM(model=model_name_or_path, 
                max_model_len=context_window,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True)
    return model


def llm_inference(prompt_file, output_dir, model, sampling_params, yes_or_no_required=False):
    if not os.path.exists(prompt_file):
        print(f"Ignore: Prompt file {prompt_file} not found.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'completion_lm.jsonl')
    finished_data = load_finished_data(output_file)
    
    with open(prompt_file, 'r') as f_in:
        f = f_in.readlines()
        with open(output_file, 'a') as f_out:
            for line in tqdm(f):
                js = json.loads(line)
                prompt = js['prompt']
                task_id = js['namespace']
                if task_id in finished_data:
                    continue
                
                completions = []
                if yes_or_no_required:
                    try_times = 0
                    max_try_times = 3
                    while try_times < max_try_times:
                        try:
                            results = model.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
                            for result in results:
                                for output in result.outputs:
                                    completions.append(output.text)

                            if re.match(r"yes|y|yea|yeah|yep|yup|sure|ok|okay|alright", completions, re.IGNORECASE):
                                completions = "yes"
                                break
                            elif re.match(r"no|n|nope|nah|nay", completions, re.IGNORECASE):
                                completions = "no"
                                break
                            else:
                                try_times += 1
                        except Exception as e:
                            print(f"Error: {e}")
                            try_times += 1
                    if try_times == max_try_times:
                        completions = "no"
                else:
                    results = model.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
                    for result in results:
                        for output in result.outputs:
                            completions.append(output.text)
                
                cases = {'namespace': task_id, 'completion': completions}
                f_out.write(json.dumps(cases) + '\n')
                f_out.flush()

def main():
    args = parse_args()
    if args.yes_or_no_required:
        args.N = 1
    if args.decoding == 'greedy':
        args.T = 0
        args.top_p = 1
        args.N = 1
    
    print(args)

    model = load_model(args.model_name_or_path, args.context_window, 
                        args.gpu_memory_utilization, args.tensor_parallel_size)

    sampling_params = SamplingParams(temperature=args.T, top_p=args.top_p, 
                                     max_tokens=args.max_tokens, n=args.N)

    llm_inference(args.prompt_file, args.output_dir, model, sampling_params, 
                  yes_or_no_required=args.yes_or_no_required)
    
if __name__ == '__main__':
    main()