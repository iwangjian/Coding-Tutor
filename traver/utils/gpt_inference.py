from openai import AzureOpenAI
import os
from tqdm import tqdm
import time, json
import multiprocessing
import argparse
from utils import load_api, load_finished_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=['gpt-3.5', 'gpt-4'])
    parser.add_argument('--api_key_file', type=str, default='')
    parser.add_argument("--azure_endpoint", type=str, default='')
    parser.add_argument('--decoding', type=str, choices=['greedy', 'sampling'])
    parser.add_argument('--T', type=float, default=0.4)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--N', type=int, default=20)
    
    return parser.parse_args()

def gpt_completion(item: tuple):
    idx, args, prompt_block, api_key, output_path = item 

    client = AzureOpenAI(
        azure_endpoint = args.azure_endpoint, 
        api_key=api_key,
        api_version="2024-02-15-preview"
    )
    if os.path.exists(output_path):
        finished_ids = load_finished_data(output_path) 
        output_f = open(output_path, 'a')
    else:
        finished_ids = []
        output_f = open(output_path, 'w')
    print(f'Worker {idx} start', 'total:', len(prompt_block), 'finished:', len(finished_ids))
    
    for sample in tqdm(prompt_block, total=len(prompt_block), desc=f'Worker {idx}'):
        sample = json.loads(sample)
        prompt = sample['prompt']
        task_id = sample['namespace']
        if task_id in finished_ids:
            continue

        sample['completion'] = []
        while len(sample['completion']) < args.N:
            flag = False
            while not flag:
                try:
                    if args.T == 0:
                        response = client.chat.completions.create(
                                    model=args.model, 
                                    messages=[{'role': 'user', 'content': prompt}],
                                    temperature=args.T,
                                    n=args.N,
                                )
                    elif args.T > 0:
                        response = client.chat.completions.create(
                                        model=args.model, 
                                        messages=[{'role': 'user', 'content': prompt}],
                                        temperature=args.T,
                                        n=args.N,
                                        top_p=args.top_p
                                )
                    flag = True
                except Exception as e:
                    print(f'Worker {idx}', e)
                    time.sleep(0.5)
            for choice in response.choices:
                assert choice.message.role == 'assistant'
                sample['completion'].append(choice.message.content)
            time.sleep(0.5)
        del sample['prompt']
        output_f.write(json.dumps(sample) + '\n')
        output_f.flush()


if __name__ == "__main__":
    args = parse_args()
    if args.model == 'gpt-3.5':
        args.model = 'gpt35-1106'  # Azure OpenAI deploy name for 'gpt-3.5-turbo-1106'
    elif args.model == 'gpt-4':
        args.model = 'GPT4-1106-preview' # Azure OpenAI deploy name for 'gpt-4-1106-preview'
    else:
        raise ValueError('Invalid model name')
    
    if args.decoding == 'greedy':
        args.T = 0
        args.top_p = 1
        args.N = 1
    
    print(args)

    with open(args.prompt_file, 'r') as f:
        prompt_file = f.readlines()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    api_pool = load_api(args.api_key_file, is_pool=True)

    task_block = []
    api_num = len(api_pool)
    l = len(prompt_file) // api_num
    for i in range(api_num):
        if i == api_num - 1:
            prompt_block = prompt_file[i*l:]
        else:
            prompt_block = prompt_file[i*l:(i+1)*l]
        api_key = api_pool[i]
        output_path = f'{args.output_dir}/completion_block{i}.jsonl'
        task_block.append((i, args, prompt_block, api_key, output_path))

    pool = multiprocessing.Pool(api_num)
    pool.map(gpt_completion, task_block)