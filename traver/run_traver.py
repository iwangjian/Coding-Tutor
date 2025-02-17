# -*- coding: utf-8 -*-
import time
import json
import os
import argparse
import tiktoken
from tqdm import tqdm
from chatarena.agent import Player, Moderator
from chatarena.agent_tutor import Tutor
from chatarena.backends import GPTChat, O1Chat, VLLMChat
from chatarena.environments.conversation_tutoring import TutoringConversation
from chatarena.arena_tutoring import TutoringArena
from utils import (
    load_json_dict,
    load_json_data, 
    load_api, 
    load_finished_data, 
    convert_to_json
)
from utils.make_prompt import (
    prompt_student, 
    prompt_tutor, 
    prompt_moderator
)
from verifier.data_utils import OnlineDataBuilder
from verifier.model_utils import load_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tutor_setting", type=str)
    parser.add_argument("--namespace_file", type=str, default='prompt/namespaces.json')
    parser.add_argument("--prompt_element_file", type=str, default='prompt/prompt_elements_final.jsonl')
    parser.add_argument("--output_dir", type=str,  required=True,
                        help="The output directory to save the simulated dialog data.")
    
    parser.add_argument("--use_KT", type=str2bool, default="true")
    parser.add_argument("--verifier_base_model_path", type=str, default=None)
    parser.add_argument("--verifier_model_dir", type=str, default=None)
    parser.add_argument("--verifier_model_parts", type=str, default="0,1,2,3,4")
    parser.add_argument("--verifier_max_length", type=int, default=2000)
    
    parser.add_argument("--tutor_model_name_or_path", type=str, default="gpt-4o")
    parser.add_argument("--tutor_max_tokens", type=int, default=300, 
                        help="The max number of tokens to generate for the tutor.")
    parser.add_argument("--tutor_num_responses", type=int, default=1)
    
    parser.add_argument("--student_model_name_or_path", type=str, default="Mixtral-8x7B-Instruct")
    parser.add_argument("--student_setting", type=str, choices=['low_level', 'med_level', 'high_level'])
    parser.add_argument("--student_max_tokens", type=int, default=300,
                        help="The max number of tokens to generate for the student.")
    parser.add_argument("--max_code_context", type=int, default=1024,
                        help="The max number of tokens for context above the target code.")
    
    parser.add_argument('--api_key_file', type=str, default=None)
    parser.add_argument('--azure_endpoint', type=str, default=None)
    parser.add_argument('--vllm_api_key', type=str, default='EMPTY')
    parser.add_argument('--vllm_endpoint_tutor', type=str, default='http://localhost:8001/v1')
    parser.add_argument('--vllm_endpoint_student', type=str, default='http://localhost:8002/v1')
    parser.add_argument("--max_interaction_round", type=int,default=8, 
                        help="The max number of interaction rounds.")
    parser.add_argument("--max_latest_messages", type=int, default=8,
                        help="The maximum number of latest messages to consider in the backend.")
    parser.add_argument("--temperature", type=float, default=0.4, 
                        help="The temperature to use in sampling.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="The top_p to use in sampling.")
    parser.add_argument("--show_description", type=str2bool, default="true", 
                        help="Whether to show the role description.")
    parser.add_argument("--show_message", type=str2bool, default="true", 
                        help="Whether to show the conversation messages.")
    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def get_output_path(args):
    if "gpt-" in args.tutor_model_name_or_path or "o1" in args.tutor_model_name_or_path:
        output_subdir = os.path.join(args.output_dir, 
                                     f"{args.tutor_setting}/{args.tutor_model_name_or_path}/{args.student_setting}")
    else:
        model_name = args.tutor_model_name_or_path.split("/")[-1]
        output_subdir = os.path.join(args.output_dir, f"{args.tutor_setting}/{model_name}/{args.student_setting}")
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    output_path = os.path.join(output_subdir, "simulated_dialogs.jsonl")
    return output_path

def interactive_simulation(args, prompt_data, tutor_backend, student_backend, moderator_backend, 
                           verifier_model=None, verifier_data_builder=None):
    """Generate dialog data between the student and the tutor."""
    output_path = get_output_path(args)
    
    # load output data to be simulated (skip finished data)
    finished_data = load_finished_data(output_path)
    print(f"Skip {len(finished_data)} finished data.")

    with open(output_path, "a", encoding='utf-8') as fw:
        for js in tqdm(prompt_data):
            if js["namespace"] in finished_data:
                continue

            if verifier_data_builder is not None:
                verifier_data_builder.set_namespace(namespace=js["namespace"])
            
            tutor = Tutor(
                role_desc=js["tutor_desc"], KT_desc=js["KT_desc"],
                backend=tutor_backend, request_prompt=js["request_prompt"],
                verifier=verifier_model, verifier_data_builder=verifier_data_builder,
                num_responses=args.tutor_num_responses,
                use_KT=args.use_KT
            )

            student = Player(
                name="student", role_desc=js["student_desc"],
                backend=student_backend,
            )

            moderator = Moderator(
                role_desc=js["moderator_desc"], 
                backend=moderator_backend,
                terminal_condition="According to the dialogue history above, do you think the tutor's goal is completed? Please answer 'yes' or 'no'.",
            )

            env = TutoringConversation(
                player_names=[tutor.name, student.name], 
                moderator=moderator, 
                moderator_period="round"
            )
            
            arena = TutoringArena(players=[tutor, student], environment=env)
            arena.launch_cli(max_steps=args.max_interaction_round * 2, 
                             show_description=args.show_description, 
                             show_message=args.show_message, 
                             interactive=False)
            
            # save the simulated dialog to file
            messages = env.get_observation()
            simulated_convs = []
            for msg in messages:
                if msg.agent_name == tutor.name:
                    utt = {"tutor": msg.content}
                else:
                    utt = {"student": msg.content}
                simulated_convs.append(utt)
            
            thoughts = env.get_thought(player_name=tutor.name)
            tutor_thoughts = []
            for thought in thoughts:
                thought_dict = {
                    "turn": thought.turn,
                    "agent_name": thought.agent_name,
                    "response_candidates": thought.candidates
                }
                tutor_thoughts.append(thought_dict)
            
            write_line = {
                "namespace": js["namespace"],
                "conversation": simulated_convs,
                "tutor_thoughts": tutor_thoughts
            }
            fw.write(json.dumps(write_line, ensure_ascii=False) + "\n")
            fw.flush()

    print(f"Saved to {output_path}")

    # for readability, convert the jsonl file to json format
    convert_to_json(output_path, output_path.replace(".jsonl", ".json"))


def run_simulation(args, prompt_data, verifier_model=None, verifier_data_builder=None):
    print(f"Total of {len(prompt_data)} prompt samples.")

    if "gpt-" in args.tutor_model_name_or_path:
        assert args.api_key_file is not None, "Please provide the API key file for OpenAI."
        api_key = load_api(args.api_key_file)
        if args.tutor_model_name_or_path == "gpt-3.5":
            tutor_model = "gpt35-1106"  # Azure OpenAI deploy name for 'gpt-3.5-turbo-1106'
        elif args.tutor_model_name_or_path == "gpt-4":
            tutor_model = 'GPT4-1106-preview' # Azure OpenAI deploy name for 'gpt-4-1106-preview'
        elif args.tutor_model_name_or_path == "gpt-4o":
            tutor_model = 'GPT4o'   # Azure OpenAI deploy name for 'gpt-4o-2024-05-13'
        else:
            raise ValueError(f"Unknown tutor model name: {args.tutor_model_name_or_path}")
        
        tutor_backend = GPTChat(
            api_key=api_key,
            azure_endpoint=args.azure_endpoint,
            model=tutor_model, 
            temperature=args.temperature, 
            top_p=args.top_p, 
            max_tokens=args.tutor_max_tokens,
            max_latest_messages=args.max_latest_messages
        )

    elif "o1" in args.tutor_model_name_or_path:
        assert args.api_key_file is not None, "Please provide the API key file for OpenAI."
        api_key = load_api(args.api_key_file)
        tutor_model = 'o1-mini'
        tutor_backend = O1Chat(
            api_key=api_key,
            azure_endpoint=args.azure_endpoint,
            model=tutor_model,
            api_version = "2024-09-01-preview",
            max_latest_messages=args.max_latest_messages
        )
    else:
        tutor_backend = VLLMChat(
            vllm_api_key=args.vllm_api_key,
            vllm_endpoint=args.vllm_endpoint_tutor,
            model_name_or_path=args.tutor_model_name_or_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.tutor_max_tokens,
            max_latest_messages=args.max_latest_messages
        )
    # set the `temperature` and `top_p` to fixed values for consistent comparison
    student_backend = VLLMChat(
        vllm_api_key=args.vllm_api_key,
        vllm_endpoint=args.vllm_endpoint_student,
        model_name_or_path=args.student_model_name_or_path,
        max_tokens=args.student_max_tokens,
        max_latest_messages=args.max_latest_messages,
        temperature=0.4,
        top_p=0.95
    )
    moderator_backend = VLLMChat(
        vllm_api_key=args.vllm_api_key,
        vllm_endpoint=args.vllm_endpoint_student,
        model_name_or_path=args.student_model_name_or_path,
        temperature=0.1,
        top_p=0.95,
        max_tokens=100,
        max_latest_messages=-1
    )

    # run interactive simulation
    interactive_simulation(args, prompt_data, tutor_backend, student_backend, moderator_backend,
                           verifier_model=verifier_model,
                           verifier_data_builder=verifier_data_builder)


def main(args):
    prompt_elements = load_json_data(args.prompt_element_file)
    tokenizer = tiktoken.encoding_for_model("gpt-4")

    if args.verifier_model_dir is not None:
        use_verifier = True
        verifier_model_parts = [p for p in args.verifier_model_parts.split(",")]
        verifier_model_paths = []
        for idx in verifier_model_parts:
            verifier_model_path = os.path.join(args.verifier_model_dir, f"part{idx}", "pytorch_model.bin")
            verifier_model_paths.append(verifier_model_path)
            if not os.path.exists(verifier_model_path):
                use_verifier = False
                raise Warning(f"Verifier model path not found: {verifier_model_path}")
        
        if use_verifier:
            verifier_template = open(f'prompt/template/verifier.txt', 'r').read()
            # Simulate the dialogues with the verifier
            namespaces = load_json_dict(args.namespace_file)
            part_lists = namespaces["part_lists"]
            assert len(part_lists) == len(verifier_model_paths), "The number of verifier models does not match the number of parts."
            
            for idx, namespaces in enumerate(part_lists):
                verifier_model_path = verifier_model_paths[idx]
                print(f"Simulating part {idx} with verifier model: {verifier_model_path}")
                
                # load verifier model
                verifier_model, verifer_tokenizer = load_model(
                    base_model_name_or_path=args.verifier_base_model_path,
                    trained_verifier_model_path=verifier_model_path
                )
                
                elements = []
                for d in prompt_elements:
                    if d["namespace"] in namespaces:
                        elements.append(d)
                
                # define verifier data builder
                verifier_data_builder = OnlineDataBuilder(
                    elements=elements,
                    data_template=verifier_template,
                    tokenizer=verifer_tokenizer,
                    max_length=args.verifier_max_length
                )
                
                prompt_data_part = []
                for d in tqdm(elements):
                    tutor_desc = prompt_tutor(d, tokenizer,
                                            setting="base",
                                            max_code_context=args.max_code_context)
                    KT_desc = prompt_tutor(d, tokenizer, setting="KT")
                    request_prompt = prompt_tutor(d, tokenizer, setting="RG")
                    student_desc = prompt_student(d, tokenizer,
                                                level=args.student_setting,
                                                max_code_context=args.max_code_context,
                                                is_pretest=False)
                    moderator_desc = prompt_moderator(d, tokenizer,max_code_context=args.max_code_context)
                    prompt_data_part.append({
                        "namespace": d['namespace'], 
                        "tutor_desc": tutor_desc,
                        "KT_desc": KT_desc,
                        "request_prompt": request_prompt,
                        "student_desc": student_desc,
                        "moderator_desc": moderator_desc
                    })
                run_simulation(args, prompt_data_part,
                               verifier_model=verifier_model,
                               verifier_data_builder=verifier_data_builder)

    else:
        # Simulate the dialogues without the verifier
        prompt_data_all = []
        for d in tqdm(prompt_elements):
            tutor_desc = prompt_tutor(d, tokenizer,
                                    setting="base",
                                    max_code_context=args.max_code_context)
            KT_desc = prompt_tutor(d, tokenizer, setting="KT")
            request_prompt = prompt_tutor(d, tokenizer, setting="RG")
            student_desc = prompt_student(d, tokenizer,
                                        level=args.student_setting,
                                        max_code_context=args.max_code_context,
                                        is_pretest=False)
            moderator_desc = prompt_moderator(d, tokenizer,max_code_context=args.max_code_context)
            prompt_data_all.append({
                "namespace": d['namespace'], 
                "tutor_desc": tutor_desc,
                "KT_desc": KT_desc,
                "request_prompt": request_prompt,
                "student_desc": student_desc,
                "moderator_desc": moderator_desc
            })

        run_simulation(args, prompt_data_all)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
