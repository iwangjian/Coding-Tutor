import os
import json
import pathlib
import datasets
import torch.distributed as dist
from typing import List
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser
from transformers.integrations import deepspeed
from verifier.model_utils import VerifierTrainer
from verifier.model_utils import load_model


@dataclass
class DataArguments:
    data_dir: str = field(
        default="output/verifier_data", metadata={"help": "Path to the training data."}
    )
    eval_part: str = field(
        default="part0", metadata={"help": "The part of the data to use for evaluation: part0, part1, part2, part3, part4."}
    )
    pretrained_model_name_or_path: str = field(
        default=None, metadata={"help": "The path to the pretrained model."}
    )
    max_length: int = field(
        default=2048, metadata={"help": "The maximum length of the input sequence."}
    )
    lora_r: int = field(
        default=8, metadata={"help": "The number of bits for the Lora quantization."}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "The number of bits for the Lora quantization."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for the Lora quantization."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    do_inference: bool = False


@dataclass
class FinetuningArguments(TrainingArguments):
    # The default arguments are defined in transformers.TrainingArguments
    # The following arguments are specified for reference
    output_dir: str = field(
        default="output/verifier_model", metadata={"help": "The directory to save the model."}   
    )
    load_in_8bit: bool = False
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    num_train_epochs: float = 3
    learning_rate: float = 1e-6
    weight_decay: float = 0.001
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    optim: str = "adamw_torch"
    fp16: bool = False
    bf16: bool = False
    lr_scheduler_type: str = "linear"
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 1
    local_rank: int = field(default=0, metadata={"help": "Local rank of the process."})
    random_seed: int = 42


def rank0_print(*args):
    if dist.get_rank() == 0:
        print(*args)

def read_jsonl(source):
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list

def build_dataset(data_list, tokenizer, max_length):
    dataset = datasets.Dataset.from_list(data_list)
    
    def process(data):
        prompt_answer = data['prompt_response']
        label = data['label']

        encoded_pair = tokenizer.encode_plus(
            prompt_answer,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoded_pair['input_ids'].squeeze(),
            'attention_mask': encoded_pair['attention_mask'].squeeze(),
            'labels': label
        }
    
    dataset = dataset.map(process, num_proc=8, remove_columns=["prompt_response", "label"])
    
    return dataset


def main():
    parser = HfArgumentParser(
        (DataArguments, FinetuningArguments)
    )
    (data_args, training_args) = parser.parse_args_into_dataclasses()
    
    
    verifier_model, tokenizer = load_model(
        data_args.pretrained_model_name_or_path,
        lora_r=data_args.lora_r,
        lora_alpha=data_args.lora_alpha,
        lora_dropout=data_args.lora_dropout,
        lora_target_modules=data_args.lora_target_modules,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        gradient_checkpointing=training_args.gradient_checkpointing
    )

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        verifier_model.print_trainable_parameters()    
    
    rank0_print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    rank0_print(f"Tokenizer bos_token_id: {tokenizer.bos_token_id}")
    rank0_print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
    rank0_print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    
    
    # Load training set and evaluation set
    eval_fp = os.path.join(data_args.data_dir, f"verifier_data_{data_args.eval_part}.jsonl")
    eval_data = read_jsonl(eval_fp)
    train_data = []
    for fp in os.listdir(data_args.data_dir):
        if fp.startswith("verifier_data_part") and fp != f"verifier_data_{data_args.eval_part}.jsonl":
            json_data = read_jsonl(os.path.join(data_args.data_dir, fp))
            train_data.extend(json_data)
    rank0_print(f"Loaded {len(train_data)} training samples")
    rank0_print(f"Loaded {len(eval_data)} validation samples")

    # Create a custom dataset
    train_dataset = build_dataset(train_data, tokenizer, max_length=data_args.max_length)
    val_dataset = build_dataset(eval_data, tokenizer, max_length=data_args.max_length)

    trainer = VerifierTrainer(
        verifier_model, training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if training_args.local_rank == 0:
        trainer.save_model()
    
    print("Training complete!")

if __name__ == "__main__":
    main()