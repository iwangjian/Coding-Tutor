import os
import torch
from typing import List, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    Trainer,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)
from .model import Verifier


class VerifierTrainer(Trainer):
    def __init__(self, model, args, tokenizer, train_dataset, eval_dataset):
        super().__init__(model, args,
                         tokenizer=tokenizer,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.model

        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)


def load_model(
    base_model_name_or_path: str,
    trained_verifier_model_path: str = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] =  ["q_proj", "v_proj"],
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = False
):
    # Load the pre-trained model and tokenizer
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
    compute_dtype = (
        torch.float16
        if fp16
        else (torch.bfloat16 if bf16 else torch.float32)
    )    

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map=device_map,
        torch_dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(  
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        use_cache=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=gradient_checkpointing)
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        base_model.is_parallelizable = True
        base_model.model_parallel = True

    # Set tokenizer's padding token and padding side
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        truncation_side='left',  # set to 'left' to truncate the input from the left
        trust_remote_code=True
    )
    if base_model.config.model_type == "llama" or base_model.config.model_type == "mistral":
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap the model with the defined PRM model
    verify_model = Verifier(
        model=base_model,
        lora_config=lora_config,
        torch_dtype=compute_dtype
    )

    if trained_verifier_model_path is not None:
        print(f"Loading trained verifier model from {trained_verifier_model_path}")
        # Load state dict with weights_only=True for security
        state_dict = torch.load(trained_verifier_model_path, weights_only=True)
        try:
            verify_model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading model state dict: {e}")
            # Optional: Print missing and unexpected keys
            missing_keys, unexpected_keys = verify_model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

        # Only support single GPU for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        verify_model.to(device)
        verify_model.eval()

    return verify_model, tokenizer
