import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from peft import LoraConfig, PeftModel


class Verifier(PeftModel):

    def __init__(self, model: PreTrainedModel, lora_config: LoraConfig, torch_dtype=torch.float):
        super().__init__(
            model,
            lora_config
        )
        lora_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

        # Transformation parameters
        self.gain = nn.Parameter(torch.randn(1,))
        self.bias = nn.Parameter(torch.randn(1,))
        self.dropout = nn.Dropout(p=0.2)
        self.vscore_head = nn.Linear(
            self.base_model.get_input_embeddings().embedding_dim, 1, bias=False
        )

        self.to(dtype=torch_dtype)
        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1
    
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return super().state_dict()
    
    def load_state_dict(self, state_dict, strict: bool = True):
        load_result = super().load_state_dict(state_dict, strict=strict)
        # Combine the missing keys and unexpected keys.
        missing_keys, unexpected_keys = [], []
        if len(load_result.missing_keys) != 0:
            missing_keys.extend(load_result.missing_keys)
        if len(load_result.unexpected_keys) != 0:
            unexpected_keys.extend(load_result.unexpected_keys)
        # Return the same thing as PyTorch load_state_dict function.
        return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
    
    def transform(self, last_hidden_states):
        """Apply linear transformation to hidden states."""
        return self.gain * last_hidden_states + self.bias

    def loss_fct(self, v_scores, v_labels):
        """Calculate MSE loss with mask for verification scores."""
        return F.mse_loss(v_scores, v_labels, reduction="mean")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        llm_hidden_states = outputs.hidden_states

        v_hidden_states = self.transform(llm_hidden_states[-1])
        v_hidden_states = v_hidden_states.mean(dim=1)
        v_scores = self.vscore_head(self.dropout(v_hidden_states)).squeeze(-1)

        if labels is not None:
            loss = self.loss_fct(v_scores, labels)
        else:
            loss = None
        
        return {"loss": loss, "score": v_scores}