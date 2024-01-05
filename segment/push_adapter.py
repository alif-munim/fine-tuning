import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

model_name = "meta-llama/Llama-2-7b-hf"
new_model = "llama-2-7b-guanaco_lora-att-d1-r64-a16-2-merged-ep1-ml5"
push_to_hub = False

save_adapter = False
adapter_model = "llama-2-7b-guanaco_lora-att-d1-r64-a16-2_cluster_1/epoch_1"

save_merged = True
merged_checkpoint = "merged_models/llama-2-7b-guanaco_lora-att-d1-r64-a16-2-merged.pt"


if save_adapter:
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        # device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, adapter_model)
    model = model.merge_and_unload()

if save_merged:
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
    model.load_state_dict(torch.load(merged_checkpoint))


# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.save_pretrained(new_model, use_temp_dir=False)
tokenizer.save_pretrained(new_model, use_temp_dir=False)

if push_to_hub:
    model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)