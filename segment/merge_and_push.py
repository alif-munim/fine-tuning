import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
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
adapter_model = "llama-2-7b-guanaco"

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

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.save_pretrained((adapter_model, use_temp_dir=False)
tokenizer.save_pretrained((adapter_model, use_temp_dir=False)

model.push_to_hub(adapter_model, use_temp_dir=False)
tokenizer.push_to_hub(adapter_model, use_temp_dir=False)