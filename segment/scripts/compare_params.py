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

push_to_hub = False
model_name = "meta-llama/Llama-2-7b-hf"

save_adapter = False
# adapter_model = "llama-2-7b-instruct_lora-att-d0-r64-a16-2_cluster_1/epoch_1"
# new_model = "llama-2-7b-instruct_lora-att-d0-r64-a16-2_cluster_1/epoch_1/model"

# save_merged = True
merged_checkpoint = "merged_models/llama-2-7b-instruct_lora-att-d0-r32-a16-2-merged-ep1-ml9.pt"
adapter_model = "llama-2-7b-instruct_lora-att-d0-r32-a16-2_cluster_0/epoch_1"
# new_model = "llama-2-7b-instruct_lora-att-d0-r64-a16-2-merged-ep1-ml9"

# checkpoint = torch.load(merged_checkpoint)
# for key, value in checkpoint.items():
#     print(f"{key}: {type(value)}")

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
initial_model = PeftModel.from_pretrained(base_model, adapter_model)
initial_state_dict = initial_model.state_dict()

merged_model = PeftModel.from_pretrained(base_model, adapter_model)
merged_model.load_state_dict(torch.load(merged_checkpoint))
merged_state_dict = merged_model.state_dict()

update_list = []
for param_name in initial_state_dict:
    if param_name in merged_state_dict:
        if not torch.equal(initial_state_dict[param_name], merged_state_dict[param_name]):
            update_list.append(param_name)
            
non_lora_list = []
for param_name in update_list:
    if 'lora' not in param_name:
        non_lora_list.append(param_name)
        
if len(non_lora_list) > 0:
    print(f"{len(non_lora_list)} non-LoRA parameters were updated:")
    for param_name in non_lora_list:
        print(param_name)
else:
    print(f"Success, only LoRA parameters were updated!")
