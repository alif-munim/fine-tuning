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

save_merged = True
save_adapter = False
push_to_hub = False
epoch_num = 1
epoch_folder = "epoch_" + str(epoch_num)

adapter_model = f"llama-2-7b-instruct_lora-att-d0-r32-a16-2_adapter_epoch{epoch_num}"
adapter_path = adapter_model

merged_name = f"llama-2-7b-instruct_lora-att-d0-r32-a16-2-lora-attn-cg-fisher-ep1-ml3"
merged_checkpoint = os.path.join("merged_models", merged_name + '.pt')
new_model = merged_name if save_merged else os.path.join(adapter_path, "model")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    # device_map=device_map,
)


if save_adapter:
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

if save_merged:
    adapter_model = f"llama-2-7b-instruct_lora-att-d0-r32-a16-2_cluster_0"
    adapter_path = os.path.join(adapter_model, epoch_folder)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    model.load_state_dict(torch.load(merged_checkpoint))
    model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.save_pretrained(new_model, use_temp_dir=False)
tokenizer.save_pretrained(new_model, use_temp_dir=False)

if push_to_hub:
    model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)