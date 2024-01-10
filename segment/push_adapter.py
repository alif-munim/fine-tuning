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

adapter_model = "llama-2-7b-instruct_lora-att-d0-r32-a16-2_cluster_0"
adapter_path = os.path.join(adapter_model, "epoch_1")

merged_name = "llama-2-7b-instruct_lora-att-d0-r32-a16-2-lora-fisher-ep1-ml9"
merged_checkpoint = os.path.join("merged_models", merged_name + '.pt')
new_model = merged_name if save_merged else os.path.join(adapter_path, "model")

base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        # device_map=device_map,
    )
model = PeftModel.from_pretrained(base_model, adapter_path)

if save_adapter:
    model = model.merge_and_unload()

if save_merged:
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