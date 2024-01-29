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


def save_model(save_type, model_name, adapter_path, output_dir, new_state_dict=None, push_to_hub=False):

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        # device_map=device_map,
    )
    
    if save_type == "adapter" :
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
    elif save_type == "merge":
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.load_state_dict(torch.load(new_state_dict))
        model = model.merge_and_unload()
    else:
        print(f"Save type must be specified. Valid options are 'adapter' or 'merge'")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.save_pretrained(output_dir, use_temp_dir=False)
    tokenizer.save_pretrained(output_dir, use_temp_dir=False)

    if push_to_hub:
        model.push_to_hub(output_dir, use_temp_dir=False)
        tokenizer.push_to_hub(output_dir, use_temp_dir=False)


def merge_adapter_checkpoints(model_name, adapter_path):
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        # device_map=device_map,
    )
    
    epochs = [1, 2, 3, 4]
    
    for epoch in epochs:
        epoch_num = "epoch_" + str(epoch)
        epoch_path = os.path.join(adapter_path, epoch_num)
        model = PeftModel.from_pretrained(base_model, epoch_path)
        model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        model.save_pretrained(epoch_path, use_temp_dir=False)
        tokenizer.save_pretrained(epoch_path, use_temp_dir=False)
        print(f"Merged adapter model at {epoch_path}")
        
            
# save_type = "adapter" # Options are "adapter" or "merge"
# push_to_hub = False
# select_epoch = False
# select_checkpoint = True

# model_name = "meta-llama/Llama-2-7b-hf"
# adapter_model = f"qlora-instruct-7b-r32-a16-adapter"
# adapter_folder = "checkpoint-7500"
# adapter_path = os.path.join(adapter_model, adapter_folder)

# merged_name = f"llama-2-7b-instruct_lora-att-d0-r32-a16-2-lora-attn-cg-fisher-ep1-ml3"
# merged_checkpoint = os.path.join("merged_models", merged_name + '.pt')
# new_model = merged_name if save_type == "merge" else os.path.join(adapter_model, "model_new")
        
# save_model("adapter", model_name, adapter_path, new_model)

# model_name = "meta-llama/Llama-2-7b-hf"
# adapter_model = f"guanaco-7b-r64-a16"
# merge_adapter_checkpoints(model_name, adapter_model)