import os
import re
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from push_adapter import save_model

# Define the model name pattern as a variable
model_name_pattern = 'instruct-7b-r64-a16'

# Define a function to find LoRA weight keys
def find_lora_weight_keys(state_dict):
    return [key for key in state_dict.keys() if 'lora' in key]

# List directories that match the model pattern
root_dir = "/scratch/alif/language-models/segment"
model_dirs = [d for d in os.listdir('.') if model_name_pattern in d and 'cluster' in d]

# Load the base model for compatibility checks
base_model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)

select_groups = [4]
epoch_num = 3
epoch_path = f"epoch_{epoch_num}"

# Process each group of models
pattern = re.compile(rf'{model_name_pattern}-(\d+)-cluster')

for group in set(match.group(1) for d in model_dirs if (match := pattern.match(d))):

    if int(group) in select_groups:
        summed_weights = None
        num_models = 0
        print(f'Processing group {group}...')

        # Accumulate weights for each adapter in the group
        for model_dir in (d for d in model_dirs if d.startswith(f'{model_name_pattern}-{group}-cluster')):
            model_path = os.path.join(root_dir, model_dir, epoch_path) 
            print(f'Loading model from directory: {model_path}')
            model = PeftModel.from_pretrained(base_model, model_path)
            num_models += 1

            # Initialize summed_weights with the first model's state_dict structure
            if summed_weights is None:
                lora_weight_keys = find_lora_weight_keys(model.state_dict())
                summed_weights = {key: torch.zeros_like(model.state_dict()[key]) for key in lora_weight_keys}

            # Sum the weights across all models
            for key in lora_weight_keys:
                summed_weights[key] += model.state_dict()[key]

        # Average the weights
        average_weights = {key: summed_weights[key] / num_models for key in lora_weight_keys}

        # Save the averaged adapter weights and config
        avg_adapter_dir = f'{model_name_pattern}-{group}-sa-ep{epoch_num}-adapter'
        os.makedirs(avg_adapter_dir, exist_ok=True)

        # Save the weights
        torch.save(average_weights, os.path.join(avg_adapter_dir, 'adapter_model.bin'))

        # Save the config - assume the config is the same for all models in the group
        # and use the first model's config as the reference
        first_model_config = PeftConfig.from_pretrained(os.path.join(model_dirs[0], epoch_path))
        first_model_config.save_pretrained(avg_adapter_dir)

        print(f'Finished processing group {group}.')


print(f"Merging averaged adapter weights into base model {base_model_name}...")
output_dir = f'{model_name_pattern}-{group}-sa-ep{epoch_num}-model'
adapter_path = avg_adapter_dir
save_model("adapter", base_model_name, adapter_path, output_dir)
print(f"Successfully saved averaged model at {output_dir}!")