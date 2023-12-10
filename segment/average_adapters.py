import os
import re
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

# Define a function to find LoRA weight keys
def find_lora_weight_keys(state_dict):
    return [key for key in state_dict.keys() if 'lora' in key]

# List directories that match the model pattern
model_dirs = [d for d in os.listdir('.') if 'llama-2-7b-guanaco' in d and 'cluster' in d]

# Load the base model for compatibility checks
base_model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)

single_group = True
select_group = 12

# Process each group of models
for group in set(re.match(r'llama-2-7b-guanaco-(\d+)_cluster', d).group(1) for d in model_dirs if re.match(r'llama-2-7b-guanaco-\d+_cluster', d)):

    
    if single_group and int(group) == select_group:
      summed_weights = None
      num_models = 0
      print(f'Processing group {group}...')
  
      # Accumulate weights for each adapter in the group
      for model_dir in (d for d in model_dirs if d.startswith(f'llama-2-7b-guanaco-{group}_cluster')):
          print(f'Loading model from directory: {model_dir}')
          model = PeftModel.from_pretrained(base_model, model_dir)
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
      avg_adapter_dir = f'llama-2-7b-guanaco-{group}_avg_adapter'
      os.makedirs(avg_adapter_dir, exist_ok=True)
      
      # Save the weights
      torch.save(average_weights, os.path.join(avg_adapter_dir, 'adapter_model.bin'))
      
      # Save the config - assume the config is the same for all models in the group
      # and use the first model's config as the reference
      first_model_config = PeftConfig.from_pretrained(model_dirs[0])
      first_model_config.save_pretrained(avg_adapter_dir)
  
      print(f'Finished processing group {group}.')
