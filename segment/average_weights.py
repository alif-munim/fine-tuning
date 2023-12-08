from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch

print('Loading peft model weights...')
# Load the models from .safetensors files
model1 = LlamaForCausalLM.from_pretrained("alif-munim/llama-2-7b-guanaco-cluster0", torch_dtype=torch.float32, trust_remote_code=True)
model2 = LlamaForCausalLM.from_pretrained("alif-munim/llama-2-7b-guanaco-cluster1", torch_dtype=torch.float32, trust_remote_code=True)

# Function to find LoRA weight keys
def find_lora_weight_keys(state_dict):
    return [key for key in state_dict.keys() if 'lora' in key]

# Find LoRA weight keys from the first model
lora_weight_keys = find_lora_weight_keys(model1.state_dict())

print('Averaging lora weights..')
# Average the weights
for key in lora_weight_keys:
    weight1 = model1.state_dict()[key]
    weight2 = model2.state_dict()[key]
    average_weight = (weight1 + weight2) / 2

    # Apply the averaged weight to one of the models
    model1.state_dict()[key].copy_(average_weight)

# Now model1 has the averaged LoRA weights

print('Successfully averaged LoRA weights!')

avg_adapter = 'llama-2-7b-guanaco-avg'
model1.save_pretrained(avg_adapter)

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, avg_adapter)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.push_to_hub(avg_adapter, use_temp_dir=False)
tokenizer.push_to_hub(avg_adapter, use_temp_dir=False)