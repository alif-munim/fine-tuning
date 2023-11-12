import torch
import os
from model import GPTConfig, GPT

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 
bias = False 
batch_size = 12 
block_size = 1024
vocab_size = 50257

device = 'cuda'


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) 


out_dir = 'out-guanaco'
ckpt_name = 'guanaco_gpt2__ckpt.pt'
ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
print(f"loaded checkpoint from {out_dir}/{ckpt_name}")

checkpoint_model_args = checkpoint['model_args']
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
print(f"initialized GPT2 from model args")

unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
print(f"copied checkpoint state dict")

model.load_state_dict(state_dict)
print(f"pushing model to huggingface hub")
# model.save_pretrained(out_dir)
model.push_to_hub(organization="alif-munim", repo_name="gpt2-small-guanaco")

print(f"testing model loading")
model.from_pretrained("alif-munim/gpt2-small-guanaco")