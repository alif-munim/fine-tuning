import time
from functools import partial

import torch
from lora.model import LoRAParametrization


out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
max_iters = 20

wandb_log = False # feel free to turn on
wandb_project = 'gpt2-shakespeare'
wandb_run_name = 'ft-' + str(time.time())

# only save checkpoints if the validation loss improves
always_save_checkpoint = False
dataset = 'shakespeare'

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
block_size = 1024

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

init_from = 'gpt2-large' # models are gpt2, gpt2-medium, gpt2-large, and gpt2-xl
use_lora = False

if init_from == 'gpt2-xl':
    # decrease grad accum from 32 to save memory
    use_lora = True
    gradient_accumulation_steps = 8
    block_size = 128

if use_lora == True:
    learning_rate = 1e-3
    lora_dropout_p = 0.0
    rank = 4
    lora_alpha = 64
    lora_config = {
        torch.nn.Embedding: {
            "weight": partial(LoRAParametrization.from_embedding, rank=rank, lora_alpha=lora_alpha)
        },
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=lora_alpha)
        },
    }
    

