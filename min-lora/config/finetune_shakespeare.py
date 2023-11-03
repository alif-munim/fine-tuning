import time

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

init_from = 'gpt2-large' # second largest gpt2 model
# init_from = 'gpt2-xl' # largest gpt2 model

if init_from == 'gpt2-xl':
    
    # decrease grad accum from 32 to save memory
    gradient_accumulation_steps = 8
    block_size = 128

elif init_from == 'gpt2-large':
    
    # 32 gradient accumulation steps to simulate large batch size
    # No change to block size of 1024
    gradient_accumulation_steps = 32


# finetune at constant LR
learning_rate = 3e-5
decay_lr = False