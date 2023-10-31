"""
GPT training script.
TODO: assertion statements
TODO: add documentation and comments
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT


out_dir = 'out'

eval_iters = 200
eval_interval = 2000
eval_only = False
log_interval = 1

always_save_checkpoint = True
init_from = 'scratch'

wandb_log = False
wandb_project = 'gpt-ft'
wandb_run_name = 'gpt2'

dataset = 'openwebtext'
grad_accum_steps = 5 # What is this?

batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 600000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5



config_keys = [k for k,v in globals().items() if k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}


backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Start single-gpu or ddp (multi-gpu) run
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    
    # What do these parameters mean?
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    
    assert grad_accum_steps % ddp_world_size == 0
    grad_accum_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    
tokens_per_iter = grad_accum_steps * ddp_world_size * batch_size * block_size
print(f'Tokens per iteration: {tokens_per_iter:,}')


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)





# Data loading
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    # Randomly select index to start block from. Labels are offset by +1 (next token preds)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    
    # Pin x and y to move to GPU asynchronously
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y




# Init
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout)




# TODO: initialize from checkpoint or pretrained OpenAI GPT2 weights
if init_from == 'scratch':
    if meta_vocab_size is None:
        print("Defaulting to GPT-2 vocab size of 50304.")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

 
    
    
# Evaluation and learning rate decay
@torch.no_grad()
def estimate_loss()
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff + (learning_rate - min_lr)



# Compile model, initialize distributed process, and start logging
if compile:
    print("Compiling model...")
    unoptimized_model = model
    model = torch.compile(model)
    
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    
    
# Training Loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model # Unwrap DDP if necessary
running_mfu = -1.0

while True:
    
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict().
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f'Saving checkpoint to {out_dir}')
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt,pt'))
                
    if iter_num == 0 and eval_only:
        break
        
    for micro_step in range(grad_accum_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / grad_accum_steps
        
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
        
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    
    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss_item() * grad_accum_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * grad_accum_steps, dt)
            running_mfu = mfu if running_mfu = -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1
    
    if iter_num > max_iters:
        break
        
if ddp:
    destroy_process_group()
        
