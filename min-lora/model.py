import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, x):
        # Second parameter 'normalized_shape' is the shape of the input
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    
    
"""
Everything required for SA (single head and multi-head) in one class.
"""
class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
    def forward(self, x):
        B, T, C = x.size()
        
        # The attn linear layer expands to 3x n_embd (embed dim)
        # When we split the output, we pass a chunk size of n_embd and split along dim 2 (channel dim)
        # As a result, we get 3 equal-sized chunks for q, k, and v
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        
        # Concatenate outputs from all attenion heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    

"""
Basic MLP with a "fan-out" and "fan-in" transformation and GeLU activation.
"""
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

"""
Transformer (decoder) blocks that will be stacked.
"""
class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x):
        
        # Calculations with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            t_emb = nn.Embedding(config.vocab_size, config.n_embd),
            p_emb = nn.Embedding(config.block_size, config.n_embd),
            dropout = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        
    # Initialize weights: _ following a function name denotes in-place operation
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
     
    def forward(self, idx, targets=None):
        
        # The input idx denotes word (token) indices according to our dictionary
        device = idx.device
        b, t = idx.size()
        
        # The possible positions of a token given the time (t) sequence dim
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.p_emb(pos)
        
        # Token embeddings
        tok_emb = self.transformer.t_emb(idx)
        
        # Add embeddings and pass through all of the transformer blocks
        # Finally, apply layer normalization to output
        x = self.transformer.dropout(tok_emb + pos_emb)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Map the processed embeddings back to the vocabulary logits
        # During inference, set loss to none and forward LM head on last token embedding
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) # Forward the LM head on the last time (B, T, C) token
            loss = None
        
        return logits, loss
    
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    
    
    def get_num_params(self, non_embedding=True):
        
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.p_emb.weight.numel()
        return n_params


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) achieved:expected V100 peak FLOPS
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        
        # Where does this calculation come from?
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 112e12 # V100 GPU peak flops is 112
        mfu = flops_achieved / flops_promised
        return mfu
    
    # TODO: implement loading from pre-trained