import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096                                   # Dimension of word embedding vectors
    n_layers: int = 32                  
    n_heads: int = 32                                 # Number of heads for queries
    n_kv_heads: Optional[int] = None                  # Number of heads for K and V
    vocab_size: int = -1                              # Set when loading tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None        # Indicate the hidden dim of the FF layer
    norm_eps: float = 1e-5
    
    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None
    

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    
    # Build theta parameter sequence
    # Formula: 10000 ^ (-2(i-1)/dim) for i = [1, 2, ..., dim / 2]
    # Shape: (head_dim / 2), applied after splitting into multi-head
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    
    # Build the position sequence (m)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    
    # Multiply "m" with outer product (all possible combinations between vectors)
    # Shape: (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    
    # Write numbers in complex (polar) form [c = R * exp(i * m * theta)], where R = 1
    # Shape: (seq_len, head_dim / 2)
    # We must convert to complex form to represent angle (sines and cosines) 
    # Re^i(theta) = Rcos(theta) + R(i)sin(theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    
    # Original shape: (batch, seq_len, h, head_dim)
    # New shape: (batch, seq_len, h, head_dim / 2)
    # Why? Every 2 consecutive dimensions become 1 complex number
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Shape: (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    # Element-wise multiplication (rotation)
    # (batch, seq_len, h, head_dim / 2)
    x_rotated = x_complex * freqs_complex 
    
    # Transform complex number into tensor of 2 dims
    # (batch, seq_len, h, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # Flatten tensor: (batch, seq_len, h, head_dim / 2, 2) -> (batch, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x: torch.Tensor):
        # (batch, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (batch, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)
    

class SelfAttention(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is none else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
  

    
class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super.__init__()
        
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # Normalization before self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before feed forward network
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (batch, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(x))
        return out
    
    
    
    
    

class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size != -1, "Vocab size must be set"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers         # Number of transformer blocks stacked before RMSNorm, Linear Layer, Softmax
        
        # A simple lookup table that stores embeddings of a fixed dictionary and size.
        # Args: num_embeddings (size of dict), embedding_dim (embed vector)
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        # Final output is normalized before being sent to linear layer
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # Applies a linear transformation to the incoming data: y = xA^T + b
        # What is a linear transformation? A transformation (function) that maps a vector from Rn to Rm
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
    
        # Pre-compute the frequency of the rotary positional encodings
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)
        
    def forward(self, tokens: torch.Tensor, start_pos: int):
        
        # KV cache stores previous and intermediate tokens, so that 1 token is processed at a time
        # KV cache is only used during inference, and not during training
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only 1 token can be processed at a time"
        
        # (batch, seq_len) -> (batch, seq_len, dim)
        h = self.tok_embeddings(tokens)
        
        # Retrieve the pairs (m, theta) corresponding to the position [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm
        output = self.output(h).float()
        return output