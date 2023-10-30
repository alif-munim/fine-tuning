import torch
import torch.nn as nn
from torch.nn import functional as F

# Set torch seed for reproducibility
torch.manual_seed(1337)


# Set hyperparameters
batch_size = 64      # Number of independent sequences to be processed in parallel
block_size = 256       # Context (sequence) length for prediction
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

n_embd = 384 # Embeddings of size 386 are split across 6 heads, so each head is 64 dim
n_heads = 6
n_layer = 6
dropout = 0.2

# Read tiny shakespeare text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# Sorted, unique characters occuring in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
unique_chars = ''.join(chars)

# Create dictionaries mapping strings to integers, and integers to strings
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

# Define lambda functions that convert between string and integer
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Tokenize the tiny shakespeare dataset
data = torch.tensor(encode(text), dtype=torch.long) # Why use torch.long?

# Split data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

"""
Batching allows us to stack chunks. They are processed independently,
but it allows us to take advantage of parallelization.
"""
def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    # Generate batch_size (4) random indices in the data to stack along for x and y
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y



"""
Get average train, val loss for evaluation
Context manager torch.no_grad tells torch no need to call backward
"""
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



"""
Create a self-attention head and corresponding operations.

Dropout randomly shuts off some subset of neurons during every forward and backward pass.
This ends up training an ensemble of subnetworks (since the dropped out neurons change),
which are then merged during inference time. 
"""
class Head(nn.Module):
    
    def __init__(self, head_dim):
        super().__init__()
        self.key = nn.Linear(n_embd, head_dim, bias=False)
        self.query = nn.Linear(n_embd, head_dim, bias=False)
        self.value = nn.Linear(n_embd, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # Add dropout to affinity matrix to prevent some nodes from communicating
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        
        return out
    
    
"""
Create multiple heads of self-attention running in parallel.
We need to split the q, k, and v and then concatenate the result.

How do we know that the multiple SA heads are learning different things?
They all attend to different segments of the query, key, and value (by splitting input)

Why add a linear projection layer?
Projection back into residual pathway (?)
"""

class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_heads, head_dim):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_dim) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(head_dim * n_heads, n_embd) # Projection layer back into residual pathway
        self.dropout = nn.Dropout(dropout) # Dropout layer before projection back into residual pathway
        
    def forward(self, x):
        outputs = [h(x) for h in self.heads]
        concat_out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(concat_out))
        return out

    

"""
We implement layer normalization (row-wise normalization, e.g. per training example)
as opposed to batch normalization (column-wise normalization, e.g. per input feature)

This is an implementation from scratch, but PyTorch has a built-in method.
"""

class LayerNorm:
    
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
    def __call__(self, x):
        x_mean = x.mean(1, keepdim=True) # Mean along row-axis (each batch example)
        x_var = x.var(1, keepdim=True) # Variance along row-axis (each batch example)
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps) # Normalize to unit variance
        self.out = self.gamma * x_hat + self.beta
        return self.out
        
    def parameters(self):
        return [self.gamma, self.beta]

    
"""
Instead of going directly from MHSA output to vocab logits, let's give the network
time to "think" by adding some feed-forward layers.

According to "Attention Is All You Need," the inner dimension of FF layers should be 
multiplied by a factor of 4. This creates a "fan-out" and "fan-in" matrix transformation.

According to "Locating and Editing Factual Associations in GPT," these MLP transformations
form a sort of key-value store for factual associations.
"""

class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        
        # Add a linear projection here as well
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection layer back into residual pathway
            nn.Dropout(dropout) # Add dropout before connection back into residual pathway
        )
        
    def forward(self, x):
        return self.net(x)
    
    

"""
We create multiple MHSA and FF blocks to duplicate in order to intersperse 
communication between tokens and computation at the token (embedding) level

Why add residual connections?
Addition distributes gradients evenly.
"""

class Block(nn.Module):
    
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_dim  = n_embd // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_dim)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        
        # Add residual connections
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x    
    

"""
We start with a simple bigram language model.
The bigram language model predicts the next token for a given token.

When to use nn.Embedding or nn.Linear?
When to use nn.Sequential or nn.ModuleList?
"""

class GPTLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Each token reads logits for next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        
        # Replace single self-attention head of size 32 with multi-head attention
        # self.sa_head = Head(n_embd)
        
        # Since we are concatenating head outputs, we divide n_embd by n_heads
        # So that the linear projection layer can map back to the vocab logits
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        # self.ffwd = FeedForward(n_embd)
        
        # Instead of initializing a single sa_heads and ffwd layer, we create blocks
        # Making the neural net deeper may lead to optimization issues
        # Which can be solved with residual connections and normalization
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_heads=4),
        #     Block(n_embd, n_heads=4),
        #     Block(n_embd, n_heads=4),
        #     nn.LayerNorm(n_embd), # Final layer norm before final linear projection to vocab
        # )
        
        # Code cleanup
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        # idx and targets are tensors of (batch_size, time) or (batch_size, block_size)
        tok_emb = self.token_embedding_table(idx) # (batch, time, embed_dim)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (time, embed_dim)
        
        x = tok_emb + pos_emb # (batch, time, emb_dim)
        # x = self.sa_heads(x) # Feed input to self-attention head       
        # Before passing the sa_heads output to the lm_head, we add some FF layers
        # x = self.ffwd(x)    
        
        # Pass embeddings to MHSA + FF blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (batch, time, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # Negative log-likelihood is a good quality measure for predictions
            # PyTorch expects (B,C,T) instead of (B,T,C)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)        
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -block_size:] # Crop input to last block_size tokens
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # Take last element (B,T,C) -> (B,C)
            probs = F.softmax(logits, dim=-1) # Get softmax probs
            idx_next = torch.multinomial(probs, num_samples=1) # Sample using probs -> (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # Concat next token to integers
        return idx
    


    


    
# We can print logits and observe the shape
# The output are next-token logits for each token in the batch
model = GPTLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer (Adam is the most popular, larger LR is okay for simpler models)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))