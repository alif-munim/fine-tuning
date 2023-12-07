import torch
import torch.nn as nn
from torch.nn import functional as F

# Set torch seed for reproducibility
torch.manual_seed(1337)


# Set hyperparameters
batch_size = 32        # Number of independent sequences to be processed in parallel
block_size = 8        # Context (sequence) length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

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
import torch
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
We start with a simple bigram language model.
The bigram language model predicts the next token for a given token.
"""

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        
        # Each token reads logits for next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        
        # idx and targets are tensors of (batch_size, time) or (batch_size, block_size)
        logits = self.token_embedding_table(idx) # (B,T,C) or (batch_size, block_size, vocab_size)
        
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
            logits, loss = self(idx)
            logits = logits[:, -1, :] # Take last element (B,T,C) -> (B,C)
            probs = F.softmax(logits, dim=-1) # Get softmax probs
            idx_next = torch.multinomial(probs, num_samples=1) # Sample using probs -> (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # Concat next token to integers
        return idx
    



    
# We can print logits and observe the shape
# The output are next-token logits for each token in the batch
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create a PyTorch optimizer (Adam is the most popular, larger LR is okay for simpler models)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


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