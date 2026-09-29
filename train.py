import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
batch_size = 64          # number of sequences processed per step
block_size = 256         # max context length (how far back the model looks)
max_iters = 1500         # total number of training steps
eval_interval = 500      # how often to evaluate train/val loss
learning_rate = 3e-4     # AdamW learning rate
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200         # number of batches sampled for loss estimation
n_embd = 128             # dimension of each token embedding (C)
n_head = 4               # number of parallel attention heads
n_layer = 6              # number of stacked transformer blocks
dropout = 0.2            # dropout rate for attention / feed-forward
# max_seq_len = 501  # must be >= block_size and >= any max_new_tokens + context length

torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
with open("tiny-shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('hii there'))
print(decode(encode('hii there')))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9*len(data))
train_data = data[:n].to(device)
val_data = data[n:].to(device)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------
class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,16)
        q = self.query(x) #(B,T,16)

        # Compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) --> (B,T,T) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # The 4 multiplier comes from section 3.3 AIAYN
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        """ 
        self.blocks = nn.Sequential(
                Block(n_embd, n_head=4),
                Block(n_embd, n_head=4),
                Block(n_embd, n_head=4),
                nn.LayerNorm(n_embd),
            )
        """
        # self.sa_head = MultiHeadAttention(4, n_embd//4)
        # self.ffwd = FeedForward(n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb #(B,T,C)
        # x = self.sa_head(x) # apply one head of self-attention
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------
def get_grad_norm(model):
    """Global L2 norm of all gradients - a quick sanity check for exploding/vanishing grads."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5

def print_attention_map(model, idx, layer_idx=0, head_idx=0, max_T=16):
    """
    Visualize the attention weights of a single head in a single block.
    Rows = the token attending, columns = the tokens it looks at.
    A diagonal-heavy map means each token mostly attends to itself / nearby tokens.
    """
    model.eval()
    with torch.no_grad():
        B, T = idx.shape
        T = min(T, max_T)
        idx = idx[:, :T]
        tok_emb = model.token_embedding_table(idx)
        pos_emb = model.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = model.blocks[layer_idx].ln1(x)
        head = model.blocks[layer_idx].sa.heads[head_idx]
        k = head.key(x)
        q = head.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
    model.train()

    print(f"\n--- Attention map: block {layer_idx}, head {head_idx} "
          f"(rows attend to columns) ---")
    tokens = [decode(idx[0, t].item()) for t in range(T)]
    header = "     " + " ".join(f"{t:>3}" for t in tokens)
    print(header)
    for r in range(T):
        row = " ".join(f"{wei[0, r, c].item():.3f}" for c in range(T))
        print(f"{tokens[r]:>3} | {row}")
    print()

# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------
model = BigramLanguageModel()
print(device)
m = model.to(device)

# Report the model size so we know what we're training
n_params = sum(p.numel() for p in m.parameters())
print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"\nStarting training: {max_iters} iters, batch={batch_size}, "
      f"block_size={block_size}, n_embd={n_embd}, n_head={n_head}, "
      f"n_layer={n_layer}, lr={learning_rate}, device={device}\n")

for iter in range(max_iters):
    t0 = time.time()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Per-step logging: loss, gradient norm, and throughput
    gnorm = get_grad_norm(m)
    dt = time.time() - t0
    print(f"iter {iter:5d} | loss {loss.item():.4f} | grad_norm {gnorm:.4f} | {dt*1000:.0f} ms")

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# Show what the model's attention looks like after training
print_attention_map(m, context, layer_idx=0, head_idx=0)
