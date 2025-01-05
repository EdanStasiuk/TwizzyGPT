import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------

batch_size = 32
block_size = 8
num_embed = 32
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 3000
eval_interval = 50
eval_iters = 200
num_head = 4
num_layer = 4
dropout = 0.2

# -------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Map from characters to integers
stoi = dict()
itos = dict()
for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the dataset and store into a torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split up data into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

class Head(nn.Module):
    """ One head of self-attention """

    # Each token emits 3 vectors: a key, a query, and a value
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Lower-triangular matrix

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)     # (B,T,C)
        q = self.query(x)   # (B,T,C)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, num_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),
            nn.ReLU(),
            nn.Linear(4 * num_embed, num_embed), # projection layer that goes back into the residual pathway
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_embed, num_head):
        super().__init__()
        head_size = num_embed // num_head
        self.sa = MultiHeadAttention(num_head, head_size) # communication
        self.ffwd = FeedForward(num_embed) # computation
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed)
        self.position_embedding_table = nn.Embedding(block_size, num_embed)
        self.blocks = nn.Sequential(*[Block(num_embed, num_head=num_head) for _ in range(num_layer)])
        self.ln_f = nn.LayerNorm(num_embed)
        self.lm_head = nn.Linear(num_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, num_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        hidden = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def get_batch(split):
    data = train_data if split == 'train' else val_data
    rand_nums = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in rand_nums])
    y = torch.stack([data[i+1:i+block_size+1] for i in rand_nums])
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


model = BigramLanguageModel()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Forward pass (logits and loss)
print(f"Logits shape: {logits.shape}, Loss: {loss.item()}")

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_idx = model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)
print(decode(generated_idx[0].tolist()))