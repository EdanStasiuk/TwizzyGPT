import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------

batch_size = 32
block_size = 8
embed_size = 32
hidden_size = 64
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 3000
eval_interval = 50
eval_iters = 200

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
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Lower-triangular matrix

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

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class RNNLanguageModel(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.sa_heads = MultiHeadAttention(4, embed_size//4) # 4 heads of 8-dimensional self-attention = 32 (embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, embed_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)

        attn_out = self.sa_heads(x)
        x = x + attn_out
        output, _ = self.rnn(x)
        
        logits = self.fc(output) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        hidden = None
        for _ in range(max_new_tokens):
            token_embeddings = self.token_embedding_table(idx[:, -block_size:])
            output, hidden = self.rnn(token_embeddings, hidden)
            logits = self.fc(output[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
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


model = RNNLanguageModel()
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