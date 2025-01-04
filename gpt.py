import torch

batch_size = 4
block_size = 8

# -------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique characters
chars = sorted(list(set(text)))

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

def get_batch(split):
    data = train_data if split == 'train' else val_data
    rand_nums = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in rand_nums])
    y = torch.stack([data[i+1:i+block_size+1] for i in rand_nums])
    return x, y

xb, yb = get_batch('train')

for batch_dimension in range(batch_size):
    for time_dimension in range(block_size):
        context = xb[batch_dimension, :time_dimension+1]
        target = yb[batch_dimension, :time_dimension]

