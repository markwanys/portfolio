############################# 1. Import Packages ###############################
import torch
import torch.nn as nn
import torch.nn.functional as F
################################################################################


################################### 2. Set Seed ################################
# Define function to set seed.
def set_seed(seed):
    # Set seed for reproducibility.
    torch.manual_seed(seed)
################################################################################


############################# 3. Data Preprocessing ############################
def data(text, train_test_split):
    # Compute key properties.
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Mapping for string to i and vice-versa.
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Train and test splits.
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_test_split*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return vocab_size, encode, decode, train_data, val_data, stoi, itos

def encode(s, stoi):
    return [stoi[c] for c in s]

def decode(l, itos):
    return ''.join([itos[i] for i in l])

# Load data.
def data_loader(split, batch_size, context_size, train_data, val_data, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
################################################################################


####################### 4. Model Building & Training ###########################
class Head(nn.Module):
    def __init__(self, n_embeddings, head_size, context_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Block to compute self-attention
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

         # Block for weighted-aggregated to embeddings
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttn(nn.Module):
    def __init__(self, n_heads, head_size, n_embeddings, context_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embeddings, head_size, context_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embeddings, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embeddings, n_heads, context_size, dropout):
        super().__init__()
        head_size = n_embeddings // n_heads
        self.sa = MultiHeadAttn(n_heads, head_size, n_embeddings, context_size, dropout)
        self.ffwd = FeedFoward(n_embeddings, dropout)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramModel(nn.Module):
    def __init__(self, vocab_size, n_embeddings, n_heads, n_layers, context_size, dropout, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(context_size, n_embeddings)
        self.blocks = nn.Sequential(*[Block(n_embeddings, n_heads, context_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, vocab_size)
        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
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

    def generate(self, idx, max_new_tokens, context_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
# Define function to train model.
def train_model(
                train_data, val_data,
                batch_size,
                context_size,
                max_iterations,
                eval_interval,
                learning_rate,
                eval_iters,
                n_embeddings,
                n_heads,
                n_layers,
                dropout,
                train_test_split,
                model,
                optimizer,
                device
                ):

    for iter in range(max_iterations):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iterations - 1:
            losses = estimate_loss(model, eval_iters, batch_size, context_size, train_data, val_data, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = data_loader('train', batch_size, context_size, train_data, val_data, device)

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model
################################################################################


########################### 5. Performance Metrics #############################
@torch.no_grad()
def estimate_loss(model, eval_iters, batch_size, context_size, train_data, val_data, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_loader(split, batch_size, context_size, train_data, val_data, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
################################################################################

