import torch
import torch.nn as nn
from torch.nn import functional as F 

#hyperparametes 
#batch_size = 16 # How many independent sequences we are going to process in paralel 
#block_size = 32
#max_iters = 100
#eval_interval = 100
#learning_rate = 1e-3
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#eval_iters = 200
#n_embd = 64
#n_head = 4
#n_layer = 4
#dropout = 0.0

# HYPERPARAMETERS (The "Big Model" Settings)
batch_size = 64        # Process more sequences at once
block_size = 256       # Longer context (memory)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4   # Lower learning rate for bigger models
device = 'cuda'        # We know it works now!
eval_iters = 200
n_embd = 384           # Much bigger embedding dimensions
n_head = 6             # More heads
n_layer = 6            # More layers
dropout = 0.2          # Add dropout to prevent overfitting

#-----------------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('python/gpt/ELQUIJOTE.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(len(text))

# Create the mapping for chars to integers
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {i:k for k,i in enumerate(chars)}
itos = {i:k for k,i in stoi.items()}
encode = lambda s: [stoi[i] for i in s ]     # takes a string and output a list of integers
decode = lambda li: ''.join(itos[i] for i in li )


# Train and test splits 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]


# The mini_batches for training it more efficient
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # We need to select x batches from the data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Move the batch to the same device as the model (GPU)
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    # To get a more precise loss, we estimated it with 300 different batches and we calculated the mean of that
    out = {}   
    model.eval()        # We change the mode to evaluation
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): 
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size,bias=False)      # It's like a emb table for key of head_size
        self.query = nn.Linear(n_embd, head_size,bias=False)
        self.value = nn.Linear(n_embd, head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)                  # It's a regularization layer, it switch off some of the nodes 

    def forward(self,x): 
        B,T,C = x.shape
        k = self.key(x)                        # B,T,C
        q = self.query(x)                      # B,T,C
        wei = q @ k.transpose(-2,-1) * C**-0.5 # B,T,C @ B,C,T -> B T T
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v   
        return out

class MultipleHeadAttention(nn.Module): 
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x): 
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module): 
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )
    def forward(self,x): 
        return self.net(x)
    
class Block(nn.Module): 

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultipleHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x): 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)        # Create a 65,32 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)    # 8 x 32
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, target = None):
        B, T = idx.shape                                                       # C = 32
        tok_emb = self.token_embedding_table(idx)                              # B,T,C
        pos_emb = self.position_embedding_table(torch.arange(T,device=device)) #   T,C      Broadcasted to bacth 
        x = tok_emb + pos_emb                                                  # B,T,C
        x = self.blocks(x)
        x = self.ln_f(x)
        logist = self.lm_head(x)                                               # B,T,vocab_size

        # Target = B, T 
        # PyTorch expects the "Classes/Vocabulary" dimension (C) to be the second dimension.
        # We need to flatten it 
        if target is None: 
            loss = None
        else : 
            B, T, C  = logist.shape
            logist = logist.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logist,target)   
        # We return the logist as (B*T,C)
        return logist, loss
    
    def generate(self,idx, max_new_tokens):
        # WE GENERATE a new characters for each batch so we create 4 for each loop
        # Idx It holds the history of the conversation so far.
        # The model predicts what comes after the last l (predicts 'o'). Fo
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits,loss = self(idx_cond)     # self is a embbeding table
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) takes the last character of each batch
            probs = F.softmax(logits, dim=-1) # (B, C) In PyTorch, dim=-1 means "apply the operation along the very last dimension."
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT()
m = model.to(device)        #For training it with GPU

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print (sum(p.numel() for p in m.parameters()), 'M parameters')  # Parameters

for iter in range(max_iters): # increase number of steps for good results...

    # every once in a while we calculate the loss on train and val set 
    if iter % eval_interval == 0: 
        losses = estimate_loss()
        print (f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)       # The context is /nS
# --- SAVING THE MODEL ---
print("Saving model to 'model.pth'...")
torch.save(model.state_dict(), 'modelElQuijote.pth')
print("Model saved successfully!")


full_text = decode(model.generate(context, max_new_tokens=5000)[0].tolist())

# Print to console
print(full_text)

# Save to a file named 'output.txt'
output_filename = 'Elquijote.txt'
with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"\nSuccessfully saved the text to {output_filename}!")