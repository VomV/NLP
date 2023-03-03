# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


import torch
import torch.nn as nn
from torch.nn import functional as F


#hyperparameters
batch_size=32 #how many independent seq to process in parallel
block_size=8 #sets the size of the context
max_iters=3000
eval_interval=300
learning_rate=1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embed = 32
num_heads = 4



torch.manual_seed(42)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

len(text)

print(text[:1000])

#All unique chars in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


#Tokenize characters
text_int = {ch:i for i, ch in enumerate(chars)}
int_text = {i:ch for i, ch in enumerate(chars)}



encode = lambda x: [text_int[i] for i in x]
decode  = lambda x: ''.join([int_text[i] for i in x])
encode('hi there')
decode(encode('hi there'))
data = torch.tensor(encode(text), dtype=torch.long)


#Split Data
n = int(0.9*len(text))
train_data = data[:n]
valid_data = data[n:]
block_size = 8
data[:block_size+1]

x = data[:block_size]
y = data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'context is {context} and target is {target}')



def get_batch(split):
    '''
        Generate a small batch of data of input x and target y
    '''

    data = train_data if split == 'train' else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    return out



xb, yb = get_batch('train')
print(xb.shape)
print(yb.shape)


for b in range(batch_size): #block size
    for c in range(block_size):
        context = xb[b, :c+1]
        target = yb[b, c]
        print(f"Context:: {context}, Target:: {target}")


class Head(nn.Module):
    '''
        One head of self-attention   
    '''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.query(x)

        wts = q @ k.transpose(-2,-1) * C**-0.5 # B,T,C @ B,C,T --> B,T,T
        wts = wts.masked_fill(self.tril[:T,:T]==0, float('-inf')) #B,T,T
        wts = F.softmax(wts, dim=-1) #B,T,T

        v = self.value(x) #B,T,C
        out = wts @ v #B,T,C

        return out

class MultiHeads(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.multihead = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.multihead], dim=-1)
        
#After self attention is completed, each token contains contextual info, and now with feedforward each token can think on itself, 
class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
                            nn.Linear(n_embed, n_embed),
                            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

#Multiple blocks of decoder
class Block(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeads(n_head, head_size)
        self.ffd = FeedForward(n_embed)

    def forward(self, x):

        x = self.sa(x)
        x = self.ffd(x)
        return x


#BiGram Language Model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        #each token directly reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
                            Block(n_embed, n_head=4),
                            Block(n_embed, n_head=4),
                            Block(n_embed, n_head=4),


        )
        # self.sa_head = MultiHeads(num_heads, n_embed//num_heads)
        # self.ffd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):

        B,T = idx.shape

        #idx and targets are both (Batch ,Time) tensor of integers
        token_emb = self.token_embedding_table(idx) # (Batch=batch_size, Time=block_size, Channel=embed_size)
        pos_emb = self.pos_embedding_table(torch.arange(T)) #(T,C)
        x = token_emb + pos_emb

        x = self.blocks(x)
        
        # x = self.sa_head(x)
        # x = self.ffd(x)
        logits = self.lm_head(x) #(Batch=batch_size, Time=block_size, Channel=vocab_size)

        if targets is not None:
            B, T, C = logits.shape

            #For pytorch loss module, reshaping
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is B,T array in current context
        for _ in range(max_new_tokens):

            #crop idx to last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the preds
            logits, loss = self(idx_cond)
            #get the last time step
            logits = logits[:, -1, :] # B,C
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # B,C
            #sample from dist
            idx_next = torch.multinomial(probs, num_samples=1) #B,1
            #append
            idx = torch.cat((idx, idx_next), dim=1) #B,T+1
        return idx



model = BigramLanguageModel()
out, loss = model(xb,yb)
print(out.shape)
print(loss)

print(decode(model.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
#Train the Model
#Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for epoch in range(max_iters):

    #eval loss on train and val sets
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {epoch}, train_loss: {losses['train']}, val_loss: {losses['eval']}")

    #sample a batch of data 
    xb, yb = get_batch('train')

    #eval loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


print(decode(model.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))