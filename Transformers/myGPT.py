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



#BiGram Language Model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        #each token directly reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):

        #idx and targets are both (Batch ,Time) tensor of integers
        token_emb = self.token_embedding_table(idx) # (Batch=batch_size, Time=block_size, Channel=embed_size)
        logits = self.lm_head(token_emb) #(Batch=batch_size, Time=block_size, Channel=vocab_size)

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
            #get the preds
            logits, loss = self(idx)
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