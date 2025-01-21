import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

_________________________________________

words = open("/content/input_data/names.txt", 'r').read().splitlines()

_________________________________________

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

__________________________________________

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
  X, Y = [], []

  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

___________________________________________________

n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)
#-----------------------------------------------------------------------------------------
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.1
b1 = torch.randn(n_hidden,                        generator=g) *0.01
#-----------------------------------------------------------------------------------------
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0



parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

_______________________________________________________

# Optimization
from math import log

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # Minibatch construction
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # Forward Pass
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)

    # linear layer
    hpreact = embcat @ W1 + b1

    # Applying non linearity
    h = torch.tanh(hpreact)

    # Output (softmax) layer also called logits (raw output)
    logits = h @ W2 + b2

    # Loss function
    loss = F.cross_entropy(logits, Yb)

    # Backward Pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update
    lr = 0.1 if i < 100000 else 0.01 # learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # Track Stats
    if i % 10000 == 0: # print every once in a while
       print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

    break

___________________________________________________________

plt.hist(h.view(-1).tolist(), 50);

__________________________________________________________

plt.figure(figsize=(20,10))
plt.imshow(h.abs() > 0.99, cmap='gray' , interpolation='nearest')

_________________________________________________________

