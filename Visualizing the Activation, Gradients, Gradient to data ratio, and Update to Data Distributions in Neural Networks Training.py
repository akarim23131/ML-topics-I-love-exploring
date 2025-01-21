import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
%matplotlib inline

------------------------------
words = open('/content/input_names/names.txt', 'r').read().splitlines()

-----------------------------
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

---------------------------------
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

-----------------------------------
# This layer is exact same as we did manually where we had hpreact = X @ W1 + b1
class Linear:

  def __init__(self, fan_in, fan_out, bias = True):
    self.weight = torch.randn((fan_in, fan_out), generator = g) #/ fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:

  def __init__(self, dim, eps=1e-5, momentum = 0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # Parameters (trained with backprob)
    self.gamma = torch.ones(dim) # [bngain]
    self.beta = torch.zeros(dim) # [bnloss]
    # Buffers (trained with a running momentum update)
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)


  def __call__(self, x):
    # Calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim = True) # batch mean means if we are training then we are using the mean estimated by the batch
      xvar = x.var(0, keepdim = True) # batch varience means if we are training then we are using the mean estimated by the batch
    else:
      xmean = self.running_mean # if we are not training then we are using running mean and that will be used in inference in test or evaluation part
      xvar = self.running_var   # if we are not training then we are using running var and that will be used in inference in test or evaluation part
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalizatoin formula from tha paper we have used, unit varience normalization
    self.out = self.gamma * xhat + self.beta # this is the output of the normalized layer
    #update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out


  def parameters(self):
    return [self.gamma, self.beta]


class tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []


n_embd = 10  # the dimensionality of the character embedding vector
n_hidden = 100 # the number of neurons in the hidden layer of MLP
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((vocab_size, n_embd),                     generator = g)
layers = [
    Linear(n_embd * block_size, n_hidden), BatchNorm1d(n_hidden), tanh(),
    Linear(           n_hidden, n_hidden), BatchNorm1d(n_hidden), tanh(),
    Linear(           n_hidden, n_hidden), BatchNorm1d(n_hidden), tanh(),
    Linear(           n_hidden, n_hidden), BatchNorm1d(n_hidden), tanh(),
    Linear(           n_hidden, n_hidden), BatchNorm1d(n_hidden), tanh(),
    Linear(           n_hidden, vocab_size),     BatchNorm1d(vocab_size),
]

with torch.no_grad():
  # last layer make less confident
  #layers[-1].weight *= 0.1
  layers[-1].gamma *= 0.1
  # all other layers apply gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *=  1 #5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

------------------------------------
# same optimizationa as last time
max_steps = 200000
batch_size = 32
lossi = []
ud = [] # update to data ratio
for i in range(max_steps):

  # constucting a minibatch
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)
  Xb, Yb = Xtr[ix], Ytr[ix]

  # forward pass
  emb = C[Xb]  # embedd the characters into vectors
  x = emb.view(emb.shape[0], -1) # concenate the vectors
  for layer in layers:
    x = layer(x)
    loss = F.cross_entropy(x, Yb) # loss function

  # backward pass
  for layer in layers:
    layer.out.retain_grad()
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 1.0 if i < 100000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  #track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters])
  if i >= 1000:
   break

------------------------------------
# Activation Distribution 
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('activation distribution')

-----------------------------------
# Gradient Distribution
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('gradient distribution')

----------------------------------
# Weights gradients distribution
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution');



-----------------------------------
# Update to Data Ratio
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
plt.legend(legends);

