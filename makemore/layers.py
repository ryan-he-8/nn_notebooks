import torch
import torch.nn.functional as F

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.W = torch.randn((fan_in, fan_out)) / (fan_in)**0.5
        self.b = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.W
        self.out += self.b if self.b != None else 0
        return self.out
    
    def parameters(self):
        return [self.W] + ([self.b] if self.b != None else [])


class BatchNorm1d:
    
    def __init__(self, dim, epsilon=1e-5, momentum=0.1):
        self.epsilon = epsilon
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim, keepdim=True) # batch mean
            xvar = x.var(dim, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.epsilon) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        # update running
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []


class Embedding:

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]


class FlattenConsecutive:

    def __init__(self, n):
        self.n = n # block_size//n = how many "blocks" wanted

    def __call__(self, x):
        B, T, C = x.shape # batch_size x block_size x emb_size
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1) # either squeezes out all dimenstions of 1 or specified dimension (if that dimension is not one then nothing happens)
        self.out = x
        return self.out
    
    def parameters(self):
        return []
    

class Sequential:
    
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        
