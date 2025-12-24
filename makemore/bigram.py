from torch import nn
import torch
import torch.nn.functional as F

class Bigram(nn.Module):

    def __init__(self, dataset):
        '''
        dataset - list of names
        '''
        super().__init__() 

        self.vocab = set()
        for name in dataset:
            for ch in name:
                self.vocab.add(ch)

        self.logits = torch.zeros((len(self.vocab)+1, len(self.vocab)+1))
        self.logits += 1

        sorted_vocab = ['.'] + sorted(list(self.vocab))
        self.stoi = {c: i for i, c in enumerate(sorted_vocab)}
        self.itos = {i: c for i, c in enumerate(sorted_vocab)}

        for name in dataset:
            new_name = '.' + name + '.'
            for i, j in zip(new_name, new_name[1:]):
                x1 = self.stoi[i]
                x2 = self.stoi[j]
                self.logits[x1][x2] += 1
        
        self.logits /= torch.sum(self.logits, dim=1, keepdim=True)

    def forward(self, x, y=None):
        
        out = torch.multinomial(self.logits[x], num_samples=1, replacement=True)

        loss = None

        if y is not None:
            loss = F.cross_entropy(self.logits[x][out], y)

        return out, loss