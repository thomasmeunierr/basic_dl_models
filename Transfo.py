import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class PositionnalEncoding(nn.Module):

    def __init__(self,max_steps,max_dims,**kwargs):

        super(PositionnalEncoding,self).__init__()

        if max_dims%2 == 1 : max_dims += 1

        p,i = np.meshgrid(np.arange(max_steps),np.arange(max_dims//2))

        pos_emb = np.zeros((1,max_steps,max_dims))

        pos_emb[0, : , ::2] = np.sin(p / (10000**(2*i/max_dims))).T
        pos_emb[0, : , 1::2] = np.cos(p / (10000**(2*i/max_dims))).T

        self.P = torch.Tensor(pos_emb)

    def forward(self,x):

        return x + self.P[ :, :x.size(-2),:x.size(-1)]


class MultiHeadAttention(nn.Module):

    def __init__(self,n_r,n_k,d_k,d_v,**kwargs):

        super(MultiHeadAttention,self).__init__()

        self.Q = nn.Linear(n_r,d_k)
        self.K = nn.Linear(n_k,d_k)
        self.V = nn.Linear(n_k,d_k)

        self.dk = d_k

    def forward(self,q,k,v):

        Q_mat = nn.functional.relu(self.Q(q))
        K_mat = nn.functional.relu(self.K(k))
        V_mat = nn.functional.relu(self.V(v))

        return torch.matmul(
                            nn.functional.softmax(
                                                torch.matmul(Q_mat,torch.transpose(K_mat)) / torch.sqrt(self.dk)),
                                                V_mat)

