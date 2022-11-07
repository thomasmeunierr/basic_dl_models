# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
print(torch.__version__)



class Embedding(nn.Module):

    def __init__(self,vocab_size,embed_dim):

        super(Embedding,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_dim)

    def forward(self,x):

        return self.embed(x)


class PositionnalEmbedding(nn.Module):

    def __init__(self,max_seq_len,embed_model_dim):

        super(PositionnalEmbedding,self).__init__()


        self.embed_dim = embed_model_dim
        pe = torch.zeros((max_seq_len,self.embed_dim))

        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):

                pe[pos][i] = np.sin( pos / (1000 ** (i/self.embed_dim)))
                pe[pos][i+1] = np.sin( pos / (1000 ** ((i+1)/self.embed_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,x):

        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self,embed_dim = 512, n_heads = 8):

        super(MultiHeadAttention,self).__init__()

        self.embed_dim = embed_dim    
        self.n_heads = n_heads   
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        self.query_matrix = nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
  

        self.output = nn.Linear(embed_dim,embed_dim)

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        batch_size = key.size(0)
        seq_length = key.size(1)

        seq_length_query = query.size(1)
        
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.output(concat) #(32,10,512) -> (32,10,512)
       
        return output



        




