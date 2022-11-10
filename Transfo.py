import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class PositionnalEncoding(nn.Module):
    def __init__(self, max_steps, max_dims, **kwargs):

        super(PositionnalEncoding, self).__init__()

        if max_dims % 2 == 1:
            max_dims += 1

        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))

        pos_emb = np.zeros((1, max_steps, max_dims))

        pos_emb[0, :, ::2] = np.sin(p / (10000 ** (2 * i / max_dims))).T
        pos_emb[0, :, 1::2] = np.cos(p / (10000 ** (2 * i / max_dims))).T

        self.P = torch.Tensor(pos_emb)

    def forward(self, x):

        return x + self.P[:, : x.size(-2), : x.size(-1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, **kwargs):

        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Q = nn.Linear(input_dim, embed_dim)
        self.K = nn.Linear(input_dim, embed_dim)
        self.V = nn.Linear(input_dim, embed_dim)

        self.embed_dim = embed_dim

    def forward(self, q, k, v):

        Q_mat = nn.functional.relu(self.Q(q))
        K_mat = nn.functional.relu(self.K(k))
        V_mat = nn.functional.relu(self.V(v))

        Q_mat = self._reshape_to_batches(Q_mat)
        K_mat = self._reshape_to_batches(K_mat)
        V_mat = self._reshape_to_batches(V_mat)

        return torch.matmul(
            nn.functional.softmax(
                torch.matmul(Q_mat, torch.transpose(K_mat, -2, -1))
                / np.sqrt(self.embed_dim)
            ),
            V_mat,
        )

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return (
            x.reshape(batch_size, seq_len, self.num_heads, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)
        )


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, max_steps, vocab_size, N):

        super(Encoder, self).__init__()

        self.N = N
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionnalEncoding(max_steps, embed_dim)
        self.multi_head = MultiHeadAttention(embed_dim, embed_dim, num_heads)
        self.add_norm = nn.BatchNorm2d(embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.add_norm_2 = nn.BatchNorm2d(embed_dim)

    def forward(self, x):

        embed = self.embed_layer(x)
        pos_encod = self.pos_encoding(embed)

        input_multi = embed + pos_encod

        for _ in range(self.N):

            q, k, v = input_multi.chunk(3, dim=-1)
            temp_input = self.multi_head(q, k, v)
            temp_input = self.add_norm(input_multi + temp_input)
            input_multi = self.add_norm_2(temp_input + self.ffn(temp_input))

        return input_multi


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, max_steps, vocab_size, N):

        super(Decoder, self).__init__()

        self.N = N
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionnalEncoding(max_steps, embed_dim)
        self.multi_head = MultiHeadAttention(embed_dim, embed_dim, num_heads)
        self.add_norm = nn.BatchNorm2d(embed_dim)
        self.multi_head_2 = nn.MultiheadAttention(embed_dim, embed_dim, num_heads)
        self.add_norm_2 = nn.BatchNorm2d(embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.add_norm_3 = nn.BatchNorm2d(embed_dim)

    def forward(self, x, encod):

        embed = self.embed_layer(x)
        pos_encod = self.pos_encoding(embed)

        input_multi = embed + pos_encod

        for _ in range(self.N):

            q, k, v = input_multi.chunk(3, dim=-1)
            temp_input = self.multi_head(q, k, v)
            temp_input = self.add_norm(input_multi + temp_input)
            encod_q, encod_k = encod.chunk(2, dim=-1)
            v = temp_input
            multi2 = self.multi_head_2(encod_q, encod_k, v)
            temp_input = self.add_norm_2(multi2 + temp_input)
            input_multi = self.add_norm_2(temp_input + self.ffn(temp_input))

        return input_multi


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, max_steps, vocab_size, N):

        super(Transformer, self).__init__()

        self.encoder = Encoder(embed_dim, num_heads, max_steps, vocab_size, N)
        self.decoder = Decoder(embed_dim, num_heads, max_steps, vocab_size, N)
        self.out_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Softmax())

    def forward(self, x):

        out_encod = self.encoder(x)
        out_decod = self.decoder(x, out_encod)

        return self.out_layer(out_decod)
        
