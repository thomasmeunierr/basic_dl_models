import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class LSTMCell(nn.Module):

    def __init__(self,input_features,output_features):

        super(LSTMCell,self).__init__()

        self.forget_gate = nn.Sequential(
                                        nn.Linear(input_features,output_features),
                                        nn.Sigmoid()
        )

        self.major_gate = nn.Sequential(
                                        nn.Linear(input_features,output_features),
                                        nn.Tanh()
        )

        self.input_gate = nn.Sequential(
                                        nn.Linear(input_features,output_features),
                                        nn.Sigmoid()
        )

        self.output_gate = nn.Sequential(
                                        nn.Linear(input_features,output_features),
                                        nn.Sigmoid()
        )

    def forward(self,x,h,c):

        input = torch.cat([x,h])

        ft = self.forget_gate(input)
        gt = self.major_gate(input)
        it = self.input_gate(input)
        ot = self.output_gate(input)

        c_out = torch.mul(c,ft) + torch.mul(gt,it)
        h_out = torch.mul(nn.functional.tanh(c_out),ot)

        return c_out,h_out