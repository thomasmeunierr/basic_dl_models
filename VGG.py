import numpy as np
import torch 
from torch import nn


class VGG(nn.Module):

    def __init__(self):

        super(VGG,self).__init__()

        self.sequential = nn.Sequential(
                                        nn.Conv2d(3,64,kernel_size = 3,stride = 1, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2,stride=2),

                                        nn.Conv2d(64,128,kernel_size = 3,stride = 1, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.MaxPool2d(2,stride=2),

                                        nn.Conv2d(128,256,kernel_size = 3,stride = 1, padding=1),
                                        nn.Conv2d(256,256,kernel_size = 3,stride = 1, padding=1),
                                        nn.BatchNorm2d(256),
                                        nn.MaxPool2d(2,stride=2),

                                        nn.Conv2d(256,512,kernel_size = 3,stride = 1, padding=1),
                                        nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.MaxPool2d(2,stride=2),

                                        nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding=1),
                                        nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.MaxPool2d(2,stride=2),

                                        nn.Flatten(-3),
                                        nn.Linear(7*7*512,4096),
                                        nn.Linear(4096,4096),
                                        nn.Linear(4096,1000),
                                        nn.Softmax()
        )

    def forward(self,x):

        return self.sequential(x)