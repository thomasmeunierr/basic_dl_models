import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetUnit(nn.Module):
    def __init__(self, input_channels, nb_channels, kernel_size=3, stride=1, **kwargs):

        super(ResNetUnit, self).__init__(**kwargs)
        self.stride = stride

        
        self.seq = nn.Sequential(
            nn.Conv2d(input_channels, nb_channels, kernel_size, stride, 1,bias = False),
            nn.BatchNorm2d(nb_channels),
            nn.ReLU(True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size, 1, 1,bias = False),
            nn.BatchNorm2d(nb_channels),
        )
        self.skip_layers = nn.Sequential()

        if stride > 1 :

            self.skip_layers = nn.Sequential(nn.Conv2d(input_channels,nb_channels,1,stride,bias = False),
                                            nn.BatchNorm2d(nb_channels))

    def forward(self, x):

        out = self.seq(x)

        skip = x

        if self.stride > 1 :

            skip = self.skip_layers(skip)
        
        return nn.functional.relu(out + skip)


class ResNet34(nn.Module):
    def __init__(self):

        super(ResNet34, self).__init__()

        self.first_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3,bias = False), nn.MaxPool2d(3, 2, 1)
        )

        layers = []
        prev = 64
        
        for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
            strides = 1 if filters == prev else 2
            layers.append(ResNetUnit(prev,filters,3,strides))
            prev = filters

        self.deep_layers = nn.Sequential(*layers)

        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), nn.Linear(512, 1000), nn.Softmax()
        )

    def forward(self, x):

        out = self.first_layers(x)
        out = self.deep_layers(out)

        return self.final_layers(out)


