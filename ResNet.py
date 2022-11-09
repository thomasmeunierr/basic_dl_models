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
            nn.Conv2d(input_channels, nb_channels, kernel_size, 1, "same"),
            nn.BatchNorm2d(),
            nn.ReLU(True),
            nn.Conv2d(nb_channels, nb_channels, kernel_size, 1, "same"),
            nn.BatchNorm2d(),
        )

        self.skip_seq = nn.Sequential(
            nn.Conv2d(input_channels, nb_channels, kernel_size, stride, "same"),
            nn.BatchNorm2d(),
        )

    def forward(self, x):

        out = self.seq(x)

        if self.stride == 1:

            return x + out

        else:

            return out + self.skip_seq(x)


class ResNet34(nn.Module):
    def __init__(self):

        super(ResNet34, self).__init__()

        self.first_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, "same"), nn.MaxPool2d(3, 2, "same")
        )

        layers = []
        for _ in range(3):

            layers.append(ResNetUnit(64, 64))

        layers.append(ResNetUnit(64, 128, 3, 2))

        for _ in range(4):

            layers.append(ResNetUnit(128, 128))

        layers.append(ResNetUnit(128, 256, 3, 2))

        for _ in range(6):

            layers.append(ResNetUnit(256, 256))

        layers.append(ResNetUnit(256, 512, 3, 2))

        for _ in range(3):

            layers.append(ResNetUnit(512, 512))

        layers.append(ResNetUnit(512, 1024, 3, 2))

        self.deep_layers = nn.Sequential(*layers)

        self.final_layers = nn.Sequential(
            nn.AvgPool2d(), nn.Flatten(), nn.Linear(1024 * 7 * 7, 10), nn.Softmax()
        )

    def forward(self, x):

        out = self.first_layers(x)
        out = self.deep_layers(out)

        return self.final_layers(out)


