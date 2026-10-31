"""CNN body, fixed for the whole study (protocol 4.1)."""

import torch.nn as nn
import torch.nn.functional as F


class ConvFeatures(nn.Module):
    """conv1 -> ReLU -> MaxPool(2) -> conv2 -> ReLU -> MaxPool(2) -> flatten: 28x28 -> 256."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)   # 28 -> 24 -> 12
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)  # 12 -> 8 -> 4

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x.flatten(1)


class CNNBody(nn.Module):
    """LeNet-class body: 208 + 3,216 + 8,224 + 330 = 11,978 trainable parameters."""

    def __init__(self):
        super().__init__()
        self.features = ConvFeatures()
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(self.features(x))))
