import numpy as np
import torch
import torch.nn as nn

from networks.ops import Interpolate

class HeadNormal(nn.Module):
    def __init__(self, features, nclasses=3):
        super(HeadNormal, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, nclasses, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        x = self.head(x)
        return x

class HeadSeg(nn.Module):
    def __init__(self, features, nclasses=2):
        super(HeadSeg, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, nclasses, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        x = self.head(x)
        return x
