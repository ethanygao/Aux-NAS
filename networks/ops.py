import numpy as np

import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def batch_norm(num_features, eps=1e-3, momentum=0.05):
    bn = nn.BatchNorm2d(num_features, eps, momentum)
    nn.init.constant_(bn.weight, 1)
    nn.init.constant_(bn.bias, 0)
    return bn


def layer_norm(num_features, eps=0.0, momentum=0.05):
    ln = nn.LayerNorm(num_features, eps, momentum)
    nn.init.constant_(ln.weight, 1)
    nn.init.constant_(ln.bias, 0)

    return ln

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features // 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(
            features // 2, features // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(
            features // 2, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out + x

class Fusion(nn.Module):
    def __init__(self, resample_dim, fuse_conv_type='residue'):
        super(Fusion, self).__init__()

        if fuse_conv_type == 'residue':
            self.conv1 = ResidualConvUnit(resample_dim)
            self.conv2 = ResidualConvUnit(resample_dim)
        elif fuse_conv_type == '3x3':
            self.conv1 = nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv2 = nn.Conv2d(resample_dim, resample_dim, kernel_size=3, stride=1, padding=1, bias=True)
        elif fuse_conv_type == '1x1':
            self.conv1 = nn.Conv2d(resample_dim, resample_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv2d(resample_dim, resample_dim, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            raise NotImplementedError

        #self.resample = nn.ConvTranspose2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1)

    def forward(self, x, previous_stage=None):
        if previous_stage == None:
            previous_stage = torch.zeros_like(x)
        output_stage1 = self.conv1(x)
        output_stage1 += previous_stage
        output_stage2 = self.conv2(output_stage1)
        output_stage2 = nn.functional.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=True)
        return output_stage2
    
class Read_ignore(nn.Module):
    def __init__(self, start_index=1):
        super(Read_ignore, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class Read_add(nn.Module):
    def __init__(self, start_index=1):
        super(Read_add, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class Read_projection(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(Read_projection, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)
        return self.project(features)

class MyConvTranspose2d(nn.Module):
    def __init__(self, conv, output_size):
        super(MyConvTranspose2d, self).__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x

class Resample(nn.Module):
    def __init__(self, p, s, h, emb_dim, resample_dim):
        super(Resample, self).__init__()
        assert (s in [4, 8, 16, 32]), "s must be in [0.5, 4, 8, 16, 32]"
        self.conv1 = nn.Conv2d(emb_dim, resample_dim, kernel_size=1, stride=1, padding=0)
        if s == 4:
            self.conv2 = nn.ConvTranspose2d(resample_dim,
                                resample_dim,
                                kernel_size=4,
                                stride=4,
                                padding=0,
                                bias=True,
                                dilation=1,
                                groups=1)
        elif s == 8:
            self.conv2 = nn.ConvTranspose2d(resample_dim,
                                resample_dim,
                                kernel_size=2,
                                stride=2,
                                padding=0,
                                bias=True,
                                dilation=1,
                                groups=1)
        elif s == 16:
            self.conv2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(resample_dim, resample_dim, kernel_size=2,stride=2, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Reassemble(nn.Module):
    def __init__(self, image_size, read, p, s, emb_dim, resample_dim):
        """
        p = patch size
        s = coefficient resample
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super(Reassemble, self).__init__()
        channels, image_height, image_width = image_size

        #Read
        self.read = Read_ignore()
        if read == 'add':
            self.read = Read_add()
        elif read == 'projection':
            self.read = Read_projection(emb_dim)

        #Concat after read
        self.concat = Rearrange('b (h w) c -> b c h w',
                                c=emb_dim,
                                h=(image_height // p),
                                w=(image_width // p))

        #Projection + Resample
        self.resample = Resample(p, s, image_height, emb_dim, resample_dim)

    def forward(self, x):
        x = self.read(x)
        x = self.concat(x)
        x = self.resample(x)
        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        return x