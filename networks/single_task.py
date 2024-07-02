import sys
import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from networks.ops import Reassemble
from networks.ops import Fusion
from networks.heads import HeadSeg, HeadNormal

torch.manual_seed(0)

class SingleTask(nn.Module):
    def __init__(self,
                 image_size         = (3, 224, 224),
                 patch_size         = 16,
                 emb_dim            = 1024,
                 resample_dim       = 256,
                 read               = 'projection',
                 num_layers_encoder = 24,
                 hooks              = [5, 11, 17, 23],
                 reassemble_s       = [4, 8, 16, 32],
                 transformer_dropout= 0,
                 seg_nclasses       = 2,
                 type               = "full",
                 model_timm         = "vit_large_patch16_384",
                 fuse_conv_type     = "residue",
                 pretrained         = True):
        """
        Focus on Depth
        type : {"full", "depth", "segmentation"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        #Splitting img into patches
        # channels, image_height, image_width = image_size
        # assert image_height % patch_size == 0 and image_width % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # num_patches = (image_height // patch_size) * (image_width // patch_size)
        # patch_dim = channels * patch_size * patch_size
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear(patch_dim, emb_dim),
        # )
        # #Embedding
        # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        #Transformer
        # encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dropout=transformer_dropout, dim_feedforward=emb_dim*4)
        # self.transformer_encoders = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_encoder)
        self.model = timm.create_model(model_timm, pretrained=pretrained)

        stages = list(self.model.blocks.children())
        del self.model.blocks

        self.stages = nn.ModuleList(stages)

        self.type_ = type

        #Register hooks
        self.activation = {}
        self.hooks = hooks

        #Reassembles Fusion
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim, fuse_conv_type=fuse_conv_type))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        #Head
        if type == "full":
            self.decoder = []
            self.decoder.append(HeadNormal(resample_dim, nclasses=3))
            self.decoder.append(HeadSeg(resample_dim, nclasses=seg_nclasses))
            self.decoder = nn.ModuleList(self.decoder)
        elif type == "normal":
            self.decoder = HeadNormal(resample_dim)
        else:
            self.decoder = HeadSeg(resample_dim, nclasses=seg_nclasses)

    def base(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)

        return x

    def head(self, features):
        previous_stage = None

        for i in np.arange(len(self.fusions)-1, -1, -1):
            x = features[i]
            x = self.reassembles[i](x)
            x = self.fusions[i](x, previous_stage)
            previous_stage = x

        if self.type_ == 'normal':
            out_normal = self.decoder(previous_stage)
            out_segmentation = None
        elif self.type_ == 'segmentation':
            out_normal = None
            out_segmentation = self.decoder(previous_stage)
        elif self.type_ == 'full':
            out_normal = self.decoder[0](previous_stage)
            out_segmentation = self.decoder[1](previous_stage)

        return out_normal, out_segmentation

    def forward(self, x):
        features = []
        x = self.base(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.hooks:
                features.append(x)

        out_normal, out_segmentation = self.head(features)

        return out_normal, out_segmentation


if __name__ == "__main__":
    net = SingleTask(emb_dim=192, resample_dim=64, read='ignore', hooks=[2, 5, 8, 11], 
                     seg_nclasses=40, type='segmentation', model_timm='vit_tiny_patch16_224',
                     fuse_conv_type='3x3')
    print(net)
    net.eval()
    in_ten = torch.randn(2, 3, 224, 224)
    out_depth, out_segmentation = net(in_ten)

    print(out_segmentation.size())
