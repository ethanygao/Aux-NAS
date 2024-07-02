import torch
import torch.nn as nn

from networks.ops import batch_norm, layer_norm


def get_nddr(cfg, in_channels, out_channels, a2p=False):
    return SingleSidedAsymmetricResidualNDDR(cfg, in_channels, out_channels, a2p=a2p)


class SingleSidedAsymmetricResidualNDDR(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, a2p=False):
        super(SingleSidedAsymmetricResidualNDDR, self).__init__()
        self.cfg = cfg

        init_weights = cfg['Aux']['NDDR']['init']
        if a2p:
            init_weights = cfg['Aux']['NDDR']['a2p_init']
        
        assert in_channels >= out_channels
        # check if out_channel divides in_channels
        assert in_channels % out_channels == 0
        multipiler = in_channels / out_channels - 1

        self.conv = nn.Conv2d(out_channels * int(multipiler), out_channels, kernel_size=1, bias=False)

        # Initialize weight
        if len(init_weights):
            weight = [torch.eye(out_channels) * init_weights[1] / float(multipiler) for _ in range(int(multipiler))]
            self.conv.weight = nn.Parameter(torch.cat(weight, dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
            
        self.norm = self._get_norm(out_channels)
        self.activation = self._get_activation()
        
        if self.norm is not None:
            nn.init.constant_(self.norm.weight, 1.)
            nn.init.constant_(self.norm.bias, 0)

    def _get_norm(self, out_channels):
        if self.cfg['Aux']['NDDR']['norm'] == 'batch':
            norm = batch_norm(out_channels, eps=1e-03, momentum=0.05)
        elif self.cfg['Aux']['NDDR']['norm'] == 'layer':
            norm = layer_norm(out_channels, eps=1e-03, momentum=0.05)
        elif self.cfg['Aux']['NDDR']['norm'] == 'none':
            norm = None
        else:
            raise NotImplementedError

        return norm

    def _get_activation(self):
        if self.cfg['Aux']['NDDR']['activ'] == 'ReLU':
            activation = nn.ReLU()
        elif self.cfg['Aux']['NDDR']['activ'] == 'GELU':
            activation = nn.GELU()
        elif self.cfg['Aux']['NDDR']['activ'] == 'none':
            activation = None
        else:
            raise NotImplementedError

        return activation

    def forward(self, features):
        """

        :param features: upstream feature maps
        :return:
        """

        if len(features[0].shape) == 3:
            tag_unsqueeze = True
            features = [torch.unsqueeze(torch.permute(t, (0, 2, 1)), dim=-1) for t in features]

        identity = features[0].clone()
        x = torch.cat(features[1:], 1)
        out = self.conv(x)
        out += identity

        # ln
        if tag_unsqueeze and self.cfg['Aux']['NDDR']['norm'] == 'layer':
            out = torch.permute(torch.squeeze(out, dim=-1), (0, 2, 1))

        out = self.norm(out) if self.norm is not None else out
        out = self.activation(out) if self.activation is not None else out

        # bn
        if tag_unsqueeze and self.cfg['Aux']['NDDR']['norm'] != 'layer':
            out = torch.permute(torch.squeeze(out, dim=-1), (0, 2, 1))

        return out