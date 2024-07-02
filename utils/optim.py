import os
import logging

import torch
import torch.optim as optim


def get_optimizer_single(config, net):
    names = set([name.split('.')[0] for name, _ in net.named_modules()]) - set(['', 'model', 'stages'])
    params_backbone = net.model.parameters()
    params_scratch = list()
    for name in names:
        params_scratch += list(eval("net."+name).parameters())

    if config['General']['optim'] == 'adam':
        optimizer_backbone = optim.Adam(params_backbone, lr=config['General']['lr_backbone'])
        optimizer_scratch = optim.Adam(params_scratch, lr=config['General']['lr_scratch'])
    elif config['General']['optim'] == 'sgd':
        optimizer_backbone = optim.SGD(params_backbone, lr=config['General']['lr_backbone'], momentum=config['General']['momentum'])
        optimizer_scratch = optim.SGD(params_scratch, lr=config['General']['lr_scratch'], momentum=config['General']['momentum'])
    return optimizer_backbone, optimizer_scratch


def get_optimizer_aux(config, net):
    params_network = [
        {'params': net.backbone_parameters(), 'lr': config['General']['lr_backbone']},
        {'params': net.decoder_parameters(), 'lr': config['General']['lr_decoder']},
        {'params': net.fusion_parameters(), 'lr': config['General']['lr_fusion']},
        {'params': net.resemble_parameters(), 'lr': config['General']['lr_resemble']},
        {'params': net.nddr_parameters(), 'lr': config['General']['lr_nddr']},
    ]

    params_arch = [
        {'params': net.arch_parameters(), 'lr': config['General']['lr_arch']},
    ]

    if config['General']['optim'] == 'adam':
        optimizer_network = optim.Adam(params_network, lr=config['General']['lr_backbone'])
        optimizer_arch = optim.Adam(params_arch, lr=config['General']['lr_arch'])
    elif config['General']['optim'] == 'sgd':
        optimizer_network = optim.SGD(params_network, lr=config['General']['lr_backbone'], momentum=config['General']['momentum'])
        optimizer_arch = optim.SGD(params_arch, lr=config['General']['lr_arch'], momentum=config['General']['momentum'])

    return optimizer_network, optimizer_arch

