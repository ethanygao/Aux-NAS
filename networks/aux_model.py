import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch

from networks.single_task import SingleTask
from networks.stagewise_search_aux import GeneralizedMTLNASNet_L1Cut

def depth_limited_connectivity_matrix(stage_config, limit=3):
    """

    :param stage_config: list of number of layers in each stage
    :param limit: limit of depth difference between connected layers, pass in -1 to disable
    :return: connectivity matrix
    """
    network_depth = np.sum(stage_config)
    stage_depths = np.cumsum([0] + stage_config)
    matrix = np.zeros((network_depth, network_depth)).astype('int')
    for i in range(network_depth):
        j_limit = stage_depths[np.argmax(stage_depths > i) - 1]
        for j in range(network_depth):
            if j <= i and i - j < limit and j >= j_limit:
                matrix[i, j] = 1.
    return matrix

def vit_t_b_connectivity():
    return depth_limited_connectivity_matrix([3, 3, 3, 3])

def get_model(config, tasks=['normal', 'segmentation'], prim_task='normal'):
    resize = config['Dataset']['transforms']['resize']
    
    # model
    prim_net_backbone_pretrained = True if config['Aux']['load_prim_net'] == 'none' else False 
    prim_net = SingleTask(
                image_size  =   (3,resize,resize),
                emb_dim     =   config['General']['emb_dim'],
                resample_dim=   config['General']['resample_dim'],
                read        =   config['General']['read'],
                seg_nclasses=   config['General']['seg_nclasses'],
                hooks       =   config['General']['hooks'],
                model_timm  =   config['General']['model_timm'],
                type        =   prim_task,
                patch_size  =   config['General']['patch_size'],
                pretrained  =   prim_net_backbone_pretrained
    )

    aux_task = [i for i in tasks if i != prim_task][0]

    aux_net_backbone_pretrained = True if config['Aux']['load_aux_net'] == 'none' else False 
    aux_net = SingleTask(
                image_size  =   (3,resize,resize),
                emb_dim     =   config['General']['emb_dim'],
                resample_dim=   config['General']['resample_dim'],
                read        =   config['General']['read'],
                seg_nclasses=   config['General']['seg_nclasses'],
                hooks       =   config['General']['hooks'],
                model_timm  =   config['General']['model_timm'],
                type        =   aux_task,
                patch_size  =   config['General']['patch_size'],
                pretrained  =   aux_net_backbone_pretrained
    )

    if config['Aux']['load_prim_net'] != 'none':
        pretrained_dict = torch.load(config['Aux']['load_prim_net'])['model_state_dict']
        prim_net.load_state_dict(pretrained_dict)
    if config['Aux']['load_aux_net'] != 'none':
        pretrained_dict = torch.load(config['Aux']['load_aux_net'])['model_state_dict']
        aux_net.load_state_dict(pretrained_dict)

    # connectivity
    prim_aux_connectivity_matrix = vit_t_b_connectivity()
    aux_prim_connectivity_matrix = vit_t_b_connectivity()

    model = GeneralizedMTLNASNet_L1Cut(config, prim_net, aux_net,
                                       prim_aux_connectivity_matrix=prim_aux_connectivity_matrix,
                                       aux_prim_connectivity_matrix=aux_prim_connectivity_matrix,
                                       tasks=tasks, prim_task=prim_task
                                       )


    return model


if __name__ == "__main__":
    import json
    import torch

    with open('configs/config_nas_test3.json', 'r') as f:
        config = json.load(f)

    net = get_model(config, tasks=config["General"]["tasks"], prim_task=config["Aux"]["prim_task"])


    print(net)
    net.eval()
    in_ten = torch.randn(2, 3, 224, 224)
    out_normal, out_segmentation = net(in_ten)

    print(out_segmentation.size())
