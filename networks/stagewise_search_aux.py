import copy
import functools
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from networks.nddr import get_nddr

from loss.losses import entropy_loss, l1_loss

from utils.utils import AttrDict
from utils.scheduler import poly



class GeneralizedMTLNASNet_L1Cut(nn.Module):
    def __init__(self, cfg, prim_net, aux_net,
                 prim_aux_connectivity_matrix,
                 aux_prim_connectivity_matrix,
                 tasks, prim_task
                ):
        """
        :param net1: task one network
        :param net2: task two network
        :param task1: task one
        :param task2: task two
        :param net1_connectivity_matrix: Adjacency list for the Single sided NDDR connections
        :param net2_connectivity_matrix: Adjacency list for the Single sided NDDR connections
        """
        super(GeneralizedMTLNASNet_L1Cut, self).__init__()

        self.cfg = cfg

        self.prim_net = prim_net
        self.aux_net = aux_net
        assert len(prim_net.stages) == len(aux_net.stages)
        print("Training with %d stages" % len(prim_net.stages))
        self.num_stages = len(prim_net.stages)
        self.prim_aux_connectivity_matrix = prim_aux_connectivity_matrix
        self.aux_prim_connectivity_matrix = aux_prim_connectivity_matrix

        if 'connectivity' not in self.cfg['Aux'].keys() or self.cfg['Aux']['connectivity'] == 'sigmoid':
            self.connectivity_fn = self.sigmoid_connectivity
        elif self.cfg['Aux']['connectivity'] == 'direct_0_1':
            self.connectivity_fn = self.direct_connectivity_0_1
        elif self.cfg['Aux']['connectivity'] == 'direct_n1_1':
            self.connectivity_fn = self.direct_connectivity_n1_1
        else:
            raise NotImplementedError

        prim_in_degrees = aux_prim_connectivity_matrix.sum(axis=1)
        aux_in_degrees = prim_aux_connectivity_matrix.sum(axis=1)
        prim_fusion_ops = []  # used for incoming feature fusion
        aux_fusion_ops = []  # used for incoming feature fusion

        # alpha_1: 2 -> 1; alpha_2: 1 -> 2
        # if reverse: net_2 is main -> cut alpha_2, else: net_1 is main -> cut alpha_1

        # assert cfg.MODEL.NDDR_TYPE == 'separate_plus_nddr'
        for stage_id in range(self.num_stages):
            n_channel = self.cfg['General']['emb_dim']
            prim_op = get_nddr(cfg,
                (prim_in_degrees[stage_id]+1)*n_channel,  # +1 for original upstream input
                n_channel)
            aux_op = get_nddr(cfg,
                (aux_in_degrees[stage_id]+1)*n_channel,  # +1 for original upstream input
                n_channel)
            prim_fusion_ops.append(prim_op)
            aux_fusion_ops.append(aux_op)

        prim_fusion_ops = nn.ModuleList(prim_fusion_ops)
        aux_fusion_ops = nn.ModuleList(aux_fusion_ops)

        if 'connectivity' not in self.cfg['Aux'].keys() or self.cfg['Aux']['connectivity'] == 'sigmoid':
            p2a_init_arch_weight = self.inverse_sigmoid(self.cfg['Aux']['Arch']['init_arch_weight'])
        elif 'direct' in self.cfg['Aux']['connectivity']:
            p2a_init_arch_weight = self.cfg['Aux']['Arch']['init_arch_weight']

        self.prim_aux_alphas = nn.Parameter(torch.tensor(prim_aux_connectivity_matrix).float() * p2a_init_arch_weight)

        if 'a2p_init_arch_weight' in self.cfg['Aux']['Arch'].keys() and self.cfg['Aux']['Arch']['a2p_init_arch_weight'] != 'none':
            if 'connectivity' not in self.cfg['Aux'].keys() or self.cfg['Aux']['connectivity'] == 'sigmoid':
                a2p_init_arch_weight = self.inverse_sigmoid(self.cfg['Aux']['Arch']['a2p_init_arch_weight'])
            elif 'direct' in self.cfg['Aux']['connectivity']:
                a2p_init_arch_weight = self.cfg['Aux']['Arch']['a2p_init_arch_weight']
        else:
            a2p_init_arch_weight = p2a_init_arch_weight

        self.aux_prim_alphas = nn.Parameter(torch.tensor(aux_prim_connectivity_matrix).float() * a2p_init_arch_weight)

        self.paths = nn.ModuleDict({
            'prim_paths': prim_fusion_ops,
            'aux_paths': aux_fusion_ops
        })

        self.prim_head_id = tasks.index(prim_task)
        self.aux_head_id = [i for i in range(len(tasks)) if i != self.prim_head_id][0]
        
        # path_cost = np.array([self.cfg['General']['emb_dim'] for stage in net1.stages])
        # path_cost = path_cost[:, None] * net1_connectivity_matrix
        # path_cost = path_cost * net1_connectivity_matrix.sum() / path_cost.sum()
        # path_cost = path_cost[np.nonzero(path_cost)]
        
        # self.register_buffer("path_costs", torch.tensor(path_cost).float())

        self._step = 0

        self._arch_parameters = dict()
        self._net_parameters = dict()
        self._nddr_parameters = dict()
        self._resemble_parameters = dict()
        self._fusion_parameters = dict()
        self._decoder_parameters = dict()
        self._backbone_parameters = dict()

        for k, v in self.named_parameters():
            # do not optimize arch parameter
            if 'alpha' in k:
                if 'set_single_side_prim_aux_search_space' in self.cfg['Aux'].keys() and \
                        self.cfg['Aux']['set_single_side_prim_aux_search_space'] == True and \
                        'aux_prim_alphas' in k:
                    continue
                if 'continue_finetune' in self.cfg['General'].keys() and \
                        os.path.exists(self.cfg['General']['continue_finetune']):
                    continue
                self._arch_parameters[k] = v
            else:
                self._net_parameters[k] = v
                if 'paths' in k:
                    self._nddr_parameters[k] = v
                elif 'reassembles' in k:
                    self._resemble_parameters[k] = v
                elif 'fusions' in k:
                    self._fusion_parameters[k] = v
                elif 'decoder' in k:
                    self._decoder_parameters[k] = v
                else:
                    self._backbone_parameters[k] = v


        # hard_weight_training, hard_arch_training, hard_evaluation, stochastic_evaluation
        # all False
        self.zero_connectivity = False
        self.double_side_zero = False
        self.single_side_prim_to_aux_one_connectivity = False
        
        # supernet False

    def regu_loss(self):
        # prune all aux-to-prim links
        arch_l1_paras = [self.aux_prim_alphas[np.nonzero(self.aux_prim_connectivity_matrix)]]
        arch_entropy_paras = [self.prim_aux_alphas[np.nonzero(self.prim_aux_connectivity_matrix)]]

        if self.cfg['Aux']['regu_losses']['entropy_regu']:
            entropy_loss_tmp = entropy_loss(arch_entropy_paras)
            entropy_weight = poly(start=0., end=self.cfg['Aux']['regu_losses']['entropy_regu_weight'],
                                  steps=self._step, total_steps=self.total_steps,
                                  period=self.cfg['Aux']['regu_losses']['entropy_regu_period'],
                                  power=1.)
            entropy_loss_v = entropy_weight * entropy_loss_tmp
        else:
            entropy_loss_v = 0.0

        if self.cfg['Aux']['regu_losses']['l1_regu']:
            l1_loss_tmp = l1_loss(arch_l1_paras)
            l1_weight = poly(start=0., end=self.cfg['Aux']['regu_losses']['l1_regu_weight'],
                             steps=self._step, total_steps=self.total_steps,
                             period=self.cfg['Aux']['regu_losses']['l1_regu_period'],
                             power=1.)

            l1_loss_v = l1_weight * l1_loss_tmp
        else:
            l1_loss_v = 0.0

        return entropy_loss_v, l1_loss_v

    def new(self):
        return copy.deepcopy(self)

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps

    def step(self):  # update temperature
        self._step += 1

    def set_zero_connectivity(self, double_side=False):
        self.zero_connectivity = True
        self.double_side_zero = double_side
        # if self.double_side_zero:
        #     print('Set Double Side Zero Connectivity')
        # else:
        #     print('Set Primary Side Zero Connectivity')

    def set_no_zero_connectivity(self):
        self.zero_connectivity = False
        print('Restore Double Side Alpha Connectivity')

    def set_single_side_prim_to_aux_one_connectivity(self):
        self.single_side_prim_to_aux_one_connectivity = True
        print('Set Single Side Prim to Aux One Connectivity')

    def inverse_sigmoid(self, prob):
        return np.log(prob / (1 - prob)) if prob != 1 else np.log(1e10)

    def sigmoid_connectivity(self, path_weights):
        return torch.sigmoid(path_weights)

    def direct_connectivity_0_1(self, path_weights):
        return torch.clamp(path_weights, min=0, max=1)

    def direct_connectivity_n1_1(self, path_weights):
        return torch.clamp(path_weights, min=-1, max=1)

    def forward(self, x):
        N, C, H, W = x.size()
        y = x.clone()
        x = self.prim_net.base(x)
        y = self.aux_net.base(y)
        xs, ys = [], []
        features_x, features_y = [], []

        for stage_id in range(self.num_stages):
            x = self.prim_net.stages[stage_id](x)
            y = self.aux_net.stages[stage_id](y)
            if isinstance(x, list):
                xs.append(x[0])
                ys.append(y[0])
            else:
                xs.append(x)
                ys.append(y)

            prim_aux_path_ids = np.nonzero(self.prim_aux_connectivity_matrix[stage_id])[0]
            aux_prim_path_ids = np.nonzero(self.aux_prim_connectivity_matrix[stage_id])[0]

            prim_aux_path_weights = self.prim_aux_alphas[stage_id][prim_aux_path_ids]
            aux_prim_path_weights = self.aux_prim_alphas[stage_id][aux_prim_path_ids]

            prim_aux_path_connectivity, aux_prim_path_connectivity = self.get_path_connectivity(prim_aux_path_weights, aux_prim_path_weights)

            # print('net1_path_connectivity:')
            # print(net1_path_connectivity)
            # print('net2_path_connectivity:')
            # print(net2_path_connectivity)

            if isinstance(x, list):
                prim_net_fusion_input = [x[0]] 
                aux_net_fusion_input = [y[0]]
            else:
                prim_net_fusion_input = [x]
                aux_net_fusion_input = [y]

            # alpha_1: 2 -> 1; alpha_2: 1 -> 2
            # if reverse: net_2 is main -> cut alpha_2, else: net_1 is main -> cut alpha_1
            for idx, input_id in enumerate(aux_prim_path_ids):
                prim_net_fusion_input.append(aux_prim_path_connectivity[idx]*ys[input_id])
            for idx, input_id in enumerate(prim_aux_path_ids):
                aux_net_fusion_input.append(prim_aux_path_connectivity[idx]*xs[input_id])

            prim_net_out = self.paths['prim_paths'][stage_id](prim_net_fusion_input)
            aux_net_out = self.paths['aux_paths'][stage_id](aux_net_fusion_input)

            if isinstance(x, list):
                x[0] = prim_net_out
                y[0] = aux_net_out
            else:
                x = prim_net_out
                y = aux_net_out

            if stage_id in self.cfg['General']['hooks']:
                features_x.append(x)
                features_y.append(y)

        prim_net_out = self.prim_net.head(features_x)[self.prim_head_id]
        aux_net_out = self.aux_net.head(features_y)[self.aux_head_id]

        out = [[], []]
        out[self.prim_head_id] = prim_net_out
        out[self.aux_head_id] = aux_net_out

        return out   # normal, seg

    def get_path_connectivity(self, prim_aux_path_weights, aux_prim_path_weights):
        # Calculating path strength based on weights
        prim_aux_path_connectivity = self.connectivity_fn(prim_aux_path_weights)
        aux_prim_path_connectivity = self.connectivity_fn(aux_prim_path_weights)
        
        assert (self.zero_connectivity and self.single_side_prim_to_aux_one_connectivity) is not True

        if self.zero_connectivity:
            if self.double_side_zero:
                prim_aux_path_connectivity = torch.zeros_like(prim_aux_path_connectivity)
                aux_prim_path_connectivity = torch.zeros_like(aux_prim_path_connectivity)
            else:
                aux_prim_path_connectivity = torch.zeros_like(aux_prim_path_connectivity)

        elif self.single_side_prim_to_aux_one_connectivity:
            prim_aux_path_connectivity = torch.ones_like(prim_aux_path_connectivity)
            aux_prim_path_connectivity = torch.zeros_like(aux_prim_path_connectivity)

        return prim_aux_path_connectivity, aux_prim_path_connectivity

    def arch_parameters(self):
        return self._arch_parameters.values()

    def named_arch_parameters(self):
        return self._arch_parameters.items()

    def net_parameters(self):
        return self._net_parameters.values()

    def named_net_parameters(self):
        return self._net_parameters.items()

    def nddr_parameters(self):
        return self._nddr_parameters.values()

    def named_nddr_parameters(self):
        return self._nddr_parameters.items()

    def resemble_parameters(self):
        return self._resemble_parameters.values()

    def named_resemble_parameters(self):
        return self._resemble_parameters.items()

    def fusion_parameters(self):
        return self._fusion_parameters.values()

    def named_fusion_parameters(self):
        return self._fusion_parameters.items()

    def decoder_parameters(self):
        return self._decoder_parameters.values()

    def named_decoder_parameters(self):
        return self._decoder_parameters.items()

    def backbone_parameters(self):
        return self._backbone_parameters.values()

    def named_backbone_parameters(self):
        return self._backbone_parameters.items()