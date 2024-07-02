import os
from os import replace
import sys
import shutil, glob
import copy
import time

from tqdm import tqdm
import json

import numpy as np
from numpy.core.numeric import Inf
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainer.BaseTrainer import BaseTrainer

from networks.single_task import SingleTask
from networks.aux_model import get_model

from utils.utils import get_losses, create_dir
from utils.optim import get_optimizer_single, get_optimizer_aux
from utils.scheduler import get_schedulers
from utils.logger import get_logger





class Trainer_Aux(BaseTrainer):
    def __init__(self, config, train_data, val_data):
        super().__init__()
        self.config = config

        # ckpt dir
        if self.config['General']['save_folder'] == 'none':
            self.model_folder = os.path.join(self.config['General']['path_model'], self.config['General']['type'],
                                             '{:s}_emb_{:d}_resample_{:d}_bs_{:d}_epochs_{:d}_norm_{}_activ_{}'.format(
                                                self.config['General']['model_timm'], self.config['General']['emb_dim'],
                                                self.config['General']['resample_dim'], self.config['General']['batch_size'],
                                                self.config['General']['epochs'], self.config['Aux']['NDDR']['norm'], 
                                                self.config['Aux']['NDDR']['activ']))
        else:
            self.model_folder = self.config['General']['save_folder']

        create_dir(self.model_folder)

        # logger
        self.logger = get_logger(self.model_folder)

        if sys.version_info >= (3, 7):
            import git
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            self.logger.info('git commit hash: {}'.format(sha))

        self.logger.info('checkpoint path: {}'.format(self.model_folder))
        config_path = os.path.join(self.model_folder, os.path.basename(self.config['ARGS']['config_file']))
        shutil.copy(self.config['ARGS']['config_file'], config_path)

        print_config = copy.deepcopy(self.config)
        self.logger.info("Configuration: {}".format(json.dumps(print_config, indent=4)))
        config_path = config_path.replace('.json', '_all.json')
        with open(config_path, 'wt') as f:
            f.write(json.dumps(self.config, indent=4))
            f.flush()

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        self.logger.info("device: {}".format(self.device))
        
        self.model = get_model(config, tasks=self.config['General']['tasks'], prim_task=self.config['Aux']['prim_task'])
        self.model.to(self.device)
        self.logger.info(self.model)

        self.train_data = train_data
        self.val_data = val_data
        
        self.num_train = len(self.train_data)
        # self.indices_train = list(range(self.num_train))
        if self.config['Aux']['type'] == 'fg_nas':
            self.steps_per_epoch = int(np.ceil(self.num_train / self.config['General']['batch_size'] / 2))
        total_steps = self.steps_per_epoch * self.config['General']['epochs']
        self.config['General']['steps'] = total_steps

        self.model.set_total_steps(total_steps)

        self.val_dataloader = DataLoader(val_data, batch_size=self.config['General']['batch_size'], shuffle=False,
                                         num_workers=4, pin_memory=True)

        self.loss_normal, self.loss_segmentation = get_losses(config)
        self.optimizer_network, self.optimizer_arch = get_optimizer_aux(config, self.model)
        self.schedulers = get_schedulers(config, [self.optimizer_network, self.optimizer_arch])

        self.normal_results = []
        self.seg_results = []

    def nas_train(self):
        epochs = self.config['General']['epochs']

        val_loss = Inf
        mIoU = 0.0
        within_1125 = 0.0
        last_save_epoch = 0
        best_loss_epoch = 0
        best_mIoU_epoch = 0
        best_within_1125_epoch = 0

        if 'set_single_side_prim_aux_one' in self.config['Aux'].keys() and self.config['Aux']['set_single_side_prim_aux_one'] == True:
            self.model.set_single_side_prim_to_aux_one_connectivity()

        for epoch in range(epochs):  # loop over the dataset multiple times
            self.logger.info("Epoch {:d}".format(epoch + 1))

            train_network_dataloader, train_arch_dataloader, _ = self._get_train_loaders_from_dataset()
            iter_train_network_data = iter(train_network_dataloader)
            iter_train_arch_data = iter(train_arch_dataloader)

            running_loss = 0.0
            running_normal_loss = 0.0
            running_seg_loss = 0.0
            self.model.train()

            if 'set_single_side_prim_aux_search_space' in self.config['Aux'].keys() and self.config['Aux']['set_single_side_prim_aux_search_space'] == True:
                self.model.set_zero_connectivity(double_side=False)
            elif 'continue_finetune' in self.config['General'].keys() and os.path.exists(self.config['General']['continue_finetune']):
                pretrained_dict = torch.load(self.config['General']['continue_finetune'])['model_state_dict']
                self.model.load_state_dict(pretrained_dict)

                self.model.set_zero_connectivity(double_side=True)
            else:
                self.model.set_no_zero_connectivity()
            
            pbar = tqdm(range(self.steps_per_epoch))
            pbar.set_description("Training")

            TT = 0
            T_DATA_A = 0
            T_FORWARD_A = 0
            T_LOSS_A = 0
            T_BACKWARD_A = 0

            T_DATA_M = 0
            T_FORWARD_M = 0
            T_LOSS_M = 0
            T_BACKWARD_M = 0

            for i in pbar:
                # arch train
                start_t = time.time()
                X, Y_normals, Y_segmentations = next(iter_train_network_data)
                T_DATA_A += time.time() - start_t
                # print('[ARCH] load data with {:.4f}'.format(time.time() - start_t))

                X, Y_normals, Y_segmentations = X.to(self.device), Y_normals.to(self.device), Y_segmentations.to(self.device)
                # zero the parameter gradients
                self.optimizer_arch.zero_grad()
                # forward + backward + optimizer
                start_t = time.time()
                output_normals, output_segmentations = self.model(X)
                T_FORWARD_A += time.time() - start_t
                TT += time.time() - start_t
                # print('[ARCH] forward with {:.4f}'.format(time.time() - start_t))

                output_normals = output_normals.squeeze(1) if output_normals != None else None

                Y_normals = Y_normals.squeeze(1) #1xHxW -> HxW
                Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW
                # get loss

                start_t = time.time()
                normal_loss = self.loss_normal(output_normals, Y_normals)
                seg_loss = self.loss_segmentation(output_segmentations, Y_segmentations)
                entropy_loss, l1_loss = self.model.regu_loss()

                loss = normal_loss * self.config['General']['normal_loss_weight'] + \
                       seg_loss * self.config['General']['seg_loss_weight'] + \
                       entropy_loss + l1_loss
                T_LOSS_A += time.time() - start_t
                # print('[ARCH] calculate loss with {:.4f}'.format(time.time() - start_t))


                start_t = time.time()
                loss.backward()
                # step optimizer
                self.optimizer_arch.step()
                T_BACKWARD_A += time.time() - start_t
                TT += time.time() - start_t
                # print('[ARCH] backward with {:.4f}'.format(time.time() - start_t))

                # net train
                start_t = time.time()
                X, Y_normals, Y_segmentations = next(iter_train_arch_data)
                T_DATA_M += time.time() - start_t
                # print('[MODEL] load data with {:.4f}'.format(time.time() - start_t))

                X, Y_normals, Y_segmentations = X.to(self.device), Y_normals.to(self.device), Y_segmentations.to(self.device)
                # zero the parameter gradients
                self.optimizer_network.zero_grad()
                # forward + backward + optimizer
                start_t = time.time()
                output_normals, output_segmentations = self.model(X)
                T_FORWARD_M += time.time() - start_t
                TT += time.time() - start_t
                # print('[MODEL] forward with {:.4f}'.format(time.time() - start_t))

                output_normals = output_normals.squeeze(1) if output_normals != None else None

                Y_normals = Y_normals.squeeze(1) #1xHxW -> HxW
                Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW
                # get loss

                start_t = time.time()
                normal_loss = self.loss_normal(output_normals, Y_normals)
                seg_loss = self.loss_segmentation(output_segmentations, Y_segmentations)

                loss = normal_loss * self.config['General']['normal_loss_weight'] + \
                       seg_loss * self.config['General']['seg_loss_weight']
                T_LOSS_M += time.time() - start_t
                # print('[MODEL] calculate loss with {:.4f}'.format(time.time() - start_t))

                start_t = time.time()
                loss.backward()
                # step optimizer
                self.optimizer_network.step()
                T_BACKWARD_M += time.time() - start_t
                TT += time.time() - start_t
                # print('[MODEL] backward with {:.4f}'.format(time.time() - start_t))

                running_normal_loss = running_normal_loss + normal_loss.item() if type(normal_loss) == torch.Tensor else running_normal_loss
                running_seg_loss = running_seg_loss + seg_loss.item() if type(seg_loss) == torch.Tensor else running_seg_loss
                running_loss += loss.item()

                if np.isnan(running_loss):
                    self.logger.info('nan loss')
                    exit(0)

                pbar.set_postfix({'loss': running_loss/(i+1), 'normal_loss': running_normal_loss/(i+1), 'seg_loss': running_seg_loss/(i+1)})
            
                # poly scheduler updates each iter
                if self.config['General']['scheduler'] == 'poly':
                    self.schedulers[0].step()
                    self.schedulers[1].step()
                self.model.step()

            self.logger.info('[TRAIN] loss {:.3f}, normal_loss {:.3f}, seg_loss {:.3f}'.format(running_loss/(i+1), running_normal_loss/(i+1), running_seg_loss/(i+1)))
            best_loss_epoch, val_loss, best_mIoU_epoch, mIoU, best_within_1125_epoch, within_1125, last_save_epoch, new_val_loss = \
                self._run_eval_and_save(epoch, val_loss, mIoU, within_1125, last_save_epoch,
                                        best_loss_epoch, best_mIoU_epoch, best_within_1125_epoch,
                                        set_zero_connectivity=True)
            del train_network_dataloader, train_arch_dataloader

            # reduce_plateau scheduler updates each iter
            if self.config['General']['scheduler'] ==  'reduce_plateau':
                self.schedulers[0].step(new_val_loss)
                self.schedulers[1].step(new_val_loss)

            p2a_connectivity = self.model.connectivity_fn(self.model.prim_aux_alphas[np.nonzero(self.model.prim_aux_connectivity_matrix)])
            a2p_connectivity = self.model.connectivity_fn(self.model.aux_prim_alphas[np.nonzero(self.model.aux_prim_connectivity_matrix)])

            self.logger.info('[p2a_alpha_connectivity]: mean {:.3f}, median {:.3f}, min {:.3f}, max {:.3f}'.format(
                torch.mean(p2a_connectivity), torch.median(p2a_connectivity), torch.min(p2a_connectivity), torch.max(p2a_connectivity)))
            self.logger.info('[a2p_alpha_connectivity]: mean {:.3f}, median {:.3f}, min {:.3f}, max {:.3f}'.format(
                torch.mean(a2p_connectivity), torch.median(a2p_connectivity), torch.min(a2p_connectivity), torch.max(a2p_connectivity)))

            p2a_connectivity, a2p_connectivity = self.model.get_path_connectivity(self.model.prim_aux_alphas[np.nonzero(self.model.prim_aux_connectivity_matrix)],
                                                                                  self.model.aux_prim_alphas[np.nonzero(self.model.aux_prim_connectivity_matrix)])

            self.logger.info('[p2a_connectivity]: mean {:.3f}, median {:.3f}, min {:.3f}, max {:.3f}'.format(
                torch.mean(p2a_connectivity), torch.median(p2a_connectivity), torch.min(p2a_connectivity), torch.max(p2a_connectivity)))
            self.logger.info('[a2p_connectivity]: mean {:.3f}, median {:.3f}, min {:.3f}, max {:.3f}'.format(
                torch.mean(a2p_connectivity), torch.median(a2p_connectivity), torch.min(a2p_connectivity), torch.max(a2p_connectivity)))

        self._final_logging(val_loss, best_loss_epoch, mIoU, best_mIoU_epoch, within_1125, best_within_1125_epoch)

    def _get_train_loaders_from_dataset(self):
        split = int(np.floor(self.num_train / 2))

        indices_train = list(range(self.num_train))
        np.random.shuffle(indices_train)

        train_network_data = torch.utils.data.Subset(self.train_data, indices_train[:split])
        train_arch_data = torch.utils.data.Subset(self.train_data, indices_train[split:])

        train_network_dataloader = DataLoader(train_network_data, batch_size=self.config['General']['batch_size'], 
                                              shuffle=True, pin_memory=True, num_workers=2)
        train_arch_dataloader    = DataLoader(train_arch_data, batch_size=self.config['General']['batch_size'], 
                                              shuffle=True, pin_memory=True, num_workers=2)
        train_dataloader         = DataLoader(self.train_data, batch_size=self.config['General']['batch_size'], 
                                              shuffle=True, pin_memory=True, num_workers=2)

        return train_network_dataloader, train_arch_dataloader, train_dataloader


class Trainer_Single(BaseTrainer):
    def __init__(self, config, train_data, val_data):
        super().__init__()
        self.config = config

        if self.config['General']['save_folder'] == 'none':
            self.model_folder = os.path.join(self.config['General']['path_model'], self.config['General']['type'],
                                         '{:s}_emb_{:d}_resample_{:d}_bs_{:d}_epochs_{:d}'.format(self.config['General']['model_timm'],
                                                                                                  self.config['General']['emb_dim'],
                                                                                                  self.config['General']['resample_dim'],
                                                                                                  self.config['General']['batch_size'],
                                                                                                  self.config['General']['epochs']))
        else:
            self.model_folder = self.config['General']['save_folder']
        
        create_dir(self.model_folder)

        # logger
        self.logger = get_logger(self.model_folder)

        if sys.version_info >= (3, 7):
            import git
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            self.logger.info('git commit hash: {}'.format(sha))

        self.logger.info('checkpoint path: {}'.format(self.model_folder))
        config_path = os.path.join(self.model_folder, os.path.basename(self.config['ARGS']['config_file']))
        shutil.copy(self.config['ARGS']['config_file'], config_path)

        print_config = copy.deepcopy(self.config)
        self.logger.info("Configuration: {}".format(json.dumps(print_config, indent=4)))
        config_path = config_path.replace('.json', '_all.json')
        with open(config_path, 'wt') as f:
            f.write(json.dumps(self.config, indent=4))
            f.flush()

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        self.logger.info("device: {}".format(self.device))

        resize = config['Dataset']['transforms']['resize']
        self.model = SingleTask(
                    image_size  =   (3,resize,resize),
                    emb_dim     =   config['General']['emb_dim'],
                    resample_dim=   config['General']['resample_dim'],
                    read        =   config['General']['read'],
                    seg_nclasses=   config['General']['seg_nclasses'],
                    hooks       =   config['General']['hooks'],
                    model_timm  =   config['General']['model_timm'],
                    type        =   config['General']['type'],
                    patch_size  =   config['General']['patch_size'],
        )

        self.model.to(self.device)
        self.logger.info(self.model)

        self.train_data = train_data
        self.val_data = val_data

        self.num_train = len(self.train_data)
        self.indices_train = list(range(self.num_train))
        self.steps_per_epoch = int(np.ceil(self.num_train / self.config['General']['batch_size']))
        total_steps = self.steps_per_epoch * self.config['General']['epochs']
        self.config['General']['steps'] = total_steps

        self.train_dataloader = DataLoader(train_data, batch_size=self.config['General']['batch_size'], shuffle=False,
                                           num_workers=4, pin_memory=True)
        self.val_dataloader =   DataLoader(val_data, batch_size=self.config['General']['batch_size'], shuffle=False,
                                           num_workers=4, pin_memory=True)

        self.loss_normal, self.loss_segmentation = get_losses(config)
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer_single(config, self.model)
        self.schedulers = get_schedulers(self.config, [self.optimizer_backbone, self.optimizer_scratch])

        self.normal_results = []
        self.seg_results = []

    def train(self):
        epochs = self.config['General']['epochs']

        val_loss = Inf
        mIoU = 0.0
        within_1125 = 0.0
        last_save_epoch = 0
        best_loss_epoch = 0
        best_mIoU_epoch = 0
        best_within_1125_epoch = 0

        for epoch in range(epochs):  # loop over the dataset multiple times
            self.logger.info("Epoch {:d}".format(epoch + 1))
            running_loss = 0.0
            running_normal_loss = 0.0
            running_seg_loss = 0.0
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            pbar.set_description("Training")
            for i, (X, Y_normals, Y_segmentations) in enumerate(pbar):
                # get the inputs; data is a list of [inputs, labels]
                X, Y_normals, Y_segmentations = X.to(self.device), Y_normals.to(self.device), Y_segmentations.to(self.device)
                # zero the parameter gradients
                self.optimizer_backbone.zero_grad()
                self.optimizer_scratch.zero_grad()
                # forward + backward + optimizer
                output_normals, output_segmentations = self.model(X)
                output_normals = output_normals.squeeze(1) if output_normals != None else None

                Y_normals = Y_normals.squeeze(1) #1xHxW -> HxW
                Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW
                # get loss
                normal_loss = self.loss_normal(output_normals, Y_normals)
                seg_loss = self.loss_segmentation(output_segmentations, Y_segmentations)

                loss = normal_loss * self.config['General']['normal_loss_weight'] + seg_loss * self.config['General']['seg_loss_weight']
                loss.backward()
                # step optimizer
                self.optimizer_scratch.step()
                self.optimizer_backbone.step()

                running_normal_loss = running_normal_loss + normal_loss.item() if type(normal_loss) == torch.Tensor else running_normal_loss
                running_seg_loss = running_seg_loss + seg_loss.item() if type(seg_loss) == torch.Tensor else running_seg_loss
                running_loss += loss.item()

                if np.isnan(running_loss):
                    print('\n',
                        X.min().item(), X.max().item(),'\n',
                        Y_normals.min().item(), Y_normals.max().item(),'\n',
                        output_normals.min().item(), output_normals.max().item(),'\n',
                        loss.item(),
                    )
                    self.logger.info('nan loss')
                    exit(0)

                pbar.set_postfix({'loss': running_loss/(i+1), 'normal_loss': running_normal_loss/(i+1), 'seg_loss': running_seg_loss/(i+1)})

                # poly scheduler updates each iter
                if self.config['General']['scheduler'] == 'poly':
                    self.schedulers[0].step()
                    self.schedulers[1].step()

            self.logger.info('[TRAIN] loss {:.3f}, normal_loss {:.3f}, seg_loss {:.3f}'.format(running_loss/(i+1), running_normal_loss/(i+1), running_seg_loss/(i+1)))
            best_loss_epoch, val_loss, best_mIoU_epoch, mIoU, best_within_1125_epoch, within_1125, last_save_epoch, new_val_loss = \
                self._run_eval_and_save(epoch, val_loss, mIoU, within_1125, last_save_epoch,
                                        best_loss_epoch, best_mIoU_epoch, best_within_1125_epoch)

            # reduce_plateau scheduler updates each iter
            if self.config['General']['scheduler'] ==  'reduce_plateau':
                self.schedulers[0].step(new_val_loss)
                self.schedulers[1].step(new_val_loss)

        self._final_logging(val_loss, best_loss_epoch, mIoU, best_mIoU_epoch, within_1125, best_within_1125_epoch)