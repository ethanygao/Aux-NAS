import os
import shutil, glob

from tqdm import tqdm
import wandb

import numpy as np
from numpy.core.numeric import Inf
import cv2

import torch

from utils.utils import compute_hist, compute_angle

class BaseTrainer(object):
    def __init__(self):
        self.val_dataloader = None
        self.logger = None
        self.model = None
        self.device = None

        self.loss_normal = None
        self.loss_segmentation = None

        self.model_folder = None
        self.config = None

    def _run_eval_and_save(self, epoch, val_loss, mIoU, within_1125, last_save_epoch,
                           best_loss_epoch, best_mIoU_epoch, best_within_1125_epoch,
                           set_zero_connectivity=False):
        new_val_loss, new_mIoU, new_within_1125 = self.run_eval(self.val_dataloader, set_zero_connectivity)

        best_loss, best_mIoU, best_within_1125 = False, False, False

        if new_val_loss < val_loss:
            best_loss_epoch = epoch
            best_loss = True
            val_loss = new_val_loss
            self.logger.info('Reach best val loss: {:.4f}'.format(val_loss))

        if new_mIoU > mIoU:
            best_mIoU_epoch = epoch
            best_mIoU = True
            mIoU = new_mIoU
            self.logger.info('Reach best val mIoU: {:.4f}'.format(mIoU))

        if new_within_1125 > within_1125:
            best_within_1125_epoch = epoch
            best_within_1125 = True
            within_1125 = new_within_1125
            self.logger.info('Reach best val within 1125: {:.4f}'.format(within_1125))

        # if best_loss or best_mIoU or best_within_1125 or epoch - last_save_epoch >= 5:
        if best_loss or best_mIoU or best_within_1125:
            self.save_model(epoch, new_val_loss, new_mIoU, within_1125, best_loss, best_mIoU, best_within_1125)
            last_save_epoch = epoch

        return best_loss_epoch, val_loss, best_mIoU_epoch, mIoU, best_within_1125_epoch, within_1125, last_save_epoch, new_val_loss


    def _final_logging(self, val_loss, best_loss_epoch, mIoU, best_mIoU_epoch, within_1125, best_within_1125_epoch):
        # self.logger.info(json.dumps(self.config, indent=4))
        self.logger.info(self.model_folder)
        self.logger.info('Best val loss: {:.4f} at Epoch {:d}'.format(val_loss, best_loss_epoch))

        if self.config['General']['type'] == 'segmentation':
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.seg_results[best_loss_epoch].items()))
            
            self.logger.info('Best mIoU: {:.4f} at Epoch {:d}'.format(mIoU, best_mIoU_epoch))
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.seg_results[best_mIoU_epoch].items()))
        elif self.config['General']['type'] == 'normal':
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.normal_results[best_loss_epoch].items()))
            
            self.logger.info('Best within 1125: {:.4f} at Epoch {:d}'.format(within_1125, best_within_1125_epoch))
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.normal_results[best_within_1125_epoch].items()))
        elif self.config['General']['type'] == 'full':
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.seg_results[best_loss_epoch].items()))
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.normal_results[best_loss_epoch].items()))

            self.logger.info('Best mIoU: {:.4f} at Epoch {:d}'.format(mIoU, best_mIoU_epoch))
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.seg_results[best_mIoU_epoch].items()))
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.normal_results[best_mIoU_epoch].items()))

            self.logger.info('Best within 1125: {:.4f} at Epoch {:d}'.format(within_1125, best_within_1125_epoch))
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.seg_results[best_within_1125_epoch].items()))
            self.logger.info(', '.join('{:s}: {:.4f}'.format(k, v) for k, v in self.normal_results[best_within_1125_epoch].items()))
        else:
            raise NotImplementedError
        
        self.logger.info('Finished Training')

    def run_eval(self, val_dataloader, set_zero_connectivity=False):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- val_dataloader -: torch dataloader
        """
        val_loss = 0.0
        val_normal_loss = 0.0
        val_seg_loss = 0.0
        self.model.eval()
        if set_zero_connectivity:
            self.model.set_zero_connectivity(double_side=False)

        X_1 = None
        Y_normals_1 = None
        Y_segmentations_1 = None
        output_normals_1 = None
        output_segmentations_1 = None

        accumulator_seg = {}
        accumulator_normal = {}

        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")
            for i, (X, Y_normals, Y_segmentations) in enumerate(pbar):
                X, Y_normals, Y_segmentations = X.to(self.device), Y_normals.to(self.device), Y_segmentations.to(self.device)
                output_normals, output_segmentations = self.model(X)
                
                Y_normals_org = Y_normals.clone()
                Y_segmentations_org = Y_segmentations.clone()
                Y_normals = Y_normals.squeeze(1)
                Y_segmentations = Y_segmentations.squeeze(1)

                if output_normals is not None:
                    output_normals_org = output_normals.clone()
                    output_normals = output_normals.squeeze(1)
                else:
                    output_normals_org = None
                
                if i==0:
                    X_1 = X
                    Y_normals_1 = Y_normals
                    Y_segmentations_1 = Y_segmentations
                    output_normals_1 = output_normals
                    output_segmentations_1 = output_segmentations

                # seg_metric
                if output_segmentations is not None:
                    hist, correct_pixels, valid_pixels = compute_hist(output_segmentations, Y_segmentations_org, self.config['General']['seg_nclasses'], 255)
                    accumulator_seg['total_hist'] = accumulator_seg.get('total_hist', 0.) + hist
                    accumulator_seg['total_correct_pixels'] = accumulator_seg.get('total_correct_pixels', 0.) + correct_pixels
                    accumulator_seg['total_valid_pixels'] = accumulator_seg.get('total_valid_pixels', 0.) + valid_pixels

                if output_normals is not None:
                    angle = compute_angle(output_normals_org, Y_normals_org, 255)
                    angles = accumulator_normal.get('angles', [])
                    angles.append(angle)
                    accumulator_normal['angles'] = angles

                del output_normals_org, Y_normals_org, Y_segmentations_org
                # get loss
                normal_loss = self.loss_normal(output_normals, Y_normals)
                seg_loss = self.loss_segmentation(output_segmentations, Y_segmentations)

                loss = normal_loss * self.config['General']['normal_loss_weight'] + \
                       seg_loss * self.config['General']['seg_loss_weight']

                val_normal_loss = val_normal_loss + normal_loss.item() if type(normal_loss) == torch.Tensor else val_normal_loss
                val_seg_loss = val_seg_loss + seg_loss.item() if type(seg_loss) == torch.Tensor else val_seg_loss
                val_loss += loss.item()                

                pbar.set_postfix({'loss': val_loss/(i+1), 'normal_loss': val_normal_loss/(i+1), 'seg_loss': val_seg_loss/(i+1)})

            self.logger.info('[EVAL] loss: {:.3f}, normal_loss: {:.3f}, seg_loss: {:.3f}'.format(val_loss/(i+1), val_normal_loss/(i+1), val_seg_loss/(i+1)))

            mIoU = 0.0
            if len(accumulator_seg) > 0:
                total_hist = accumulator_seg['total_hist']
                total_correct_pixels = accumulator_seg['total_correct_pixels']
                total_valid_pixels = accumulator_seg['total_valid_pixels']
                IoUs = np.diag(total_hist) / (np.sum(total_hist, axis=0) + np.sum(total_hist, axis=1) - np.diag(total_hist) + 1e-16)
                mIoU = np.mean(IoUs)
                pixel_acc = total_correct_pixels / total_valid_pixels

                seg_result = {'mIoU': mIoU, 'pixel_acc': pixel_acc}
                self.seg_results.append(seg_result)

                self.logger.info('[EVAL][Segmentation] Mean IoU: {:.4f}; Pixel Acc: {:.4f}'.format(mIoU, pixel_acc))
            
            within_1125 = 0.0
            if len(accumulator_normal) > 0:
                angles = accumulator_normal['angles']
                angles = np.concatenate(angles, axis=0)
                
                mea = np.mean(angles)
                med = np.median(angles)
                rmse = np.sqrt(np.mean(angles ** 2))
                within_1125 = np.mean(np.less_equal(angles, 11.25)) * 100
                within_2250 = np.mean(np.less_equal(angles, 22.5)) * 100
                within_3000 = np.mean(np.less_equal(angles, 30.0)) * 100
                within_4500 = np.mean(np.less_equal(angles, 45.0)) * 100

                normal_result = {'mean': mea, 'median': med, 'rmse': rmse, '11.25': within_1125, 
                                 '22.5': within_2250, '30': within_3000, '45': within_4500}
                self.normal_results.append(normal_result)

                self.logger.info('[EVAL][Normal] Mean: {:.4f}; Median: {:.4f}; RMSE: {:.4f}'.format(mea, med, rmse))
                self.logger.info('[EVAL][Normal] Within 11.25: {:2.4f}; Within 22.5: {:2.4f}; Within 30: {:2.4f}; Within 45: {:2.4f}'.format(
                        within_1125, within_2250, within_3000, within_4500))

        return val_loss/(i+1), mIoU, within_1125

    def save_model(self, epoch, loss, mIoU, within_1125, best_loss=False, best_mIoU=False, best_within_1125=False,
                   optimizer_save_dict=None):
        model_name = os.path.join(self.model_folder, 
                                  'model_{:04d}_loss_{:.4f}_mIoU_{:.4f}_within_1125_{:.4f}.pt'.format(epoch, loss, mIoU, within_1125))

        save_dict = {}
        save_dict['model_state_dict'] = self.model.state_dict()
        if optimizer_save_dict is not None:
            save_dict.update(optimizer_save_dict)

        torch.save(save_dict, model_name)
        self.logger.info('Model saved at : {}'.format(model_name))

        if best_loss:
            check_best_ckpts = glob.glob(os.path.join(self.model_folder, 'best_loss_*.pt'))
            for ckpt in check_best_ckpts:
                os.remove(ckpt)

            shutil.copyfile(model_name, model_name.replace('model', 'best_loss_model'))

        if best_mIoU:
            check_best_ckpts = glob.glob(os.path.join(self.model_folder, 'best_mIoU_*.pt'))
            for ckpt in check_best_ckpts:
                os.remove(ckpt)

            shutil.copyfile(model_name, model_name.replace('model', 'best_mIoU_model'))

        if best_within_1125:
            check_best_ckpts = glob.glob(os.path.join(self.model_folder, 'best_within_1125_*.pt'))
            for ckpt in check_best_ckpts:
                os.remove(ckpt)

            shutil.copyfile(model_name, model_name.replace('model', 'best_within_1125_model'))

        os.remove(model_name)