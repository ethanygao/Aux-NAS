import os, errno
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch

from glob import glob
from PIL import Image
from torchvision import transforms, utils

from loss.losses import ScaleAndShiftInvariantLoss, NormalLoss, compute_scale_and_shift

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

## dataset utils
def get_total_paths(path, ext):
    return glob(os.path.join(path, '*'+ext))

def get_splitted_dataset(config, dataset_name, path_images, path_normal, path_segmentation):
    list_files = [os.path.basename(im) for im in path_images]

    path_images = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_images'], im[:-4]+config['Dataset']['extensions']['ext_images']) for im in list_files]
    path_normals = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_normals'], im[:-4]+config['Dataset']['extensions']['ext_normals']) for im in list_files]
    path_segmentation = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_segmentations'], im[:-4]+config['Dataset']['extensions']['ext_segmentations']) for im in list_files]
    return path_images, path_normal, path_segmentation

def get_transforms(config):
    im_size = config['Dataset']['transforms']['resize']
    transform_image = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_normal = transforms.Compose([
        transforms.Resize((im_size, im_size)),
    ])
    transform_seg = transforms.Compose([
        transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.NEAREST),
    ])
    return transform_image, transform_normal, transform_seg


## losses utils
def get_losses(config):
    def NoneFunction(a, b):
        return 0
    loss_normal = NoneFunction
    loss_segmentation = NoneFunction
    type = config['General']['type']
    if type == "full" or type=="normal":
        if config['General']['loss_normal'] == 'cosine':
            loss_normal = NormalLoss(ignore_index=int(config['General']['seg_ignore_index']))
    if type == "full" or type=="segmentation":
        if config['General']['loss_segmentation'] == 'ce':
            loss_segmentation = nn.CrossEntropyLoss(ignore_index=int(config['General']['seg_ignore_index']))

    return loss_normal, loss_segmentation


## eval utils
def compute_hist(prediction, gt, n_classes, ignore_label):
    N, C, H, W = gt.size()
    prediction = F.interpolate(prediction, (H, W), mode='bilinear', align_corners=True)
    prediction = torch.argmax(prediction, dim=1).flatten().cpu().numpy()
    gt = gt.flatten().cpu().numpy()
    keep = np.logical_not(gt == ignore_label)
    merge = prediction[keep] * n_classes + gt[keep]
    hist = np.bincount(merge, minlength=n_classes**2)
    hist = hist.reshape((n_classes, n_classes))
    correct_pixels = np.diag(hist).sum()
    valid_pixels = keep.sum()
    return hist, correct_pixels, valid_pixels

def compute_depth_metrics(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    alpha = np.mean(np.log(gt) - np.log(pred))
    si = np.mean((np.log(pred) - np.log(gt) + alpha) ** 2)

    return np.array([abs_rel, sq_rel, rmse, rmse_log, si, a1, a2, a3])

def compute_angle(prediction, gt, ignore_label):
    N, C, H, W = gt.size()
    prediction = F.interpolate(prediction, (H, W), mode='bilinear', align_corners=True)
    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    mask = ((gt == ignore_label).sum(dim=1) - 3).nonzero().squeeze()
    prediction = prediction[mask]
    gt = gt[mask]
    cosine_distance = F.cosine_similarity(gt, prediction)
    cosine_distance = cosine_distance.cpu().numpy()
    cosine_distance = np.minimum(np.maximum(cosine_distance, -1.0), 1.0)
    angles = np.arccos(cosine_distance) / np.pi * 180.0
    return angles
