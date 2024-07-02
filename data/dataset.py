import os
import random
from glob import glob
from time import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils.utils import get_total_paths, get_splitted_dataset, get_transforms

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to('cpu').float())
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def read_all_images(image_list):
    all_images = []
    for img in image_list:
        img_tmp = Image.open(img)
        all_images.append(np.array(img_tmp))
        img_tmp.close()

    return all_images


class AutoFocusDataset(Dataset):
    """
        Dataset class for the AutoFocus Task. Requires for each image, its normal ground-truth and
        segmentation mask
        Args:
            :- config -: json config file
            :- dataset_name -: str
            :- split -: split ['train', 'val', 'test']
    """
    def __init__(self, config, dataset_name, split=None):
        self.split = split
        self.config = config

        path_images = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_images'])
        path_normals = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_normals'])
        path_segmentations = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_segmentations'])

        self.paths_images = get_total_paths(path_images, config['Dataset']['extensions']['ext_images'])
        self.paths_normals = get_total_paths(path_normals, config['Dataset']['extensions']['ext_normals'])
        self.paths_segmentations = get_total_paths(path_segmentations, config['Dataset']['extensions']['ext_segmentations'])

        assert (self.split in ['train', 'test', 'val']), "Invalid split!"
        assert (len(self.paths_images) == len(self.paths_normals)), "Different number of instances between the input and the normal maps"
        assert (len(self.paths_images) == len(self.paths_segmentations)), "Different number of instances between the input and the segmentation maps"
        assert (config['Dataset']['splits']['split_train']+config['Dataset']['splits']['split_test']+config['Dataset']['splits']['split_val'] == 1), "Invalid splits (sum must be equal to 1)"
        # check for segmentation

        # utility func for splitting
        self.paths_images, self.paths_normals, self.paths_segmentations = get_splitted_dataset(config, dataset_name, self.paths_images, self.paths_normals, self.paths_segmentations)

        if len(self.paths_images) < 50000:
            start_time = time()
            self.all_images = read_all_images(self.paths_images)
            self.all_normals = read_all_images(self.paths_normals)
            self.all_segmentations = read_all_images(self.paths_segmentations)

            print('loaded all {:d} images within {:.3f}s'.format(len(self.paths_images), time() - start_time))
        else:
            self.all_images = None
            self.all_normals = None
            self.all_segmentations = None

        # Get the transforms
        self.transform_image, self.transform_normal, self.transform_seg = get_transforms(config)
        self.to_tensor = transforms.ToTensor()

        # get p_flip from config
        self.p_flip = config['Dataset']['transforms']['p_flip'] if split=='train' else 0
        self.p_crop = config['Dataset']['transforms']['p_crop'] if split=='train' else 0
        self.p_rot = config['Dataset']['transforms']['p_rot'] if split=='train' else 0
        self.resize = config['Dataset']['transforms']['resize']

    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.paths_images)

    def __getitem__(self, idx):
        """
            Getter function in order to get the triplet of images / normal maps and segmentation masks
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.all_images is None:
            image = Image.open(self.paths_images[idx])
            normal = Image.open(self.paths_normals[idx])
            segmentation = Image.open(self.paths_segmentations[idx])
        else:
            image = Image.fromarray(self.all_images[idx])
            normal = Image.fromarray(self.all_normals[idx])
            segmentation = Image.fromarray(self.all_segmentations[idx])

        image = self.transform_image(image)

        normal = self.transform_normal(normal)
        normal = self.to_tensor(np.array(normal) - 255.) + 255.
        normal = normal.float()

        segmentation = self.transform_seg(segmentation)
        segmentation = self.to_tensor(np.array(segmentation) - 255.) + 255.
        segmentation = segmentation.long()

        imgorig = image.clone()

        if random.random() < self.p_flip:
            image = TF.hflip(image)
            normal = TF.hflip(normal)
            segmentation = TF.hflip(segmentation)

        if random.random() < self.p_crop:
            random_size = random.randint(self.resize // 3 * 2, self.resize-1)
            max_size = self.resize - random_size
            left = int(random.random()*max_size)
            top = int(random.random()*max_size)
            image = TF.crop(image, top, left, random_size, random_size)
            normal = TF.crop(normal, top, left, random_size, random_size)
            segmentation = TF.crop(segmentation, top, left, random_size, random_size)
            image = transforms.Resize((self.resize, self.resize))(image)
            normal = transforms.Resize((self.resize, self.resize))(normal)
            segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)

        if random.random() < self.p_rot:
            #rotate
            random_angle = random.random()*20 - 10 #[-10 ; 10]
            mask = torch.ones((1,self.resize,self.resize)) #useful for the resize at the end
            mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            image = TF.rotate(image, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            normal = TF.rotate(normal, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            segmentation = TF.rotate(segmentation, random_angle, interpolation=transforms.InterpolationMode.NEAREST)
            #crop to remove black borders due to the rotation
            left = torch.argmax(mask[:,0,:]).item()
            top = torch.argmax(mask[:,:,0]).item()
            coin = min(left,top)
            size = self.resize - 2*coin
            image = TF.crop(image, coin, coin, size, size)
            normal = TF.crop(normal, coin, coin, size, size)
            segmentation = TF.crop(segmentation, coin, coin, size, size)
            #Resize
            image = transforms.Resize((self.resize, self.resize))(image)
            normal = transforms.Resize((self.resize, self.resize))(normal)
            segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)
        # show([imgorig, image, normal, segmentation])
        # exit(0)

        return image, normal, segmentation