import os
import json
import argparse

import numpy as np

from torch.utils.data import ConcatDataset

from trainer.Trainer import Trainer_Single as Trainer
from data.dataset import AutoFocusDataset


parser = argparse.ArgumentParser()
parser.add_argument("--config-file", default="configs/config.json", type=str)

args = parser.parse_args()

with open(args.config_file, 'r') as f:
    config = json.load(f)

config['ARGS'] = {}
for k, v in vars(args).items():
    config['ARGS'][k] = v
    
np.random.seed(config['General']['seed'])

list_data = config['Dataset']['paths']['list_datasets']

## train set
autofocus_datasets_train = []
for dataset_name in list_data:
    train_dataset_name = os.path.join(dataset_name, config['Dataset']['paths']['path_train'])
    autofocus_datasets_train.append(AutoFocusDataset(config, train_dataset_name, 'train'))

train_data = ConcatDataset(autofocus_datasets_train)

## validation set
autofocus_datasets_val = []
for dataset_name in list_data:
    val_dataset_name = os.path.join(dataset_name, config['Dataset']['paths']['path_val'])
    autofocus_datasets_val.append(AutoFocusDataset(config, val_dataset_name, 'val'))

val_data = ConcatDataset(autofocus_datasets_val)

trainer = Trainer(config, train_data, val_data)
trainer.train()
