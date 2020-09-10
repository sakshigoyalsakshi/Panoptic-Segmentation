import os

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import train_config
from datasets import get_dataset
from utils.utils import enet_weighing, enet_weighing_sem

torch.backends.cudnn.benchmark = True

args = train_config.get_args()

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader
datasets = []
# for c, w in zip(args['train_dataset']['kwargs']['type'], args['train_dataset']['kwargs']['size']):
#     datasets.append(get_dataset(
#         args['train_dataset']['name'], args['train_dataset']['kwargs']['root_dir'], c, None, w,
#         args['train_dataset']['kwargs']['transform']))
#
# train_dataset = torch.utils.data.ConcatDataset(datasets)
train_dataset = get_dataset(
        args['train_dataset']['name'], args['train_dataset']['kwargs']['root_dir'], 'train', None, None,
        args['train_dataset']['kwargs']['transform'])
print(len(train_dataset))

train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, drop_last=True,
    num_workers=1, pin_memory=True if args['cuda'] else False)

weights = enet_weighing_sem(train_dataset_it, 12)
print(weights)
