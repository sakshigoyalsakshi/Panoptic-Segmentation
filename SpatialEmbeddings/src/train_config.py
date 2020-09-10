"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

CITYSCAPES_DIR = os.environ.get('CITYSCAPES_DIR')

args = dict(

    cuda=True,
    display=True,
    display_it=5,

    save=True,
    save_dir='./exp_dialation_test/',
    resume_path='./exp_dialation/checkpoint.pth',

    # specifiy whether to train on crops
    on_crops=True,

    pretrain_encoder={
        'apply': True,
        'path': './pretrained_models/erfnet_encoder_pretrained.pth.tar',
    },

    train_dataset={
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': CITYSCAPES_DIR,
            'type': ['crops_24', 'crops_25', 'crops_26', 'crops_27', 'crops_28', 'crops_31', 'crops_32', 'crops_33'],
            'size': [375, 375, 375, 375, 375, 375, 375, 375],
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomCrop',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'semantic_label'),
                        'size': (512, 512),             # 1024 X 1024 for fine tuning
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'semantic_label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor),
                    }
                },
            ]),
        },
        'batch_size': 16,          # 2 for fine tuning
        'workers': 8               # 1 for fine tuning
    },

    val_dataset={
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': CITYSCAPES_DIR,
            'type': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'semantic_label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor),
                    }
                },
            ]),
        },
        'batch_size': 16,          # 2 for fine tuning
        'workers': 8               # 1 for fine tuning
 \
    },

    model={
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [13, 3]
        }
    },

    lr=5e-4,                       # 5e-5 for fine tuning
    n_epochs=200,                  # 250 for fine tuning

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': 1,

        # Foreground weights for Seed map Loss
        'foreground_weight':

        # Weights For crops, Comment for fine tuning
            [8.39668024, 9.8414818, 6.45759122, 9.0720037, 8.98162236, 8.59004596, 9.75674618, 9.61851834],

        # Uncomment for fine tuning
        # [9.52455428, 10.35256841, 7.11769108, 10.26738037, 10.26032928, 10.27105471, 10.41938415, 10.26141231],


        # Weights for Semantic segmentation Loss for 'stuff' classes
        'foreground_weight_sem':

        # Weights For crops, Comment for fine tuning
            [3.15260688, 7.30972138, 3.90342899, 10.13544035, 9.85256874, 8.84798197,
             10.16290448, 9.78282582, 4.35870971, 9.63300013, 7.74450984, 4.01761167, 0]

        # Uncomment for fine tuning
        # [2.6428655, 7.78222996, 4.02478003, 10.13522688, 9.94484147, 9.57578521,
        #  10.31093128, 10.04195818, 4.56329034, 9.81163643, 7.16105178, 6.22400169, 0]
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
)


def get_args():
    return copy.deepcopy(args)
