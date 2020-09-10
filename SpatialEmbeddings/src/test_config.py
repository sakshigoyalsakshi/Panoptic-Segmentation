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
    save=True,
    save_dir='./masks/',
    checkpoint_path='./exp/checkpoint.pth',
    pretrain_encoder={
        'apply': True,
        'path': './pretrained_models/erfnet_encoder_pretrained.pth.tar',
    },

    dataset={
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
        }
    },

    model={
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [13, 3],
        }
    }
)


def get_args():
    return copy.deepcopy(args)
