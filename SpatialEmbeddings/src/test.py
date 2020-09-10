"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time
import numpy as np
from PIL import Image
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

import test_config
import torch
from datasets import get_dataset
from torch.autograd import Variable
from models import get_model
from utils.utils import Cluster, TrainIdToIDS, Colorize, FinalId, flow_to_color, Colorize2

torch.set_printoptions(edgeitems=1000)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

args = test_config.get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs']['root_dir'], args['dataset']['kwargs']['type'],
    None, None, args['dataset']['kwargs']['transform'])

dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

pretrainedEnc = True
if args['pretrain_encoder']['apply']:
    print("Loading encoder pretrained in imagenet")
    from erfnet_imagenet import ERFNet as ERFNet_imagenet

    pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
    pretrainedEnc.load_state_dict(torch.load(args['pretrain_encoder']['path'])['state_dict'])
    pretrainedEnc = next(pretrainedEnc.children()).features.encoder

model = get_model(args['model']['name'], args['model']['kwargs']['num_classes'], pretrainedEnc)
model = torch.nn.DataParallel(model).to(device)

if os.path.exists(args['checkpoint_path']):
    print("model_loaded")
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)
else:
    assert (False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

model.eval()

# cluster module
cluster = Cluster()

with torch.no_grad():
    for ind, sample in enumerate(tqdm(dataset_it)):
        im = sample['image']
        output = model(im)
        im_name, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))[0].split("_leftImg8bit")
        label = output[0][0].max(0)[1]
        label = TrainIdToIDS(label)

        instance_map = cluster.cluster(output[1][0], label, threshold=0.9)
        ins = torchvision.transforms.ToPILImage()(instance_map.int().cpu())
        ins.save(args['save_dir'] + im_name + "_gtFine_instanceIds" + ".png")
