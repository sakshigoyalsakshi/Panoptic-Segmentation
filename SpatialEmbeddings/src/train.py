"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import train_config
from criterions.my_loss import SpatialEmbLoss
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger, Visualizer, Colorize
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

args = train_config.get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

# Fixing batch normalization statistics
if not args['on_crops']:
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader
if args['on_crops']:

    datasets = []
    for c, w in zip(args['train_dataset']['kwargs']['type'], args['train_dataset']['kwargs']['size']):
        datasets.append(get_dataset(
            args['train_dataset']['name'], args['train_dataset']['kwargs']['root_dir'], c, None, w,
            args['train_dataset']['kwargs']['transform']))

    train_dataset = torch.utils.data.ConcatDataset(datasets)
    print(len(train_dataset))

else:
    train_dataset = get_dataset(
        args['train_dataset']['name'], args['train_dataset']['kwargs']['root_dir'], 'train', None, 3000,
        args['train_dataset']['kwargs']['transform'])

train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs']['root_dir'], args['val_dataset']['kwargs']['type'],
    None, None, args['val_dataset']['kwargs']['transform'])

val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

pretrainedEnc = True
if args['pretrain_encoder']['apply']:
    print("Loading encoder pretrained in imagenet")
    from erfnet_imagenet import ERFNet as ERFNet_imagenet

    pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
    pretrainedEnc.load_state_dict(torch.load(args['pretrain_encoder']['path'])['state_dict'])
    pretrainedEnc = next(pretrainedEnc.children()).features.encoder

# set model
model = get_model(args['model']['name'], args['model']['kwargs']['num_classes'], pretrainedEnc)
model.init_output(args['loss_opts']['n_sigma'])
model = torch.nn.DataParallel(model).to(device)

# set criterion
criterion = SpatialEmbLoss(**args['loss_opts'])
criterion = torch.nn.DataParallel(criterion).to(device)

# set optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)


def lambda_(epoch):
    return pow((1 - ((epoch) / args['n_epochs'])), 0.9)


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_, )

# clustering
cluster = Cluster()

# Logger
logger = Logger(('train', 'val', 'iou'), 'loss')

# writing to tensorboard
writer = SummaryWriter()

# resume
start_epoch = 0
best_iou = 0
if args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    start_epoch = state['epoch'] + 1
    best_iou = state['best_iou']
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optim_state_dict'])
    logger.data = state['logger_data']


def train(epoch):
    # define meters
    loss_meter = AverageMeter()

    # put model into training mode
    model.train()

    if not args['on_crops']:
        model.apply(set_bn_eval)

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):
        im = sample['image']
        instances = sample['instance'].squeeze()
        class_labels = sample['label'].squeeze()
        semantic_label = sample['semantic_label']

        output = model(im)
        output_sem = output[0]
        output_ins = output[1]
        loss, ce_loss, ins_loss = criterion(output_sem, output_ins, instances, class_labels, semantic_label,
                                            **args['loss_w'])
        loss = loss.mean()
        ce_loss = ce_loss.mean()
        ins_loss = ins_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        writer.add_scalars("train_losses",
                           {"Total_Loss": loss, "CE_Loss": ce_loss, "Ins_Loss": ins_loss},
                           epoch)

    return loss_meter.avg


def val(epoch):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample['image']
            instances = sample['instance'].squeeze()
            class_labels = sample['label'].squeeze()
            semantic_label = sample['semantic_label']

            output = model(im)
            output_sem = output[0]
            output_ins = output[1]
            loss, ce_loss, ins_loss = criterion(output_sem, output_ins, instances, class_labels, semantic_label, **
            args['loss_w'], iou=True, iou_meter=iou_meter)
            ce_loss = ce_loss.mean()
            ins_loss = ins_loss.mean()
            loss = loss.mean()
            loss_meter.update(loss.item())

            writer.add_scalars("Val_losses",
                               {"Total_Loss": loss, "CE_Loss": ce_loss, "Ins_Loss": ins_loss},
                               epoch)

    return loss_meter.avg, iou_meter.avg


def save_checkpoint(state, is_best, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_iou_model.pth'))


for epoch in range(start_epoch, args['n_epochs']):

    print('Starting epoch {}'.format(epoch))
    scheduler.step(epoch)

    train_loss = train(epoch)
    val_loss, val_iou = val(epoch)

    print('===> train loss: {:.2f}'.format(train_loss))
    print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))

    logger.add('train', train_loss)
    logger.add('val', val_loss)
    logger.add('iou', val_iou)
    logger.plot(save=args['save'], save_dir=args['save_dir'])

    is_best = val_iou > best_iou
    best_iou = max(val_iou, best_iou)

    if args['save']:
        state = {
            'epoch': epoch,
            'best_iou': best_iou,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data
        }
        save_checkpoint(state, is_best)
