"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import multiprocessing
import collections
import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, ToPILImage
import seaborn as sb
from tqdm import tqdm

import torch


def enet_weighing_sem(dataloader, num_classes, c=1.1):
    class_count = 0
    total = 0
    class_count = np.zeros(num_classes)
    for sample in dataloader:
        label = sample['semantic_label']
        relabel = Relabel(255, 19)
        label = relabel(label)
        label = label.squeeze().cpu().numpy()

        # Flatten label
        flat_label = label.flatten()
        for cl in range(num_classes):
            mask = label == cl
            class_count[cl] += (np.sum(mask))

        total += flat_label.size

    # Compute propensity score and then the weights for each class
    print(class_count)
    print(total)
    propensity_score = class_count / total
    print(propensity_score)
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def enet_weighing(dataloader, num_classes, c=1.1):
    class_count = 0
    total = 0
    class_count = np.zeros(num_classes)
    for sample in dataloader:
        label = sample['label']
        label = label.squeeze().cpu().numpy()

        # Flatten label
        flat_label = label.flatten()
        for cl in range(num_classes):
            mask = label == cl + 1
            class_count[cl] += (np.sum(mask))

        total += flat_label.size

    # Compute propensity score and then the weights for each class
    print(class_count)
    print(total)
    propensity_score = class_count / total
    print(propensity_score)
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


class FinalId:
    def __init__(self, instance_label, semantic_label, class_id):
        self.inst_label = instance_label
        self.sem_label = semantic_label
        self.class_id = class_id

    def convert(self):
        unique = list(np.unique(np.array(self.sem_label).flatten()))
        [unique.remove(x) for x in self.class_id if x in unique]
        new_image = self.convert_instance_map()
        for i in unique:
            new_image[np.array(self.sem_label) == i] = i
        return Image.fromarray(new_image)

    def convert_instance_map(self):
        height, width = np.array(self.inst_label).shape
        new_map = np.zeros([height, width], dtype=np.int32)
        unique = list(np.unique(np.array(self.inst_label).flatten()))
        unique.remove(0)
        for i in unique:
            new_map[np.array(self.inst_label) == i] = self.class_id[0] * 1000 + (i - 1)
        return new_map


class Colorize:
    def __init__(self):
        self.colors = self.create()

    def create(self):
        colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                  (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                  (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                  (0, 0, 0)]

        return colors

    def __call__(self, image, type=None):
        unique = np.unique(image.flatten())
        sem_seg_color = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        ind = 0
        for i in unique:
            sem_seg_color[image == i, :] = self.colors[ind]
            ind += 1
        if type == "PIL":
            return Image.fromarray(sem_seg_color)
        else:
            return sem_seg_color


def TrainIdToIDSFULL(image):
    cityscapes_trainIds2labelIds = Compose([
        Relabel(19, 255),
        Relabel(18, 33),
        Relabel(17, 32),
        Relabel(16, 31),
        Relabel(15, 28),
        Relabel(14, 27),
        Relabel(13, 26),
        Relabel(12, 25),
        Relabel(11, 24),
        Relabel(10, 23),
        Relabel(9, 22),
        Relabel(8, 21),
        Relabel(7, 20),
        Relabel(6, 19),
        Relabel(5, 17),
        Relabel(4, 13),
        Relabel(3, 12),
        Relabel(2, 11),
        Relabel(1, 8),
        Relabel(0, 7),
        Relabel(255, 0),
    ])
    return cityscapes_trainIds2labelIds(image)


def TrainIdToIDS(image):
    cityscapes_trainIds2labelIds = Compose([
        Relabel(19, 255),
        Relabel(7, 20),
        Relabel(0, 7),
        Relabel(11, 0),
        Relabel(10, 23),
        Relabel(9, 22),
        Relabel(8, 21),
        Relabel(6, 19),
        Relabel(5, 17),
        Relabel(4, 13),
        Relabel(3, 12),
        Relabel(2, 11),
        Relabel(1, 8),
        Relabel(255, 0),
    ])
    return cityscapes_trainIds2labelIds(image)


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x / y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class) / len(self.avg_per_class)


class Cluster:

    def __init__(self, ):

        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)
        self.class_ids = [24, 25, 26, 27, 28, 31, 32, 33]
        self.class_train_ids = [11, 12, 13, 14, 15, 16, 17, 18]

        self.xym = xym.cuda()

    def cluster_with_gt(self, prediction, instance, n_sigma=1, ):

        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).byte().cuda()

        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = instance.eq(id).view(1, height, width)

            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s * 10)  # n_sigma x 1 x 1

            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))

            proposal = (dist > 0.5)
            instance_map[proposal] = id

        return instance_map

    def cluster(self, prediction, labels, n_sigma=1, threshold=0.5):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_maps = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 8])
        masks = (seed_maps > 0.5)
        for i in range(8):
            count = 1
            if masks[i].sum() > 128:
                spatial_emb_masked = spatial_emb[masks[i].expand_as(spatial_emb)].view(2, -1)
                sigma_masked = sigma[masks[i].expand_as(sigma)].view(n_sigma, -1)
                seed_map_masked = seed_maps[i][masks[i]].view(1, -1)
                unclustered = torch.ones(masks[i].sum()).byte().cuda()
                instance_map_masked = torch.zeros(masks[i].sum()).long().cuda()
                while unclustered.sum() > 128:
                    seed = (seed_map_masked * unclustered.float()).argmax().item()
                    seed_score = (seed_map_masked * unclustered.float()).max().item()
                    if seed_score < threshold:
                        break
                    center = spatial_emb_masked[:, seed:seed + 1]
                    unclustered[seed] = 0
                    s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                    dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                              center, 2) * s, 0, keepdim=True))
                    proposal = (dist > 0.5).squeeze()
                    if proposal.sum() > 128:
                        if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                            instance_map_masked[proposal.squeeze()] = self.class_ids[i] * 1000 + count
                            count += 1
                    unclustered[proposal] = 0
                labels[masks[i].squeeze()] = instance_map_masked

        return labels


class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


class Colorize2:

    def __init__(self, n=22):
        # self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        # for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def colormap_cityscapes(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([128, 64, 128])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([70, 70, 70])
    cmap[3, :] = np.array([102, 102, 156])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])

    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([107, 142, 35])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])

    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])

    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([119, 11, 32])
    cmap[19, :] = np.array([0, 0, 0])

    return cmap
