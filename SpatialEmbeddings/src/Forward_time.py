import os
import time

import test_config
import torch
from datasets import get_dataset
from models import get_model
from utils.utils import Cluster

torch.backends.cudnn.benchmark = True

args = test_config.get_args()

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs']['root_dir'], args['dataset']['kwargs']['type'],
    None, None, args['dataset']['kwargs']['transform'])
dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8, pin_memory=True if args['cuda'] else False)

pretrainedEnc = True
if args['pretrain_encoder']['apply']:
        print("Loading encoder pretrained in imagenet")
        from erfnet_imagenet import ERFNet as ERFNet_imagenet

        pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
        pretrainedEnc.load_state_dict(torch.load(args['pretrain_encoder']['path'])['state_dict'])
        pretrainedEnc = next(pretrainedEnc.children()).features.encoder


model = get_model(args['model']['name'], args['model']['kwargs']['num_classes'], pretrainedEnc)
model = torch.nn.DataParallel(model).to(device)

# load snapshot

if os.path.exists(args['checkpoint_path']):
    print("model_loaded")
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)
else:
    assert (False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

model.eval()
batch_size = 1
cluster = Cluster()

time_train = []
it = iter(dataset_it)

for i in range(len(dataset_it)):
    im = next(it)['image']
    start_time = time.time()
    with torch.no_grad():
        output = model(im)
        label = output[0][0].max(0)[1]
        cluster.cluster(output[1][0], label, threshold=0.9)
        torch.cuda.synchronize()

    if i > 15:
        fwt = time.time() - start_time
        time_train.append(fwt)
        print("Forward time per img (b=%d): %.3f (Mean: %.3f)" % (
            batch_size, fwt / batch_size, sum(time_train) / len(time_train) / batch_size))

    time.sleep(1)  # to avoid overheating the GPU too much


