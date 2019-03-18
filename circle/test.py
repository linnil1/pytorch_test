import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import sys
import matplotlib.pyplot as plt
import tqdm

from models import *
from dataset import Count, GaussianDataset

output_dir = 'save/'
dataset_name = 'gau'
method = 'UNET'
net = UNet(1, 1)
loss_fn = CountLoss()
version = 'v1'
batch = 5

# init
save_name = method + '_' + dataset_name + '_' + version
save_name = os.path.join(output_dir, save_name)

# load net
print(save_name)
data = torch.load("{}.pt".format(save_name))
net.load_state_dict(data)
print(net)
net.cuda()

# load dataset
validset = GaussianDataset(mode="valid",
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
data_loader_val = DataLoader(validset,
                             batch_size=batch,
                             shuffle=True,
                             num_workers=8)
valid_count = Count()

# calculate error on the validation dataset
net.eval()
with torch.no_grad():
    for blob in data_loader_val:
        dmap = net(blob['smap'].cuda())
        gtmap = blob['dmap']
        plt.subplot(131)
        plt.imshow(blob['smap'][1,0].cpu().numpy())
        plt.subplot(132)
        plt.imshow(dmap[1,0].cpu().numpy())
        plt.subplot(133)
        plt.imshow(gtmap[1, 0].cpu().numpy())
        plt.show()
        valid_count.add(dmap, gtmap)
        break

valid_mae, valid_mse = valid_count.getResult()

log_text = '%s VALID: MAE: %.4f, MSE: %0.4f' % \
           (save_name, valid_mae, valid_mse)
print(log_text)
