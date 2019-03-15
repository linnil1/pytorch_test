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

try:
    from termcolor import cprint
except ImportError:
    cprint = None


output_dir = 'save/'
dataset_name = 'gau'
method = 'UNET'
net = DeConv()
loss_fn = CountLoss()
version = 'v4'
start_epoches = 0
epoches = 250
lr = 1e-4
weight_decay = 0
patient_until = 0
patient = 16
rand_seed = 123
resolution=8
batch = 10

# init
save_name = method + '_' + dataset_name + '_' + version
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
save_name = os.path.join(output_dir, save_name)

# load net
if start_epoches:
    # data = torch.load("{}_{}.pt".format(save_name, start_epoches))
    data = torch.load("{}.pt".format(save_name))
    net.load_state_dict(data)
print(net)
net.cuda()

# load dataset
trainset = GaussianDataset(mode="train",
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
validset = GaussianDataset(mode="valid",
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
data_loader_trn = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=8)
data_loader_val = DataLoader(validset, batch_size=batch, shuffle=True, num_workers=8)

# optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)

# save data
history = []
best_mse = 1e99


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


for epoch in range(start_epoches + 1, epoches + 1):
    train_count = Count()
    valid_count = Count()
    train_loss = 0

    net.train()
    for blob in tqdm.tqdm(data_loader_trn, ncols=57, leave=True):
        optimizer.zero_grad()
        gtmap = blob['dmap'].cuda()
        dmap = net(blob['smap'].cuda())
        loss = loss_fn(dmap, gtmap)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss
            train_count.add(dmap, gtmap)

    # calculate error on the validation dataset
    net.eval()
    with torch.no_grad():
        for blob in data_loader_val:
            dmap = net(blob['smap'].cuda())
            gtmap = blob['dmap']
            valid_count.add(dmap, gtmap)

    train_mae, train_mse = train_count.getResult()
    valid_mae, valid_mse = valid_count.getResult()
    history.append({'epoch': epoch,
                    'train_mae': train_mae,
                    'train_mse': train_mse,
                    'valid_mae': valid_mae,
                    'valid_mse': valid_mse,
                    'loss': train_loss})
    np.savez(save_name + '.npz', history=history)

    # save best
    if valid_mse < best_mse:
        best_mae = valid_mae
        best_mse = valid_mse
        # best_model = '{}_{}.pt'.format(save_name, epoch)
        best_model = '{}.pt'.format(save_name)
        torch.save(net.state_dict(), best_model)
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, BEST MODEL: %s' % (best_mae, best_mse, best_model)
        log_print(log_text, color='red', attrs=['bold'])
        now_pat = patient
    else:
        if epoch > patient_until:
            now_pat -= 1
            if now_pat < 0:
                break

    log_text = 'EPOCH: %d, MAE: %0.4f MSE: %0.4f VALID: MAE: %.4f, MSE: %0.4f, LOSS: %f' % \
               (epoch, train_mae, train_mse, valid_mae, valid_mse, train_loss)
    log_print(log_text, color='green', attrs=['bold'])


# plot
history = history[:10]
plt.suptitle(method + ' on Agar dataset')
x = [h['epoch'] for h in history]
plt.subplot(131)
plt.plot(x, [h['valid_mae'] for h in history], label='valid_mae')
plt.plot(x, [h['train_mae'] for h in history], label='train_mae')
plt.legend()
plt.subplot(132)
plt.plot(x, [h['valid_mse'] for h in history], label='valid_mse')
plt.plot(x, [h['train_mse'] for h in history], label='train_mse')
plt.legend()
plt.subplot(133)
plt.plot(x, [h['loss'] for h in history], label='loss')
plt.legend()
plt.xlabel("epoch")
plt.savefig(save_name + '.png')
plt.show()
