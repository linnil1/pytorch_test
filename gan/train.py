import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
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


# pix2pix method
output_dir = 'save/'
dataset_name = 'gan_pix'
method = 'UNET'
net = UNet(1, 1)
net_d = Discriminator(1)
loss_fn = nn.MSELoss()
loss_fn_d = nn.BCELoss()
coeff_d = .01
version = 'v2'
start_epoches = 0
epoches = 250
lr = 1e-4
weight_decay = 0
patient_until = 0
patient = 16
rand_seed = 123
batch = 5
norm = 1000
patch_size = 70

# init
save_name = method + '_' + dataset_name + '_' + version
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
save_name = os.path.join(output_dir, save_name)

# two
# load net
# if start_epoches:
#     data = torch.load("{}.pt".format(save_name))
#     net.load_state_dict(data)

net.cuda()
net_d.cuda()

# load dataset
trainset = GaussianDataset(mode="train",
                           norm=norm,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
validset = GaussianDataset(mode="valid",
                           norm=norm,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
data_loader_trn = DataLoader(trainset,
                             batch_size=batch,
                             shuffle=True,
                             num_workers=8)
data_loader_val = DataLoader(validset,
                             batch_size=batch,
                             shuffle=True,
                             num_workers=8)

# optimizer
optimizer_net = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        weight_decay=weight_decay)
optimizer_d = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net_d.parameters()),
        lr=lr,
        weight_decay=weight_decay)

# save data
history = []
best_mse = 1e99


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def crop(data):
    x = torch.randint(0, data.size()[-2] - patch_size, [1])
    y = torch.randint(0, data.size()[-1] - patch_size, [1])
    return data[..., x:x+patch_size, y:y+patch_size]


for epoch in range(start_epoches + 1, epoches + 1):
    train_count = Count(norm)
    valid_count = Count(norm)
    train_loss = 0

    net.train()
    for blob in tqdm.tqdm(data_loader_trn, ncols=100, leave=False):
        b = blob['smap'].size(0)
        real = Variable(torch.ones(b, 1, 1, 1), requires_grad=False).cuda()
        fake = Variable(torch.zeros(b, 1, 1, 1), requires_grad=False).cuda()

        # net
        smap = blob['smap'].cuda()
        dmap = blob['dmap'].cuda()
        optimizer_net.zero_grad()
        density = net(smap)
        pred_fake = net_d(crop(density), crop(smap))
        loss_net = loss_fn(density, dmap)
        loss_d = loss_fn_d(pred_fake, real)
        loss = loss_net + coeff_d * loss_d
        loss.backward()
        optimizer_net.step()

        # discriminate
        optimizer_d.zero_grad()
        pred_real = net_d(crop(dmap), crop(smap))
        loss_d_real = loss_fn_d(pred_real, real)
        loss_d_fake = loss_fn_d(pred_fake.detach(), fake)
        loss = coeff_d * (loss_d_fake + loss_d_real)
        loss.backward()
        optimizer_d.step()

        with torch.no_grad():
            train_loss += loss_net
            train_count.add(density, dmap)

    # calculate error on the validation dataset
    net.eval()
    with torch.no_grad():
        for blob in tqdm.tqdm(data_loader_val, ncols=100, leave=False):
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

    # log
    log_text = 'EPOCH: %d LOSS: %0.4f' % (epoch, train_loss)
    log_text += '  TRAIN: MAE: %0.4f MSE: %0.4f' % (train_mae, train_mse)
    log_text += '  VALID: MAE: %0.4f MSE: %0.4f' % (valid_mae, valid_mse)
    log_print(log_text, color='green', attrs=['bold'])

    # save best
    if valid_mse < best_mse:
        best_mae = valid_mae
        best_mse = valid_mse
        # best_model = '{}_{}.pt'.format(save_name, epoch)
        best_model = '{}.pt'.format(save_name)
        torch.save(net.state_dict(), best_model[:-3] + '_net.pt')
        torch.save(net_d.state_dict(), best_model[:-3] + '_dis.pt')
        log_text = 'BEST MAE: %0.1f, BEST MSE: %0.1f, ' % (best_mae, best_mse)
        log_text += 'BEST MODEL: %s' % (best_model,)
        log_print(log_text, color='red', attrs=['bold'])
        now_pat = patient
    else:
        if epoch > patient_until:
            now_pat -= 1
            if now_pat < 0:
                break


# plot
history = history[5:]
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
