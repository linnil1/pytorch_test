from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import PIL
import os
import h5py

norm = 1000


class GaussianDataset(Dataset):
    def __init__(self, mode, transform=None):
        self.train_dir= "/home/tmp/gau/"
        self.transform = transform
        self.mode = mode
        self.names= sorted([i[:-3] for i in os.listdir(self.train_dir)])
        self.testnum = len(self.names) // 10
        self.norm = norm

    def __len__(self):
        if self.mode == "train":
            return len(self.names) - self.testnum
        elif self.mode == "valid":
            return self.testnum

    def __getitem__(self, idx):
        if self.mode == "valid":
            idx += len(self.names) - self.testnum

        f = h5py.File(os.path.join(self.train_dir, self.names[idx] + ".h5"), 'r')
        smap = Image.fromarray(np.array(f['map']))
        dmap = Image.fromarray(np.array(f['density']) * self.norm)
        # dmap = Image.fromarray(np.array(f['density']) * 1)

        if np.random.random() < 0.5:
            smap = smap.transpose(Image.FLIP_LEFT_RIGHT)
            dmap = dmap.transpose(Image.FLIP_LEFT_RIGHT)

        return {'smap': self.transform(smap),
                'dmap': self.transform(dmap)}


class Count():
    def __init__(self):
        self.gt = []
        self.et = []
        self.norm = norm

    def add(self, gt_data, density_map):
        gt = gt_data.data.cpu().numpy()
        et = density_map.data.cpu().numpy()
        assert(len(gt) == len(et))
        for i in range(len(gt)):
            self.gt.append(np.sum(gt[i]))
            self.et.append(np.sum(et[i]))

    def getResult(self):
        self.gt = np.array(self.gt) / self.norm
        self.et = np.array(self.et) / self.norm
        mae = np.mean(np.abs(self.gt - self.et))
        mse = np.sqrt(np.mean((self.gt - self.et) ** 2))
        return mae, mse


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = GaussianDataset(mode='train',
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                              ]))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

    for i_batch, sample_batched in enumerate(dataloader):
        plt.subplot(121)
        plt.imshow(transforms.ToPILImage()(sample_batched['simg'][0]))
        plt.subplot(122)
        plt.imshow(sample_batched['dmap'][0].numpy()[0])
        plt.show()
        break
