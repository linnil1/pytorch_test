import h5py
from PIL import Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density_fix(gt):
    density = np.array(gt, dtype=np.float32)
    sigma = 15
    density = gaussian_filter(density, sigma, mode='constant')
    return density

save_path = './data/'
os.makedirs(save_path, exist_ok=True)
shape = (512, 512)
res = 1
n_max = 50

for i in range(220):
    print('generate', i) 
    n = np.random.randint(n_max)
    name = "img_{:03}.h5".format(i)
    x = np.random.randint(shape[0], size=n)
    y = np.random.randint(shape[1], size=n)
    m = np.zeros(shape)
    m[x, y] = 1

    d = gaussian_filter_density_fix(m)
    with h5py.File(save_path + name, 'w') as hf:
        hf['map'] = m
        hf['density'] = d
