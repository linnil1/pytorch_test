import h5py
from PIL import Image, ImageDraw
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
n_max = 100
n_div = 10

for i in range(200):
    print('generate', i) 
    n = int(np.random.normal(n_max, n_div))
    name = "img_{:03}.h5".format(i)
    x = np.random.randint(shape[0], size=n)
    y = np.random.randint(shape[1], size=n)

    point = np.zeros(shape, dtype=np.float)
    img = Image.new('L', shape)
    draw = ImageDraw.Draw(img)
    for i in range(n):
        s = np.random.normal(20, 5)
        if s < 5:
            continue
        draw.ellipse((y[i], x[i], y[i] + s, x[i] + s), fill = 'white')
        s = np.random.normal(0, abs(s) / 2, size=2)
        s = [0, 0]
        nx = x[i] + int(s[0])
        ny = y[i] + int(s[1])
        if nx < 0: nx = 0
        if ny < 0: ny = 0
        if nx >= shape[0]: nx = shape[0] - 1
        if ny >= shape[1]: ny = shape[1] - 1
        point[nx, ny] = 1

    d = gaussian_filter_density_fix(point)
    with h5py.File(save_path + name, 'w') as hf:
        hf['point'] = point
        hf['image'] = img
        hf['density'] = d

    """
    plt.subplot(1, 2, 1)
    plt.imshow(d)
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.show()
    """
