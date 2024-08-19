import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

import torch
import torch.nn.functional as F

from make_gabors import GaborSet


import warnings
warnings.filterwarnings('ignore')


seed=7607
BATCH_SIZE = 10
MAX_CONTRAST = 1.75
MIN_CONTRAST = 0.01 * MAX_CONTRAST

MIN_SIZE = 2
MAX_SIZE = 9

num_contrasts = 6
contrast_increment = (MAX_CONTRAST / MIN_CONTRAST) ** (1 / (num_contrasts - 1))

num_sizes = 8
size_increment = (MAX_SIZE / MIN_SIZE) ** (1 / (num_sizes - 1))

random.seed(seed)
np.random.seed(seed)

canvas_size = [MAX_SIZE, MAX_SIZE]
x_start = 2
x_end = 3
y_start = 2
y_end = 3
min_size = MIN_SIZE

min_sf = (1.3 ** -1)
num_sf = 10
sf_increment = 1.3
min_contrast = MIN_CONTRAST

num_orientations = 12
num_phases = 8

center_range = [x_start, x_end, y_start, y_end]
sizes = min_size * size_increment ** np.arange(num_sizes)
sfs = min_sf * sf_increment ** np.arange(num_sf)
c = min_contrast * contrast_increment ** np.arange(num_contrasts)

g = GaborSet(
    canvas_size,
    center_range,
    sizes,
    sfs,
    c,
    num_orientations,
    num_phases,
)

num_stims = np.prod(g.num_params)
print(num_stims)

output_dir = f'./sample_gabor_images_{MAX_SIZE}'

# load 10 Cifar 10 images 
import cifar10_utils
data_dir = './data'
images = cifar10_utils.load_data(data_dir, N=10) # N grayscaled images from cifar10
print(images.shape)

for i in range(43000, 43002):
    # gabor image and parameters
    gabor_image = g.gabor_from_idx(i)
    location, size, spatial_frequency, contrast, orientation, phase = g.params_from_idx(i)
     
    # Normalize the image data
    min_val = np.min(gabor_image)
    max_val = np.max(gabor_image)
    
    if max_val > min_val: # avoid division by zero
        image_normalized = (gabor_image - min_val) / (max_val - min_val)
    else:
        image_normalized = np.zeros_like(image)
    
    gabor_image = (image_normalized * 255).astype(np.uint8)
    image_pil = Image.fromarray(gabor_image)
    image_pil.save(f'{output_dir}/image_{i}_{size:.1f}_sf_{spatial_frequency:.1f}_{orientation:.1f}.png')

    for j in range(10):
        image = images[j]
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_pil.save(f'{output_dir}/image_{j}.png')
        # save convolution of image and gabor
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gabor_tensor = torch.tensor(gabor_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        convolved = F.conv2d(image_tensor, gabor_tensor, padding='valid').squeeze().numpy()
        convolved_pil = Image.fromarray((convolved * 255).astype(np.uint8))
        convolved_pil.save(f'{output_dir}/c_{j}_{i}_{size:.1f}_sf_{spatial_frequency:.1f}_{orientation:.1f}.png')
    