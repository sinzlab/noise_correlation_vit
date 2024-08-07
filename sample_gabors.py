import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import os


from make_gabors import GaborSet


import warnings
warnings.filterwarnings('ignore')


seed=7607
BATCH_SIZE = 10
MAX_CONTRAST = 1.75
MIN_CONTRAST = 0.01 * MAX_CONTRAST

MIN_SIZE = 2
MAX_SIZE = 5

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


for i in range(43000, 43002):
    image = g.gabor_from_idx(i)
    location, size, spatial_frequency, contrast, orientation, phase = g.params_from_idx(i)
     
    # Normalize the image data
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val > min_val: # avoid division by zero
        image_normalized = (image - min_val) / (max_val - min_val)
    else:
        image_normalized = np.zeros_like(image)
    
    image = (image_normalized * 255).astype(np.uint8)
    image_pil = Image.fromarray(image)
    image_pil.save(f'{output_dir}/image_{i}_{size:.1f}_sf_{spatial_frequency:.1f}_{orientation:.1f}.png')