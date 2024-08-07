import os
import tarfile
import urllib.request
import pickle
import numpy as np
from skimage.color import rgb2gray

def download_and_extract_cifar10(data_dir):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.basename(url)
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(filepath):
        print(f"Downloading CIFAR-10 dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")

    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete.")

# load only the first batch of CIFAR-10 dataset
def load_first_cifar10_batch(data_dir):
    cifar10_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    file = os.path.join(cifar10_dir, 'data_batch_1')
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    images = dict[b'data']
    images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    return images

def convert_to_grayscale(images):
    grayscale_images = np.array([rgb2gray(image) for image in images])
    return grayscale_images

# load first N images from the first batch of CIFAR-10 dataset
def load_data(data_dir, N=1000):
    images = load_first_cifar10_batch(data_dir)
    images = convert_to_grayscale(images[:N])
    return images



if __name__ == "__main__":
    data_dir = './data'
    download_and_extract_cifar10(data_dir)
    images = load_data(data_dir, N=1000)
    print(images.shape)