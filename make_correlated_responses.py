import cifar10_utils

import numpy as np
import json
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm

import random
seed=7607

BATCH_SIZE = 100

class CorrResponsesSet:
    def __init__(self, gabor_params_file, images, canvas_size, z = 10, Lambda = None, noisy = True):
        self.gabor_params_file = gabor_params_file
        self.images = images
        self.canvas_size = canvas_size
        self.gabor_params_list = self.load_gabor_params()
        self.n_gabors = self.num_gabors()
        self.z = z  # latent state
        if Lambda is None:
            self.Lambda = np.random.uniform(0, 1, self.n_gabors)
            print(f"Generated random Lambda: {self.Lambda}")
        else:
            self.Lambda = Lambda
        self.noisy = noisy  # add noise to the responses  

    def load_gabor_params(self):
        with open(self.gabor_params_file, 'r') as json_file:
            gabor_params_list = json.load(json_file)
        return gabor_params_list
    
    def num_gabors(self):
        return len(self.gabor_params_list)

    def gabor(self, location, size, spatial_frequency, contrast, orientation, phase):
        x, y = np.meshgrid(
            np.arange(self.canvas_size[0]) - location[0],
            np.arange(self.canvas_size[1]) - location[1],
        )
        R = np.array(
            [
                [np.cos(orientation), -np.sin(orientation)],
                [np.sin(orientation), np.cos(orientation)],
            ]
        )
        coords = np.stack([x.flatten(), y.flatten()])
        x, y = R.dot(coords).reshape((2,) + x.shape)
        envelope = contrast * np.exp(-(x ** 2 + y ** 2) / (2 * (size / 4) ** 2))
        grating = np.cos(spatial_frequency * x * (2 * np.pi) + phase)
        return envelope * grating
    
    def add_latent_state(self, i):
        return self.z*self.Lambda[i]

    def response_from_stimulus_state(self, image, i):
        params = self.gabor_params_list[i]
        gabor_filter = self.gabor(
            params["location"],
            params["size"],
            params["spatial_frequency"],
            params["contrast"],
            params["orientation"],
            params["phase"],
        )
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gabor_tensor = torch.tensor(gabor_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        convolved = F.conv2d(image_tensor, gabor_tensor, padding='valid')
        latent_state = self.add_latent_state(i)
        stimulus_state = convolved + latent_state
        elu_output = F.elu(stimulus_state)

        if self.noisy:
            return np.random.poisson(elu_output.squeeze().numpy().flatten())
        else:
            return elu_output.squeeze().numpy().flatten()
    
    # responses to all images for one Gabor filter
    def responses_from_stimulus_state(self, i):
        responses = [self.response_from_stimulus_state(image, i) for image in self.images]
        return np.concatenate(responses)

    # responses of all gabors by batches 
    def response_batches(self, batch_size):
        for batch_start in np.arange(0, self.n_gabors, batch_size):
            batch_end = np.minimum(batch_start + batch_size, self.n_gabors)
            responses = [self.responses_from_stimulus_state(i) for i in range(batch_start, batch_end)]
            yield np.array(responses)

    def responses(self):
        return np.array([self.responses_from_stimulus_state(i) for i in range(self.n_gabors)])

if __name__ == "__main__":
    import cifar10_utils

    canvas_size = [9, 9]

    data_dir = './data'
    gabor_params_file = f'./gabor_params/gabor_params_{canvas_size[0]}.json'  
    output_dir = f'./correlated_responses_{canvas_size[0]}'

    images = cifar10_utils.load_data(data_dir, N=1000) # N grayscaled images from cifar10
    print(images.shape)

    responses_set = CorrResponsesSet(gabor_params_file, images, canvas_size)  

    batch_size = 100
    num_gabors = responses_set.num_gabors()
    print(num_gabors)
    for batch_start in tqdm(range(0, num_gabors, batch_size)):
        batch_end = min(batch_start + batch_size, num_gabors)
        responses = [responses_set.responses_from_stimulus_state(i) for i in range(batch_start, batch_end)]
        responses_array = np.array(responses)
        output_file = os.path.join(output_dir, f'responses_batch_{batch_start}.npy')
        np.save(output_file, responses_array)
        print(f"Saved responses batch {batch_start} to {output_file}")

    # save all responses at once
    # all_responses = np.array([responses_set.responses_from_gabor_params(responses_set.gabor_params_list[i]) for i in range(num_gabors)])
    # np.save(os.path.join(output_dir, 'all_responses.npy'), all_responses)
    # print("Saved all responses to 'all_responses.npy'")
    
    # for batch_idx, batch_start in tqdm(enumerate(np.arange(0, num_stims, BATCH_SIZE)), total=num_stims/BATCH_SIZE):
    #     batch_end = np.minimum(batch_start + BATCH_SIZE, num_stims)
    #     images = [g.gabor_from_idx(i) for i in range(batch_start, batch_end)]
    #     images_or = np.array(images)
    #     np.save(f'/user/turishcheva/optimal_gabors_saved/raw_images/{batch_idx}.npy', images_or)