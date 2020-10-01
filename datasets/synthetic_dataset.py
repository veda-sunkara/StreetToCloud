import os
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def gen_syn_image():
    base_img = np.zeros((256, 256))
    X, Y = np.mgrid[0:256, 0:256]

    n_points = 256**2
    x_idx = np.random.choice(np.arange(256), round(n_points*0.1))
    y_idx = np.random.choice(np.arange(256), round(n_points*0.1))
    base_img[x_idx, y_idx] = 1
    
    mask_img = base_img.copy()

    x_idx = np.random.choice(np.arange(256), round(n_points*0.1))
    y_idx = np.random.choice(np.arange(256), round(n_points*0.1))
    base_img[x_idx, y_idx] = 0.5

    return base_img, mask_img

class Synthetic_Dataset(Dataset):
    def __init__(self, base_dir, csv_path, is_train=True, crowd_points_path=None, annotation_level=None):
        self.num_images = 400
        self.toT = transforms.ToTensor()

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        data = {}
        # create a random image
        img, mask = gen_syn_image()

        img = self.toT(img).float()
        mask = self.toT(mask).long()

        data['img'] = img
        data['target'] = mask.squeeze()
    
        return data
