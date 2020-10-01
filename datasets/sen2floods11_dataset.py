import os
import rasterio
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def load_image(path):
    img = (rasterio.open(path).read()) / 17418  # normalize image to [0,1]
    img = np.nan_to_num(img) * 255
    img = np.rollaxis(img, 0, 3)

    # read each channel of image as separate PIL image
    pil_imgs = []
    for c in range(img.shape[-1]):
        channel_img = img[:,:,c]
        channel_img = Image.fromarray(channel_img.astype(np.uint8))
        pil_imgs.append(channel_img)
    return pil_imgs

def load_target(path):
    img = rasterio.open(path).read()
    img[img == -1] = 255
    img = np.rollaxis(img, 0, 3).astype(np.uint8)
    img = Image.fromarray(img[:,:,0])
    return img

def create_cs_img(h, w, cs_points):
    cs_img = np.zeros((h,w))
    cs_img[cs_points] = 1
    cs_img = Image.fromarray(cs_img.astype(np.uint8))
    return cs_img

def convert_pil_to_tensor(pil_img):
    np_img = np.array(pil_img)
    if len(np_img.shape) == 3:
        np_img = np_img.transpose(2, 0, 1)
    else:
        np_img = np_img[np.newaxis,:,:]
    tensor = torch.from_numpy(np_img)
    return tensor


class Data_Transforms(object):
    def __init__(self, crop_size=[256, 256]):
        self.crop_size = crop_size

    def __call__(self, images, mask, cs_img=None):
        # random crop
        img = images[0]
        i, j, h, w = transforms.RandomCrop.get_params(img, self.crop_size)

        for idx in range(len(images)):
            images[idx] = TF.crop(images[idx], i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if not cs_img is None:
            cs_img = TF.crop(cs_img, i, j, h, w)

        # h flip
        if np.random.random() > 0.5:
            for idx in range(len(images)):
                images[idx] = TF.hflip(images[idx])
            mask = TF.hflip(mask)

            if not cs_img is None:
                cs_img = TF.hflip(cs_img)

        # v flip
        if np.random.random() > 0.5:
            for idx in range(len(images)):
                images[idx] = TF.vflip(images[idx])
            mask = TF.vflip(mask)

            if not cs_img is None:
                cs_img = TF.vflip(cs_img)

        for idx in range(len(images)):
            images[idx] = convert_pil_to_tensor(images[idx]).float()
        img = torch.stack(images, dim=0).squeeze()

        mask = convert_pil_to_tensor(mask).long()
        mask = mask.squeeze()

        if not cs_img is None:
            cs_img = convert_pil_to_tensor(cs_img).float()

        if not cs_img is None:
            return img, mask, cs_img
        else:
            return img, mask


class Test_Data_Transforms(Data_Transforms):
    def __call__(self, images, mask, cs_img=None):
        for idx in range(len(images)):
            images[idx] = convert_pil_to_tensor(images[idx]).float()
        img = torch.stack(images, dim=0).squeeze()

        mask = convert_pil_to_tensor(mask).long()
        mask = mask.squeeze()

        if not cs_img is None:
            cs_img = convert_pil_to_tensor(cs_img).float()

        if not cs_img is None:
            return img, mask, cs_img
        else:
            return img, mask


class Sen2Floods11_Dataset(Dataset):
    def __init__(self, base_dir, csv_path, is_train=True, crowd_points_path=None, annotation_level=None):
        self.crowd_points_path = crowd_points_path

        if is_train:
            self.transforms = Data_Transforms()
        else:
            self.transforms = Test_Data_Transforms()

        # get image paths
        csv_file = pd.read_csv(csv_path).to_numpy()

        # load crowd points for each image
        if crowd_points_path:
            import pickle
            # load pickle file
            crowd_points_file = pickle.load(open(crowd_points_path, 'rb'))

        self.dataset = []
        for i in range(csv_file.shape[0]):
            data_path = os.path.join(base_dir, csv_file[i][0]).replace('S1', 'S2')

            mask_path = os.path.join(base_dir, csv_file[i][1])
            if annotation_level == 'coarse':
                mask_path = mask_path.replace('QC_v2', 'QC_v2_shrunk')

            if self.crowd_points_path:
                # load points and 
                img_name = os.path.splitext(os.path.split(csv_file[i][1])[1])[0]
                crowd_points = crowd_points_file[img_name]
                self.dataset.append([data_path, mask_path, crowd_points])
            else:
                self.dataset.append([data_path, mask_path])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = {}
        if self.crowd_points_path:
            img_path, mask_path, crowd_points = self.dataset[index]
        else:
            img_path, mask_path = self.dataset[index]

        img = load_image(img_path)
        mask = load_target(mask_path)

        if self.transforms:
            if self.crowd_points_path:
                cs_img = create_cs_img(mask.size[0], mask.size[1], crowd_points)
                img, mask, cs_img = self.transforms(img, mask, cs_img)
                data['cs_img'] = cs_img
            else:
                img, mask = self.transforms(img, mask)
        data['img'] = img
        data['target'] = mask
    
        return data


if __name__ == '__main__':
    base_dir = '/home/purri/research/water_dots/Sen1_dataset/'
    csv_path = os.path.join(base_dir, 'flood_train_data.csv')
    csv_file = pd.read_csv(csv_path).to_numpy()

    dataset = []
    min_val, max_val = 0, 0
    mmin_val, mmax_val = 0, 0
    for i in range(csv_file.shape[0]):
        data_path = os.path.join(base_dir, csv_file[i][0]).replace('S1', 'S2')
        mask_path = os.path.join(base_dir, csv_file[i][1])

        img = np.nan_to_num(rasterio.open(data_path).read())
        mask = np.nan_to_num(rasterio.open(mask_path).read())

        min_val = min(img.min(), min_val)
        max_val = max(img.max(), max_val)

        mmin_val = min(mask.min(), mmin_val)
        mmax_val = max(mask.max(), mmax_val)
    
    print('Extrema for Sen2 images')
    print(min_val, max_val)  # [0, 17418]
    print('Extrema for Sen2 masks')
    print(mmin_val, mmax_val) # [-1, 1]
