import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

def shrink_annotation(img, threshold=0.3, kernal_size=55):
    # apply Gaussian filter to the image
    blur = cv2.GaussianBlur(img,(kernal_size,kernal_size),0)
    blur[blur < threshold] = 0
    blur[blur > threshold] = 1
    return blur


# get all annotation images 
base_dir = '/home/purri/research/water_dots/Sen1_dataset/'

anno_folder_names = ['QC_v2', 'NoQC']

for anno_folder_name in anno_folder_names:
    print(anno_folder_name)
    anno_dir = os.path.join(base_dir, anno_folder_name)
    anno_paths = sorted(glob(anno_dir + '/*.tif'))

    if len(anno_paths) == 0:
        print('No images found for {} folder'.format(anno_folder_name))

    save_dir = os.path.join(base_dir, anno_folder_name+'_shrunk')

    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)

    for anno_path in tqdm(anno_paths):

        # load image
        try:
            img = np.asarray(Image.open(anno_path))
        except UnidentifiedImageError:
            continue

        # get water image
        xw, yw = np.where(img == 1)  # get all water pixels
        water_img = np.zeros(img.shape)
        water_img[xw, yw] = 1

        # shrink the water annotations
        shrunk_img = shrink_annotation(water_img)

        assert img.shape == shrunk_img.shape

        # save shrunk image
        img_name = os.path.split(anno_path)[1]
        save_path = os.path.join(save_dir, img_name)       
        Image.fromarray(shrunk_img).save(save_path)
