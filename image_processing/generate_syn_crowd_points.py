import os
import cv2
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt


def sample_crowd_points(img, clusters='low', noise='low'):
    img = (img * 255).astype(np.uint8) # convert from float to integer values

    # get edges of annotation
    edges = cv2.Canny(img, 30, 60)  # 100, 200

    # sample edges of annotation
    x, y = np.where(edges == 255)

    n_points = x.shape[0]

    if n_points == 0:
        return None

    if clusters == 'low':
        c_prob = 0.0020
    elif clusters == 'high':
        c_prob = 0.0005

    rx = np.random.choice(x, size=int(c_prob*n_points))
    ry = np.random.choice(y, size=int(c_prob*n_points))

    points = np.stack((ry, rx), axis=1)

    if clusters == 'high':
        points = np.concatenate((points, points, points, points), axis=0)

    # add noise to points
    if noise == 'low':
        n = np.random.randint(low=-5, high=5, size=points.shape)
    elif noise == 'high':
        n = np.random.randint(low=-10, high=10, size=points.shape)
   
    # add noise to points
    points += n

    # clip values to edge of image
    points = np.clip(points, a_min=0, a_max=img.shape[0])

    return points


# get all annotation images 
base_dir = '/home/purri/research/water_dots/Sen1_dataset/'
anno_folder_names = ['QC_v2', 'NoQC']

np.random.seed(0)  # set random seed

clusters = 'high'
noise = 'high'

for anno_folder_name in anno_folder_names:
    print(anno_folder_name)
    anno_dir = os.path.join(base_dir, anno_folder_name)
    anno_paths = sorted(glob(anno_dir + '/*.tif'))

    if len(anno_paths) == 0:
        print('No images found for {} folder'.format(anno_folder_name))

    crowd_data = {}
    for anno_path in tqdm(anno_paths):
        img_name = os.path.splitext(os.path.split(anno_path)[1])[0]

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
        crowd_points = sample_crowd_points(water_img)

        crowd_data[img_name] = crowd_points

        # plt.subplot(1,2,1)
        # plt.imshow(water_img); plt.title('Original Annotation')
        # plt.subplot(1,2,2)
        # plt.imshow(water_img); plt.title('Original Annotation')
        # plt.scatter(crowd_points[:,0], crowd_points[:,1])
        # plt.show()

    # save crowd points
    save_path = os.path.join(base_dir, anno_folder_name+'_cluster_'+clusters+'_noise_'+noise+'.p')
    pickle.dump(crowd_data, open(save_path, 'wb'))

