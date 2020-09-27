import os

from .sen1floods11_dataset import Sen1Floods11_Dataset
from .sen2floods11_dataset import Sen2Floods11_Dataset

name_to_dataset = {'sen1': Sen1Floods11_Dataset,
                   'sen2': Sen2Floods11_Dataset,
                  }


def get_dataset(dataset_name, base_dir, csv_file_path, is_train, crowd_points_path):
    dataset_name = dataset_name.lower()
    dataset_object = name_to_dataset[dataset_name]
    return dataset_object(base_dir, csv_file_path, is_train, crowd_points_path)

def get_cs_points_path(base_dir, cluster_type, noise_type):
    try:
        cluster_type = cluster_type.lower()
        noise_type = noise_type.lower()
    except AttributeError:
        # either cluster_type or noise_type is None
        return None

    if not cluster_type in ['low', 'high']:
        print('FATAL: Invalid cluster type {}'.format(cluster_type))
        exit()

    if not noise_type in ['low', 'high']:
        print('FATAL: Invalid noise type {}'.format(noise_type))
        exit()

    cs_points_path = os.path.join(base_dir, 'QC_v2_cluster_{}_noise_{}.p'.format(cluster_type, noise_type))

    return cs_points_path