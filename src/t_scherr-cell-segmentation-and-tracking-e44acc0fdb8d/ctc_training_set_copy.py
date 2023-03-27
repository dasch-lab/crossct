import json
import math
import tifffile as tiff
from pathlib import Path
from skimage.morphology import binary_closing
from segmentation.training.train_data_representations import *
from segmentation.utils.utils import get_nucleus_ids
import os
import glob
from skimage import io

def convert_ctc_data_set(path, path_new, radius, mode='GT', k_pena=2, name = None, rap = None, t = None):
    """ Create training data representations of the frames and data set specified.

    :param path: Path to the directory containing the Cell Tracking Challenge data.
        :type path: Path
    :param radius: Radius needed for distance label creation.
        :type radius: int
    :param mode: Use GT or ST data.
        :type mode: str
    :param frames: Frames to use for the training set (not all frames provide full annotations).
        :type frames: list
    :return: None
    """

    # Get ids of the label images
    #label_ids = sorted((path / ('01_' + mode) / 'SEG').glob('*.tif'))
    #img_list = sorted(glob.glob(os.path.join(path, '*.png')))
    #lab_list = sorted(glob.glob(os.path.join(path, '*_label.tiff')))
    lab_list = sorted(glob.glob(os.path.join(path, '*_masks.png')))
    print(lab_list[0])
    #lab_list.extend(sorted(glob.glob(os.path.join(path, '*_omnipose_*_label.png'))))

    # Create label images
    for i, label_id in enumerate(lab_list):

        #label = tiff.imread(label_id)
        label = io.imread(label_id)

        #img = io.imread(img_list[i])

        #label_bin = binary_label(label=label)
        #label_boundary = boundary_label_2d(label=label, algorithm='dilation')
        #label_border = border_label_2d(label=label, algorithm='dilation')
        #label_adapted_border = adapted_border_label_2d(label=label)
        #label_dist_chebyshev = chebyshev_dist_label_2d(label=label, normalize_dist=True, radius=radius)
        label_dist, label_dist_neighbor = dist_label_2d(label=label, neighbor_radius=radius)
        #label_pena = pena_label_2d(label=label, k_neighbors=k_pena)

        # Add pseudo color channel and min-max normalize to 0 - 65535
        # img = np.expand_dims(img, axis=-1)
        # img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
        # img = np.clip(img, 0, 65535).astype(np.uint16)
        #label = np.expand_dims(label, axis=-1).astype(np.uint16)
        #label_bin = 255 * np.expand_dims(label_bin.astype(np.uint8), axis=-1)
        #label_boundary = np.expand_dims(label_boundary.astype(np.uint8), axis=-1)
        #label_border = np.expand_dims(label_border.astype(np.uint8), axis=-1)
        #label_adapted_border = np.expand_dims(label_adapted_border.astype(np.uint8), axis=-1)
        #label_dist_chebyshev = np.expand_dims(label_dist_chebyshev, axis=-1).astype(np.float32)
        label_dist = np.expand_dims(label_dist, axis=-1).astype(np.float32)
        label_dist_neighbor = np.expand_dims(label_dist_neighbor, axis=-1).astype(np.float32)
        #label_pena = np.expand_dims(label_pena.astype(np.uint8), axis=-1)

        # Save
        #lab_name = label_id.split('/')[-1].split('_')[0] + '_' + name + '_' + rap + '_' + t + '_label.tiff'
        lab_name = label_id.split('/')[-1].split('_')[0] + '_' + rap + '_' + t + '_label.tiff'
        print(lab_name)
        #tiff.imsave(os.path.join(path_new, 'images', img_name), img)
        #tiff.imsave(os.path.join(path_new, 'labels', lab_name), label)
        #tiff.imsave(os.path.join(path_new, 'label_bin', lab_name), label_bin)
        #tiff.imsave(os.path.join(path_new, 'label_boundary', lab_name), label_boundary)
        #tiff.imsave(os.path.join(path_new, 'label_border', lab_name), label_border)
        #tiff.imsave(os.path.join(path_new, 'label_adapted_border', lab_name), label_adapted_border)
        #tiff.imsave(os.path.join(path_new, 'label_dist_chebyshev', lab_name), label_dist_chebyshev)
        tiff.imsave(os.path.join(path_new, 'label_dist', lab_name), label_dist)
        tiff.imsave(os.path.join(path_new, 'label_dist_neighbor', lab_name), label_dist_neighbor)
        #tiff.imsave(os.path.join(path_new, 'label_pena', lab_name), label_pena)

    return None


def create_ctc_training_set(path, path_new, name = None, rap = None, t = None):
    """ Create the CTC Training Set.

    :param path: Path to the directory containing the Cell Tracking Challenge data.
        :type path: Path
    :return: None
    """

    # Create directories
    os.makedirs(path_new, exist_ok=True)
    #os.makedirs(os.path.join(path_new, 'images'), exist_ok=True)
    #os.makedirs(os.path.join(path_new, 'labels'), exist_ok=True)
    #os.makedirs(os.path.join(path_new, 'label_bin'), exist_ok=True)
    #os.makedirs(os.path.join(path_new, 'label_boundary'), exist_ok=True)
    #os.makedirs(os.path.join(path_new, 'label_adapted_border'), exist_ok=True)
    #os.makedirs(os.path.join(path_new, 'label_dist_chebyshev'), exist_ok=True)
    os.makedirs(os.path.join(path_new, 'label_dist'), exist_ok=True)
    os.makedirs(os.path.join(path_new, 'label_dist_neighbor'), exist_ok=True)
    #os.makedirs(os.path.join(path_new, 'label_pena'), exist_ok=True)
    
    convert_ctc_data_set(path=path, path_new=path_new, radius=None, mode='GT', name = name , rap = rap , t = t )


    # # Print border information
    # methods = ['boundary', 'border', 'adapted_border', 'dist_neighbor', 'pena_touching', 'pena_gap']
    # print('Border information fraction [1e-3]')
    # for method in methods:
    #     border_pixels, all_pixels = 0, 0
    #     for mode in ['train', 'val']:
    #         if method in ['pena_touching', 'pena_gap']:
    #             label_ids = (path / 'ctc_training_set' / mode).glob(method.split('_')[0] + '*')
    #         else:
    #             label_ids = (path / 'ctc_training_set' / mode).glob(method + '*')
    #         for label_id in label_ids:
    #             label = tiff.imread(str(label_id))
    #             if method == 'dist_neighbor':
    #                 label = label > 0.5
    #             elif method == 'pena_touching':
    #                 label = label == 2
    #             elif method == 'pena_gap':
    #                 label = label == 3
    #             else:
    #                 label = label == 2
    #             border_pixels += np.sum(label)
    #             all_pixels += label.shape[0] * label.shape[1]
    #     print("   {}: {:.2f}".format(method, 1000 * border_pixels / all_pixels))

    return None

if __name__ == "__main__":
    path = '/disk1/neurips/dataset_cellpose/train'
    path_new = '/disk1/neurips/dataset/distance_neighbour_representation'
    create_ctc_training_set(path, path_new, name = None, rap = 'cellpose', t = 'train')
