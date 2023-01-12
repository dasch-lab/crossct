#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:12:04 2022

convert instance labels to three class labels:
0: background
1: interior
2: boundary
@author: jma
"""

import os
join = os.path.join
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint16')
    else:
        img_norm = img
    return img_norm.astype(np.uint16)

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint16 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint16)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior

def read_names(img_path, gt_path, cells = None):
    
    if cells is None:
        img_names = sorted([join(img_path, im) for im in os.listdir(img_path) if im.split('.')[-1] != 'DS_Store'])

    else:
        img_names = []

        for name in cells:
            img_names.extend([join(img_path, name, im) for im in os.listdir(join(img_path, name)) if im.split('.')[-1] != 'DS_Store'])
        
        img_names = sorted(img_names)
        
    gt_names = [join(gt_path, img_name.split('/')[-1].split('.')[0]+'_label.tiff') for img_name in img_names]

    return img_names, gt_names

def read_image(img_name):
    if img_name.endswith('.tif') or img_name.endswith('.tiff'):
            img_data = tif.imread(img_name)
    else:
        img_data = io.imread(img_name)
    return img_data

def process_image(img_names, target_path, image_bool, name):
    
    for img_name in img_names:

        img_data = read_image(img_name)
        
        # normalize image data
        if len(img_data.shape) == 2:
            img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
        elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
            img_data = img_data[:,:, :3]
        else:
            pass
        pre_img_data = np.zeros(img_data.shape, dtype=np.uint16)
        for i in range(3):
            img_channel_i = img_data[:,:,i]
            if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)

        io.imsave(join(target_path, name, img_name.split('/')[-1].split('.')[0]+'.png'), pre_img_data.astype(np.uint16), check_contrast=False)
        print(join(target_path, name, img_name.split('/')[-1].split('.')[0]+'.png'))


def process_labels(gt_names, target_path):
    
    for gt_name in gt_names:

        gt_data = tif.imread(gt_name)
        # conver instance bask to three-class mask: interior, boundary
        interior_map = create_interior_map(gt_data.astype(np.int16))
        io.imsave(join(target_path, 'labels', gt_name.split('.')[0]+'.png'), interior_map.astype(np.uint16), check_contrast=False)
        print(join(target_path, 'labels', gt_name.split('.')[0]+'.png'))


def main():
    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='/disk1/neurips/dataset/train_labeled', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument('-iu', '--input_path_un', default='/disk1/neurips/dataset/train_unlabeled', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='/disk1/neurips/dataset/train_pre_3class', type=str, help='preprocessing data path')    
    parser.add_argument("-cl", '--images_tot', default=True, type=bool, help='choose images or images_tot')
    args = parser.parse_args()
    
    source_path = args.input_path
    source_path_un = args.input_path_un
    target_path = args.output_path
    image_bool = args.images_tot
    
    gt_path =  join(source_path, 'labels')

    if image_bool: # read images names
        img_path = join(source_path, 'images_tot')
        img_names, gt_names = read_names(img_path, gt_path)
    else: # read images names in each class folder
        img_path = join(source_path, 'images')
        cells = [dir for dir in os.listdir(img_path) if dir != '.DS_Store']
        img_names, gt_names = read_names(img_path, gt_path, cells)
    
    img_names_un = sorted([join(source_path_un, im) for im in os.listdir(source_path_un) if im.split('.')[-1] != 'DS_Store'])
    

    pre_img_path = join(target_path, 'images')
    pre_gt_path = join(target_path, 'labels')
    pre_un_path = join(target_path, 'unlabeled')
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)
    os.makedirs(pre_un_path, exist_ok=True)
    
    #process_image(img_names, target_path, image_bool, 'images')
    gt_names = ['/disk1/neurips/dataset/train_labeled/labels/cell_00686_label.tiff']
    process_labels(gt_names, target_path)
    #process_image(img_names_un, target_path, True, 'unlabeled')

    
if __name__ == "__main__":
    main()