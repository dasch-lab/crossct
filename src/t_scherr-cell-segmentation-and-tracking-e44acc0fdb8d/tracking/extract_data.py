"""Utilities to extract segmentation masks and compute movement statistics"""
import os
import re

import numpy as np
import pandas as pd


def get_mask_positions(data, background_id=0):
    """
    Extracts for each mask id its positions within the array.
    Args:
        data: a np.array containing masks (regions with the same (integer) value)
        background_id: the value of the background to ignore

    Returns:a pd.DataFrame containing for each mask id its positions within data as tuples

    """
    if data.size < 1e9:  # aim for speed at cost of high memory consumption
        masked_data = (data != background_id)
        flat_data = data[masked_data]  # d: data , mask attribute
        dummy_index = np.where(masked_data.ravel())[0]
        df = pd.DataFrame.from_dict({'mask_id': flat_data, 'flat_index': dummy_index})
        df = df.groupby('mask_id').apply(lambda x: np.unravel_index(x.flat_index, data.shape))
    else:  # aim for lower memory consumption at cost of speed
        flat_data = data[(data != background_id)]  # d: data , mask attribute
        dummy_index = np.where((data != background_id).ravel())[0]
        mask_indices = np.unique(flat_data)
        df = {'mask_id': [], 'index': []}
        data_shape = data.shape
        for mask_id in mask_indices:
            df['index'].append(np.unravel_index(dummy_index[flat_data == mask_id], data_shape))
            df['mask_id'].append(mask_id)
        df = pd.DataFrame.from_dict(df)
        df = df.set_index('mask_id')
        df = df['index'].apply(lambda x: x)  # convert to same format as for other case
    return df


def get_img_files(img_dir):
    """Extracts all tif files from a directory."""
    img_file_pattern = re.compile(r'(\D*)(\d+)(\D*)\.(' + '|'.join(('tif', 'tiff')) + ')')
    files = {int(img_file_pattern.match(file).groups()[1]): (img_dir / file).as_posix()
             for file in os.listdir(img_dir)
             if file.endswith(('tif', 'tiff'))}
    return files
