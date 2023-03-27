import numpy as np
import tifffile as tiff
from pathlib import Path
from torch.utils.data import Dataset


class CellSegDataset(Dataset):
    """ Pytorch data set for instance cell nuclei segmentation """

    def __init__(self, root_dir, label_type, mode='train', transform=lambda x: x):

        self.img_ids = sorted(Path.joinpath(root_dir, mode).glob('img*.tif'))
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.label_type = label_type

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img = tiff.imread(str(img_id))

        if self.label_type == 'adapted_border':

            border_seed_label_id = img_id.parent / (self.label_type + img_id.name.split('img')[-1])
            cell_label_id = img_id.parent / ('bin' + img_id.name.split('img')[-1])

            border_seed_label = tiff.imread(str(border_seed_label_id)).astype(np.uint8)
            cell_label = (tiff.imread(str(cell_label_id)) > 0).astype(np.uint8)
            sample = {'image': img,
                      'border_label': border_seed_label,
                      'cell_label': cell_label,
                      'id': img_id.stem}

        elif self.label_type == 'dual_unet':

            dist_label_id = img_id.parent / ('dist_chebyshev' + img_id.name.split('img')[-1])
            dist_label = tiff.imread(str(dist_label_id)).astype(np.float32)

            boundary_id = img_id.parent / ('boundary' + img_id.name.split('img')[-1])
            boundary_label = (tiff.imread(str(boundary_id)) == 2).astype(np.uint8)

            cell_label = (tiff.imread(str(boundary_id)) == 1).astype(np.uint8)

            sample = {'image': img,
                      'cell_label': cell_label,
                      'cell_dist_label': dist_label,
                      'border_label': boundary_label,
                      'id': img_id.stem}

        elif self.label_type == 'distance':

            dist_label_id = img_id.parent / ('dist_cell' + img_id.name.split('img')[-1])
            dist_neighbor_label_id = img_id.parent / ('dist_neighbor' + img_id.name.split('img')[-1])

            dist_label = tiff.imread(str(dist_label_id)).astype(np.float32)
            dist_neighbor_label = tiff.imread(str(dist_neighbor_label_id)).astype(np.float32)

            sample = {'image': img,
                      'cell_label': dist_label,
                      'border_label': dist_neighbor_label,
                      'id': img_id.stem}

        elif self.label_type in ['boundary', 'border', 'pena']:

            label_id = img_id.parent / (self.label_type + img_id.name.split('img')[-1])
            label = tiff.imread(str(label_id)).astype(np.uint8)
            sample = {'image': img, 'label': label, 'id': img_id.stem}

        else:
            raise Exception('Unknown label type')

        sample = self.transform(sample)

        return sample
