import json
import math
import tifffile as tiff
from pathlib import Path
from skimage.morphology import binary_closing
from segmentation.training.train_data_representations import *
from segmentation.utils.utils import get_nucleus_ids


def convert_ctc_data_set(path, radius, mode='GT', frames='all', k_pena=2):
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

    # Load train-val split
    with open(Path.cwd() / 'segmentation' / 'training' / 'ctc_training_set_train_val_ids.json') as f:
        train_val_ids = json.load(f)

    # Get ids of the label images
    label_ids = sorted((path / ('01_' + mode) / 'SEG').glob('*.tif'))

    # Create label images
    for i, label_id in enumerate(label_ids):

        if isinstance(frames, list):
            if '3D' in path.stem:
                if label_id.stem.split('man_seg_')[-1] not in frames:
                    continue
            else:
                if label_id.stem.split('man_seg')[-1] not in frames:
                    continue

        file_id = label_id.name.split('man_seg')[-1]
        label = tiff.imread(str(label_id))

        if '3D' in path.stem:
            file_id = label_id.name.split('man_seg_')[-1]
            frame = label_id.stem.split('man_seg_')[-1].split('_')[0]
            slice = label_id.stem.split('man_seg_')[-1].split('_')[-1]
            img = tiff.imread(str(label_id.parents[2] / '01' / ('t' + frame + '.tif')))
            img = img[int(slice)]
            # Apply some closing
            nucleus_ids = get_nucleus_ids(label)
            hlabel = np.zeros(shape=label.shape, dtype=label.dtype)
            for nucleus_id in nucleus_ids:
                hlabel += nucleus_id * binary_closing(label == nucleus_id, np.ones((5, 5))).astype(label.dtype)
            label = hlabel
        else:
            img = tiff.imread(str(label_id.parents[2] / '01' / ('t' + file_id)))

        label_bin = binary_label(label=label)
        label_boundary = boundary_label_2d(label=label, algorithm='dilation')
        label_border = border_label_2d(label=label, algorithm='dilation')
        label_adapted_border = adapted_border_label_2d(label=label)
        label_dist_chebyshev = chebyshev_dist_label_2d(label=label, normalize_dist=True, radius=radius)
        label_dist, label_dist_neighbor = dist_label_2d(label=label, neighbor_radius=radius)
        label_pena = pena_label_2d(label=label, k_neighbors=k_pena)

        # Add pseudo color channel and min-max normalize to 0 - 65535
        img = np.expand_dims(img, axis=-1)
        img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
        img = np.clip(img, 0, 65535).astype(np.uint16)
        label = np.expand_dims(label, axis=-1).astype(np.uint16)
        label_bin = 255 * np.expand_dims(label_bin.astype(np.uint8), axis=-1)
        label_boundary = np.expand_dims(label_boundary.astype(np.uint8), axis=-1)
        label_border = np.expand_dims(label_border.astype(np.uint8), axis=-1)
        label_adapted_border = np.expand_dims(label_adapted_border.astype(np.uint8), axis=-1)
        label_dist_chebyshev = np.expand_dims(label_dist_chebyshev, axis=-1).astype(np.float32)
        label_dist = np.expand_dims(label_dist, axis=-1).astype(np.float32)
        label_dist_neighbor = np.expand_dims(label_dist_neighbor, axis=-1).astype(np.float32)
        label_pena = np.expand_dims(label_pena.astype(np.uint8), axis=-1)

        # Save
        ctc_set_path = path.parent / 'ctc_training_set'
        file_name = path.stem + '_' + file_id
        tiff.imsave(str(ctc_set_path / 'all' / ('img_' + file_name)), img)
        tiff.imsave(str(ctc_set_path / 'all' / ('mask_' + file_name)), label)
        tiff.imsave(str(ctc_set_path / 'all' / ('bin_' + file_name)), label_bin)
        tiff.imsave(str(ctc_set_path / 'all' / ('boundary_' + file_name)), label_boundary)
        tiff.imsave(str(ctc_set_path / 'all' / ('border_' + file_name)), label_border)
        tiff.imsave(str(ctc_set_path / 'all' / ('adapted_border_' + file_name)), label_adapted_border)
        tiff.imsave(str(ctc_set_path / 'all' / ('dist_chebyshev_' + file_name)), label_dist_chebyshev)
        tiff.imsave(str(ctc_set_path / 'all' / ('dist_cell_' + file_name)), label_dist)
        tiff.imsave(str(ctc_set_path / 'all' / ('dist_neighbor_' + file_name)), label_dist_neighbor)
        tiff.imsave(str(ctc_set_path / 'all' / ('pena_' + file_name)), label_pena)

        # Crop images and check cell type
        cell_type = file_name.split('_{}'.format(file_id))[0]

        if cell_type == 'BF-C2DL-HSC':
            # Zero pad to 1024x1024 for easier cropping:
            img = np.pad(img, ((7, 7), (7, 7), (0, 0)), mode='constant')
            mask = np.pad(label, ((7, 7), (7, 7), (0, 0)), mode='constant')
            label_bin = np.pad(label_bin, ((7, 7), (7, 7), (0, 0)), mode='constant')
            label_boundary = np.pad(label_boundary, ((7, 7), (7, 7), (0, 0)), mode='constant')
            label_border = np.pad(label_border, ((7, 7), (7, 7), (0, 0)), mode='constant')
            label_adapted_border = np.pad(label_adapted_border, ((7, 7), (7, 7), (0, 0)), mode='constant')
            label_dist_chebyshev = np.pad(label_dist_chebyshev, ((7, 7), (7, 7), (0, 0)), mode='constant')
            label_dist = np.pad(label_dist, ((7, 7), (7, 7), (0, 0)), mode='constant')
            label_dist_neighbor = np.pad(label_dist_neighbor, ((7, 7), (7, 7), (0, 0)), mode='constant')
            label_pena = np.pad(label_pena, ((7, 7), (7, 7), (0, 0)), mode='constant')
        elif cell_type == 'BF-C2DL-MuSC':
            # Crop to 1024x1024 since the cells are only in the image center:
            img = img[:1024, :1024, :]
            mask = label[:1024, :1024, :]
            label_bin = label_bin[:1024, :1024, :]
            label_boundary = label_boundary[:1024, :1024, :]
            label_border = label_border[:1024, :1024, :]
            label_adapted_border = label_adapted_border[:1024, :1024, :]
            label_dist_chebyshev = label_dist_chebyshev[:1024, :1024, :]
            label_dist = label_dist[:1024, :1024, :]
            label_dist_neighbor = label_dist_neighbor[:1024, :1024, :]
            label_pena = label_pena[:1024, :1024, :]
        elif cell_type == 'Fluo-N2DL-HeLa':
            # Crop to 512x1024 (the border cells may not be annotated depending on the field of interest)
            img = img[94:606, 38:1062, :]
            mask = label[94:606, 38:1062, :]
            label_bin = label_bin[94:606, 38:1062, :]
            label_boundary = label_boundary[94:606, 38:1062, :]
            label_border = label_border[94:606, 38:1062, :]
            label_adapted_border = label_adapted_border[94:606, 38:1062, :]
            label_dist_chebyshev = label_dist_chebyshev[94:606, 38:1062, :]
            label_dist = label_dist[94:606, 38:1062, :]
            label_dist_neighbor = label_dist_neighbor[94:606, 38:1062, :]
            label_pena = label_pena[94:606, 38:1062, :]
        elif cell_type == 'Fluo-N3DH-CE':
            img = np.pad(img, ((0, 0), (30, 30), (0, 0)), mode='constant')
            mask = np.pad(label, ((0, 0), (30, 30), (0, 0)), mode='constant')
            label_bin = np.pad(label_bin, ((0, 0), (30, 30), (0, 0)), mode='constant')
            label_boundary = np.pad(label_boundary, ((0, 0), (30, 30), (0, 0)), mode='constant')
            label_border = np.pad(label_border, ((0, 0), (30, 30), (0, 0)), mode='constant')
            label_adapted_border = np.pad(label_adapted_border, ((0, 0), (30, 30), (0, 0)), mode='constant')
            label_dist_chebyshev = np.pad(label_dist_chebyshev, ((0, 0), (30, 30), (0, 0)), mode='constant')
            label_dist = np.pad(label_dist, ((0, 0), (30, 30), (0, 0)), mode='constant')
            label_dist_neighbor = np.pad(label_dist_neighbor, ((0, 0), (30, 30), (0, 0)), mode='constant')
            label_pena = np.pad(label_pena, ((0, 0), (30, 30), (0, 0)), mode='constant')
        else:
            raise Exception('Cell type {} not known'.format(cell_type))

        # Use 256x256 crops
        nx = math.floor(img.shape[1] / 256)
        ny = math.floor(img.shape[0] / 256)
        for y in range(ny):
            for x in range(nx):
                img_crop = img[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]
                mask_crop = mask[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]
                label_bin_crop = label_bin[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]
                label_boundary_crop = label_boundary[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]
                label_border_crop = label_border[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]
                label_adapted_border_crop = label_adapted_border[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]
                label_dist_chebyshev_crop = label_dist_chebyshev[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]
                label_dist_crop = label_dist[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]
                label_dist_neighbor_crop = label_dist_neighbor[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]
                label_pena_crop = label_pena[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256, :]

                # Save
                crop_name = file_name.split('.tif')[0] + '_{:03d}_{:03d}.tif'.format(y, x)
                for mode in ['train', 'val']:

                    if crop_name.split('.tif')[0] in train_val_ids[mode]:
                        tiff.imsave(str(ctc_set_path / mode / ('img_' + crop_name)), img_crop)
                        tiff.imsave(str(ctc_set_path / mode / ('mask_' + crop_name)), mask_crop)
                        tiff.imsave(str(ctc_set_path / mode / ('bin_' + crop_name)), label_bin_crop)
                        tiff.imsave(str(ctc_set_path / mode / ('boundary_' + crop_name)), label_boundary_crop)
                        tiff.imsave(str(ctc_set_path / mode / ('border_' + crop_name)), label_border_crop)
                        tiff.imsave(str(ctc_set_path / mode / ('adapted_border_' + crop_name)), label_adapted_border_crop)
                        tiff.imsave(str(ctc_set_path / mode / ('dist_chebyshev_' + crop_name)), label_dist_chebyshev_crop)
                        tiff.imsave(str(ctc_set_path / mode / ('dist_cell_' + crop_name)), label_dist_crop)
                        tiff.imsave(str(ctc_set_path / mode / ('dist_neighbor_' + crop_name)), label_dist_neighbor_crop)
                        tiff.imsave(str(ctc_set_path / mode / ('pena_' + crop_name)), label_pena_crop)

    return None


def create_ctc_training_set(path):
    """ Create the CTC Training Set.

    :param path: Path to the directory containing the Cell Tracking Challenge data.
        :type path: Path
    :return: None
    """

    # Create directories
    Path.mkdir(path / 'ctc_training_set', exist_ok=True)
    Path.mkdir(path / 'ctc_training_set' / 'all', exist_ok=True)
    Path.mkdir(path / 'ctc_training_set' / 'train', exist_ok=True)
    Path.mkdir(path / 'ctc_training_set' / 'val', exist_ok=True)

    # Get training data representations, e.g., boundary or distance label, crop and split into train and val set
    frames_bf_c2dl_hsc_gt = ['0701', '0713', '0760', '0806', '0838', '0868', '1122', '1162', '1170', '1199', '1219',
                             '1357', '1369', '1397', '1460', '1468', '1471', '1476', '1489', '1490', '1528', '1558',
                             '1567', '1583', '1618', '1707', '1743']
    frames_bf_c2dl_musc_gt = ['1066', '1103', '1136', '1137', '1140', '1186', '1196', '1200', '1203', '1221', '1246',
                              '1254', '1267', '1301', '1322']
    frames_fluo_n2dl_hela_gt = ['13', '52']
    frames_fluo_n2dl_hela_st = []
    frames_fluo_n3dh_ce_gt = ['028_018', '078_017', '162_010']
    for i in range(0, 92, 6):
        frames_fluo_n2dl_hela_st.append('{:03d}'.format(i))
    print('   ... BF-C2DL-HSC ...')
    convert_ctc_data_set(path=path / 'BF-C2DL-HSC', radius=25, mode='GT', frames=frames_bf_c2dl_hsc_gt)
    print('   ... BF-C2DL-MuSC ...')
    convert_ctc_data_set(path=path / 'BF-C2DL-MuSC', radius=100, mode='GT', frames=frames_bf_c2dl_musc_gt)
    print('   ... Fluo-N2DL-HeLa ...')
    convert_ctc_data_set(path=path / 'Fluo-N2DL-HeLa', radius=50, mode='GT', frames=frames_fluo_n2dl_hela_gt, k_pena=3)
    convert_ctc_data_set(path=path / 'Fluo-N2DL-HeLa', radius=50, mode='ST', frames=frames_fluo_n2dl_hela_st, k_pena=3)
    print('   ... Fluo-N3DH-CE ...')
    convert_ctc_data_set(path=path / 'Fluo-N3DH-CE', radius=100, mode='GT', frames=frames_fluo_n3dh_ce_gt, k_pena=3)

    # Print border information
    methods = ['boundary', 'border', 'adapted_border', 'dist_neighbor', 'pena_touching', 'pena_gap']
    print('Border information fraction [1e-3]')
    for method in methods:
        border_pixels, all_pixels = 0, 0
        for mode in ['train', 'val']:
            if method in ['pena_touching', 'pena_gap']:
                label_ids = (path / 'ctc_training_set' / mode).glob(method.split('_')[0] + '*')
            else:
                label_ids = (path / 'ctc_training_set' / mode).glob(method + '*')
            for label_id in label_ids:
                label = tiff.imread(str(label_id))
                if method == 'dist_neighbor':
                    label = label > 0.5
                elif method == 'pena_touching':
                    label = label == 2
                elif method == 'pena_gap':
                    label = label == 3
                else:
                    label = label == 2
                border_pixels += np.sum(label)
                all_pixels += label.shape[0] * label.shape[1]
        print("   {}: {:.2f}".format(method, 1000 * border_pixels / all_pixels))

    return None
