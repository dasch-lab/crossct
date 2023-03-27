import numpy as np
import cv2
from itertools import product
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt, grey_closing
from skimage import measure
from skimage.morphology import disk
from segmentation.utils.utils import get_nucleus_ids


def binary_label(label):
    """ Binary label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :return: Binary label image.
    """
    return label > 0


def boundary_label_2d(label, algorithm='dilation'):
    """ Boundary label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param algorithm: canny or dilation-based boundary creation.
        :type algorithm: str
    :return: Boundary label image.
    """

    label_bin = binary_label(label)

    if algorithm == 'canny':

        if len(get_nucleus_ids(label)) > 255:
            raise Exception('Canny method works only with uint8 images but more than 255 nuclei detected.')

        boundary = cv2.Canny(label.astype(np.uint8), 1, 1) > 0
        label_boundary = np.maximum(label_bin, 2 * boundary)

    elif algorithm == 'dilation':

        kernel = np.ones(shape=(3, 3), dtype=np.uint8)

        # Pre-allocation
        boundary = np.zeros(shape=label.shape, dtype=np.bool)

        nucleus_ids = get_nucleus_ids(label)

        for nucleus_id in nucleus_ids:
            nucleus = (label == nucleus_id)
            nucleus_boundary = ndimage.binary_dilation(nucleus, kernel) ^ nucleus
            boundary += nucleus_boundary

        label_boundary = np.maximum(label_bin, 2 * boundary)

    return label_boundary


def border_label_2d(label, algorithm='dilation'):
    """ Border label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param algorithm: canny or dilation-based boundary creation.
        :type algorithm: str
    :return: Border label image.
    """

    label_bin = binary_label(label)

    if algorithm == 'canny':

        if len(get_nucleus_ids(label)) > 255:
            raise Exception('Canny method works only with uint8 images but more than 255 nuclei detected.')

        boundary = cv2.Canny(label.astype(np.uint8), 1, 1) > 0

        border = cv2.Canny(label_bin.astype(np.uint8), 1, 1) > 0
        border = boundary ^ border
        label_border = np.maximum(label_bin, 2 * border)

    elif algorithm == 'dilation':

        kernel = np.ones(shape=(3, 3), dtype=np.uint8)

        # Pre-allocation
        boundary = np.zeros(shape=label.shape, dtype=np.bool)

        nucleus_ids = get_nucleus_ids(label)

        for nucleus_id in nucleus_ids:
            nucleus = (label == nucleus_id)
            nucleus_boundary = ndimage.binary_dilation(nucleus, kernel) ^ nucleus
            boundary += nucleus_boundary

        border = boundary ^ (ndimage.binary_dilation(label_bin, kernel) ^ label_bin)
        label_border = np.maximum(label_bin, 2 * border)

    return label_border


def adapted_border_label_2d(label):
    """ Adapted border label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :return: Adapted border label image.
    """

    if len(get_nucleus_ids(label)) > 255:
        raise Exception('Canny method works only with uint8 images but more than 255 nuclei detected.')

    kernel = np.ones(shape=(3, 3), dtype=np.uint8)

    label_bin = binary_label(label)

    boundary = cv2.Canny(label.astype(np.uint8), 1, 1) > 0

    border = cv2.Canny(label_bin.astype(np.uint8), 1, 1) > 0
    border = boundary ^ border

    border_adapted = ndimage.binary_dilation(border.astype(np.uint8), kernel)
    cell_adapted = ndimage.binary_erosion(label_bin.astype(np.uint8), kernel)

    border_adapted = ndimage.binary_closing(border_adapted, kernel)
    label_adapted_border = np.maximum(cell_adapted, 2 * border_adapted)

    return label_adapted_border


def chebyshev_dist_label_2d(label, normalize_dist, radius):
    """ Cell distance label creation (Chebyshev distance).

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param radius: Defines the area to look for neighbors (smaller radius in px decreases the computation time)
        :type radius: int
    :return: Cell distance label image.
    """
    # Relabel label to avoid some errors/bugs
    label_dist = np.zeros(shape=label.shape, dtype=np.float)

    props = measure.regionprops(label)

    # Find centroids, crop image, calculate distance transform
    for i in range(len(props)):

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - radius, 0)):int(min(centroid[0] + radius, label.shape[0])),
                       int(max(centroid[1] - radius, 0)):int(min(centroid[1] + radius, label.shape[1]))
                       ]
        nucleus_crop_dist = distance_transform_cdt(nucleus_crop, metric='chessboard')
        if np.max(nucleus_crop_dist) > 0 and normalize_dist:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
        label_dist[
        int(max(centroid[0] - radius, 0)):int(min(centroid[0] + radius, label.shape[0])),
        int(max(centroid[1] - radius, 0)):int(min(centroid[1] + radius, label.shape[1]))
        ] += nucleus_crop_dist

    return label_dist


def dist_label_2d(label, neighbor_radius=None, apply_grayscale_closing=True):
    """ Cell and neigbhor distance label creation (Euclidean distance).

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param neighbor_radius: Defines the area to look for neighbors (smaller radius in px decreases the computation time)
        :type neighbor_radius: int
    :param apply_grayscale_closing: close gaps in between neighbor labels.
        :type apply_grayscale_closing: bool
    :return: Cell distance label image, neighbor distance label image.
    """
    # Relabel label to avoid some errors/bugs
    label_dist = np.zeros(shape=label.shape, dtype=np.float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=np.float)

    props = measure.regionprops(label)
    if neighbor_radius is None:
        mean_diameter = []
        for i in range(len(props)):
            mean_diameter.append(props[i].equivalent_diameter)
        mean_diameter = np.mean(np.array(mean_diameter))
        neighbor_radius = 3 * mean_diameter

    # Find centroids, crop image, calculate distance transform
    for i in range(len(props)):

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == props[i].label)
        centroid, diameter = np.round(props[i].centroid), int(np.ceil(props[i].equivalent_diameter))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
                       int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1]))
                       ]
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)
        label_dist[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1]))
        ] += nucleus_crop_dist

        # Get crop containing neighboring nuclei
        nucleus_neighbor_crop = np.copy(label[
                                int(max(centroid[0] - neighbor_radius, 0)):int(
                                    min(centroid[0] + neighbor_radius, label.shape[0])),
                                int(max(centroid[1] - neighbor_radius, 0)):int(
                                    min(centroid[1] + neighbor_radius, label.shape[1]))
                                ])

        # Convert background to nucleus id
        nucleus_neighbor_crop_nucleus = nucleus_neighbor_crop == props[i].label
        nucleus_neighbor_crop[nucleus_neighbor_crop == 0] = props[i].label
        nucleus_neighbor_crop[nucleus_neighbor_crop != props[i].label] = 0
        nucleus_neighbor_crop = nucleus_neighbor_crop > 0
        nucleus_neighbor_crop_dist = distance_transform_edt(nucleus_neighbor_crop)
        nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist * nucleus_neighbor_crop_nucleus
        if np.max(nucleus_neighbor_crop_dist) > 0:
            nucleus_neighbor_crop_dist = nucleus_neighbor_crop_dist / np.max(nucleus_neighbor_crop_dist)
        else:
            nucleus_neighbor_crop_dist = 1
        nucleus_neighbor_crop_dist = (1 - nucleus_neighbor_crop_dist) * nucleus_neighbor_crop_nucleus
        label_dist_neighbor[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1]))
        ] += nucleus_neighbor_crop_dist

    if apply_grayscale_closing:
        label_dist_neighbor = grey_closing(label_dist_neighbor, size=(5, 5))
    label_dist_neighbor = label_dist_neighbor ** 3
    if (label_dist_neighbor.max() - label_dist_neighbor.min()) > 0.5:
        label_dist_neighbor = (label_dist_neighbor - label_dist_neighbor.min()) / (label_dist_neighbor.max() - label_dist_neighbor.min())
    else:
        label_dist_neighbor = np.zeros(shape=label.shape, dtype=np.float)

    return label_dist, label_dist_neighbor


def pena_label_2d(label, k_neighbors=2, se_radius=4):
    """ Pena label creation for the J4 method (background, cell, touching, gap).

    Reference: Pena et al. "J regularization improves imbalanced mutliclass segmentation". In: 2020 IEEE 17th
        International Symposium on Biomedical Imaging (ISBI). 2020.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param k_neighbors: Neighborhood parameter needed for the creation of the touching class.
        :type k_neighbors: int
    :param se_radius: Structuring element (hypersphere) radius needed for the creation of the gap class.
    :return: Pena/J4 label image.
    """

    # Bottom hat transformation:
    label_bin = label > 0
    se = disk(se_radius)
    label_bottom_hat = ndimage.binary_closing(label_bin, se) ^ label_bin

    neighbor_mask = compute_neighbor_instances(label, k_neighbors)

    label_bg = (~label_bin) & (~label_bottom_hat)
    label_gap = (~label_bin) & label_bottom_hat
    label_touching = label_bin & (neighbor_mask > 1)
    label_cell = ~(label_bg | label_gap | label_touching)

    # 0: background, 1: cell, 2: touching, 3: gap
    label_pena = np.maximum(label_bg, 2 * label_cell)
    label_pena = np.maximum(label_pena, 3 * label_touching)
    label_pena = np.maximum(label_pena, 4 * label_gap)
    label_pena -= 1

    return label_pena


def compute_neighbor_instances(instance_mask, k_neighbors):
    """ Function to find instances in the neighborhood. """
    indices = [list(range(s)) for s in instance_mask.shape]

    mask_shape = instance_mask.shape
    padded_mask = np.pad(instance_mask, pad_width=k_neighbors, constant_values=0)
    n_neighbors = np.zeros_like(instance_mask)

    crop_2d = lambda x, y: (slice(x[0], y[0]), slice(x[1], y[1]))
    crop_3d = lambda x, y: (slice(x[0], y[0]), slice(x[1], y[1]), slice(x[2], y[2]))
    if len(mask_shape) == 2:
        crop_func = crop_2d
    elif len(mask_shape) == 3:
        crop_func = crop_3d
    else:
        raise AssertionError(f'instance mask shape is not 2 or 3 dimensional {instance_mask.shape}')

    for index in product(*indices):
        top_left = np.array(index) - k_neighbors + k_neighbors  # due to shift from padding
        bottom_right = np.array(index) + 2 * k_neighbors + 1
        crop_box = crop_func(top_left, bottom_right)
        crop = padded_mask[crop_box]
        n_neighbors[index] = len(set(crop[crop > 0]))

    return n_neighbors
