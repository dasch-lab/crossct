import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion
from skimage.segmentation import watershed
from skimage import measure


def dual_unet_postprocessing(prediction, input_3d=False):
    """ Post-processing for Dual U-Net predictions.

    :param prediction: cell prediction.
        :type prediction:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
    :return: Instance segmentation mask.
    """

    # Simply threshold the main output/prediction
    if input_3d:
        mask = prediction > 0.5
        seeds = binary_erosion(mask, np.ones(shape=(1, 3, 3), dtype=np.uint8))
    else:
        mask = prediction[:, :, 0] > 0.5
        seeds = binary_erosion(mask, np.ones(shape=(3, 3), dtype=np.uint8))
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 2:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    if not input_3d:
        prediction_instance = np.expand_dims(prediction_instance, axis=-1)

    return prediction_instance.astype(np.uint16)


def boundary_postprocessing(prediction, input_3d=False):
    """ Post-processing for boundary label prediction.

    :param prediction: Boundary label prediction.
        :type prediction:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
    :return: Instance segmentation mask, binary raw prediction (0: background, 1: cell, 2: boundary).
    """

    # Binarize the channels
    prediction_bin = np.argmax(prediction, axis=-1).astype(np.uint16)

    # Get mask to flood with watershed
    mask = (prediction_bin == 1)  # only interior cell class belongs to cells

    # Get seeds for marker-based watershed
    if input_3d:
        seeds = (prediction[:, :, :, 1] * (1 - prediction[:, :, :, 2])) > 0.5
    else:
        seeds = (prediction[:, :, 1] * (1 - prediction[:, :, 2])) > 0.5
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 2:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    if not input_3d:
        prediction_instance = np.expand_dims(prediction_instance, axis=-1)
        prediction_bin = np.expand_dims(prediction_bin, axis=-1)

    return prediction_instance.astype(np.uint16), prediction_bin.astype(np.uint8)


def border_postprocessing(prediction, input_3d=False):
    """ Post-processing for border label prediction.

    :param prediction: Border label prediction.
        :type prediction:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
    :return: Instance segmentation mask, binary raw prediction (0: background, 1: cell, 2: border).
    """

    # Binarize the channels
    prediction_bin = np.argmax(prediction, axis=-1).astype(np.uint16)

    # Get mask to flood with watershed
    mask = (prediction_bin > 0)  # border class belongs to cells

    # Get seeds for marker-based watershed
    if input_3d:
        seeds = (prediction[:, :, :, 1] * (1 - prediction[:, :, :, 2])) > 0.5
    else:
        seeds = (prediction[:, :, 1] * (1 - prediction[:, :, 2])) > 0.5
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 2:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    if not input_3d:
        prediction_instance = np.expand_dims(prediction_instance, axis=-1)
        prediction_bin = np.expand_dims(prediction_bin, axis=-1)

    return prediction_instance.astype(np.uint16), prediction_bin.astype(np.uint8)


def pena_postprocessing(prediction, input_3d=False):
    """ Post-processing for pena label prediction (background, cell, touching, gap).

    :param prediction: pena label prediction.
        :type prediction:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
    :return: Instance segmentation mask, binary raw prediction (0: background, 1: cell, 2: border, 3: gap).
    """

    # Binarize the channels
    prediction_bin = np.argmax(prediction, axis=-1).astype(np.uint16)

    # Get mask to flood with watershed
    mask = (prediction_bin == 1) | (prediction_bin == 2)  # gap belongs to background

    # Get seeds for marker-based watershed
    if input_3d:
        seeds = (prediction[:, :, :, 1] * (1 - prediction[:, :, :, 2]) * (1 - prediction[:, :, :, 3])) > 0.5
    else:
        # seeds = prediction_bin == 1  ## results in merged objects
        seeds = (prediction[:, :, 1] * (1 - prediction[:, :, 2]) * (1 - prediction[:, :, 3])) > 0.5
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 2:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    if not input_3d:
        prediction_instance = np.expand_dims(prediction_instance, axis=-1)
        prediction_bin = np.expand_dims(prediction_bin, axis=-1)

    return prediction_instance.astype(np.uint16), prediction_bin.astype(np.uint8)


def adapted_border_postprocessing(border_prediction, cell_prediction, input_3d=False):
    """ Post-processing for adapted border label prediction.

    :param border_prediction: Adapted border prediction (3 channels).
        :type border_prediction:
    :param cell_prediction: Cell prediction (1 channel).
        :type cell_prediction:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
    :return: Instance segmentation mask, binary border prediction (0: background, 1: cell, 2: border).
    """

    # Binarize the channels
    prediction_border_bin = np.argmax(border_prediction, axis=-1).astype(np.uint16)

    # Get mask to flood with watershed
    mask = cell_prediction > 0.5

    # Get seeds for marker-based watershed
    if input_3d:
        seeds = (border_prediction[:, :, :, 1] * (1 - border_prediction[:, :, :, 2])) > 0.5
    else:
        seeds = border_prediction[:, :, 1] * (1 - border_prediction[:, :, 2]) > 0.5
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 2:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    if not input_3d:
        prediction_instance = np.expand_dims(prediction_instance, axis=-1)
        prediction_border_bin = np.expand_dims(prediction_border_bin, axis=-1)

    return prediction_instance.astype(np.uint16), prediction_border_bin.astype(np.uint8)


def distance_postprocessing(border_prediction, cell_prediction, input_3D=False):
    """ Post-processing for distance label (cell + neighbor) prediction.

    :param border_prediction: Neighbor distance prediction.
        :type border_prediction:
    :param cell_prediction: Cell distance prediction.
        :type cell_prediction:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
    :return: Instance segmentation mask.
    """

    # Smooth predictions slightly + clip border prediction (to avoid negative values being positive after squaring)
    if input_3D:
        sigma_cell = (0.5, 1.5, 1.5)
        sigma_border = (0.5, 1.5, 1.5)
        cell_prediction = gaussian_filter(cell_prediction, sigma=sigma_cell)
        border_prediction = np.clip(border_prediction, 0, 1)
        border_prediction = gaussian_filter(border_prediction, sigma=sigma_border)
    else:
        sigma = 1.5
        cell_prediction = gaussian_filter(cell_prediction, sigma=sigma)
        border_prediction = np.clip(border_prediction, 0, 1)
        border_prediction = gaussian_filter(border_prediction, sigma=sigma)

    # Thresholds
    th_cell = 0.09
    th_seed = 0.5

    # Get mask for watershed
    mask = cell_prediction > th_cell

    # Get seeds for marker-based watershed
    borders = border_prediction ** 2
    seeds = (cell_prediction - borders) > th_seed
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 2:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)

    return prediction_instance.astype(np.uint16)
