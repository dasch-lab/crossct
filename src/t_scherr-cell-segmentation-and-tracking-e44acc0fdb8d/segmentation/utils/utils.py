import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff


def get_nucleus_ids(img):
    """ Get nucleus ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values


def hela_foi_correction(mask):
    """ Field of interest correction for CTC data set Fluo-N2DL-HeLa.

    :param mask: Instance segmentation mask
        :type mask:
    :return: foi corrected mask
    """

    foi = mask[25:mask.shape[0] - 25, 25:mask.shape[1] - 25, 0]
    ids_foi = get_nucleus_ids(foi)
    ids_prediction = get_nucleus_ids(mask)
    for id_prediction in ids_prediction:
        if id_prediction not in ids_foi:
            mask[mask == id_prediction] = 0
    return mask


def min_max_normalization(img, min_value=None, max_value=None):
    """ Minimum maximum normalization.

    :param img:
    :param min_value:
    :param max_value:
    :return:
    """

    if max_value is None:
        max_value = img.max()

    if min_value is None:
        min_value = img.min()

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply min-max-normalization
    img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1

    return img.astype(np.float32)


def unique_path(directory, name_pattern):
    """ Get unique file name to save trained model.

    :param directory: Path to the model directory
        :type directory: pathlib path object.
    :param name_pattern: Pattern for the file name
        :type name_pattern: str
    :return: pathlib path
    """
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def write_train_info(configs, path):
    """ Write training configurations into a json file.

    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param path: path to the directory to store the json file.
        :type path: pathlib Path object
    :return: None
    """

    with open(path / (configs['run_name'] + '.json'), 'w', encoding='utf-8') as outfile:
        json.dump(configs, outfile, ensure_ascii=False, indent=2)

    return None


def write_eval_info(results, path):
    """ Write evaluation results into a json file.

    :param results: Dictionary with evaluation results.
        :type results: dict
    :param path: path to the directory to store the json file.
        :type path: pathlib Path object
    :return: None
    """

    with open(path / 'metrics.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)

    return None


def get_models(results, path, methods, cell_types, mode='best'):
    """ Get median model for each label type and data set.

        :param results: Dictionary with evaluation results.
            :type results: dict
        :param path: Path to the directory to store the metrics json file.
            :type path: pathlib Path object
        :param methods: List of methods.
            :type methods: list
        :param cell_types: List of cell types.
            :type cell_types: list
        :param mode: 'best' or 'median' models.
            :type mode: str
        :return: None
        """

    model_dict = {}

    for label_type in methods:

        model_dict[label_type] = {}

        for cell_type in cell_types:

            op_csb_scores, model_list = [], []

            for model in results[label_type][cell_type]:
                op_csb_scores.append(results[label_type][cell_type][model]['OP_CSB'])
                model_list.append(model)

            op_csb_scores = np.array(op_csb_scores)
            if mode == 'median':
                pos = np.argwhere(op_csb_scores == np.median(op_csb_scores))
            elif mode == 'best':
                pos = np.argwhere(op_csb_scores == np.max(op_csb_scores))

            model_dict[label_type][cell_type] = model_list[pos[0, 0]]

    with open(path / '{}_models.json'.format(mode), 'w', encoding='utf-8') as outfile:
        json.dump(model_dict, outfile, ensure_ascii=False, indent=2)

    return None


def get_paper_crops(path_results, path_datasets, cell_types, methods, mode='best'):
    """ Function to produce the crops of the best and median models shown in our publication.

    :param path_results: Path to the results directory with the metrics json file.
        :type path_results: pathlib Path object
    :param path_datasets: Path to the data sets.
        :type path_datasets: pathlib Path object
    :param cell_types: List of cell types.
        :type cell_types: list
    :param methods: List of methods.
        :type methods: list
    :param mode: 'best' or 'median' models.
        :type mode: str
    :return: None
    """

    files = {'BF-C2DL-HSC': ['1748', '1748'],
             'BF-C2DL-MuSC': ['1278', '1278'],
             'Fluo-N2DL-HeLa': ['079', '079'],
             'Fluo-N3DH-CE': ['106', '_106_020', 20]}

    crops = {'BF-C2DL-HSC': [580, 85, 140],
             'BF-C2DL-MuSC': [150, 375, 360],
             'Fluo-N2DL-HeLa': [243, 196, 140],
             'Fluo-N3DH-CE': [320, 390, 140]}

    # Make directory for results
    (path_results / '{}_results'.format(mode)).mkdir(exist_ok=True)

    # Get median/best models
    with open(path_results / '{}_models.json'.format(mode)) as infile:
        median_models = json.load(infile)

    for method in methods:

        for cell_type in cell_types:

            median_model = median_models[method][cell_type]

            result_ids = sorted((path_results / median_model / cell_type).glob('*{}*'.format(files[cell_type][0])))

            for result_id in result_ids:
                img = tiff.imread(str(result_id))

                # Crop image to wanted size
                if len(img.shape) == 3:
                    img = img[files[cell_type][2]]
                h_start, h = crops[cell_type][0], crops[cell_type][2]
                w_start, w = crops[cell_type][1], crops[cell_type][2]
                img = img[h_start:h_start + h, w_start:w_start + w]

                tiff.imsave(path_results / '{}_results'.format(mode) / '{}_{}_{}.tif'.format(cell_type,
                                                                                             median_model,
                                                                                             result_id.stem), img)
    # Ground truth crops
    for cell_type in cell_types:
        img_id = path_datasets / cell_type / "02" / ("t" + files[cell_type][0] + '.tif')
        gt_id = path_datasets / cell_type / "02_GT" / "SEG" / ("man_seg" + files[cell_type][1] + '.tif')
        img = tiff.imread(str(img_id))
        gt = tiff.imread(str(gt_id))
        if len(img.shape) == 3:
            img = img[files[cell_type][2]]
        h_start, h = crops[cell_type][0], crops[cell_type][2]
        w_start, w = crops[cell_type][1], crops[cell_type][2]
        img = img[h_start:h_start + h, w_start:w_start + w]
        gt = gt[h_start:h_start + h, w_start:w_start + w]
        tiff.imsave(path_results / '{}_results'.format(mode) / '{}_img_{}.tif'.format(cell_type, files[cell_type][1]), img)
        tiff.imsave(path_results / '{}_results'.format(mode) / '{}_gt_{}.tif'.format(cell_type, files[cell_type][1]), gt)

        return None


def plot_metrics(path, cell_types, methods):
    """ Create box plots of the results.

    :param path: Path to the results directory.
        :type path: pathlib Path object
    :param cell_types: List of cell types.
        :type cell_types: list
    :param methods: List of methods.
        :type methods: list
    :return: None
    """

    sns.set_style(style="darkgrid")

    with open(path / 'metrics.json') as infile:
        metric_scores = json.load(infile)

    methods = pd.Series(methods, dtype="category")

    for cell_type in cell_types:
        lt, scores, st = [], [], []
        for label_type in methods:
            for model in metric_scores[label_type][cell_type]:
                for metric in ['DET', 'SEG', 'OP_CSB']:
                    scores.append(metric_scores[label_type][cell_type][model][metric])
                    st.append(metric)
                    if label_type == 'boundary':
                        lt.append('Boundary')
                    elif label_type == 'border':
                        lt.append('Border')
                    elif label_type == 'adapted_border':
                        lt.append('Adapted Border')
                    elif label_type == 'dual_unet':
                        lt.append('Dual U-Net')
                    elif label_type == 'pena':
                        lt.append('J4')
                    elif label_type == 'distance':
                        lt.append('Proposed')

        metrics_df = pd.DataFrame({'Method': lt,
                                   'Score': scores,
                                   'Metric': st
                                   })
        metrics_df['Metric'] = metrics_df['Metric'].astype('category')

        plt.figure()
        ax = sns.boxplot(x='Method', y='Score', hue="Metric",
                         hue_order=['SEG', 'DET', 'OP_CSB'], data=metrics_df, palette="deep")  # pastel, dark, deep
        ax = sns.swarmplot(x='Method', y='Score', hue="Metric", hue_order=['SEG', 'DET', 'OP_CSB'],
                           dodge=True, data=metrics_df, color=".25", size=3)

        ax.set_xlabel('')
        [ax.axvline(x, color='w', linestyle='-') for x in np.arange(0.5, len(methods) - 1, 1)]
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[0:3], labels[0:3], loc='lower right')
        plt.xticks(ax.get_xticks(), rotation=45)
        plt.savefig(str(path / ('metrics_' + cell_type + '.pdf')), bbox_inches='tight', dpi=300)
        plt.close()

    return None


def zero_pad_model_input(img):
    """ Zero-pad model input to get for the model needed sizes.

    :param img: Model input image.
        :type:

    :return: zero-padded img, [0s padded in y-direction, 0s padded in x-direction]
    """

    if img.shape[0] < 64:
        # Zero-pad to 128
        y_pads = 64 - img.shape[0]

    elif img.shape[0] == 64:
        # No zero-padding needed
        y_pads = 0

    elif img.shape[0] < 128:
        # Zero-pad to 128
        y_pads = 128 - img.shape[0]

    elif img.shape[0] == 128:
        # No zero-padding needed
        y_pads = 0

    elif 128 < img.shape[0] < 256:
        # Zero-pad to 256
        y_pads = 256 - img.shape[0]

    elif img.shape[0] == 256:
        # No zero-padding needed
        y_pads = 0

    elif 256 < img.shape[0] < 512:
        # Zero-pad to 512
        y_pads = 512 - img.shape[0]

    elif img.shape[0] == 512:
        # No zero-padding needed
        y_pads = 0

    elif 512 < img.shape[0] < 768:
        # Zero-pad to 768
        y_pads = 768 - img.shape[0]

    elif img.shape[0] == 768:
        # No zero-padding needed
        y_pads = 0

    elif 768 < img.shape[0] < 1024:
        # Zero-pad to 1024
        y_pads = 1024 - img.shape[0]

    elif img.shape[0] == 1024:
        # No zero-padding needed
        y_pads = 0

    elif 1024 < img.shape[0] < 1280:
        # Zero-pad to 2048
        y_pads = 1280 - img.shape[0]

    elif img.shape[0] == 1280:
        # No zero-padding needed
        y_pads = 0

    elif 1280 < img.shape[0] < 1920:
        # Zero-pad to 1920
        y_pads = 1920 - img.shape[0]

    elif img.shape[0] == 1920:
        # No zero-padding needed
        y_pads = 0

    elif 1920 < img.shape[0] < 2048:
        # Zero-pad to 2048
        y_pads = 2048 - img.shape[0]

    elif img.shape[0] == 2048:
        # No zero-padding needed
        y_pads = 0

    elif 2048 < img.shape[0] < 2560:
        # Zero-pad to 2560
        y_pads = 2560 - img.shape[0]

    elif img.shape[0] == 2560:
        # No zero-padding needed
        y_pads = 0

    elif 2560 < img.shape[0] < 4096:
        # Zero-pad to 4096
        y_pads = 4096 - img.shape[0]

    elif img.shape[0] == 4096:
        # No zero-padding needed
        y_pads = 0

    elif 4096 < img.shape[0] < 8192:
        # Zero-pad to 8192
        y_pads = 8192 - img.shape[0]

    elif img.shape[0] == 8192:
        # No zero-padding needed
        y_pads = 0
    else:
        raise Exception('Padding error. Image too big?')

    if img.shape[1] < 64:
        # Zero-pad to 128
        x_pads = 64 - img.shape[1]

    elif img.shape[1] == 64:
        # No zero-padding needed
        x_pads = 0

    elif img.shape[1] < 128:
        # Zero-pad to 128
        x_pads = 128 - img.shape[1]

    elif img.shape[1] == 128:
        # No zero-padding needed
        x_pads = 0

    elif 128 < img.shape[1] < 256:
        # Zero-pad to 256
        x_pads = 256 - img.shape[1]

    elif img.shape[1] == 256:
        # No zero-padding needed
        x_pads = 0

    elif 256 < img.shape[1] < 512:
        # Zero-pad to 512
        x_pads = 512 - img.shape[1]

    elif img.shape[1] == 512:
        # No zero-padding needed
        x_pads = 0

    elif 512 < img.shape[1] < 768:
        # Zero-pad to 768
        x_pads = 768 - img.shape[1]

    elif img.shape[1] == 768:
        # No zero-padding needed
        x_pads = 0

    elif 768 < img.shape[1] < 1024:
        # Zero-pad to 1024
        x_pads = 1024 - img.shape[1]

    elif img.shape[1] == 1024:
        # No zero-padding needed
        x_pads = 0

    elif 1024 < img.shape[1] < 1280:
        # Zero-pad to 1024
        x_pads = 1280 - img.shape[1]

    elif img.shape[1] == 1280:
        # No zero-padding needed
        x_pads = 0

    elif 1280 < img.shape[1] < 1920:
        # Zero-pad to 1920
        x_pads = 1920 - img.shape[1]

    elif img.shape[1] == 1920:
        # No zero-padding needed
        x_pads = 0

    elif 1920 < img.shape[1] < 2048:
        # Zero-pad to 2048
        x_pads = 2048 - img.shape[1]

    elif img.shape[1] == 2048:
        # No zero-padding needed
        x_pads = 0

    elif 2048 < img.shape[1] < 2560:
        # Zero-pad to 2560
        x_pads = 2560 - img.shape[1]

    elif img.shape[1] == 2560:
        # No zero-padding needed
        x_pads = 0

    elif 2560 < img.shape[1] < 4096:
        # Zero-pad to 4096
        x_pads = 4096 - img.shape[1]

    elif img.shape[1] == 4096:
        # No zero-padding needed
        x_pads = 0

    elif 4096 < img.shape[1] < 8192:
        # Zero-pad to 8192
        x_pads = 8192 - img.shape[1]

    elif img.shape[1] == 8192:
        # No zero-padding needed
        x_pads = 0
    else:
        raise Exception('Padding error. Image too big?')

    img = np.pad(img, ((y_pads, 0), (x_pads, 0), (0, 0)), mode='constant')

    return img, [y_pads, x_pads]
