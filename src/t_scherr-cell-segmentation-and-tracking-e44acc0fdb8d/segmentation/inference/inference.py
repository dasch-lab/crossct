import gc
import json
import tifffile as tiff
import torch
import torch.nn.functional as F

from segmentation.inference.postprocessing import *
from segmentation.utils.utils import min_max_normalization, zero_pad_model_input, hela_foi_correction
from segmentation.utils.unets import build_unet


def inference_2d_ctc(model, data_path, result_path, device, cell_type):
    """ Inference function for 2D Cell Tracking Challenge data sets.

    :param model: Path to the model to use for inference.
        :type model: pathlib Path object.
    :param data_path: Path to the directory containing the Cell Tracking Challenge data sets.
        :type data_path: pathlib Path object
    :param result_path: Path to the results directory.
        :type result_path: pathlib Path object
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param cell_type: Date set to process.
        :type cell_type: str
    :return: None
    """

    # Load model json file to get architecture + filters
    with open(model.parent / (model.stem + '.json')) as f:
        model_settings = json.load(f)

    if model_settings['label_type'] in ['distance', 'dual_unet']:
        ch_out = 1
    elif model_settings['label_type'] == 'pena':
        ch_out = 4
    else:
        ch_out = 3

    num_gpus = 1
    net = build_unet(unet_type=model_settings['architecture'][0],
                     act_fun=model_settings['architecture'][2],
                     pool_method=model_settings['architecture'][1],
                     normalization=model_settings['architecture'][3],
                     device=device,
                     num_gpus=num_gpus,
                     ch_in=1,
                     ch_out=ch_out,
                     filters=model_settings['architecture'][4],
                     print_path=None)

    if num_gpus > 1:
        net.module.load_state_dict(torch.load(str(model), map_location=device))
    else:
        net.load_state_dict(torch.load(str(model), map_location=device))
    net.eval()
    torch.set_grad_enabled(False)

    # Get images to predict
    files = sorted((data_path / cell_type / '02').glob('*.tif'))

    # Prediction process (iterate over images/files)
    for i, file in enumerate(files):

        # Load image
        img = tiff.imread(str(file))

        # Save not all raw predictions to save memory
        if i % 10 == 0 or i == 1748 or i == 1278 or i == 79 or i == 106:
            save_raw_pred = True
        else:
            save_raw_pred = False
        save_path, file_id = result_path / model.stem / cell_type, file.stem.split('t')[-1] + '.tif'

        # Get position of the color channel
        if len(img.shape) == 2:  # grayscale image, add pseudo color channel
            img = np.expand_dims(img, axis=-1)
        elif len(img.shape) == 3 and img.shape[0] == 1:
            img = np.swapaxes(img, 0, -1)

        # Min-max normalize the image to [0, 65535] (uint16 range)
        img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
        img = np.clip(img, 0, 65535).astype(np.uint16)

        # Check if zero-padding is needed to apply the model
        img, pads = zero_pad_model_input(img=img)

        # First convert image into the range [-1, 1] to get a zero-centered input for the network
        net_input = min_max_normalization(img=img, min_value=0, max_value=65535)

        # Bring input into the shape [batch, channel, height, width]
        net_input = np.transpose(np.expand_dims(net_input, axis=0), [0, 3, 1, 2])

        # Prediction
        print('         ... processing {0}{1} ...'.format(file.stem, file.suffix))
        net_input = torch.from_numpy(net_input).to(device)

        if model_settings['label_type'] in ['boundary', 'border', 'pena']:
            prediction = net(net_input)
            prediction = F.softmax(prediction, dim=1)
            prediction = prediction[0, :, pads[0]:, pads[1]:].permute(1, 2, 0).cpu().numpy()

            if model_settings['label_type'] == 'border':
                prediction_instance, prediction_bin = border_postprocessing(prediction)
            elif model_settings['label_type'] == 'pena':
                prediction_instance, prediction_bin = pena_postprocessing(prediction)
            else:
                prediction_instance, prediction_bin = boundary_postprocessing(prediction)

            if cell_type == 'Fluo-N2DL-HeLa':
                prediction_instance = hela_foi_correction(prediction_instance)

            tiff.imsave(str(save_path / ('mask' + file_id)), prediction_instance[:, :, 0], compress=1)

            if save_raw_pred:
                tiff.imsave(str(save_path / ('bin' + file_id)), prediction_bin[:, :, 0], compress=1)
                tiff.imsave(str(save_path / ('cell' + file_id)), prediction[:, :, 1].astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('border' + file_id)), prediction[:, :, 2].astype(np.float32), compress=1)
                if model_settings['label_type'] == 'pena':
                    tiff.imsave(str(save_path / ('gap' + file_id)), prediction[:, :, 3].astype(np.float32), compress=1)

        elif model_settings['label_type'] == 'adapted_border':
            border_prediction, cell_prediction = net(net_input)
            border_prediction = F.softmax(border_prediction, dim=1)
            cell_prediction = torch.sigmoid(cell_prediction)
            border_prediction = border_prediction[0, :, pads[0]:, pads[1]:].permute(1, 2, 0).cpu().numpy()
            cell_prediction = cell_prediction[0, 0, pads[0]:, pads[1]:].cpu().numpy()

            prediction_instance, border_prediction_bin = adapted_border_postprocessing(
                border_prediction=border_prediction,
                cell_prediction=cell_prediction)

            if cell_type == 'Fluo-N2DL-HeLa':
                prediction_instance = hela_foi_correction(prediction_instance)

            tiff.imsave(str(save_path / ('mask' + file_id)), prediction_instance[:, :, 0], compress=1)

            if save_raw_pred:
                tiff.imsave(str(save_path / ('bin' + file_id)), border_prediction_bin[:, :, 0], compress=1)
                tiff.imsave(str(save_path / ('seed' + file_id)), border_prediction[:, :, 1].astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('border' + file_id)), border_prediction[:, :, 2].astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('cell' + file_id)), cell_prediction.astype(np.float32), compress=1)

        elif model_settings['label_type'] == 'dual_unet':
            prediction_border, prediction_cell_dist, prediction_cell = net(net_input)
            prediction_border = torch.sigmoid(prediction_border)
            prediction_cell = torch.sigmoid(prediction_cell)
            prediction_cell = prediction_cell[0, 0, pads[0]:, pads[1]:, None].cpu().numpy()
            prediction_cell_dist = prediction_cell_dist[0, 0, pads[0]:, pads[1]:, None].cpu().numpy()
            prediction_border = prediction_border[0, 0, pads[0]:, pads[1]:, None].cpu().numpy()

            prediction_instance = dual_unet_postprocessing(prediction=prediction_cell)

            if cell_type == 'Fluo-N2DL-HeLa':
                prediction_instance = hela_foi_correction(prediction_instance)

            tiff.imsave(str(save_path / ('mask' + file_id)), prediction_instance[:, :, 0], compress=1)

            if save_raw_pred:
                tiff.imsave(str(save_path / ('cell' + file_id)), prediction_cell[:, :, 0].astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('cell_dist' + file_id)), prediction_cell_dist[:, :, 0].astype(np.float32),
                            compress=1)
                tiff.imsave(str(save_path / ('border' + file_id)), prediction_border[:, :, 0].astype(np.float32), compress=1)

        elif model_settings['label_type'] == 'distance':
            prediction_border, prediction_cell = net(net_input)
            prediction_cell = prediction_cell[0, 0, pads[0]:, pads[1]:, None].cpu().numpy()
            prediction_border = prediction_border[0, 0, pads[0]:, pads[1]:, None].cpu().numpy()

            prediction_instance = distance_postprocessing(border_prediction=prediction_border,
                                                          cell_prediction=prediction_cell)

            if cell_type == 'Fluo-N2DL-HeLa':
                prediction_instance = hela_foi_correction(prediction_instance)

            tiff.imsave(str(save_path / ('mask' + file_id)), prediction_instance[:, :, 0], compress=1)

            if save_raw_pred:
                tiff.imsave(str(save_path / ('cell' + file_id)), prediction_cell[:, :, 0].astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('border' + file_id)), prediction_border[:, :, 0].astype(np.float32), compress=1)

    # Clear memory
    del net
    gc.collect()

    return None


def inference_3d_ctc(model, data_path, result_path, device, cell_type):
    """ Inference function for 2D Cell Tracking Challenge data sets.

    :param model: Path to the model to use for inference.
        :type model: pathlib Path object.
    :param data_path: Path to the directory containing the Cell Tracking Challenge data sets.
        :type data_path: pathlib Path object
    :param result_path: Path to the results directory.
        :type result_path: pathlib Path object
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param cell_type: Date set to process.
        :type cell_type: str
    :return: None
    """

    # Load model json file to get architecture + filters
    with open(model.parent / (model.stem + '.json')) as f:
        model_settings = json.load(f)

    if model_settings['label_type'] in ['distance', 'dual_unet']:
        ch_out = 1
    elif model_settings['label_type'] == 'pena':
        ch_out = 4
    else:
        ch_out = 3

    num_gpus = 1
    net = build_unet(unet_type=model_settings['architecture'][0],
                     act_fun=model_settings['architecture'][2],
                     pool_method=model_settings['architecture'][1],
                     normalization=model_settings['architecture'][3],
                     device=device,
                     num_gpus=num_gpus,
                     ch_in=1,
                     ch_out=ch_out,
                     filters=model_settings['architecture'][4],
                     print_path=None)

    if num_gpus > 1:
        net.module.load_state_dict(torch.load(str(model), map_location=device))
    else:
        net.load_state_dict(torch.load(str(model), map_location=device))
    net.eval()
    torch.set_grad_enabled(False)

    # Get images to predict
    files = sorted((data_path / cell_type / '02').glob('*.tif'))

    # Prediction process (iterate over images/files)
    for i, file in enumerate(files):

        # Load image
        img = tiff.imread(str(file))

        # Save not all raw predictions to save memory
        if i % 10 == 0 or i == 106:
            save_raw_pred = True
        else:
            save_raw_pred = False
        save_path, file_id = result_path / model.stem / cell_type, file.stem.split('t')[-1] + '.tif'

        # Min-max normalize the volume to [0, 65535] (uint16 range)
        img = 65535 * (img.astype(np.float32) - img.min()) / (img.max() - img.min())
        img = np.clip(img, 0, 65535).astype(np.uint16)

        # Pre-allocate arrays for results
        if model_settings['label_type'] in ['boundary', 'border']:
            prediction = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2], 3), dtype=np.float32)
        elif model_settings['label_type'] == 'pena':
            prediction = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2], 4), dtype=np.float32)
        elif model_settings['label_type'] == 'adapted_border':
            prediction_cell = np.zeros(shape=img.shape, dtype=np.float32)
            prediction_border = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2], 3), dtype=np.float32)
        elif model_settings['label_type'] == 'dual_unet':
            prediction_cell = np.zeros(shape=img.shape, dtype=np.float32)
            prediction_cell_dist = np.zeros(shape=img.shape, dtype=np.float32)
            prediction_border = np.zeros(shape=img.shape, dtype=np.float32)
        elif model_settings['label_type'] == 'distance':
            prediction_cell = np.zeros(shape=img.shape, dtype=np.float32)
            prediction_border = np.zeros(shape=img.shape, dtype=np.float32)

        img = img[:, :, :, None]

        print('         ... processing {0}{1} ...'.format(file.stem, file.suffix))

        # Go through the slices and make slice-wise predictions
        for j in range(img.shape[0]):
            img_slice = img[j]

            img_slice, pads = zero_pad_model_input(img=img_slice)

            # First normalize image into the range [-1, 1] to get a zero-centered input for the network
            net_input = min_max_normalization(img=img_slice, min_value=0, max_value=65535)

            # Bring input into the shape [batch, channel, height, width]
            net_input = np.transpose(np.expand_dims(net_input, axis=0), [0, 3, 1, 2])

            net_input = torch.from_numpy(net_input).to(device)

            if model_settings['label_type'] in ['boundary', 'border', 'pena']:
                prediction_slice = net(net_input)
                prediction_slice = F.softmax(prediction_slice, dim=1)
                prediction_slice = prediction_slice[0, :, pads[0]:, pads[1]:].permute(1, 2, 0).cpu().numpy()

                prediction[j] = prediction_slice

            elif model_settings['label_type'] == 'adapted_border':
                prediction_slice_border, prediction_slice_cell = net(net_input)
                prediction_slice_border = F.softmax(prediction_slice_border, dim=1)
                prediction_slice_cell = torch.sigmoid(prediction_slice_cell)

                prediction_slice_border = prediction_slice_border[0, :, pads[0]:, pads[1]:].permute(1, 2, 0).cpu().numpy()
                prediction_slice_cell = prediction_slice_cell[0, 0, pads[0]:, pads[1]:].cpu().numpy()

                prediction_cell[j] = prediction_slice_cell
                prediction_border[j] = prediction_slice_border

            elif model_settings['label_type'] == 'dual_unet':
                prediction_slice_border, prediction_slice_cell_dist, prediction_slice_cell = net(net_input)
                prediction_slice_border = torch.sigmoid(prediction_slice_border)
                prediction_slice_cell = torch.sigmoid(prediction_slice_cell)
                prediction_slice_cell_dist = prediction_slice_cell_dist[0, 0, pads[0]:, pads[1]:].cpu().numpy()
                prediction_slice_cell = prediction_slice_cell[0, 0, pads[0]:, pads[1]:].cpu().numpy()
                prediction_slice_border = prediction_slice_border[0, 0, pads[0]:, pads[1]:].cpu().numpy()

                prediction_cell[j] = prediction_slice_cell
                prediction_cell_dist[j] = prediction_slice_cell_dist
                prediction_border[j] = prediction_slice_border

            elif model_settings['label_type'] == 'distance':
                prediction_slice_border, prediction_slice_cell = net(net_input)
                prediction_slice_cell = prediction_slice_cell[0, 0, pads[0]:, pads[1]:].cpu().numpy()
                prediction_slice_border = prediction_slice_border[0, 0, pads[0]:, pads[1]:].cpu().numpy()

                prediction_cell[j] = prediction_slice_cell
                prediction_border[j] = prediction_slice_border

        # Post-processing of whole frame/volume
        if model_settings['label_type'] in ['boundary', 'border', 'pena']:
            if model_settings['label_type'] == 'border':
                prediction_instance, prediction_bin = border_postprocessing(prediction, input_3d=True)
            elif model_settings['label_type'] == 'pena':
                prediction_instance, prediction_bin = pena_postprocessing(prediction, input_3d=True)
            else:
                prediction_instance, prediction_bin = boundary_postprocessing(prediction, input_3d=True)

            tiff.imsave(str(save_path / ('mask' + file_id)), prediction_instance, compress=1)

            if save_raw_pred:
                tiff.imsave(str(save_path / ('bin' + file_id)), prediction_bin, compress=1)
                tiff.imsave(str(save_path / ('cell' + file_id)), prediction[:, :, :, 1].astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('border' + file_id)), prediction[:, :, :, 2].astype(np.float32), compress=1)
                if model_settings['label_type'] == 'pena':
                    tiff.imsave(str(save_path / ('gap' + file_id)), prediction[:, :, :, 3].astype(np.float32),
                                compress=1)

        elif model_settings['label_type'] == 'adapted_border':

            # Post-processing
            prediction_instance, border_prediction_bin = adapted_border_postprocessing(
                border_prediction=prediction_border,
                cell_prediction=prediction_cell,
                input_3d=True
            )

            tiff.imsave(str(save_path / ('mask' + file_id)), prediction_instance, compress=1)

            if save_raw_pred:
                tiff.imsave(str(save_path / ('bin' + file_id)), border_prediction_bin, compress=1)
                tiff.imsave(str(save_path / ('seed' + file_id)), prediction_border[:, :, :, 1].astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('border' + file_id)), prediction_border[:, :, :, 2].astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('cell' + file_id)), prediction_cell.astype(np.float32), compress=1)

        elif model_settings['label_type'] == 'dual_unet':

            # Post-processing
            prediction_instance = dual_unet_postprocessing(prediction=prediction_cell, input_3d=True)

            tiff.imsave(str(save_path / ('mask' + file_id)), prediction_instance, compress=1)

            if save_raw_pred:
                tiff.imsave(str(save_path / ('cell' + file_id)), prediction_cell.astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('cell_dist' + file_id)), prediction_cell_dist.astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('border' + file_id)), prediction_border.astype(np.float32), compress=1)

        elif model_settings['label_type'] == 'distance':

            # Post-processing
            prediction_instance = distance_postprocessing(border_prediction=prediction_border,
                                                          cell_prediction=prediction_cell,
                                                          input_3D=True)

            tiff.imsave(str(save_path / ('mask' + file_id)), prediction_instance, compress=1)

            if save_raw_pred:
                tiff.imsave(str(save_path / ('cell' + file_id)), prediction_cell.astype(np.float32), compress=1)
                tiff.imsave(str(save_path / ('border' + file_id)), prediction_border.astype(np.float32), compress=1)

    # Clear memory
    del net
    gc.collect()

    return None
