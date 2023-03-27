import json
import os
import platform
import re
import shutil
import subprocess
import torch
from pathlib import Path


def iou_pytorch(predictions, labels, device):
    """ Simple IoU-metric.

    :param predictions: Batch of predictions.
        :type predictions:
    :param labels: Batch of ground truths / label images.
        :type labels:
    :param device: cuda (gpu) or cpu.
        :type device:
    :return: Intersection over union.
    """

    a = predictions.clone().detach()

    # Apply sigmoid activation function in one-class problems
    # a = torch.sigmoid(a)

    # Flatten predictions and apply threshold
    a = a.view(-1) > torch.tensor([0.5], requires_grad=False).to(device)

    # Flatten labels
    b = labels.clone().detach().view(-1).bool()

    # Calculate intersection over union
    intersection = torch.sum((a * b).float())
    union = torch.sum(torch.max(a, b).float())
    iou = intersection / (union + 1e-6)

    return iou.cpu().numpy()


def ctc_metrics(data_path, results_path, software_path):
    """ Cell Tracking Challenge detection and segmentation metrics (DET, SEG).

    :param data_path: Path to directory containing the results.
        :type data_path: pathlib Path object
    :param software_path: Path to the evaluation software.
        :type software_path: pathlib Path object
    :return: None
    """

    if data_path.stem == 'BF-C2DL-HSC' or data_path.stem == 'BF-C2DL-MuSC':
        t = '4'
    else:
        t = '3'

    # software_path = Path('/srv/scherr/EvaluationSoftware/')

    # Clear temporary result directory if exists
    if os.path.exists(str(data_path / '02_RES')):
        shutil.rmtree(str(data_path / '02_RES'))

    # Create new clean directory
    Path.mkdir(data_path / '02_RES', exist_ok=True)

    # Chose the executable in dependency of the operating system
    if platform.system() == 'Linux':
        path_seg_executable = software_path / 'Linux' / 'SEGMeasure'
        path_det_executable = software_path / 'Linux' / 'DETMeasure'
    elif platform.system() == 'Windows':
        path_seg_executable = software_path / 'Win' / 'SEGMeasure.exe'
        path_det_executable = software_path / 'Win' / 'DETMeasure.exe'
    elif platform.system() == 'Darwin':
        path_seg_executable = software_path / 'Mac' / 'SEGMeasure'
        path_det_executable = software_path / 'Mac' / 'DETMeasure'
    else:
        raise ValueError('Platform not supported')

    mask_ids = results_path.glob('mask*')

    # copy masks to 02/_RES
    for mask_id in mask_ids:
        shutil.copyfile(str(mask_id), str(data_path / '02_RES' / mask_id.name))

    # Apply the evaluation software to calculate the cell tracking challenge SEG measure
    output = subprocess.Popen([str(path_seg_executable), str(data_path), '02', t],
                              stdout=subprocess.PIPE)
    result, _ = output.communicate()
    seg_measure = re.findall(r'\d\.\d*', result.decode('utf-8'))
    seg_measure = float(seg_measure[0])

    output = subprocess.Popen([str(path_det_executable), str(data_path), '02', t],
                              stdout=subprocess.PIPE)
    result, _ = output.communicate()
    det_measure = re.findall(r'\d\.\d*', result.decode('utf-8'))
    det_measure = float(det_measure[0])

    # Copy result files
    shutil.copyfile(str(data_path / '02_RES' / 'DET_log.txt'), str(results_path / 'DET_log.txt'))
    shutil.copyfile(str(data_path / '02_RES' / 'SEG_log.txt'), str(results_path / 'SEG_log.txt'))

    # Remove temporary directory
    shutil.rmtree(str(data_path / '02_RES'))

    return {'SEG': seg_measure, 'DET': det_measure, 'OP_CSB': 0.5 * (seg_measure + det_measure)}
