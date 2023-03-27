import argparse
import json
import numpy as np
import random
import torch
import warnings

from pathlib import Path
from PyQt5.QtCore import QFileInfo

from segmentation.inference.inference import inference_2d_ctc, inference_3d_ctc
from segmentation.training.cell_segmentation_dataset import CellSegDataset
from segmentation.training.ctc_training_set import create_ctc_training_set
from segmentation.training.mytransforms import augmentors
from segmentation.training.training import train
from segmentation.utils import utils, unets
from segmentation.utils.metrics import ctc_metrics

warnings.filterwarnings("ignore", category=UserWarning)


def main():

    random.seed()
    np.random.seed()

    # Get arguments
    parser = argparse.ArgumentParser(description='Cell Segmentation')
    parser.add_argument('--create_ctc_set', '-c',
                        default=False,
                        action='store_true',
                        help='Create the CTC Training Set.')
    parser.add_argument('--train', '-t',
                        default=False,
                        action='store_true',
                        help='Train new models')
    parser.add_argument('--evaluate', '-e',
                        default=False,
                        action='store_true',
                        help='Evaluate models')
    parser.add_argument('--plot', '-p',
                        default=False,
                        action='store_true',
                        help='Plot results')
    args = parser.parse_args()

    path_datasets = Path.cwd() / 'datasets'
    path_results = Path.cwd() / 'segmentation_results'
    path_models = Path.cwd() / 'segmentation_models'
    cell_types = ['BF-C2DL-HSC', 'BF-C2DL-MuSC', 'Fluo-N2DL-HeLa', 'Fluo-N3DH-CE']
    methods = ['boundary', 'border', 'adapted_border', 'dual_unet', 'pena', 'distance']

    # Create CTC Training Set
    if args.create_ctc_set:
        print('Create CTC Training Set (Pena label generation needs some time)')
        create_ctc_training_set(path=path_datasets)
    else:
        if not QFileInfo(str(path_datasets / 'ctc_training_set' / 'train')).exists():
            return print('No CTC Training Set found. Run with flag -c to create the data set.')

    mode = []
    if args.train:
        mode.append('train')

    if args.evaluate:
        mode.append('eval')
        if not QFileInfo(str(Path.cwd() / 'EvaluationSoftware')).exists():
            return print('No CTC evaluation software found. Download the software'
                         '(http://celltrackingchallenge.net/evaluation-methodology/)\n'
                         'and unzip it into {}.'.format(Path.cwd()))

    if args.plot:
        mode.append('plot')

    if not mode:
        return print('Train models (-t), evaluate models (-e), or plot results (-p)')

    # Load settings
    with open(Path.cwd() / 'cell_segmentation_settings.json') as f:
        settings = json.load(f)

    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    num_gpus = torch.cuda.device_count()

    if 'train' in mode:  # Train model from scratch

        # Make directory for the trained models
        path_models.mkdir(exist_ok=True)

        for architecture in settings['architectures']:

            for i in range(settings['iterations']):  # Train multiple models

                run_name = utils.unique_path(path_models, architecture[1] + '_model_{:02d}.pth').stem

                train_configs = {'architecture': architecture[0],
                                 'batch_size': settings['batch_size'],
                                 'break_condition': settings['break_condition'],
                                 'label_type': architecture[1],
                                 'learning_rate': settings['learning_rate'],
                                 'lr_patience': settings['learning_rate_patience'],
                                 'loss': architecture[2],
                                 'max_epochs': settings['max_epochs'],
                                 'num_gpus': num_gpus,
                                 'run_name': run_name
                                 }

                if train_configs['label_type'] in ['distance', 'dual_unet']:
                    ch_out = 1
                elif train_configs['label_type'] == 'pena':
                    ch_out = 4
                else:
                    ch_out = 3

                net = unets.build_unet(unet_type=train_configs['architecture'][0],
                                       act_fun=train_configs['architecture'][2],
                                       pool_method=train_configs['architecture'][1],
                                       normalization=train_configs['architecture'][3],
                                       device=device,
                                       num_gpus=num_gpus,
                                       ch_in=1,
                                       ch_out=ch_out,
                                       filters=train_configs['architecture'][4],
                                       print_path=path_models)

                # The training images are uint16 crops of a min-max normalized image
                data_transforms = augmentors(label_type=train_configs['label_type'], min_value=0, max_value=65535)
                train_configs['data_transforms'] = str(data_transforms)

                # Load training and validation set
                datasets = {x: CellSegDataset(root_dir=path_datasets / 'ctc_training_set',
                                              label_type=train_configs['label_type'],
                                              mode=x,
                                              transform=data_transforms[x])
                            for x in ['train', 'val']}

                # Train model
                train(net=net, datasets=datasets, configs=train_configs, device=device, path_models=path_models)

                # Write information to json-file
                utils.write_train_info(configs=train_configs, path=path_models)

    if 'eval' in mode:  # Eval all trained models on the test data set

        # Make results directory
        path_results.mkdir(exist_ok=True)

        # Metric scores dict
        metric_scores = {}

        for label_type in methods:

            print('Evaluate {} models'.format(label_type))

            metric_scores[label_type] = {}
            for cell_type in cell_types:
                metric_scores[label_type][cell_type] = {}

            # Get paths of trained models
            models = sorted(path_models.glob('{}_model*.pth'.format(label_type)))

            if not models:
                print('No models available.')
                continue

            # Inference all trained models
            for model in models:

                print('   Inference of {}'.format(model.stem))

                (path_results / model.stem).mkdir(exist_ok=True)

                for cell_type in cell_types:

                    (path_results / model.stem / cell_type).mkdir(exist_ok=True)

                    # Check if already inference results available
                    if QFileInfo(str(path_results / model.stem / cell_type)).exists():
                        if not (not list((path_results / model.stem / cell_type).glob('mask*'))):
                            print("      {}: predictions already available. Delete if you want to recalculate.".format(cell_type))
                            # Calculate metric scores
                            if QFileInfo(str(path_results / "metrics.json")).exists():
                                with open(path_results / 'metrics.json') as infile:
                                    metric_scores[label_type][cell_type][model.stem] = json.load(infile)[label_type][cell_type][model.stem]
                            else:
                                metric_scores[label_type][cell_type][model.stem] = ctc_metrics(data_path=path_datasets / cell_type,
                                                                                               results_path=path_results / model.stem / cell_type,
                                                                                               software_path=Path.cwd() / 'EvaluationSoftware')
                            continue
                    else:
                        # Make result directory
                        Path.mkdir(path_results / model.stem / cell_type, exist_ok=True)

                    print('      {}: inference'.format(cell_type))
                    if '2D' in cell_type:
                        inference_2d_ctc(model=model,
                                         data_path=path_datasets,
                                         result_path=path_results,
                                         device=device,
                                         cell_type=cell_type)
                    else:
                        inference_3d_ctc(model=model,
                                         data_path=path_datasets,
                                         result_path=path_results,
                                         device=device,
                                         cell_type=cell_type)

                    # Calculate metric scores
                    metric_scores[label_type][cell_type][model.stem] = ctc_metrics(data_path=path_datasets / cell_type,
                                                                                   results_path=path_results / model.stem / cell_type,
                                                                                   software_path=Path.cwd() / 'EvaluationSoftware')

        # Save evaluation metric scores
        utils.write_eval_info(results=metric_scores, path=path_results)

        # Save best OP_CSB models
        utils.get_models(results=metric_scores, path=path_results, methods=methods, cell_types=cell_types, mode='best')

    if 'plot' in mode:

        # Check if metrics file + median models are available
        if not QFileInfo(str(path_results / 'metrics.json')).exists():
            print('No metrics file available. Evaluate first (-e, --evaluate).')
            return

        utils.plot_metrics(path=path_results, cell_types=cell_types, methods=methods)

        # Paper plots for best model
        utils.get_paper_crops(path_results=path_results,
                              path_datasets=path_datasets,
                              cell_types=cell_types,
                              methods=methods,
                              mode='best')


if __name__ == "__main__":

    main()
