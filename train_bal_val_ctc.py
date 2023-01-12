#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
from cProfile import label
from distutils import core
import os
join = os.path.join
import sys
# sys.path.append("/disk1/neurips/omnipose-main/omnipose")

import tifffile as tif
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
import glob
from torch import nn

import monai
from monai.data import decollate_batch, PILReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    # RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    # RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,    
    EnsureTyped,
    EnsureType,
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from utils import ramps, losses
from tqdm import tqdm
import logging
from itertools import cycle
from torch.nn.modules.loss import CrossEntropyLoss
from PIL import Image
# import tifffile as tif
# from skimage import io


from skimage import io, segmentation, morphology, measure, exposure
from skimage.filters import gaussian
from train_baseline_ctc import iou_F1
from train_baseline_ctc import split_dataset_training as split_dataset_training_val





Image.MAX_IMAGE_PIXELS = None
# from unified_focalLoss import AsymmetricFocalLoss, AsymmetricUnifiedFocalLoss

monai.config.print_config()
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print('Successfully import all requirements!')

parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
# Dataset parameters
parser.add_argument('--labeled_path', default='/disk1/neurips/dataset/dataset_tot', type=str,
                    help='labeled training data path; subfolders: images, labels')
parser.add_argument('--instance_path', default='/disk1/neurips/dataset/dataset_tot/instance_labels', type=str,
                    help='labeled training data path; subfolders: images, labels')
parser.add_argument('--ctc_path', default='', type=str,
                    help='labeled training data path; subfolders: images, labels')
parser.add_argument('--unlabeled_path', default='/disk1/neurips/dataset/train_unlabeled_preprocessed', type=str,
                    help='unlabeled training data path; subfolder: images')
parser.add_argument('--work_dir', default='./work_dir',
                    help='path where to save models and logs')
parser.add_argument('--seed', default=2022, type=int)
parser.add_argument('--resume', default=False,
                    help='resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int)

# Model parameters
parser.add_argument('--model1_name', default='unet', help='select model 1: unet ...')
parser.add_argument('--model2_name', default='swinunetr', help='select mode: unetr, swinunetr')
parser.add_argument('--num_class', default=3, type=int, help='segmentation classes')
parser.add_argument('--input_size', default=256, type=int, help='segmentation classes') # 512

# Training parameters
parser.add_argument('--pre_trained', default=True, help='load checkpoint')
parser.add_argument('--dir_checkpoint', default='/disk1/neurips/baseline/pretrained_models', type=str, help='path of the checkpoint to resume')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
parser.add_argument('--max_epochs', default=2000, type=int)
parser.add_argument('--val_interval', default=2, type=int) 
parser.add_argument('--epoch_tolerance', default=100, type=int)
parser.add_argument('--initial_lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--val_frac', type=float, default=0.3, help='learning rate')
parser.add_argument('--max_iterations', type=int, default=50000, help='maximum epoch number to train')
parser.add_argument('--optimizer', type=str, default="Adam", help="Adam or SGD")

parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
# def main():
#     args = parser.parse_args()
args = parser.parse_args()
args.ctc = '/disk1/neurips/dataset/distance_neighbour_representation'
args.num_class = 5


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def load_file_by_name(path, name):
    
    list_name = glob.glob(os.path.join(path, name))
    return list_name

def save_img_lab_names(img_path = None, gt_path = None, name = '*'):
    if img_path is None:
        lab_name = load_file_by_name(gt_path, name)
        return lab_name
    elif gt_path is None:
        img_name = load_file_by_name(img_path, name)
        return img_name
    else:
        img_name = load_file_by_name(img_path, name)
        lab_name = load_file_by_name(gt_path, name)
        return img_name, lab_name

def split_dataset(img_list, gt_list, val_frac):

    img_num = len(img_list)
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num*val_frac)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    train_files = [{"img": join(img_list[i]), "label": join(gt_list[i])} for i in train_indices]
    val_files = [{"img": join(img_list[i]), "label": join(gt_list[i])} for i in val_indices]
    print('Train: {}'.format(len(train_files)))
    print('Val: {}'.format(len(val_files)))
    
    return train_files, val_files

def split_dataset_training(img_list, gt1_list, gt2_list, gt3_list, val_frac):

    img_num = len(img_list)
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num*val_frac)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    train_files = [{"img": join(img_list[i]), "label1": join(gt1_list[i]), "label2": join(gt2_list[i]), "label3": join(gt3_list[i]), } for i in train_indices]
    val_files = [{"img": join(img_list[i]), "label1": join(gt1_list[i]), "label2": join(gt2_list[i]), "label3": join(gt3_list[i])}  for i in val_indices]
    print('Train: {}'.format(len(train_files)))
    print('Val: {}'.format(len(val_files)))
    
    return train_files, val_files

def main():
    #%% set training/validation split
    np.random.seed(args.seed)
    #model_path = join(args.work_dir, "ct_" + args.model1_name + "_" + args.model2_name + "_" + \
    #                  "val_frac_" + str(args.val_frac) + "_lr_" + str(args.initial_lr) + "_BG_" + str(args.optimizer) + \
    #                    "_new_pseudo")
    model_path = join(args.work_dir, "ct_" + args.model1_name + "_" + args.model2_name + "_batch_" + str(args.batch_size) + \
                      "_patch_" + str(args.input_size) + "_val_frac_" + str(args.val_frac) + "_lr_" + str(args.initial_lr) + str(args.optimizer) + \
                        "_ctc_int16_float32_pretrained_baseline_2")
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(__file__, join(model_path, run_id + '_' + os.path.basename(__file__)))
    img_path = join(args.labeled_path, 'images')
    gt_path = join(args.labeled_path, 'labels')
    unlabeled_path = join(args.unlabeled_path, 'images')

    #img_omn, lab_omn = save_img_lab_names(img_path, gt_path, '*_omnipose_*.png')
    #img_cell, lab_cell = save_img_lab_names(img_path, gt_path, '*_cellpose_*.png')
    img_nips, lab1_nips = save_img_lab_names(img_path, gt_path, 'cell_*.png')
    lab2_nips = save_img_lab_names(gt_path = join(args.ctc, 'label_dist'), name ='cell_*_label.tiff')
    lab3_nips = save_img_lab_names(gt_path = join(args.ctc, 'label_dist_neighbor'), name ='cell_*_label.tiff')
    lab4_nips = save_img_lab_names(gt_path = args.instance_path, name = 'cell_*.tiff')
    #img_nips.sort()
    #lab3_nips.sort()
    #lab1_nips.sort()
    #lab2_nips.sort()
    #lab4_nips.sort()


    #img_names = sorted(os.listdir(img_path))
    #gt_names = sorted(os.listdir(gt_path))
    unlabeled_names = sorted(os.listdir(unlabeled_path))
    un_num = len(unlabeled_names)
    print("Number of unlabeled files: {}".format(un_num))
    #gt_names = [img_name.split('.')[0]+'_label.png' for img_name in img_names]
    
    #img_num = len(img_omn) + len(img_cell) + len(img_nips) 
    img_num = len(img_nips) 
    print("Number of labeled files: {}".format(img_num))
    val_frac = args.val_frac

    #train_omn, val_omn = split_dataset(img_omn, lab_omn, val_frac)
    #train_cell, val_cell = split_dataset(img_cell, lab_cell, val_frac)
    #train_nips, val_nips = split_dataset(img_nips, lab_nips, val_frac)
    #train_nips, val_nips = split_dataset_training(img_nips, lab1_nips, lab2_nips, lab3_nips, val_frac)
    train_nips, val_nips = split_dataset_training_val(img_nips, lab1_nips, lab2_nips, lab3_nips, lab4_nips, val_frac)

    #train_files = train_omn + train_cell + train_nips
    #val_files = val_omn + val_cell + val_nips
    train_files = train_nips
    val_files = val_nips
    print(f"training image num: {len(train_files)}, validation image num: {len(val_files)}")
    unlabeled_files = [{"img": join(unlabeled_path, unlabeled_names[i])} for i in range(un_num)]
    
    #%% define transforms for image and segmentation
    ''' train_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8), # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["label"], allow_missing_keys=True), # label: (1, H, W)
            AsChannelFirstd(keys=['img'], channel_dim=-1, allow_missing_keys=True), # image: (3, H, W)
            ScaleIntensityd(keys=["img"], allow_missing_keys=True), # Do not scale label
            SpatialPadd(keys=["img","label"], spatial_size=args.input_size),
            RandSpatialCropd(keys=["img", "label"], roi_size=args.input_size, random_size=False),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform 
            RandGaussianNoised(keys=['img'], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1,2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1,2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(keys=["img", "label"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=['area', 'nearest']),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=['img'], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label"]),
        ]
    )'''

    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader, dtype=np.uint8), # image three channels (H, W, 3); label: (H, W)
            LoadImaged(keys=["label1"], reader=PILReader, dtype=np.int16), # image three channels (H, W, 3); label: (H, W)
            LoadImaged(keys=["label2", "label3"], reader=PILReader, dtype=np.float32), # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["label1", "label2", "label3"], allow_missing_keys=True), # label: (1, H, W)
            AsChannelFirstd(keys=['img'], channel_dim=-1, allow_missing_keys=True), # image: (3, H, W)
            ScaleIntensityd(keys=["img"], allow_missing_keys=True), # Do not scale label
            SpatialPadd(keys=["img","label1", "label2", "label3"], spatial_size=args.input_size),
            RandSpatialCropd(keys=["img", "label1", "label2", "label3"], roi_size=args.input_size, random_size=False),
            RandAxisFlipd(keys=["img", "label1", "label2", "label3"], prob=0.5),
            RandRotate90d(keys=["img", "label1", "label2", "label3"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform 
            RandGaussianNoised(keys=['img'], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1,2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1,2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            #RandZoomd(keys=["img", "label1", "label2", "label3", "label4", "label5", "label6", "label7"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=['area', 'nearest']),
            EnsureTyped(keys=["img", "label1", "label2", "label3"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader, dtype=np.uint8),
            LoadImaged(keys=["label1", "label4"], reader=PILReader, dtype=np.int16),
            LoadImaged(keys=["label2", "label3"], reader=PILReader, dtype=np.float32),
            AddChanneld(keys=["label1", "label2", "label3"], allow_missing_keys=True),
            AsChannelFirstd(keys=['img'], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label1", "label2", "label3", "label4"]),
        ]
    )

    train_transforms_unlabeled = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader, dtype=np.uint8), # image three channels (H, W, 3); label: (H, W)
            AsChannelFirstd(keys=['img'], channel_dim=-1, allow_missing_keys=True), # image: (3, H, W)
            ScaleIntensityd(keys=["img"], allow_missing_keys=True), # Do not scale label
            SpatialPadd(keys=["img"], spatial_size=args.input_size),
            RandSpatialCropd(keys=["img"], roi_size=args.input_size, random_size=False),
            RandAxisFlipd(keys=["img"], prob=0.5),
            RandRotate90d(keys=["img"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform 
            RandGaussianNoised(keys=['img'], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1,2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1,2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            #RandZoomd(keys=["img"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=['area', 'nearest']),
            EnsureTyped(keys=["img"]),
        ]
    )

    #% define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print('sanity check:', check_data["img"].shape, torch.max(check_data["img"]), check_data["label1"].shape, torch.max(check_data["label1"]))

    check_unlabeled_ds = monai.data.Dataset(data=unlabeled_files, transform=train_transforms_unlabeled)
    check_unlabeled_loader = DataLoader(check_unlabeled_ds, batch_size=1, num_workers=4)
    check_unlabeled_data = monai.utils.misc.first(check_unlabeled_loader)
    print('sanity check:', check_unlabeled_data["img"].shape, torch.max(check_data["img"]))

    #%% create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    train_unlabeled_ds = monai.data.Dataset(data=unlabeled_files, transform=train_transforms_unlabeled)
    train_unlabeled_loader = DataLoader(
        train_unlabeled_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    post_pred = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)])
    post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
    # create UNet, DiceLoss and Adam optimizer
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model1_name.lower() == 'unet':
        if args.input_size == 256:
            chs = (16, 32, 64, 128, 256)
            sts = (2, 2, 2, 2)
        elif args.input_size == 512:
            chs = (16, 32, 64, 128, 256, 512)
            sts = (2, 2, 2, 2, 2)
        model1 = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=args.num_class,
            channels= chs,
            strides=sts,
            num_res_units=2,
        ).to(device)

    if args.model2_name.lower() == "swinunetr":
        model2 = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=2,
        ).to(device)


    initial_lr = args.initial_lr

    # TO TRY: AdamW
    if args.optimizer == "Adam":
        optimizer1 = torch.optim.AdamW(model1.parameters(), initial_lr)
        optimizer2 = torch.optim.AdamW(model2.parameters(), initial_lr)
    else:
        optimizer1 = torch.optim.SGD(model1.parameters(), initial_lr, momentum=0.9, weight_decay=0.0001)
        optimizer2 = torch.optim.SGD(model2.parameters(), initial_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = monai.losses.DiceCELoss(softmax=True)
    dice_loss = monai.losses.DiceLoss()
    l2_loss = nn.MSELoss()

    if args.pre_trained:
        if args.dir_checkpoint is not None:
            dir_checkpoint = args.dir_checkpoint
            checkpoint1 = torch.load(os.path.join(dir_checkpoint, "best_unet_Loss_model_instance.pth"))
            model1.load_state_dict(checkpoint1['model_state_dict'])
            optimizer1.load_state_dict(checkpoint1['optimizer_state_dict'])
            epoch1 = checkpoint1['epoch']
            loss1 = checkpoint1['loss']

            checkpoint2 = torch.load(os.path.join(dir_checkpoint, "best_swin_Loss_model_instance.pth"))
            model2.load_state_dict(checkpoint2['model_state_dict'])
            optimizer2.load_state_dict(checkpoint2['optimizer_state_dict'])
            epoch2 = checkpoint2['epoch']
            loss2 = checkpoint2['loss']
        else:
            print("Checkpoint path needed!!!")
            sys.exit(1)

        
    
    # start a typical PyTorch training
    max_iterations = args.max_iterations
    iter_num = 0
    max_epochs = max_iterations // len(train_unlabeled_loader) + 1
    iterator = tqdm(range(max_epochs), ncols=70)
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric1_1 = -1
    best_metric2_1 = -1
    best_metric1_2 = -1
    best_metric2_2 = -1
    best_metric_epoch1_1 = -1
    best_metric_epoch2_1 = -1
    best_metric_epoch1_2 = -1
    best_metric_epoch2_2 = -1
    epoch_loss_values1_1 = list()
    epoch_loss_values2_1 = list()
    epoch_loss_values1_2 = list()
    epoch_loss_values1_2 = list()
    metric_values1_1 = list()
    metric_values1_2 = list()
    metric_values2_1 = list()
    metric_values2_2 = list()

    #writer = SummaryWriter(model_path)
    for epoch in iterator:
        model1.train()
        model2.train()
        epoch_loss = 0
        for step, (batch_data, batch_data_unlabeled) in enumerate(zip(cycle(train_loader), train_unlabeled_loader)):
            inputs = batch_data["img"].to(device)
            label1 = batch_data["label1"].to(device) # instance
            labels_onehot = monai.networks.one_hot(label1, 3) # (b,cls,256,256)
            # tif.imwrite(join("/disk1/neurips", 'label2.tiff'), batch_data["label2"][3,:,:,:].numpy())
            # tif.imwrite(join("/disk1/neurips", 'label3.tiff'), batch_data["label3"][3,:,:,:].numpy())


            label2 = batch_data["label2"].to(device) # binary
            label3 = batch_data["label3"].to(device) # boundary

            labels = torch.cat([labels_onehot, label2, label3], dim = 1)

            print('')
            print('Image ',inputs.shape)
            print('Label1 ', label1.shape)
            print('Labels', labels.shape)
            unlabeled_inputs = batch_data_unlabeled["img"].to(device)
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            outputs1 = model1(inputs)
            outputs1_soft = torch.softmax(outputs1, dim=1)

            outputs1_unlabeled = model1(unlabeled_inputs)
            outputs1_unlabeled_soft = torch.softmax(outputs1_unlabeled, dim=1)

            outputs2 = model2(inputs)
            outputs2_soft = torch.softmax(outputs2, dim=1)

            outputs2_unlabeled = model2(unlabeled_inputs)
            outputs2_unlabeled_soft = torch.softmax(outputs2_unlabeled, dim=1)


            #labels_onehot = labels
            # Lsup for both models
            # prova = labels_onehot[:].squeeze().long()
            # prova2 = labels_onehot[:].unsqueeze(1)
            
            #supervised_loss1 = 0.5 * \
            #    (ce_loss(outputs1, labels_onehot[:].long()) + dice_loss(outputs1_soft, labels_onehot[:]))
            #supervised_loss2 = 0.5 * \
            #    (ce_loss(outputs2, labels_onehot[:].long()) + dice_loss(outputs2_soft, labels_onehot[:]))
            
            # supervised_loss1 = 0.5 * \
            #     (ce_loss(outputs1, labels_onehot[:].squeeze().long()) + dice_loss(outputs1_soft, labels_onehot[:]))
            # supervised_loss2 = 0.5 * \
            #     (ce_loss(outputs2, labels_onehot[:].squeeze().long()) + dice_loss(outputs2_soft, labels_onehot[:]))

            loss_neigh_dist1 = l2_loss(outputs1[:, 3, :, :], labels[:, 3, :, :]) + l2_loss(outputs1[:, 4, :, :], labels[:, 4, :, :]) 

            loss_neigh_dist2 = l2_loss(outputs2[:, 3, :, :], labels[:, 3, :, :]) + l2_loss(outputs2[:, 4, :, :], labels[:, 4, :, :]) 

            loss_3class1 = ce_loss(outputs1[:, 0:3, :, :], labels_onehot.long()) \
            + dice_loss(outputs1_soft[:, 0:3, :, :], labels_onehot)

            loss_3class2 = ce_loss(outputs2[:, 0:3, :, :], labels_onehot.long()) \
            + dice_loss(outputs2_soft[:, 0:3, :, :], labels_onehot)


            supervised_loss1 = 0.5 * \
                (loss_neigh_dist1 + loss_3class1)
            
            supervised_loss2 = 0.5 * \
                (loss_neigh_dist2 + loss_3class2)

            pseudo_outputs1 = torch.argmax(
                outputs1_unlabeled_soft.detach()[:, 0:3, :, :], dim=1, keepdim=False)
            pseudo_outputs1 = pseudo_outputs1.unsqueeze(1)
            pseudo_outputs1 = monai.networks.one_hot(pseudo_outputs1, 3) # (b,cls,256,256)

            pseudo_outputs2 = torch.argmax(
                outputs2_unlabeled_soft.detach()[:, 0:3, :, :], dim=1, keepdim=False)
            pseudo_outputs2 = pseudo_outputs2.unsqueeze(1)
            pseudo_outputs2 = monai.networks.one_hot(pseudo_outputs2, 3) # (b,cls,256,256)

            #pseudo_l2_loss1 = l2_loss(outputs1_unlabeled_soft[:,3:,:,:], pseudo_outputs2[:,3:,:,:])
            #pseudo_l2_loss2 = l2_loss(outputs2_unlabeled_soft[:,3:,:,:], pseudo_outputs1[:,3:,:,:])

            pseudo_supervision1 = dice_loss(
                outputs1_unlabeled_soft[:,0:3,:,:], pseudo_outputs2
            )
            pseudo_supervision2 = dice_loss(
                outputs2_unlabeled_soft[:,0:3,:,:], pseudo_outputs1
            )

            # oppure metto la l2 come:
            pseudo_l2_loss = l2_loss(outputs1_unlabeled[:,3,:,:], outputs2_unlabeled[:,3,:,:]) + l2_loss(outputs1_unlabeled[:,4,:,:], outputs2_unlabeled[:,4,:,:])

            consistency_weight = get_current_consistency_weight(
                iter_num // (args.max_iterations/args.consistency_rampup))

            #model1_loss = supervised_loss1 + consistency_weight * (pseudo_supervision1 + pseudo_l2_loss1)
            #model2_loss = supervised_loss2 + consistency_weight * (pseudo_supervision2 + pseudo_l2_loss2)
            #print("Model 1 loss: {}".format(model1_loss))
            #print("Model 2 loss: {}".format(model2_loss))

            # Oppure
            model1_loss = supervised_loss1 + consistency_weight * (pseudo_supervision1 + pseudo_l2_loss)
            model2_loss = supervised_loss2 + consistency_weight * (pseudo_supervision2 + pseudo_l2_loss)
            print("Model 1 loss: {}".format(model1_loss))
            print("Model 2 loss: {}".format(model2_loss))
            
            loss = model1_loss + model2_loss
            print("Total loss: {}".format(loss))
            
            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = initial_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_
            
            '''writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))
            '''
            
            if iter_num > 0 and iter_num % 25 == 0:
                model1.eval()
                checkpoint1 = {'epoch': iter_num,
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'loss': epoch_loss_values1_1,
                }
                model2.eval()
                checkpoint2 = {'epoch': iter_num,
                'model_state_dict': model2.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'loss': epoch_loss_values2_1,
                }
                with torch.no_grad():
                    val_images = None
                    val_labels = None
                    val_outputs = None
                    f1_tot_1 = 0
                    f1_tot_2 = 0

                    for val_data in val_loader:
                        val_images, val_labels, val_instance  = val_data["img"].to(device), val_data["label1"].to(device), val_data["label4"].to(device)
                        val_labels_onehot = monai.networks.one_hot(val_labels, 3)
                        if args.input_size == 256:
                            roi_size = (256, 256)
                        elif args.input_size == 512:
                            roi_size = (512, 512)
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model1)
                        val_outputs1 = [post_pred(i) for i in decollate_batch(val_outputs[:, 0:3, :, :])]
                        val_labels_onehot = [post_gt(i) for i in decollate_batch(val_labels_onehot)]
                        val_labels_instance = [post_gt(i) for i in decollate_batch(val_instance)]
                        # compute metric for current iteration
                        print('3 class metric')
                        print(os.path.basename(val_data['img_meta_dict']['filename_or_obj'][0]), 
                            dice_metric(y_pred=val_outputs1, y=val_labels_onehot))
                        
                        val_pred_out = torch.nn.functional.softmax(val_outputs[:,0:3, :,:], dim=1) # (B, C, H, W)
                        val_pred_npy = val_pred_out[0,1].cpu().numpy()
                        val_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(val_pred_npy>0.5)),16)
                        print('Instance metric')
                        f1 = iou_F1(val_pred_mask, val_labels_instance)
                        f1_tot_1 = f1_tot_1 + f1
                    
                        print(os.path.basename(val_data['img_meta_dict']['filename_or_obj'][0]), f1)
                    
                    # aggregate the final mean dice result
                    metric1_1 = dice_metric.aggregate().item()
                    metric2_1 = f1_tot_1/len(val_files)
                    # reset the status for next validation round
                    dice_metric.reset()
                    metric_values1_1.append(metric1_1)
                    metric_values2_1.append(metric2_1)
                        
                    if metric1_1 > best_metric1_1:
                        best_metric1_1 = metric1_1
                        best_metric_epoch1_1 = iter_num
                        torch.save(checkpoint1, join(model_path, "best_Dice_model1.pth"))
                        print("saved new best metric model")
                    print(
                        "current epoch model 1: {}, current mean dice: {:.4f}, best mean dice: {:.4f} at epoch {}".format(
                            iter_num, metric1_1, best_metric1_1, best_metric_epoch1_1
                        )
                    )
                    #writer.add_scalar("val_mean_dice_1", metric1_1, iter_num)
                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    #plot_2d_or_3d_image(val_images, iter_num, writer, index=0, tag="image")
                    #plot_2d_or_3d_image(val_labels, iter_num, writer, index=0, tag="label")
                    #plot_2d_or_3d_image(val_outputs, iter_num, writer, index=0, tag="output")

                    if metric2_1 > best_metric2_1:
                        best_metric2_1 = metric2_1
                        best_metric_epoch2_1 = iter_num
                        torch.save(checkpoint1, join(model_path, "best_Loss_instance_model1.pth"))
                        print("saved new best instance model")
                        print(join(model_path, "best_Loss_instance_model1.pth"))
                    print(
                        "current epoch: {} current mean f1 score: {:.4f} best mean f1 score: {:.4f} at epoch {}".format(
                            iter_num, metric2_1, best_metric2_1, best_metric_epoch2_1
                        )
                    )
                    #writer.add_scalar("val_mean_f1", metric2_1, iter_num)
                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    #plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
                    #plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
                    #plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")

                    val_images = None
                    val_labels = None
                    val_outputs = None
                    for val_data in val_loader:
                        val_images, val_labels, val_instance  = val_data["img"].to(device), val_data["label1"].to(device), val_data["label4"].to(device)
                        val_labels_onehot = monai.networks.one_hot(val_labels, 3)
                        if args.input_size == 256:
                            roi_size = (256, 256)
                        elif args.input_size == 512:
                            roi_size = (512, 512)
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model2)
                        val_outputs1 = [post_pred(i) for i in decollate_batch(val_outputs[:, 0:3, :, :])]
                        val_labels_onehot = [post_gt(i) for i in decollate_batch(val_labels_onehot)]
                        val_labels_instance = [post_gt(i) for i in decollate_batch(val_instance)]
                        # compute metric for current iteration
                        print('3 class metric')
                        print(os.path.basename(val_data['img_meta_dict']['filename_or_obj'][0]), 
                            dice_metric(y_pred=val_outputs1, y=val_labels_onehot))
                        
                        val_pred_out = torch.nn.functional.softmax(val_outputs[:,0:3, :,:], dim=1) # (B, C, H, W)
                        val_pred_npy = val_pred_out[0,1].cpu().numpy()
                        val_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(val_pred_npy>0.5)),16)
                        print('Instance metric')
                        f1 = iou_F1(val_pred_mask, val_labels_instance)
                        f1_tot_2 = f1_tot_2 + f1
                    
                        print(os.path.basename(val_data['img_meta_dict']['filename_or_obj'][0]), f1)
                        
                    # aggregate the final mean dice result
                    metric1_2 = dice_metric.aggregate().item()
                    metric2_2 = f1_tot_2/len(val_files)
                    # reset the status for next validation round
                    dice_metric.reset()
                    metric_values1_2.append(metric1_2)
                    metric_values2_2.append(metric2_2)

                    if metric1_2 > best_metric1_2:
                        best_metric1_2 = metric1_2
                        best_metric_epoch1_2 = iter_num
                        torch.save(checkpoint2, join(model_path, "best_Dice_model2.pth"))
                        print("saved new best metric model")
                    print(
                        "current epoch model 2: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            iter_num, metric1_2, best_metric1_2, best_metric_epoch1_2
                        )
                    )
                    #writer.add_scalar("val_mean_dice_2", metric1_2, iter_num)
                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    #plot_2d_or_3d_image(val_images, iter_num, writer, index=0, tag="image")
                    #plot_2d_or_3d_image(val_labels, iter_num, writer, index=0, tag="label")
                    #plot_2d_or_3d_image(val_outputs, iter_num, writer, index=0, tag="output")

                    if metric2_2 > best_metric2_2:
                        best_metric2_2 = metric2_2
                        best_metric_epoch2_2 = iter_num
                        torch.save(checkpoint2, join(model_path, "best_Loss_instance_model2.pth"))
                        print("saved new best instance model")
                        print(join(model_path, "best_Loss_instance_model2.pth"))
                    print(
                        "current epoch: {} current mean f1 score: {:.4f} best mean f1 score: {:.4f} at epoch {}".format(
                            iter_num, metric2_2, best_metric2_2, best_metric_epoch2_2
                        )
                    )
                    #writer.add_scalar("val_mean_f1", metric2_2, iter_num)
                    # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                    #plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
                    #plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
                    #plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")
                model1.train()
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path1 = os.path.join(
                    model_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path1)
                logging.info("save model1 to {}".format(save_mode_path1))

                save_mode_path2 = os.path.join(
                    model_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path2)
                logging.info("save model2 to {}".format(save_mode_path2))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break


    print(f"train completed, best_metric1: {best_metric1_1:.4f} at epoch: {best_metric_epoch1_1}")
    print(f"train completed, best_metric1: {best_metric2_1:.4f} at epoch: {best_metric_epoch2_1}")
    print(f"train completed, best_metric2: {best_metric1_2:.4f} at epoch: {best_metric_epoch1_2}")
    print(f"train completed, best_metric2: {best_metric2_2:.4f} at epoch: {best_metric_epoch2_2}")
    #writer.close()
    
    #torch.save(checkpoint1, join(model_path, 'final_model1.pth'))
    #np.savez_compressed(join(model_path, 'train_log1.npz'), val_dice=metric_values1, epoch_loss=epoch_loss_values1)

    #torch.save(checkpoint2, join(model_path, 'final_model2.pth'))
    #np.savez_compressed(join(model_path, 'train_log2.npz'), val_dice=metric_values2, epoch_loss=epoch_loss_values2)


if __name__ == "__main__":
    main()