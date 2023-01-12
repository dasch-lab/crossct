
import os
join = os.path.join
import argparse
import numpy as np
import torch
import monai
from monai.inferers import sliding_window_inference
from models.unetr2d import UNETR2D
import time
from skimage import io, segmentation, morphology, measure, exposure
import tifffile as tif
from PIL import Image
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def boundary_postprocessing(prediction, input_3d=False):
    """ Post-processing for boundary label prediction.

    :param prediction: Boundary label prediction.
        :type prediction:
    :param input_3d: True (3D data), False (2D data).
        :type input_3d: bool
    :return: Instance segmentation mask, binary raw prediction (0: background, 1: cell, 2: boundary).
    """

    # Binarize the channels
    prediction_bin = np.argmax(prediction, axis=0).astype(np.uint16)

    # Get mask to flood with watershed
    mask = (prediction_bin == 1)  # only interior cell class belongs to cells

    # Get seeds for marker-based watershed
    if input_3d:
        seeds = (prediction[:, :, :, 1] * (1 - prediction[:, :, :, 2])) > 0.5
    else:
        seeds = (prediction[1, :, :] * (1 - prediction[2, :, :])) > 0.5
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

def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='/disk1/neurips/dataset/validation/images', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='/disk1/neurips/baseline/outputs/ct_unet_0.3_0.1_batch64_Adam_256_ctc_int16_float32_pretrained_baseline_2', type=str, help='output path')
    parser.add_argument('--model_path', default='/disk1/neurips/baseline/work_dir/ct_unet_swinunetr_batch_64_patch_256_val_frac_0.3_lr_0.01Adam_ctc_int16_float32_pretrained_baseline_2', help='path where to save models and segmentation results')
    #parser.add_argument('--model_path', default='./work_dir/_New_Preprocessing_DiceCE_swinunetr_3class', help='path where to save models and segmentation results')
    parser.add_argument('--show_overlay', default=False, action="store_true", help='save segmentation overlay')

    # Model parameters
    parser.add_argument('--model_name', default='unet', help='select mode: unet, unetr, swinunetr')
    parser.add_argument('--num_class', default=5, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=256, type=int, help='segmentation classes') # 256
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    img_names = sorted(os.listdir(join(input_path)))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.input_size == 256:
        chs = (16, 32, 64, 128, 256)
        sts = (2, 2, 2, 2)
    elif args.input_size == 512:
        chs = (16, 32, 64, 128, 256, 512)
        sts = (2, 2, 2, 2, 2)
    if args.model_name.lower() == 'unet':
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=args.num_class,
            channels=chs,
            strides=sts,
            num_res_units=2,
        ).to(device)


    if args.model_name.lower() == 'unetr':
        model = UNETR2D(
            in_channels=3,
            out_channels=args.num_class,
            img_size=(args.input_size, args.input_size),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)


    if args.model_name.lower() == 'swinunetr':
        model = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size), 
            in_channels=3, 
            out_channels=args.num_class,
            feature_size=24, # should be divisible by 12
            spatial_dims=2
            ).to(device)

    if args.model_name == "unet":
        #checkpoint = torch.load(join(args.model_path, 'best_Dice_model1.pth'), map_location=torch.device(device))
        checkpoint = torch.load(join(args.model_path, 'best_Loss_instance_model1.pth'), map_location=torch.device(device))
    elif args.model_name == "swinunetr":
        #checkpoint = torch.load(join(args.model_path, 'best_Dice_model2.pth'), map_location=torch.device(device))
        checkpoint = torch.load(join(args.model_path, 'best_Loss_instance_model2.pth'), map_location=torch.device(device))

    model.load_state_dict(checkpoint['model_state_dict'])
    #from torchsummary import summary
    #summary(model, input_size=(3, args.input_size, args.input_size))

    #from thop import profile
    #input = torch.randn(1,3, 256, 256).to(device)
    #macs, params = profile(model, inputs=(input, ))
    #print(macs)
    #print(params)


    #%%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4
    model.eval()
    with torch.no_grad():
        for img_name in img_names:
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(input_path, img_name))
            else:
                img_data = io.imread(join(input_path, img_name))
            
            # normalize image data
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:,:, :3]
            else:
                pass
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
            
            t0 = time.time()
            test_npy01 = pre_img_data/np.max(pre_img_data)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
            test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)
            test_pred_out_3classes = torch.nn.functional.softmax(test_pred_out[:, 0:3, :, :], dim=1) # (B, C, H, W)
            test_pred_npy_3classes = test_pred_out_3classes[0,1].cpu().numpy()
            test_pred_out_dist_neighbour = test_pred_out[:, 3:, :, :]
            test_pred_npy_dist_neighbour = test_pred_out_dist_neighbour.cpu().numpy()
            test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(test_pred_npy_3classes>0.5)),16)

            tif.imwrite(join(output_path, img_name.split('.')[0]+'_label.tiff'), test_pred_mask)
            #tif.imwrite(join(output_path, img_name.split('.')[0]+'_distance.tiff'), test_pred_npy_dist_neighbour[:,0,:,:])
            #tif.imwrite(join(output_path, img_name.split('.')[0]+'_neighbour.tiff'), test_pred_npy_dist_neighbour[:,1,:,:])
            #de = distance_postprocessing(test_pred_npy_dist_neighbour[:,1,:,:], test_pred_npy_dist_neighbour[:,0,:,:])
            #tif.imwrite(join(output_path, img_name.split('.')[0]+'_instance.tiff'), np.squeeze(de))

            

            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {pre_img_data.shape}; costing: {t1-t0:.2f}s')
            
            if args.show_overlay:
                boundary = segmentation.find_boundaries(test_pred_mask, connectivity=1, mode='inner')
                boundary = morphology.binary_dilation(boundary, morphology.disk(2))
                img_data[boundary, :] = 255
                io.imsave(join(output_path, 'overlay_' + img_name), img_data, check_contrast=False)
            
        
if __name__ == "__main__":
    main()





