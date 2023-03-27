import gc
import numpy as np
import time
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from segmentation.training.losses import get_loss
from segmentation.utils.metrics import iou_pytorch


def train(net, datasets, configs, device, path_models):
    """ Train the model.

    :param net: Model/Network to train.
        :type net:
    :param datasets: Dictionary containing the training and the validation data set.
        :type datasets: dict
    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param path_models: Path to the directory to save the models.
        :type path_models: pathlib Path object
    :return: None
    """

    print('-' * 20)
    print('Train {0} on {1} images, validate on {2} images'.format(configs['run_name'],
                                                                   len(datasets['train']),
                                                                   len(datasets['val'])))

    # Data loader for training and validation set
    apply_shuffling = {'train': True, 'val': False}
    dataloader = {x: torch.utils.data.DataLoader(datasets[x],
                                                 batch_size=configs['batch_size'],
                                                 shuffle=apply_shuffling,
                                                 pin_memory=True,
                                                 num_workers=0)
                  for x in ['train', 'val']}

    # Loss function and optimizer
    criterion = get_loss(configs['loss'], configs['label_type'])

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=configs['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=True)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=configs['lr_patience'], verbose=True,
                                  min_lr=6e-5)

    # Auxiliary variables for training process
    epochs_wo_improvement, train_loss, val_loss, best_loss, train_iou, val_iou = 0, [], [], 1e4, [], []
    since = time.time()

    # Training process
    for epoch in range(configs['max_epochs']):

        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, configs['max_epochs']))
        print('-' * 10)

        start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluation mode

            running_loss, running_iou = 0.0, 0.0

            # Iterate over data
            for samples in dataloader[phase]:

                # Get img_batch and label_batch and put them on GPU if available
                if len(samples) == 2:
                    img_batch, label_batch = samples
                    img_batch, label_batch = img_batch.to(device), label_batch.to(device)
                elif len(samples) == 3:
                    img_batch, border_label_batch, cell_label_batch = samples
                    img_batch = img_batch.to(device)
                    cell_label_batch, border_label_batch = cell_label_batch.to(device), border_label_batch.to(device)
                elif len(samples) == 4:
                    img_batch, border_label_batch, cell_dist_label_batch, cell_label_batch = samples
                    img_batch = img_batch.to(device)
                    cell_label_batch = cell_label_batch.to(device)
                    border_label_batch = border_label_batch.to(device)
                    cell_dist_label_batch = cell_dist_label_batch.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):

                    if configs['label_type'] in ['boundary', 'border', 'pena']:
                        pred_batch = net(img_batch)
                        loss = criterion(pred_batch, label_batch)
                    elif configs['label_type'] == 'adapted_border':
                        border_pred_batch, cell_pred_batch = net(img_batch)
                        loss_border = criterion['border'](border_pred_batch, border_label_batch)
                        loss_cell = criterion['cell'](cell_pred_batch, cell_label_batch)
                        loss = 0.0075 * loss_border + loss_cell
                    elif configs['label_type'] == 'distance':
                        border_pred_batch, cell_pred_batch = net(img_batch)
                        loss_border = criterion['border'](border_pred_batch, border_label_batch)
                        loss_cell = criterion['cell'](cell_pred_batch, cell_label_batch)
                        loss = loss_border + loss_cell
                    elif configs['label_type'] == 'dual_unet':
                        border_pred_batch, cell_dist_pred_batch, cell_pred_batch = net(img_batch)
                        loss_border = criterion['border'](border_pred_batch, border_label_batch)
                        loss_cell_dist = criterion['cell_dist'](cell_dist_pred_batch, cell_dist_label_batch)
                        loss_cell = criterion['cell'](cell_pred_batch, cell_label_batch)
                        loss = 0.01 * loss_border + 0.01 * loss_cell + loss_cell_dist

                    # Backward (optimize only if in training phase)
                    if phase == 'train':

                        loss.backward()
                        optimizer.step()

                    # IoU metric
                    if configs['label_type'] in ['boundary', 'border', 'pena']:
                        iou = iou_pytorch(pred_batch[:, 1, :, :], label_batch == 1, device)
                    elif configs['label_type'] == 'adapted_border':
                        iou_border = iou_pytorch(border_pred_batch[:, 1, :, :], border_label_batch == 1, device)
                        iou_cells = iou_pytorch(cell_pred_batch, cell_label_batch, device)
                        iou = (iou_border + iou_cells) / 2
                    elif configs['label_type'] == 'distance':
                        iou_border = iou_pytorch(border_pred_batch,
                                                 border_label_batch > torch.tensor([0.5],
                                                                                   requires_grad=False).to(device),
                                                 device)
                        iou_cells = iou_pytorch(cell_pred_batch,
                                                cell_label_batch > torch.tensor([0.5],
                                                                                requires_grad=False).to(device),
                                                device)
                        iou = (iou_border + iou_cells) / 2
                    elif configs['label_type'] == 'dual_unet':
                        iou = iou_pytorch(cell_pred_batch, cell_label_batch, device)

                # Statistics
                running_loss += loss.item() * img_batch.size(0)
                running_iou += iou * img_batch.size(0)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_iou = running_iou / len(datasets[phase])

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_iou.append(epoch_iou)
                print('Training loss: {:.4f}, iou: {:.4f}'.format(epoch_loss, epoch_iou))
            else:
                val_loss.append(epoch_loss)
                val_iou.append(epoch_iou)
                print('Validation loss: {:.4f}, iou: {:.4f}'.format(epoch_loss, epoch_iou))

                scheduler.step(epoch_loss)

                if epoch_loss < best_loss:
                    print('Validation loss improved from {:.4f} to {:.4f}. Save model.'.format(best_loss, epoch_loss))
                    best_loss = epoch_loss

                    # The state dict of data parallel (multi GPU) models need to get saved in a way that allows to
                    # load them also on single GPU or CPU
                    if configs['num_gpus'] > 1:
                        torch.save(net.module.state_dict(), str(path_models / (configs['run_name'] + '.pth')))
                    else:
                        torch.save(net.state_dict(), str(path_models / (configs['run_name'] + '.pth')))
                    epochs_wo_improvement = 0

                else:
                    print('Validation loss did not improve.')
                    epochs_wo_improvement += 1

        # Epoch training time
        print('Epoch training time: {:.1f}s'.format(time.time() - start))

        # Break training if plateau is reached
        if epochs_wo_improvement == configs['break_condition']:
            print(str(epochs_wo_improvement) + ' epochs without validation loss improvement --> break')
            break

    # Total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 20)
    configs['training_time'], configs['trained_epochs'] = time_elapsed, epoch + 1

    # Save loss
    stats = np.transpose(np.array([list(range(1, len(train_loss) + 1)), train_loss, train_iou, val_loss, val_iou]))
    np.savetxt(fname=str(path_models / (configs['run_name'] + '_loss.txt')), X=stats,
               fmt=['%3i', '%2.5f', '%1.4f', '%2.5f', '%1.4f'],
               header='Epoch, training loss, training iou, validation loss, validation iou', delimiter=',')

    # Clear memory
    del net
    gc.collect()

    return None
