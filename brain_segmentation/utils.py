import os
import random
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from picai_eval.eval import evaluate_case
from report_guided_annotation import extract_lesion_candidates
from skimage.metrics import hausdorff_distance
import torch.nn.functional as F


def one_hot_encode(tensor, num_classes):
    """
    Convert a tensor of class indices to one-hot encoded format.

    :param tensor: Tensor of shape (N, H, W) with class indices.
    :param num_classes: Number of classes.
    :return: One-hot encoded tensor of shape (N, num_classes, H, W).
    """
    assert tensor.dtype == torch.long, "Target tensor must be of type torch.long"
    return F.one_hot(tensor, num_classes).permute(0, 3, 1, 2).float()


def dice_score_per_class(preds, targets, num_classes, smooth=1.0):
    """
    Compute the Dice score for each class.

    :param preds: Predicted logits of shape (N, C, H, W).
    :param targets: Ground truth labels of shape (N, H, W).
    :param num_classes: Number of classes.
    :param smooth: Smoothing factor to avoid division by zero.
    :return: Tensor of Dice scores for each class.
    """
    # Convert predictions to class indices
    preds = torch.argmax(preds, dim=1)

    # One-hot encode the predictions and targets
    preds_one_hot = one_hot_encode(preds, num_classes)
    targets_one_hot = one_hot_encode(targets, num_classes)

    # Compute the intersection and union for each class
    intersection = (preds_one_hot * targets_one_hot).sum(dim=(2, 3))
    union = preds_one_hot.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

    # Compute the Dice score for each class
    dice_scores = (2.0 * intersection + smooth) / (union + smooth)

    return dice_scores

def dice_score_per_class_3d(preds, targets, num_classes, smooth=1.0):
    """
    Compute the Dice score for each class.

    :param preds: Predicted logits of shape (N, C, H, W).
    :param targets: Ground truth labels of shape (N, H, W).
    :param num_classes: Number of classes.
    :param smooth: Smoothing factor to avoid division by zero.
    :return: Tensor of Dice scores for each class.
    """
    # Convert predictions to class indices
    preds = torch.argmax(preds, dim=1)

    # One-hot encode the predictions and targets
    preds_one_hot = one_hot_encode(preds, num_classes)
    targets_one_hot = one_hot_encode(targets, num_classes)

    # Compute the intersection and union for each class
    intersection = (preds_one_hot * targets_one_hot).sum(dim=(0, 2, 3))
    union = preds_one_hot.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

    # Compute the Dice score for each class
    dice_scores = (2.0 * intersection + smooth) / (union + smooth)

    return dice_scores

def transform_mask_to_single_channel(mask_3d):
    # Create the single-channel mask by combining the three channels
    mask_1d = mask_3d[0, ...] * 1 + mask_3d[1, ...] * 2 + mask_3d[2, ...] * 3
    return mask_1d

def plot_segmentation2D(img2D, prev_masks, gt2D, save_path, count, image_dice=None):
    """
        Plot each slice of a 3D image, its corresponding previous mask, and ground truth mask.

        Parameters:
        img3D (numpy.ndarray): The 3D image array of shape (depth, height, width).
        prev_masks (numpy.ndarray): The 3D array of previous masks of shape (depth, height, width).
        gt3D (numpy.ndarray): The 3D array of ground truth masks of shape (depth, height, width).
        slice_axis (int): The axis along which to slice the image (0=depth, 1=height, 2=width).
        """
    os.makedirs(save_path, exist_ok=True)
    # Determine the number of slices based on the selected axis

    # Iterate over each slice
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))

        # Plot image slice
    ax = axes[0]
    ax.imshow(img2D, cmap='gray')
    ax.set_title(f'Image')
    ax.axis('off')

        # Plot previous mask slice
    cmap = plt.cm.get_cmap('viridis', 4)
    cmap.colors[0, 3] = 0
    ax = axes[1]
    ax.imshow(img2D, cmap='gray')
    ax.imshow(prev_masks, cmap=cmap, alpha=0.5)
    ax.set_title(f'Predict Mask')
    ax.axis('off')

        # Plot ground truth slice
    cmap = plt.cm.get_cmap('viridis', 4)
    cmap.colors[0, 3] = 0
    ax = axes[2]
    ax.imshow(img2D, cmap='gray')
    ax.imshow(gt2D, cmap=cmap, alpha=0.5)
    ax.set_title(f'Ground Truth')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{count}.png'))
    plt.close()
def compute_results_detect(logits, target, results):
    preds = []
    logits = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
    preds.append(extract_lesion_candidates(logits, threshold=0.5)[0])
    for y_det, y_true in zip(preds,
                             [target]):
        y_list, *_ = evaluate_case(
            y_det=y_det,
            y_true=y_true,
        )

        # aggregate all validation evaluations
        results.append(y_list)
    return results

def binary_dice(y_true, y_pred):
    smooth = 1e-7
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def multi_dice(y_true,y_pred,num_classes):
    dice_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        dice = binary_dice(true,pred)
        dice_list.append(dice)
    
    dice_list = [round(case, 4) for case in dice_list]
    
    return dice_list, round(np.mean(dice_list),4)


def hd_2d(true,pred):
    hd_list = []
    for i in range(true.shape[0]):
        if np.sum(true[i]) != 0 and np.sum(pred[i]) != 0:
            hd_list.append(hausdorff_distance(true[i],pred[i]))
    
    return np.mean(hd_list)

def multi_hd(y_true,y_pred,num_classes):
    hd_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        hd = hd_2d(true,pred)
        hd_list.append(hd)
    
    hd_list = [round(case, 4) for case in hd_list]
    
    return hd_list, round(np.mean(hd_list),4)


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image



def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print('%.3f GFLOPs' %(macs/10e9))
    print('%.3f M' % (params/10e6))



def get_weight_path(
    ckpt_path: Union[Path, str]
):

    if os.path.isfile(ckpt_path):
        return ckpt_path
    elif os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        if os.path.exists(str(ckpt_path) + ".pth"):
            return str(ckpt_path) + ".pth"
        return None

def get_weight_list(ckpt_path,choice=None):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if choice is not None and eval(str(fold.name)[-1]) not in choice:
            continue
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
            # print(os.path.join(fold.path,weight_path[-1]))
    return path_list


def remove_weight_path(ckpt_path,retain=3):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=5):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain=retain)
        else:
            remove_weight_path(ckpt_path,retain=retain)
            break  

def poly_lr(epoch, max_epochs,ck_epoch = 0, initial_lr = 1e-2, exponent=0.9):
    return initial_lr * (1 - (epoch - ck_epoch) / (max_epochs - ck_epoch))**exponent


def get_cross_validation_by_sample(path_list, fold_num, current_fold):

    sample_list = list(set([os.path.basename(case).split('_')[0] for case in path_list]))
    sample_list.sort()
    print(sample_list)
    print('number of sample:',len(sample_list))
    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])

    train_path = []
    validation_path = []
    for case in path_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length ", len(train_path),
          "Val set length", len(validation_path))
    return train_path, validation_path

if __name__ == "__main__":

    ckpt_path = './new_ckpt/Cervical/2d/v1.0'
    dfs_remove_weight(ckpt_path)