import os
import random
from pathlib import Path
from typing import Union
import cv2
from torchvision import transforms

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from picai_eval.eval import evaluate_case
from skimage.metrics import hausdorff_distance

from eval_utils import extract_lesion_candidates
from scipy.spatial.distance import cdist
from enum import Enum

class ModelName(Enum):
    medsam = 'MedSAMAuto'
    swin_unetr = 'Swin-UNETR'
    unet = 'UNet'
    itunet = 'ITUNet'
    samcnn = 'SAMCNN'
    masam = 'MASAM'
    samed = 'SAMed'


def calculate_max_tumor_distance(mask, spacing):
    """
    Calculate the maximum Euclidean distance between tumor voxels in a 3D MRI mask.

    Parameters:
    - mask (np.ndarray): 3D binary mask where tumor voxels are labeled as 1.
    - spacing (list or tuple): The voxel spacing for the z, y, and x axes in mm, e.g., [3.0, 0.5, 0.5].

    Returns:
    - max_distance (float): The maximum Euclidean distance between any two tumor voxels in mm.
    """

    # Find the coordinates of all tumor voxels (where mask == 1)
    tumor_voxel_coords = np.argwhere(mask == 1)

    # If no tumor is found, return zero
    if len(tumor_voxel_coords) == 0:
        return 0

    # Convert voxel coordinates to real-world coordinates by applying the spacing
    real_world_coords = np.multiply(tumor_voxel_coords, spacing)

    # Calculate the pairwise Euclidean distances between all tumor voxels
    distances = cdist(real_world_coords, real_world_coords, metric='euclidean')

    # Find and return the maximum distance
    max_distance = np.max(distances)

    return max_distance



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
    cmap = plt.cm.get_cmap('viridis', 2)
    cmap.colors[0, 3] = 0
    ax = axes[1]
    ax.imshow(img2D, cmap='gray')
    ax.imshow(prev_masks, cmap=cmap, alpha=0.5)
    ax.set_title(f'Predict Mask')
    ax.axis('off')

        # Plot ground truth slice
    cmap = plt.cm.get_cmap('viridis', 2)
    cmap.colors[0, 3] = 0
    ax = axes[2]
    ax.imshow(img2D, cmap='gray')
    ax.imshow(gt2D, cmap=cmap, alpha=0.5)
    ax.set_title(f'Ground Truth')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'slice_{count}'))
    plt.close()

def compute_results_detect(logits, target, gland_output, results, threshold, post_process):
    preds = []
    logits = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
    preds.append(extract_lesion_candidates(logits, gland_output, threshold=threshold, post_process=post_process)[0])
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

class Normalize_2d(object):
    def __call__(self, sample):
        new_sample = {}
        for key, value in sample.items():
            if key == 'ct':
                ct = value
                if isinstance(ct, torch.Tensor):
                    ct = ct.numpy()
                for i in range(ct.shape[0]):
                    for j in range(ct.shape[1]):
                        if np.max(ct[i, j]) != 0:
                            ct[i, j] = ct[i, j] / np.max(ct[i, j])
                new_sample[key] = ct
            else:
                new_sample[key] = value
        return new_sample

class Resize_2d(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, sample):
        new_sample = {}
        transform = transforms.Resize(size=self.size)
        seg_transform = transforms.Resize(size=self.size,
                                          interpolation=transforms.functional.InterpolationMode.NEAREST)
        for key, value in sample.items():
            if key == 'ct':
                ct = value
                if isinstance(ct, np.ndarray):
                    ct = torch.tensor(ct, dtype=torch.float32)
                new_ct = []
                for j in range(ct.shape[1]):
                    new_ct.append(transform(ct[:, j]))
                new_sample[key] = torch.stack(new_ct, dim=1)
            else:
                seg = value
                if isinstance(seg, np.ndarray):
                    seg = torch.tensor(seg, dtype=torch.uint8)
                new_seg = []
                for j in range(seg.shape[1]):
                    new_seg.append(seg_transform(seg[:, j]))
                new_sample[key] = torch.stack(new_seg, dim=1)
        return new_sample

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