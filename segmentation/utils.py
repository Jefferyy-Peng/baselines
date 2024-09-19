import os
import random
from pathlib import Path
from typing import Union
import cv2

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from picai_eval.eval import evaluate_case
from skimage.metrics import hausdorff_distance
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral,\
         create_pairwise_gaussian, softmax_to_unary, unary_from_softmax

from eval_utils import extract_lesion_candidates
from scipy.spatial.distance import cdist


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

def get_crf_img(inputs, outputs):
    for i in range(outputs.shape[0]):
        img = inputs[i]
        softmax_prob = outputs[i]
        unary = unary_from_softmax(softmax_prob)
        unary = np.ascontiguousarray(unary)
        d = dcrf.DenseCRF(img.shape[0] * img.shape[1], 2)
        d.setUnaryEnergy(unary)
        feats = create_pairwise_gaussian(sdims=(10,10), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = create_pairwise_bilateral(sdims=(50,50), schan=(20,20,20),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)
        res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
        if i == 0:
            crf = np.expand_dims(res,axis=0)
        else:
            res = np.expand_dims(res,axis=0)
            crf = np.concatenate((crf,res),axis=0)
    return crf

def post_process(output_root, inputs, outputs, input_path=None,
                 crf_flag=True, erode_dilate_flag=True,
                 save=True, overlap=True):
    inputs = (np.array(inputs.squeeze()).astype(np.float32)) * 255
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np.concatenate((inputs,inputs,inputs), axis=3)
    outputs = np.array(outputs)

    # Conditional Random Field
    if crf_flag:
        outputs = get_crf_img(inputs, outputs)
    else:
        outputs = outputs.argmax(1)

    # Erosion and Dilation
    if erode_dilate_flag:
        outputs = erode_dilate(outputs, kernel_size=7)
    if save == False:
        return outputs

    outputs = outputs*255
    for i in range(outputs.shape[0]):
        path = input_path[i].split('/')
        output_folder = os.path.join(output_root, path[-2])
        try:
            os.mkdir(output_folder)
        except:
            pass
        output_path = os.path.join(output_folder, path[-1])
        if overlap:
            img = outputs[i]
            img = np.expand_dims(img, axis=2)
            zeros = np.zeros(img.shape)
            img = np.concatenate((zeros,zeros,img), axis=2)
            img = np.array(img).astype(np.float32)
            img = inputs[i] + img
            if img.max() > 0:
                img = (img/img.max())*255
            else:
                img = (img/1) * 255
            cv2.imwrite(output_path, img)
        else:
            img = outputs[i]
            cv2.imwrite(output_path, img)
    return None

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

def compute_results_detect(logits, target, gland_output, results):
    preds = []
    logits = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
    preds.append(extract_lesion_candidates(logits, gland_output, threshold=0.75)[0])
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