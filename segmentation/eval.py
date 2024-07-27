import copy
import glob
import os
import pickle
import random
import re
from collections import OrderedDict
import seaborn as sns
from monai.transforms import Resize

from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Sized,
                    Tuple, Union)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from report_guided_annotation import extract_lesion_candidates

from data_loader import (DataGenerator, Normalize, RandomFlip2D,
                         RandomRotate2D, To_Tensor, MultiLevelDataGenerator)
from segmentation.MedSAMAuto import MedSAMAUTO, MedSAMAUTOZONE, MedSAMAUTOMULTI, MedSAMAUTOCNN
from segmentation.config import FOLD_NUM, CURRENT_FOLD
from segmentation.model_single import ModelEmb, SegDecoderCNN
from segmentation.segment_anything import sam_model_registry
from segmentation.run import get_cross_validation_by_sample
from segmentation.segment_anything.modeling import TwoWayTransformer, MaskDecoder
from picai_eval import Metrics
from picai_eval.eval import evaluate_case

from segmentation.utils import compute_results_detect


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


seed_value = 42
set_seed(seed_value)


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

def plot_eval(net, val_path, ckpt_path, log_dir, device):
    epoch_pattern = re.compile(r'epoch:(\d+)-')

    # Initialize variables to keep track of the largest epoch and the corresponding file
    largest_epoch = -1
    ckpt_file = None

    # Iterate over all files in the directory
    for filename in os.listdir(ckpt_path):
        # Match the pattern to find the epoch number
        match = epoch_pattern.search(filename)
        if match:
            epoch = int(match.group(1))
            # Update the largest epoch and file if the current epoch is larger
            if epoch > largest_epoch:
                largest_epoch = epoch
                ckpt_file = filename

    ckpt_file = os.path.join(ckpt_path, ckpt_file)
    state_dict = torch.load(ckpt_file, map_location=device)['state_dict']
    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    plot_path = os.path.join(log_dir, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    val_transformer = transforms.Compose([
        Normalize(),
        # tio.CropOrPad(target_shape=(32, 128, 128)),
        To_Tensor(num_class=2, input_channel=3)
    ])

    val_dataset = DataGenerator(val_path, num_class=2, transform=val_transformer)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    count = 0
    with torch.no_grad():
        for step, sample in enumerate(val_loader):
            data = sample['image']
            target = sample['label'][:, 1].unsqueeze(1)

            data = data.cuda()
            target = target.cuda()
            with autocast(False):
                output = net(data)
                if isinstance(output, tuple):
                    output = output[0]
            plot_segmentation2D(data.squeeze(0).permute(1, 2, 0).detach().cpu(), output.squeeze(0)[0].detach().cpu(), target.squeeze(0)[0].detach().cpu(), plot_path, count)
            count += 1

def compute_dice(predict, target):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'

    dice = (2 * (predict * target).sum() + 1) / (predict.sum() + target.sum() + 1)

    return dice

def add_contour(original_image, mask_lesion, mask_pz, mask_cz, mask_gland, random_color=False, contour_thickness=7,):
    color1 = np.array([240, 128, 128, 0.9])  # Red
    color2 = np.array([144, 238, 144, 0.9])  # Green
    color3 = np.array([221, 160, 221, 0.9])  # Blue
    color4 = np.array([173, 216, 230, 0.9])  # yellow
    # Create a copy to avoid altering the original image
    image_copy = copy.copy(original_image).detach().cpu().numpy()
    image_copy = np.ascontiguousarray((image_copy * 255).astype(np.uint8))

    # Assuming mask is binary [0, 1], prepare for contour detection
    mask_uint8_lesion = (mask_lesion * 255).detach().cpu().numpy().astype(np.uint8)
    mask_uint8_cz = (mask_cz * 255).detach().cpu().numpy().astype(np.uint8)
    mask_uint8_pz = (mask_pz * 255).detach().cpu().numpy().astype(np.uint8)
    mask_uint8_gland = (mask_gland * 255).detach().cpu().numpy().astype(np.uint8)

    contours_lesion, _ = cv2.findContours(mask_uint8_lesion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_cz, _ = cv2.findContours(mask_uint8_cz, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_pz, _ = cv2.findContours(mask_uint8_pz, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_gland, _ = cv2.findContours(mask_uint8_gland, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours directly on the copied image
    cv2.drawContours(image_copy, contours_cz, -1, color2, contour_thickness)
    cv2.drawContours(image_copy, contours_pz, -1, color3, contour_thickness)
    cv2.drawContours(image_copy, contours_gland, -1, color4, contour_thickness)
    cv2.drawContours(image_copy, contours_lesion, -1, color1, contour_thickness)


    # alpha = 0.5  # Transparency factor
    # cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)
    return image_copy

def plot_segmentation2D_multilevel(img2D, lesion_prev_masks, zone_prev_masks, gland_prev_masks, lesion_gt2D, zone_gt2D, gland_gt2D, save_path, count, image_dice=None):
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
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
    axes[0].imshow(img2D[..., 0].unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy())
    axes[0].set_title('original image')
    image_pred = add_contour(img2D[..., 0].unsqueeze(-1).expand(-1, -1, 3), lesion_prev_masks, zone_prev_masks[0], zone_prev_masks[1], gland_prev_masks)
    image_gt = add_contour(img2D[..., 0].unsqueeze(-1).expand(-1, -1, 3), lesion_gt2D.squeeze(0), zone_gt2D[0], zone_gt2D[1], gland_gt2D.squeeze(0))
    lesion_dice = compute_dice(lesion_prev_masks.int(), lesion_gt2D[0])
    pz_dice = compute_dice(zone_prev_masks[0].int(), zone_gt2D[0])
    tz_dice = compute_dice(zone_prev_masks[1].int(), zone_gt2D[1])
    gland_dice = compute_dice(gland_prev_masks.int(), gland_gt2D[0])
    fig.suptitle(f'lesion_dice: {lesion_dice}, pz_dice: {pz_dice}, tz_dice: {tz_dice}, gland_dice: {gland_dice}')
    axes[1].imshow(image_pred)
    axes[1].set_title('predicted results')
    axes[2].imshow(image_gt)
    axes[2].set_title('ground truth')

    plt.savefig(os.path.join(save_path, f'slice_{count}'))

    return lesion_dice, pz_dice, tz_dice, gland_dice

    # # Iterate over each slice
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
    #
    #     # Plot image slice
    # ax = axes[0]
    # ax.imshow(img2D, cmap='gray')
    # ax.set_title(f'Image')
    # ax.axis('off')
    #
    #     # Plot previous mask slice
    # cmap = plt.cm.get_cmap('viridis', 2)
    # cmap.colors[0, 3] = 0
    # ax = axes[1]
    # ax.imshow(img2D, cmap='gray')
    # ax.imshow(prev_masks, cmap=cmap, alpha=0.5)
    # ax.set_title(f'Predict Mask')
    # ax.axis('off')
    #
    #     # Plot ground truth slice
    # cmap = plt.cm.get_cmap('viridis', 2)
    # cmap.colors[0, 3] = 0
    # ax = axes[2]
    # ax.imshow(img2D, cmap='gray')
    # ax.imshow(gt2D, cmap=cmap, alpha=0.5)
    # ax.set_title(f'Ground Truth')
    # ax.axis('off')
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, f'slice_{count}'))
    # plt.close()

def plot_segmentation3D_lesion(img3D, lesion_prev_masks, lesion_gt3D, lesion_ap, lesion_auc, save_path, count, image_dice=None):
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
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
    axes[0].imshow(img3D[..., 0].unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy())
    axes[0].set_title('original image')
    image_pred = add_contour(img3D[..., 0].unsqueeze(-1).expand(-1, -1, 3), lesion_prev_masks, torch.zeros_like(lesion_prev_masks), torch.zeros_like(lesion_prev_masks), torch.zeros_like(lesion_prev_masks))
    image_gt = add_contour(img3D[..., 0].unsqueeze(-1).expand(-1, -1, 3), lesion_gt2D.squeeze(0), torch.zeros_like(lesion_prev_masks), torch.zeros_like(lesion_prev_masks), torch.zeros_like(lesion_prev_masks))
    lesion_dice = compute_dice(lesion_prev_masks.int(), lesion_gt2D[0])
    fig.suptitle(f'lesion_dice: {lesion_dice}')
    axes[1].imshow(image_pred)
    axes[1].set_title('predicted results')
    axes[2].imshow(image_gt)
    axes[2].set_title('ground truth')

    plt.savefig(os.path.join(save_path, f'slice_{count}'))

    return lesion_dice

def search_ckpt_path(ckpt_path):
    epoch_pattern = re.compile(r'epoch:(\d+)-')

    # Initialize variables to keep track of the largest epoch and the corresponding file
    largest_epoch = -1
    ckpt_file = None

    # Iterate over all files in the directory
    for filename in os.listdir(ckpt_path):
        # Match the pattern to find the epoch number
        match = epoch_pattern.search(filename)
        if match:
            epoch = int(match.group(1))
            # Update the largest epoch and file if the current epoch is larger
            if epoch > largest_epoch:
                largest_epoch = epoch
                ckpt_file = filename

    return ckpt_file

def plot_eval_multi_level(net, val_path, ckpt_path, log_dir, device, activation):
    ckpt_file = os.path.join(ckpt_path, search_ckpt_path(ckpt_path))
    lesion_pid = pickle.load(open(os.path.join(PATH_DIR, '../lesion_pid.p'), 'rb'))
    # use zone_segdata_all for all data
    zone_pid = pickle.load(open('./dataset/zone_segdata_all/zone_pid.p', 'rb'))
    gland_pid = pickle.load(open('./dataset/gland_segdata/gland_pid.p', 'rb'))

    state_dict = torch.load(ckpt_file, map_location=device)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.` prefix
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    net = net.to(device)
    # net = DataParallel(net)
    plot_path = os.path.join(log_dir, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    val_transformer = transforms.Compose([
        Normalize(),
        # tio.CropOrPad(target_shape=(32, 128, 128)),
        To_Tensor(num_class=2, input_channel=3)
    ])

    val_dataset = MultiLevelDataGenerator(val_path, 'val', num_class=2, transform=val_transformer, zone_pid=zone_pid, gland_pid=gland_pid, lesion_pid=lesion_pid)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    count = 0

    dice_dict = {}
    lesion_results = []
    with torch.no_grad():
        for step, (sample, pid, slice) in enumerate(tqdm(val_loader)):
            lesion_targets = []
            zone_targets = []
            gland_targets = []
            for name, value in sample.items():
                if name == 'ct':
                    data = value
                elif 'lesion' in name:
                    lesion_targets.append(value)
                elif 'gland' in name:
                    gland_targets.append(value)
                elif 'zone' in name:
                    zone_targets.append(value)
            lesion_target = torch.stack(lesion_targets).permute(1, 0, 2, 3)
            zone_target = torch.stack(zone_targets).permute(1, 0, 2, 3)
            gland_target = torch.stack(gland_targets).permute(1, 0, 2, 3)

            data = data.to(device)
            lesion_target = lesion_target.to(device)
            zone_target = zone_target.to(device)
            gland_target = gland_target.to(device)

            with autocast(False):
                if activation:
                    logits = torch.sigmoid(net(data))
                else:
                    logits = net(data)
                output = logits > 0.5
                gland_output = output[:, 0]
                zone_output = output[:, 1:3]
                lesion_output = output[:, 3]
                # if isinstance(output, tuple):
                #     output = output[0]
            # multi_level_target = torch.cat([gland_target, zone_target, lesion_target], dim=1).permute(0, 2, 3, 1).detach().cpu().numpy()
            # preds = []
            # for slices in logits[:,-1,:,:].detach().cpu().numpy():
            #     preds.append(extract_lesion_candidates(np.expand_dims(slices, axis=-1), threshold=0.5)[0])
            # for y_det, y_true in zip(preds, [lesion_target[:, 0]]):
            #     y_list, *_ = evaluate_case(
            #         y_det=y_det,
            #         y_true=y_true.permute(1, 2, 0).detach().cpu().numpy(),
            #     )
            #
            #     # aggregate all validation evaluations
            #     lesion_results.append(y_list)
            lesion_dice, pz_dice, tz_dice, gland_dice = plot_segmentation2D_multilevel(data.squeeze(0).permute(1, 2, 0), lesion_output.squeeze(0), zone_output.squeeze(0), gland_output.squeeze(0), lesion_target.squeeze(0), zone_target.squeeze(0), gland_target.squeeze(0), log_dir, pid[0]+'-'+slice[0])
            dice_dict[pid[0] if isinstance(pid, list) else pid] = [lesion_dice, pz_dice, tz_dice, gland_dice]
            count += 1
    # lesion_results = {idx: result for idx, result in enumerate(lesion_results)}
    # valid_metrics = Metrics(lesion_results)
    # auc = valid_metrics.auroc
    # ap = valid_metrics.AP
    # score = valid_metrics.score
    # print(f'auc: {auc}, ap:{ap}, score: {score}')
    # os.system(f'cd {log_dir}')
    # os.system(f'touch result.txt')
    # os.system(f'echo "auc: {auc}, ap:{ap}, score: {score}" >> result.txt')

def plot_eval_detect(net, val_path, ckpt_path, log_dir, device, activation, mode='normal',):
    ckpt_file = os.path.join(ckpt_path, search_ckpt_path(ckpt_path))

    image_tsne = TSNE(n_components=2, random_state=42)
    dense_tsne = TSNE(n_components=2, random_state=42)

    state_dict = torch.load(ckpt_file, map_location=device)['state_dict']

    revert_transform = Resize((256, 256), mode='bilinear')
    seg_transform = Resize((256, 256), mode='nearest')

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.` prefix
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    net.to(device)
    net = DataParallel(net, device_ids=[0, 1, 2, 3, 4, 5])
    plot_path = os.path.join(log_dir, 'plots')
    os.makedirs(plot_path, exist_ok=True)

    class Normalize_2d(object):
        def __call__(self, sample):
            ct = sample['ct']
            seg = sample['seg']
            for i in range(ct.shape[0]):
                for j in range(ct.shape[1]):
                    if np.max(ct[i, j]) != 0:
                        ct[i, j] = ct[i, j] / np.max(ct[i, j])

            new_sample = {'ct': ct, 'seg': seg}
            return new_sample

    val_transformer = transforms.Compose(
        [Normalize_2d(), To_Tensor()])

    val_dataset = DataGenerator(val_path, transform=val_transformer)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=32,
        pin_memory=True
    )
    count = 0

    dice_dict = {}
    lesion_results = []
    image_embeddings = []
    dense_embeddings = []
    image_targets = []
    with torch.no_grad():
        for step, sample in enumerate(tqdm(val_loader)):
            data = sample['ct']
            target = sample['seg']

            data = data.squeeze().transpose(1, 0)
            data = data.to(device)
            target = target.to(device)
            if mode == 'normal':
                output = net(data)
            elif mode == 'viz_representation':
                output, image_embedding, dense_embedding = net(data)
                image_embeddings.append(
                    torch.mean(image_embedding.reshape(image_embedding.shape[0], image_embedding.shape[1], -1), dim=2))
                dense_embeddings.append(
                    torch.mean(dense_embedding.reshape(dense_embedding.shape[0], dense_embedding.shape[1], -1), dim=2))
                image_targets.append(torch.Tensor([this_target.max() > 0 for this_target in target.squeeze(0)]))
            if isinstance(output, tuple):
                output = output[0]

            output = output.float()
            if activation:
                output = torch.sigmoid(output)  # N*H*W
            output = output.detach().cpu()
            lesion_output = output[:, -1, :, :].unsqueeze(1)
            lesion_output = torch.from_numpy(np.array([revert_transform(slice) for slice in lesion_output])).squeeze(1)
            target = target.detach().cpu()
            target = target[0].unsqueeze(1)
            target = torch.from_numpy(np.array([seg_transform(slice) for slice in target])).squeeze(1)
            # if plot:
            #     for
            #     plot_segmentation2D()


            lesion_results = compute_results_detect(lesion_output.numpy(), target.numpy(),
                                                    lesion_results)
            # if step > 2:
            #     break

    if mode == 'viz_representation':
        image_embeddings = torch.cat(image_embeddings)
        dense_embeddings = torch.cat(dense_embeddings)
        image_targets = torch.cat(image_targets)
        image_embeddings_2d = image_tsne.fit_transform(image_embeddings.detach().cpu().numpy())
        dense_embeddings_2d = dense_tsne.fit_transform(dense_embeddings.detach().cpu().numpy())

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plot Embeddings1
        axes[0].scatter(image_embeddings_2d[image_targets == 0, 0], image_embeddings_2d[image_targets == 0, 1], label='no lesion present', alpha=0.6)
        axes[0].scatter(image_embeddings_2d[image_targets == 1, 0], image_embeddings_2d[image_targets == 1, 1], label='lesion present', alpha=0.6)
        axes[0].set_title('2D Visualization of Image Embeddings')
        axes[0].set_xlabel('TSNE Component 1')
        axes[0].set_ylabel('TSNE Component 2')

        # Plot Embeddings2
        axes[1].scatter(dense_embeddings_2d[image_targets == 0, 0], dense_embeddings_2d[image_targets == 0, 1],
                        label='no lesion present', alpha=0.6)
        axes[1].scatter(dense_embeddings_2d[image_targets == 1, 0], dense_embeddings_2d[image_targets == 1, 1],
                        label='lesion present', alpha=0.6)
        axes[1].set_title('2D Visualization of Dense Embeddings')
        axes[1].set_xlabel('TSNE Component 1')
        axes[1].set_ylabel('TSNE Component 2')

        # Show the plot
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{mode}.png'))

    lesion_results = {idx: result for idx, result in enumerate(lesion_results)}
    FP_patient_level = 0
    TP_patient_level = 0
    FN_patient_level = 0
    TN_patient_level = 0
    lesion_results_list = []
    for idx, lesion_result in lesion_results.items():
        this_FP_patient_level = 0
        this_TP_patient_level = 0
        this_FN_patient_level = 0
        this_TN_patient_level = 0
        if len(lesion_result) == 0:
            this_TN_patient_level = 1
        for this_lesion_result in lesion_result:
            if this_lesion_result[0] == 0 and this_lesion_result[2] == 0:
                this_FP_patient_level = 1
                this_TP_patient_level = 0
                this_FN_patient_level = 0
                this_TN_patient_level = 0
                break
            elif this_lesion_result[0] == 1 and this_lesion_result[2] != 0:
                this_TP_patient_level = 1
            elif this_lesion_result[0] == 1 and this_lesion_result[2] == 0:
                this_FN_patient_level = 1
            else:
                raise NotImplementedError
        for this_lesion_result in lesion_result:
            lesion_results_list.append(this_lesion_result)
        if this_TP_patient_level == 1 and this_FN_patient_level == 1:
            print('TP and FN coexist')
        FP_patient_level += this_FP_patient_level
        TP_patient_level += this_TP_patient_level
        FN_patient_level += this_FN_patient_level
        TN_patient_level += this_TN_patient_level
    FP_lesion_level = 0
    TP_lesion_level = 0
    FN_lesion_level = 0
    TN_lesion_level = 0
    for lesion_result_tuple in lesion_results_list:
        if lesion_result_tuple[0] == 0 and lesion_result_tuple[2] == 0:
            FP_lesion_level += 1
        elif lesion_result_tuple[0] == 1 and lesion_result_tuple[2] != 0:
            TP_lesion_level += 1
        elif lesion_result_tuple[0] == 1 and lesion_result_tuple[2] == 0:
            FN_lesion_level += 1
        else:
            raise NotImplementedError
    conf_matrix_lesion_level = np.array([[TN_lesion_level, FP_lesion_level], [FN_lesion_level, TP_lesion_level]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_lesion_level, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Lesion Level Confusion Matrix')
    plt.savefig(os.path.join(log_dir, 'Lesion Level Confusion Matrix.png'))

    conf_matrix_patient_level = np.array([[TN_patient_level, FP_patient_level], [FN_patient_level, TP_patient_level]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_patient_level, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Patient Level Confusion Matrix')
    plt.savefig(os.path.join(log_dir, 'Patient Level Confusion Matrix.png'))

    valid_metrics = Metrics(lesion_results)
    auc = valid_metrics.auroc
    ap = valid_metrics.AP
    score = valid_metrics.score
    print(f'auc: {auc}, ap:{ap}, score: {score}')
    os.system(f'cd {log_dir}')
    os.system(f'touch result.txt')
    os.system(f'echo "auc: {auc}, ap:{ap}, score: {score}" >> result.txt')





if __name__ == '__main__':
    PATH_DIR = './dataset/lesion_segdata_human_all/data_2d'
    PATH_LIST = glob.glob(os.path.join(PATH_DIR, '*.hdf5'))
    train_path, val_path = get_cross_validation_by_sample(PATH_LIST, FOLD_NUM, 1)

    # PATH_AP = './dataset/lesion_segdata_human_all/data_3d'
    # AP_LIST = glob.glob(os.path.join(PATH_AP, '*.hdf5'))
    # train_AP, val_AP = get_cross_validation_by_sample(AP_LIST, FOLD_NUM, 1)

    mode = 'normal'

    activation = True

    # net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #                                        in_channels=3, out_channels=4, init_features=32, pretrained=False)

    sam_model = sam_model_registry['vit_b'](checkpoint='medsam_vit_b.pth')
    dense_model = ModelEmb()
    multi_mask_decoder = MaskDecoder(
        num_multimask_outputs=4,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    net = MedSAMAUTOMULTI(
        image_encoder=sam_model.image_encoder,
        mask_decoder=multi_mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        dense_encoder=dense_model,
        image_size=512,
        mode=mode
    )

    # mask_decoder_model = SegDecoderCNN(num_classes=4, num_depth=4)
    #
    # net = MedSAMAUTOCNN(
    #     image_encoder=sam_model.image_encoder,
    #     mask_decoder=mask_decoder_model,
    #     prompt_encoder=sam_model.prompt_encoder,
    #     dense_encoder=None,
    #     image_size=512
    # )

    PHASE = 'seg'

    ckpt_path = './new_ckpt/{}/{}/fold1'.format('seg','MedSAMAuto_Unified_equal_rate_lr_0.0001_weight_decay_0.001')
    # ckpt_path = './new_ckpt/{}/{}/fold1'.format('seg', 'UNet_Unified_equal_rate_lr_0.0001_weight_decay_0.001')

    log_dir = './new_log/eval/MedSAM3LevelALLDataEqualRateUpdate'
    # log_dir = './new_log/eval/UNet3LevelALLDataEqualRate'
    if PHASE == 'seg':
        plot_eval_multi_level(net, val_path, ckpt_path, log_dir, 'cuda:0', activation)
    else:
        plot_eval_detect(net, val_AP, ckpt_path, log_dir, 'cuda:0', activation, mode)