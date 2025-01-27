import copy
import glob
import os
import pickle
import random
import re
from collections import OrderedDict
import seaborn as sns
from einops import rearrange
from monai.networks.nets import SwinUNETR
from monai.transforms import Resize, ScaleIntensityD, ResizeD, ToTensorD

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

from data_loader import (DataGenerator, Normalize, RandomFlip2D,
                         RandomRotate2D, To_Tensor, MultiLevelDataGenerator, DataGenerator_no_resize,
                         MultiLevel3DDataGenerator)
from segmentation.MedSAMAuto import MedSAMAUTO, MedSAMAUTOZONE, MedSAMAUTOMULTI, MedSAMAUTOCNN, TextEncoder, \
    MedSAMAUTOMULTIALIGNTYPE2FINE, MedSAMAUTOMULTIALIGNTYPE1
from segmentation.config import FOLD_NUM, CURRENT_FOLD
from segmentation.data_loader_new import MultiLevelDataGeneratorAlignDummy
from segmentation.lora_image_encoder import LoRA_Sam
from segmentation.model import itunet_2d
from segmentation.model_single import ModelEmb, SegDecoderCNN
from segmentation.sam_fact_tt_image_encoder import Fact_tt_Sam
from segmentation.segment_anything import sam_model_registry
from segmentation.run import get_cross_validation_by_sample
from segmentation.segment_anything.modeling import TwoWayTransformer, MaskDecoder
from picai_eval import Metrics
from picai_eval.eval import evaluate_case

from segmentation.segment_anything_from_MASAM import sam_model_registry_MASAM
from segmentation.segment_anything_from_SAMed import sam_model_registry_SAMed
from segmentation.utils import compute_results_detect, ModelName, Normalize_2d, Resize_2d
from segmentation.eval_utils import erode_dilate, search_ckpt_path
from vit import VisionTransformer


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

def plot_segmentation2D_multilevel(img2D, lesion_prev_masks, zone_prev_masks, gland_prev_masks, lesion_gt2D, zone_gt2D, gland_gt2D, count, image_dice=None):
    """
        Plot each slice of a 3D image, its corresponding previous mask, and ground truth mask.

        Parameters:
        img3D (numpy.ndarray): The 3D image array of shape (depth, height, width).
        prev_masks (numpy.ndarray): The 3D array of previous masks of shape (depth, height, width).
        gt3D (numpy.ndarray): The 3D array of ground truth masks of shape (depth, height, width).
        slice_axis (int): The axis along which to slice the image (0=depth, 1=height, 2=width).
        """
    # os.makedirs(save_path, exist_ok=True)
    # Determine the number of slices based on the selected axis
    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
    # axes[0, 0].imshow(img2D[..., 0].unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy())
    # axes[0, 0].set_title('channel 0')
    # image_pred = add_contour(img2D[..., 0].unsqueeze(-1).expand(-1, -1, 3), lesion_prev_masks, zone_prev_masks[0], zone_prev_masks[1], gland_prev_masks)
    # image_gt = add_contour(img2D[..., 0].unsqueeze(-1).expand(-1, -1, 3), lesion_gt2D.squeeze(0), zone_gt2D[0], zone_gt2D[1], gland_gt2D.squeeze(0))
    lesion_dice = compute_dice(lesion_prev_masks.int(), lesion_gt2D[0])
    pz_dice = compute_dice(zone_prev_masks[0].int(), zone_gt2D[0])
    tz_dice = compute_dice(zone_prev_masks[1].int(), zone_gt2D[1])
    gland_dice = compute_dice(gland_prev_masks.int(), gland_gt2D[0])
    # fig.suptitle(f'lesion_dice: {lesion_dice}, pz_dice: {pz_dice}, tz_dice: {tz_dice}, gland_dice: {gland_dice}')
    # axes[0, 1].imshow(image_pred)
    # axes[0, 1].set_title('predicted results')
    # axes[0, 2].imshow(image_gt)
    # axes[0, 2].set_title('ground truth')
    # 
    # axes[1, 0].imshow(img2D[..., 1].unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy())
    # axes[1, 0].set_title('channel 1')
    # image_pred = add_contour(img2D[..., 1].unsqueeze(-1).expand(-1, -1, 3), lesion_prev_masks, zone_prev_masks[0],
    #                          zone_prev_masks[1], gland_prev_masks)
    # image_gt = add_contour(img2D[..., 1].unsqueeze(-1).expand(-1, -1, 3), lesion_gt2D.squeeze(0), zone_gt2D[0],
    #                        zone_gt2D[1], gland_gt2D.squeeze(0))
    # axes[1, 1].imshow(image_pred)
    # axes[1, 1].set_title('predicted results')
    # axes[1, 2].imshow(image_gt)
    # axes[1, 2].set_title('ground truth')
    # 
    # axes[2, 0].imshow(img2D[..., 2].unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy())
    # axes[2, 0].set_title('channel 2')
    # image_pred = add_contour(img2D[..., 2].unsqueeze(-1).expand(-1, -1, 3), lesion_prev_masks, zone_prev_masks[0],
    #                          zone_prev_masks[1], gland_prev_masks)
    # image_gt = add_contour(img2D[..., 2].unsqueeze(-1).expand(-1, -1, 3), lesion_gt2D.squeeze(0), zone_gt2D[0],
    #                        zone_gt2D[1], gland_gt2D.squeeze(0))
    # axes[2, 1].imshow(image_pred)
    # axes[2, 1].set_title('predicted results')
    # axes[2, 2].imshow(image_gt)
    # axes[2, 2].set_title('ground truth')

    # plt.savefig(os.path.join(save_path, f'slice_{count}'))

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

def seg_eval_multi_level(net, tumor_split, model_name, val_path, ckpt_path, device, activation, is_post_process, threshold, image_size=1024):
    ckpt_file = os.path.join(ckpt_path, search_ckpt_path(ckpt_path))
    lesion_pid = pickle.load(open(os.path.join(os.path.dirname(val_path[0]), '../lesion_pid.p'), 'rb'))
    # use zone_segdata_all for all data
    if '158' in val_path[0]:
        zone_pid = pickle.load(open('./dataset/zone_segdata_158/zone_pid.p', 'rb'))
    elif 'MSD' in val_path[0]:
        zone_pid = pickle.load(open('./dataset/zone_segdata_MSD/zone_pid.p', 'rb'))
    else:
        zone_pid = pickle.load(open('/data/nvme1/meng/picai/zone_segdata_all/zone_pid.p', 'rb'))
    gland_pid = None if '158' in val_path[0] or 'MSD' in val_path[0] else pickle.load(open('/data/nvme1/meng/picai/gland_segdata/gland_pid.p', 'rb'))

    with open('/data/nvme1/meng/picai/lesion_segdata_combined/data_split.p', 'rb') as f:
        loaded_dict = pickle.load(f)

    state_dict = torch.load(ckpt_file, map_location=device)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.` prefix
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    net = net.to(device)
    net = DataParallel(net)

    if tumor_split:
        val_transformer = transforms.Compose([
            Normalize_2d(),
            Resize_2d(image_size),
            To_Tensor()
        ])
        val_dataset = MultiLevel3DDataGenerator(val_path, 'split', image_size, num_class=2, transform=val_transformer,
                                              zone_pid=zone_pid, gland_pid=gland_pid, lesion_pid=lesion_pid)
    else:
        if model_name == ModelName.masam:
            val_transformer = transforms.Compose([
                Normalize_2d(),
                Resize_2d(image_size),
                To_Tensor()
            ])
            val_dataset = MultiLevel3DDataGenerator(val_path, 'random', image_size, num_class=2,
                                                    transform=val_transformer, zone_pid=zone_pid,
                                                    gland_pid=gland_pid,
                                                    lesion_pid=lesion_pid)
        else:
            val_transformer = transforms.Compose([
                Normalize(),
                # tio.CropOrPad(target_shape=(32, 128, 128)),
                To_Tensor(num_class=2, input_channel=3)
            ])
            val_dataset = MultiLevelDataGenerator(val_path, 'val', image_size, num_class=2, transform=val_transformer, zone_pid=zone_pid, gland_pid=gland_pid, lesion_pid=lesion_pid)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    lesion_dices = []
    positive_lesion_dices = []
    pz_dices = []
    tz_dices = []
    gland_dices = []
    pid_list = []
    slice_list = []
    with torch.no_grad():
        for step, loaded in enumerate(tqdm(val_loader)):
            if tumor_split:
                sample, pid, slice, path = loaded
                if path[0] not in loaded_dict[tumor_split]:
                    continue
            else: sample, pid, slice = loaded
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
            lesion_target = torch.stack(lesion_targets).squeeze(0).squeeze(0).permute(1, 0, 2,
                                                                                      3) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else torch.stack(
                lesion_targets).permute(1, 0, 2, 3)
            zone_target = torch.stack(zone_targets).squeeze(1).squeeze(1).permute(1, 0, 2,
                                                                                  3) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else torch.stack(
                zone_targets).permute(1, 0, 2, 3)
            gland_target = torch.stack(gland_targets).squeeze(0).squeeze(0).permute(1, 0, 2,
                                                                                    3) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else torch.stack(
                gland_targets).permute(1, 0, 2, 3)

            data = data.squeeze(0).permute(1,0,2,3).to(device) if tumor_split else data.to(device)
            lesion_target = lesion_target.to(device)
            zone_target = zone_target.to(device)
            gland_target = gland_target.to(device)

            with autocast(False):
                if model_name == ModelName.masam:
                    model_output = rearrange(net(data, True, image_size)['masks'].permute(1,0,2,3).unsqueeze(0), '1 c (b d) h w -> b c d h w', b=1)
                elif model_name == ModelName.samed:
                    model_output = net(data, True, image_size)['masks']
                elif model_name == ModelName.itunet:
                    model_output = net(data)[0]
                else:
                    model_output = net(data)
                if activation:
                    logits = torch.sigmoid(model_output)
                else:
                    logits = model_output
            # post_process('./', data.detach().cpu(), logits)
            output = logits > threshold
            # if is_post_process:
            #     output = torch.from_numpy(erode_dilate(output.squeeze(0).detach().cpu().numpy())).unsqueeze(0).to(device)
            gland_output = output[:, 0]
            zone_output = output[:, 1:3]
            lesion_output = output[:, 3]
            # intersect = (gland_output & lesion_output).sum().item()
            # lesion_size = lesion_output.sum().item()
            # ratio = intersect / lesion_size
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
            if tumor_split:
                for id, img in enumerate(data):
                    lesion_dice, pz_dice, tz_dice, gland_dice = plot_segmentation2D_multilevel(
                        img.permute(1, 2, 0), lesion_output[id], zone_output[id],
                        gland_output[id], lesion_target[id], zone_target[id], gland_target[id],
                        pid[0] + '-' + str(id))
                    lesion_dices.append(lesion_dice)
                    pz_dices.append(pz_dice)
                    tz_dices.append(tz_dice)
                    gland_dices.append(gland_dice)
                    if lesion_target[id].max() > 0:
                        positive_lesion_dices.append(lesion_dice)
            else:
                if model_name==ModelName.swin_unetr or model_name == ModelName.masam:
                    for id, img in enumerate(data.squeeze(0).permute(1,0,2,3)):
                        lesion_dice, pz_dice, tz_dice, gland_dice = plot_segmentation2D_multilevel(
                            img.permute(1, 2, 0), lesion_output.squeeze(0)[id], zone_output.squeeze(0)[:, id],
                            gland_output.squeeze(0)[id], lesion_target[id], zone_target[id], gland_target[id],
                            pid[0] + '-' + str(id))
                        if 'MSD' in val_path[0]:
                            check_metric = zone_target[id]
                        else:
                            check_metric = lesion_target[id]
                        if check_metric.max() > 0:
                            lesion_dices.append(lesion_dice)
                            pz_dices.append(pz_dice)
                            tz_dices.append(tz_dice)
                            gland_dices.append(gland_dice)
                            pid_list.append(pid[0])
                            slice_list.append(id)
                else:
                    lesion_dice, pz_dice, tz_dice, gland_dice = plot_segmentation2D_multilevel(data.squeeze(0).permute(1, 2, 0), lesion_output.squeeze(0), zone_output.squeeze(0), gland_output.squeeze(0), lesion_target.squeeze(0), zone_target.squeeze(0), gland_target.squeeze(0), pid[0]+'-'+slice[0])
                    if 'MSD' in val_path[0]:
                        check_metric = zone_target
                    else:
                        check_metric = lesion_target
                    if check_metric.max() > 0:
                        lesion_dices.append(lesion_dice)
                        pz_dices.append(pz_dice)
                        tz_dices.append(tz_dice)
                        gland_dices.append(gland_dice)
                        pid_list.append(pid[0])
                        slice_list.append(slice)
    result_dict = {}
    for id, pid in enumerate(pid_list):
        result_dict[f'{pid}-*-{slice_list[id]}'] = (lesion_dices, pz_dices, tz_dices, gland_dices)
    return result_dict

def seg_eval_multi_level_ours(net, tumor_split, val_path, ckpt_path, device, activation, is_post_process, threshold, image_size=1024):
    ckpt_file = os.path.join(ckpt_path, search_ckpt_path(ckpt_path))
    lesion_pid = pickle.load(open(os.path.join(os.path.dirname(val_path[0]), '../lesion_pid.p'), 'rb'))
    # use zone_segdata_all for all data
    if '158' in val_path[0]:
        zone_pid = pickle.load(open('./dataset/zone_segdata_158/zone_pid.p', 'rb'))
    elif 'MSD' in val_path[0]:
        zone_pid = pickle.load(open('./dataset/zone_segdata_MSD/zone_pid.p', 'rb'))
    else:
        zone_pid = pickle.load(open('/data/nvme1/meng/picai/zone_segdata_all/zone_pid.p', 'rb'))
    gland_pid = None if '158' in val_path[0] or 'MSD' in val_path[0] else pickle.load(
        open('/data/nvme1/meng/picai/gland_segdata/gland_pid.p', 'rb'))

    with open('/data/nvme1/meng/picai/lesion_segdata_combined/data_split.p', 'rb') as f:
        loaded_dict = pickle.load(f)

    state_dict = torch.load(ckpt_file, map_location=device)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.` prefix
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    net = net.to(device)
    net = DataParallel(net)

    val_transformer = transforms.Compose([
        Normalize(),
        # tio.CropOrPad(target_shape=(32, 128, 128)),
        To_Tensor(num_class=2, input_channel=3)
    ])
    val_dataset = MultiLevelDataGeneratorAlignDummy(val_path, 'val', num_class=2, transform=val_transformer,
                                                    zone_pid=zone_pid,
                                                    gland_pid=gland_pid, lesion_pid=lesion_pid)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    lesion_dices = []
    positive_lesion_dices = []
    pz_dices = []
    tz_dices = []
    gland_dices = []
    pid_list = []
    slice_list = []
    with torch.no_grad():
        for step, loaded in enumerate(tqdm(val_loader)):
            if tumor_split:
                sample, pid, slice, path = loaded
                if path[0] not in loaded_dict[tumor_split]:
                    continue
            else: sample, pid, slice = loaded
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
            text_toekns = sample['tokens']
            lesion_target = torch.stack(lesion_targets).squeeze(0).squeeze(0).permute(1, 0, 2, 3) if tumor_split else torch.stack(lesion_targets).permute(1, 0, 2, 3)
            zone_target = torch.stack(zone_targets).squeeze(1).squeeze(1).permute(1, 0, 2, 3) if tumor_split else torch.stack(zone_targets).permute(1, 0, 2, 3)
            gland_target = torch.stack(gland_targets).squeeze(0).squeeze(0).permute(1, 0, 2, 3) if tumor_split else torch.stack(gland_targets).permute(1, 0, 2, 3)

            data = data.squeeze(0).permute(1,0,2,3).to(device) if tumor_split else data.to(device)
            lesion_target = lesion_target.to(device)
            zone_target = zone_target.to(device)
            gland_target = gland_target.to(device)
            text_tokens = text_toekns.to(device)

            with autocast(False):
                if activation:
                    logits = torch.sigmoid(net(data, text_tokens.squeeze(0)))
                else:
                    logits = net(data, text_tokens.squeeze(0))
            # post_process('./', data.detach().cpu(), logits)
            output = logits > threshold
            # if is_post_process:
            #     output = torch.from_numpy(erode_dilate(output.squeeze(0).detach().cpu().numpy())).unsqueeze(0).to(device)
            gland_output = output[:, 0]
            zone_output = output[:, 1:3]
            lesion_output = output[:, 3]
            # intersect = (gland_output & lesion_output).sum().item()
            # lesion_size = lesion_output.sum().item()
            # ratio = intersect / lesion_size
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

            lesion_dice, pz_dice, tz_dice, gland_dice = plot_segmentation2D_multilevel(data.squeeze(0).permute(1, 2, 0), lesion_output.squeeze(0), zone_output.squeeze(0), gland_output.squeeze(0), lesion_target.squeeze(0), zone_target.squeeze(0), gland_target.squeeze(0), pid[0]+'-'+slice[0])
            if 'MSD' in val_path[0]:
                check_metric = zone_target
            else:
                check_metric = lesion_target
            if check_metric.max() > 0:
                lesion_dices.append(lesion_dice)
                pz_dices.append(pz_dice)
                tz_dices.append(tz_dice)
                gland_dices.append(gland_dice)
                pid_list.append(pid[0])
                slice_list.append(slice)
    result_dict = {}
    for id, pid in enumerate(pid_list):
        result_dict[f'{pid}-*-{slice_list[id]}'] = (lesion_dices, pz_dices, tz_dices, gland_dices)
    return result_dict

if __name__ == '__main__':
    def get_sorted_list(dataset_name):
        mode = 'normal'
        from config import CHECKPOINT_PATH
        is_post_process = True
        threshold = 0.5

        tumor_split = None

        # ckpt_path = './new_ckpt/{}/{}/fold1'.format('seg', 'UNet_Unified_equal_rate_lr_0.0001_weight_decay_0.001')
        if dataset_name == 'picai':
            PATH_DIR = '/data/nvme1/meng/picai/lesion_segdata_combined/data_2d'
        elif dataset_name == '158':
            PATH_DIR = './dataset/lesion_segdata_158/data_2d'
        elif dataset_name == 'MSD':
            PATH_DIR = './dataset/lesion_segdata_MSD/data_2d'
        PATH_LIST = glob.glob(os.path.join(PATH_DIR, '*.hdf5'))
        train_path, val_path = get_cross_validation_by_sample(PATH_LIST, FOLD_NUM, 1)

        model_name = ModelName.unet
        activation = False
        image_size = 256
        net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                               in_channels=3, out_channels=4, init_features=32, pretrained=False)
        ckpt_path = './new_ckpt/{}/{}/fold1'.format('seg',f'UNet_Focal_Unified_equal_rate_0.8_weighted_loss_combined_label_lr_0.0001_weight_decay_0.001')
        unet_dict = seg_eval_multi_level(net, tumor_split, model_name, PATH_LIST if '158' in val_path[0] or 'MSD' in val_path[0] else val_path, ckpt_path, 'cuda:0', activation, is_post_process, threshold, image_size=image_size)

        if dataset_name == 'picai':
            PATH_DIR = '/data/nvme1/meng/picai/lesion_segdata_combined/data_3d'
        elif dataset_name == '158':
            PATH_DIR = './dataset/lesion_segdata_158/data_3d'
        elif dataset_name == 'MSD':
            PATH_DIR = './dataset/lesion_segdata_MSD/data_3d'
        PATH_LIST = glob.glob(os.path.join(PATH_DIR, '*.hdf5'))
        train_path, val_path = get_cross_validation_by_sample(PATH_LIST, FOLD_NUM, 1)
        activation = True
        image_size = 256
        sam, img_embedding_size = sam_model_registry_MASAM['vit_b'](image_size=image_size,
                                                                    num_classes=3,
                                                                    checkpoint='/data/nvme1/meng/cvpr25_results/sam_vit_b_01ec64.pth',
                                                                    pixel_mean=[0., 0., 0.],
                                                                    pixel_std=[1., 1., 1.])
        model_name = ModelName.masam
        net = Fact_tt_Sam(sam, 32, s=1.0)
        ckpt_path = './new_ckpt/{}/{}/fold1'.format('seg',f'MASAM_Focal_Unified_equal_rate_batch_4_tumorsplit_0.001_image_1024_dataset_picai_valmode_2d_lr_0.0001_weight_decay_0.001')
        masam_dict = seg_eval_multi_level(
            net, tumor_split, model_name, PATH_LIST if '158' in val_path[0] or 'MSD' in val_path[0] else val_path, ckpt_path, 'cuda:0', activation, is_post_process, threshold,
            image_size=image_size)

        if dataset_name == 'picai':
            PATH_DIR = '/data/nvme1/meng/picai/lesion_segdata_combined/data_2d'
        elif dataset_name == '158':
            PATH_DIR = './dataset/lesion_segdata_158/data_2d'
        elif dataset_name == 'MSD':
            PATH_DIR = './dataset/lesion_segdata_MSD/data_2d'
        PATH_LIST = glob.glob(os.path.join(PATH_DIR, '*.hdf5'))
        train_path, val_path = get_cross_validation_by_sample(PATH_LIST, FOLD_NUM, 1)
        activation = True
        sam_model = sam_model_registry['vit_b'](checkpoint=os.path.join(CHECKPOINT_PATH, 'medsam_vit_b.pth'))

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

        text_encoder = TextEncoder(embed_dim=256, text_cfg_path=CHECKPOINT_PATH)
        atten_encoder = VisionTransformer(input_resolution=1024, patch_size=16, width=256, layers=3, heads=8,
                                          output_dim=256)

        net = MedSAMAUTOMULTIALIGNTYPE1(
            image_encoder=sam_model.image_encoder,
            mask_decoder=multi_mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
            text_encoder=text_encoder,
            dense_encoder=dense_model,
            image_size=512,
            attention_encoder=atten_encoder,
            mode=mode)
        ckpt_path = './new_ckpt/{}/Seg_Align/{}/fold1'.format('seg',f'2024-11-08T18:12:39_lr_0.0001_weight_decay_0.001')
        ours_dict = seg_eval_multi_level_ours(
            net, tumor_split, PATH_LIST if '158' in val_path[0] or 'MSD' in val_path[0] else val_path, ckpt_path, 'cuda:0', activation, is_post_process, threshold,
            image_size=image_size)

        info_dict = {'id':[], 'mean seg boost': [], 'gland boost':[], 'zone boost':[], 'lesion boost':[]}
        for sample_name, sample in unet_dict.items():
            gland_boost = ours_dict[sample_name][3] - unet_dict[sample_name][3] + ours_dict[sample_name][3] - masam_dict[sample_name][3]
            pz_boost = ours_dict[sample_name][1] - unet_dict[sample_name][1] + ours_dict[sample_name][1] - masam_dict[sample_name][1]
            tz_boost = ours_dict[sample_name][2] - unet_dict[sample_name][2] + ours_dict[sample_name][2] - masam_dict[sample_name][2]
            lesion_boost = ours_dict[sample_name][0] - unet_dict[sample_name][0] + ours_dict[sample_name][0] - masam_dict[sample_name][0]
            info_dict['gland boost'].append(gland_boost)
            info_dict['pz boost'].append(pz_boost)
            info_dict['tz boost'].append(tz_boost)
            info_dict['lesion boost'].append(lesion_boost)
            if dataset_name == 'MSD':
                info_dict['mean seg boost'].append((pz_boost + tz_boost) / 2)
            elif dataset_name == '158':
                info_dict['mean seg boost'].append((pz_boost + tz_boost + lesion_boost) / 3)
            else:
                info_dict['mean seg boost'].append((gland_boost + pz_boost + tz_boost + lesion_boost) / 4)
            info_dict['id'].append(sample_name)

        combined = list(zip(info_dict['mean seg boost'], info_dict['gland boost'], info_dict['pz boost'], info_dict['tz boost'], info_dict['lesion boost'], info_dict['id'], info_dict['slice id']))

        # Sort the combined list based on the first list (list1)
        sorted_combined = sorted(combined, key=lambda x: x[0])

        # Unzip the sorted list back into individual lists
        sorted_list1, sorted_list2, sorted_list3, sorted_list4, sorted_list5, sorted_list6, sorted_list7 = zip(*sorted_combined)

        # Convert tuples back to lists (if needed)
        sorted_list1 = list(sorted_list1)
        sorted_list2 = list(sorted_list2)
        sorted_list3 = list(sorted_list3)
        sorted_list4 = list(sorted_list4)
        sorted_list5 = list(sorted_list5)
        sorted_list6 = list(sorted_list6)
        sorted_list7 = list(sorted_list7)
        return sorted_list1, sorted_list2, sorted_list3, sorted_list4, sorted_list5, sorted_list6, sorted_list7

    # picai_mean_seg_boost, picai_gland_seg_boost, picai_pz_seg_boost, picai_tz_seg_boost, picai_lesion_seg_boost, picai_id, picai_slice = get_sorted_list('picai')
    # _158_mean_seg_boost, _158_gland_seg_boost, _158_pz_seg_boost, _158_tz_seg_boost, _158_lesion_seg_boost, _158_id, _158_slice = get_sorted_list('158')
    MSD_mean_seg_boost, MSD_gland_seg_boost, MSD_pz_seg_boost, MSD_tz_seg_boost, MSD_lesion_seg_boost, MSD_id, MSD_slice = get_sorted_list('MSD')

    print()