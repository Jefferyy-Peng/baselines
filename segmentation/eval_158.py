import copy
import glob
import math
import os
import pickle
import random
import re
from collections import OrderedDict
import seaborn as sns
from monai.networks.nets import SwinUNETR
from monai.transforms import Resize, ScaleIntensityD, ResizeD, ToTensorD
from einops import rearrange

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
from segmentation.MedSAMAuto import MedSAMAUTO, MedSAMAUTOZONE, MedSAMAUTOMULTI, MedSAMAUTOCNN
from segmentation.config import FOLD_NUM, CURRENT_FOLD
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
    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
    # axes[0, 0].imshow(img2D[..., 0].unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy())
    # axes[0, 0].set_title('channel 0')
    # width = math.ceil(img2D.shape[0] * (7/1024))
    # image_pred = add_contour(img2D[..., 0].unsqueeze(-1).expand(-1, -1, 3), lesion_prev_masks, zone_prev_masks[0], zone_prev_masks[1], gland_prev_masks, contour_thickness=width)
    # # Note that in prostate158 pz is 2 and tz is 1, while in picai pz is 1 and tz is 2, so need to flip the gt here
    # image_gt = add_contour(img2D[..., 0].unsqueeze(-1).expand(-1, -1, 3), lesion_gt2D.squeeze(0), zone_gt2D[1], zone_gt2D[0], gland_gt2D.squeeze(0), contour_thickness=width)
    lesion_dice = compute_dice(lesion_prev_masks.int(), lesion_gt2D[0])
    pz_dice = compute_dice(zone_prev_masks[0].int(), zone_gt2D[1])
    tz_dice = compute_dice(zone_prev_masks[1].int(), zone_gt2D[0])
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
    #                          zone_prev_masks[1], gland_prev_masks, contour_thickness=width)
    # image_gt = add_contour(img2D[..., 1].unsqueeze(-1).expand(-1, -1, 3), lesion_gt2D.squeeze(0), zone_gt2D[1],
    #                        zone_gt2D[0], gland_gt2D.squeeze(0), contour_thickness=width)
    # axes[1, 1].imshow(image_pred)
    # axes[1, 1].set_title('predicted results')
    # axes[1, 2].imshow(image_gt)
    # axes[1, 2].set_title('ground truth')
    #
    # axes[2, 0].imshow(img2D[..., 2].unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy())
    # axes[2, 0].set_title('channel 2')
    # image_pred = add_contour(img2D[..., 2].unsqueeze(-1).expand(-1, -1, 3), lesion_prev_masks, zone_prev_masks[0],
    #                          zone_prev_masks[1], gland_prev_masks, contour_thickness=width)
    # image_gt = add_contour(img2D[..., 2].unsqueeze(-1).expand(-1, -1, 3), lesion_gt2D.squeeze(0), zone_gt2D[1],
    #                        zone_gt2D[0], gland_gt2D.squeeze(0), contour_thickness=width)
    # axes[2, 1].imshow(image_pred)
    # axes[2, 1].set_title('predicted results')
    # axes[2, 2].imshow(image_gt)
    # axes[2, 2].set_title('ground truth')
    #
    # plt.savefig(os.path.join(save_path, 'plots', f'slice_{count}'))

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

def plot_eval_multi_level(net, model_name, val_path, ckpt_path, log_dir, device, activation, is_post_process, threshold, image_size=1024, dataset='picai'):
    ckpt_file = os.path.join(ckpt_path, search_ckpt_path(ckpt_path))
    lesion_pid = pickle.load(open(os.path.join(PATH_DIR, '../lesion_pid.p'), 'rb'))
    # use zone_segdata_all for all data
    zone_pid = pickle.load(open('./dataset/zone_segdata_158/zone_pid.p', 'rb')) if dataset == '158' else pickle.load(open('./dataset/zone_segdata_all/zone_pid.p', 'rb'))
    gland_pid = None if dataset == '158' else pickle.load(open('./dataset/gland_segdata/lesion_pid.p', 'rb'))

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
    if model_name == ModelName.swin_unetr:
        val_transformer = transforms.Compose([
            ScaleIntensityD(keys=["ct"]),
            ResizeD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"],
                    spatial_size=(32 if model_name == ModelName.swin_unetr else 24, image_size, image_size),
                    mode=("trilinear", "nearest", "nearest", "nearest", "nearest")),
            # Resize the ct to 128x128x64
            ToTensorD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"])
        ])
    elif model_name == ModelName.masam:
        val_transformer = transforms.Compose([
            Normalize_2d(),
            Resize_2d(image_size),
            To_Tensor()
        ])
    else:
        val_transformer = transforms.Compose([
            Normalize(),
            # tio.CropOrPad(target_shape=(32, 128, 128)),
            To_Tensor(num_class=2, input_channel=3)
        ])

    val_dataset = MultiLevel3DDataGenerator(val_path, 'random', image_size, num_class=2,
                                                        transform=val_transformer, zone_pid=zone_pid,
                                                        gland_pid=gland_pid,
                                                        lesion_pid=lesion_pid) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else MultiLevelDataGenerator(val_path, 'val', image_size, num_class=2, transform=val_transformer, zone_pid=zone_pid, gland_pid=gland_pid, lesion_pid=lesion_pid)
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
    with torch.no_grad():
        for step, (sample, pid, slice) in enumerate(tqdm(val_loader)):
            if os.path.exists(os.path.join(log_dir, 'plots', 'slice_' + pid[0]+'-'+str(slice[0].item())+'.png' if model_name==ModelName.swin_unetr or model_name == ModelName.masam else 'slice_' + pid[0]+'-'+ slice[0] + '.png')):
                continue
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
            lesion_target = torch.stack(lesion_targets).squeeze(0).squeeze(0).permute(1, 0, 2, 3) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else torch.stack(lesion_targets).permute(1, 0, 2, 3)
            zone_target = torch.stack(zone_targets).squeeze(1).squeeze(1).permute(1, 0, 2, 3) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else torch.stack(zone_targets).permute(1, 0, 2, 3)
            gland_target = torch.stack(gland_targets).squeeze(0).squeeze(0).permute(1, 0, 2, 3) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else torch.stack(gland_targets).permute(1, 0, 2, 3)

            data = data.to(device)
            lesion_target = lesion_target.to(device)
            zone_target = zone_target.to(device)
            gland_target = gland_target.to(device)

            with autocast(False):
                if model_name == ModelName.itunet:
                    model_output = net(data)[0]
                elif model_name == ModelName.masam:
                    model_output = rearrange(net(data, True, image_size)['masks'].permute(1,0,2,3).unsqueeze(0), '1 c (b d) h w -> b c d h w', b=1)
                elif model_name == ModelName.samed:
                    model_output = net(data, True, image_size)['masks']
                else:
                    model_output = net(data)
                if activation:
                    logits = torch.sigmoid(model_output)
                else:
                    logits = model_output
            # post_process('./', data.detach().cpu(), logits)
            output = logits > threshold
            if is_post_process:
                output = torch.from_numpy(erode_dilate(output.squeeze(0).detach().cpu().numpy())).unsqueeze(0).to(device)
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
            if model_name==ModelName.swin_unetr or model_name == ModelName.masam:
                for id, img in enumerate(data.squeeze(0).permute(1,0,2,3)):
                    lesion_dice, pz_dice, tz_dice, gland_dice = plot_segmentation2D_multilevel(
                        img.permute(1, 2, 0),lesion_output.squeeze(0)[id], zone_output.squeeze(0)[:,id], gland_output.squeeze(0)[id], lesion_target[id], zone_target[id], gland_target[id], log_dir, pid[0]+'-'+str(id))
                    lesion_dices.append(lesion_dice)
                    pz_dices.append(pz_dice)
                    tz_dices.append(tz_dice)
                    gland_dices.append(gland_dice)
                    if lesion_target[id].max() > 0:
                        positive_lesion_dices.append(lesion_dice)
            else:
                lesion_dice, pz_dice, tz_dice, gland_dice = plot_segmentation2D_multilevel(data.squeeze(0).permute(1, 2, 0), lesion_output.squeeze(0), zone_output.squeeze(0), gland_output.squeeze(0), lesion_target.squeeze(0), zone_target.squeeze(0), gland_target.squeeze(0), log_dir, pid[0]+'-'+slice[0])
                lesion_dices.append(lesion_dice)
                pz_dices.append(pz_dice)
                tz_dices.append(tz_dice)
                gland_dices.append(gland_dice)
                if lesion_target.max() > 0:
                    positive_lesion_dices.append(lesion_dice)
    lesion_mean_dice = torch.Tensor(lesion_dices).mean()
    positive_lesion_mean_dice = torch.Tensor(positive_lesion_dices).mean()
    pz_mean_dice = torch.Tensor(pz_dices).mean()
    tz_mean_dice = torch.Tensor(tz_dices).mean()
    gland_mean_dice = torch.Tensor(gland_dices).mean()
    os.system(f'cd {log_dir} && touch dice_result.txt && echo "lesion_mean_dice: {lesion_mean_dice}, pz_mean_dice:{pz_mean_dice}, tz_mean_dice: {tz_mean_dice}, gland_mean_dice: {gland_mean_dice}, positive_lesion_mean_dice: {positive_lesion_mean_dice}" >> dice_result.txt')
    # lesion_results = {idx: result for idx, result in enumerate(lesion_results)}
    # valid_metrics = Metrics(lesion_results)
    # auc = valid_metrics.auroc
    # ap = valid_metrics.AP
    # score = valid_metrics.score
    # print(f'auc: {auc}, ap:{ap}, score: {score}')
    # os.system(f'cd {log_dir}')
    # os.system(f'touch result.txt')
    # os.system(f'echo "auc: {auc}, ap:{ap}, score: {score}" >> result.txt')

def plot_eval_detect(net, model_name, val_path, ckpt_path, log_dir, device, activation, post_process, threshold, mode='normal',image_size=1024, dataset='picai'):
    ckpt_file = os.path.join(ckpt_path, search_ckpt_path(ckpt_path))

    # with open('./dataset/lesion_segdata_combined/data_split.p', 'rb') as f:
    #     loaded_dict = pickle.load(f)

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
    if model_name == ModelName.swin_unetr or model_name == ModelName.masam:
        val_transformer = transforms.Compose([
            ScaleIntensityD(keys=["ct"]),
            ResizeD(keys=["ct", "seg"],
                    spatial_size=(32 if model_name == ModelName.swin_unetr else 24, image_size, image_size),
                    mode=("trilinear", "nearest")),
            # Resize the ct to 128x128x64
            ToTensorD(keys=["ct", "seg"])]
        )
    else:
        val_transformer = transforms.Compose(
            [Normalize_2d(), To_Tensor()])

    val_dataset = DataGenerator_no_resize(val_path, transform=val_transformer,mode='val') if model_name == ModelName.swin_unetr or model_name == ModelName.masam else DataGenerator(val_path, transform=val_transformer,mode='val', image_size=image_size)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )
    count = 0

    dice_dict = {}
    lesion_results = []
    image_embeddings = []
    dense_embeddings = []
    image_targets = []
    with torch.no_grad():
        for step, (sample, path) in enumerate(tqdm(val_loader)):
            # if path[0] in loaded_dict['small']:
            #     continue
            data = sample['ct']
            target = sample['seg']

            if model_name == ModelName.swin_unetr or model_name == ModelName.masam:
                data = data
            else:
                data = data.squeeze().transpose(1, 0)
            data = data.to(device)
            target = target.to(device)
            if mode == 'normal':
                if model_name == ModelName.masam:
                    output = rearrange(net(data, True, image_size)['masks'].permute(1, 0, 2, 3).unsqueeze(0),
                                   '1 c (b d) h w -> b c d h w',
                                   b=target.shape[0])
                elif model_name == ModelName.samed:
                    output = net(data, True, image_size)['masks']
                else:
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

            output = output[0].float() if model_name == ModelName.itunet else output.float()
            if activation:
                output = torch.sigmoid(output)  # N*H*W
            output = output.detach().cpu()
            lesion_output = output[:, -1, :, :].squeeze(0).unsqueeze(
                1) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else output[:, -1, :,
                                                                                               :].unsqueeze(1)
            lesion_output = torch.from_numpy(np.array([revert_transform(slice) for slice in lesion_output])).squeeze(1)
            gland_output = output[:, 0, :, :].squeeze(0).unsqueeze(
                1) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else output[:, 0, :,
                                                                                               :].unsqueeze(1)
            gland_output = torch.from_numpy(np.array([revert_transform(slice) for slice in gland_output])).squeeze(1)
            target = target.detach().cpu()
            target = target[0].permute(1, 0, 2,
                                       3) if model_name == ModelName.swin_unetr or model_name == ModelName.masam else \
            target[0].unsqueeze(1)
            target = torch.from_numpy(np.array([seg_transform(slice) for slice in target])).squeeze(1)
            # if plot:
            #     for
            #     plot_segmentation2D()


            lesion_results = compute_results_detect(lesion_output.numpy(), target.numpy(), gland_output.numpy(),
                                                    lesion_results, threshold, post_process)
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
    conf_matrix_lesion_level = np.array([[TN_patient_level, FP_lesion_level], [FN_lesion_level, TP_lesion_level]])
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
    os.system(f'cd {log_dir} && touch result.txt && echo "auc: {auc}, ap:{ap}, score: {score}" >> result.txt')


if __name__ == '__main__':
    PHASE = 'detect'

    model_name = ModelName.samed

    mode = 'normal'
    dataset = '158'
    # dataset = 'picai'

    is_post_process = True
    threshold = 0.5
    if model_name == ModelName.unet:
        activation = False
        image_size = 256
        net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                           in_channels=3, out_channels=4, init_features=32, pretrained=False)
    elif model_name == ModelName.itunet:
        activation = True
        image_size = 384
        net = itunet_2d(n_channels=3, n_classes=4,
                  image_size=(image_size, image_size), transformer_depth=24)
    elif model_name == ModelName.medsam:
        activation = True
        image_size = 1024
        sam_model = sam_model_registry['vit_b'](checkpoint='/data/nvme1/meng/cvpr25_results/medsam_vit_b.pth')
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
    elif model_name == ModelName.swin_unetr:
        activation = True
        image_size = 256
        net = SwinUNETR(img_size=(32, image_size, image_size),
                          in_channels=3,
                          out_channels=4,
                          feature_size=48,
                          use_checkpoint=True,
                          )
    elif model_name == ModelName.masam:
        activation = True
        image_size = 256
        sam, img_embedding_size = sam_model_registry_MASAM['vit_b'](image_size=image_size,
                                                                    num_classes=3,
                                                                    checkpoint='/data/nvme1/meng/cvpr25_results/sam_vit_b_01ec64.pth',
                                                                    pixel_mean=[0., 0., 0.],
                                                                    pixel_std=[1., 1., 1.])
        net = Fact_tt_Sam(sam, 32, s=1.0)
    elif model_name == ModelName.samed:
        activation = True
        image_size = 256
        sam, img_embedding_size = sam_model_registry_SAMed['vit_b'](image_size=image_size,
                                                                    num_classes=3,
                                                                    checkpoint='/data/nvme1/meng/cvpr25_results/sam_vit_b_01ec64.pth',
                                                                    pixel_mean=[0., 0., 0.],
                                                                    pixel_std=[1., 1., 1.])

        net = LoRA_Sam(sam, r=4)

    # mask_decoder_model = SegDecoderCNN(num_classes=4, num_depth=4)
    #
    # net = MedSAMAUTOCNN(
    #     image_encoder=sam_model.image_encoder,
    #     mask_decoder=mask_decoder_model,
    #     prompt_encoder=sam_model.prompt_encoder,
    #     dense_encoder=None,
    #     image_size=512
    # )


    ckpt_path = './new_ckpt/{}/{}/fold1'.format('seg',f'SAMed_Focal_Unified_equal_rate_batch_70_tumorsplit_0.001_image_256_dataset_picai_valmode_3d_lr_0.0001_weight_decay_0.001')
    # ckpt_path = './new_ckpt/{}/{}/fold1'.format('seg', 'UNet_Unified_equal_rate_lr_0.0001_weight_decay_0.001')

    log_dir = f'./new_log/eval/SAMed_Focal_Unified_equal_rate_batch_70_tumorsplit_0.001_image_256_dataset_picai_valmode_3d_lr_0.0001_weight_decay_0.001_threshold_{threshold}_dataset_{dataset}'
    # log_dir = './new_log/eval/UNet3LevelALLDataEqualRate'
    if PHASE == 'seg':
        PATH_DIR = './dataset/lesion_segdata_158/data_3d' if model_name == ModelName.swin_unetr or model_name == ModelName.masam else './dataset/lesion_segdata_158/data_2d'
        PATH_LIST = glob.glob(os.path.join(PATH_DIR, '*.hdf5'))
        # train_path, val_path = get_cross_validation_by_sample(PATH_LIST, FOLD_NUM, 1)
        plot_eval_multi_level(net, model_name, PATH_LIST, ckpt_path, log_dir, 'cuda:0', activation, False, threshold, image_size=image_size, dataset=dataset)
    else:
        PATH_AP = './dataset/lesion_segdata_158/data_3d'
        AP_LIST = glob.glob(os.path.join(PATH_AP, '*.hdf5'))
        plot_eval_detect(net, model_name, AP_LIST, ckpt_path, log_dir, 'cuda:0', activation, is_post_process, threshold, mode, image_size=image_size, dataset=dataset)