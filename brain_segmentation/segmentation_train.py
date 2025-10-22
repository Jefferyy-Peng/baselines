import json
import os
from collections import OrderedDict

import torch
import wandb
from matplotlib import pyplot as plt
from safetensors.torch import load_file
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
from accelerate import Accelerator
from torch.optim import Adam
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
import scipy.ndimage as ndi

from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

from monai.config import KeysCollection

from brain_segmentation.loss import Deep_Supervised_Loss
from MedSAMAuto import MedSAMAUTOMULTI
from brain_segmentation.unet import UNet2d
from segment_anything import sam_model_registry
from model_single import ModelEmb
from segmentation.segment_anything.modeling import MaskDecoder, TwoWayTransformer
import torch.nn as nn
from monai.transforms import (
    Compose,
    RandRotate90d,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    EnsureChannelFirstd, ResizeWithPadOrCropd, ResizeD,
)

def plot_segmentation_grid(image, pred_mask, gt_mask, slice_interval=5, alpha=0.5, file_name=''):
    """
    Plot original image, predicted mask, and ground truth mask in a grid.

    Args:
        image (Tensor): shape (D, H, W), grayscale image
        pred_mask (Tensor): shape (D, H, W), predicted class indices
        gt_mask (Tensor): shape (D, H, W), ground truth class indices
        slice_interval (int): interval between slices to show
        alpha (float): overlay transparency
    """
    assert image.shape == pred_mask.shape == gt_mask.shape
    D, H, W = image.shape
    selected_slices = list(range(0, D, slice_interval))

    # Make sure output dir exists
    vis_dir = './brain_segmentation/val_vis'
    os.makedirs(vis_dir, exist_ok=True)

    # Build overlay colormap once (background transparent, classes 1..C-1 colored)
    num_classes = int(max(pred_mask.max().item(), gt_mask.max().item()) + 1)
    base_cmap = cm.get_cmap('tab10', num_classes)
    class_colors = np.vstack([[0, 0, 0, 0]] + [base_cmap(i) for i in range(1, num_classes)])
    overlay_cmap = ListedColormap(class_colors)

    # Handle base name / extension
    base, ext = os.path.splitext(file_name)
    if ext == "":
        ext = ".png"

    for d in selected_slices:
        img = image[d].detach().cpu().numpy()
        pred = pred_mask[d].detach().cpu().numpy()
        gt = gt_mask[d].detach().cpu().numpy()

        # One 1x3 figure per slice
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title(f"Slice {d} - Image");
        axs[0].axis('off')

        axs[1].imshow(img, cmap='gray', interpolation='none')
        axs[1].imshow(pred, cmap=overlay_cmap, alpha=alpha, vmin=0, vmax=num_classes - 1, interpolation='none')
        axs[1].set_title("Prediction");
        axs[1].axis('off')

        axs[2].imshow(img, cmap='gray', interpolation='none')
        axs[2].imshow(gt, cmap=overlay_cmap, alpha=alpha, vmin=0, vmax=num_classes - 1, interpolation='none')
        axs[2].set_title("Ground Truth");
        axs[2].axis('off')

        plt.tight_layout()
        out_path = os.path.join(vis_dir, f"{base}_slice{d:03d}{ext}")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

class SyMRI2DDataset(Dataset):
    def __init__(self, processed_dir, split='train', test_patients=None,
                 num_classes=1, channels=3):
        assert split in ['train', 'test']
        self.num_classes = num_classes
        self.channels = channels
        self.slice_paths = []

        all_patients = sorted([
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ])

        if test_patients is None:
            selected_patients = all_patients
        elif split == 'train':
            selected_patients = [x for x in all_patients if x not in test_patients]
        else:
            selected_patients = test_patients or []

        for patient in selected_patients:
            patient_dir = os.path.join(processed_dir, patient)
            for f in sorted(os.listdir(patient_dir)):
                if f.endswith('.pt'):
                    self.slice_paths.append(os.path.join(patient_dir, f))

        # MONAI transforms
        # common_keys: KeysCollection = ["r2", "t1w", "psir", "t2w"]
        common_keys: KeysCollection = ["t1w"]

        label_key = "label"

        if split == 'train':
            self.transform = Compose([
                ResizeD(keys=common_keys, spatial_size=(512, 512), mode='bilinear'),  # for images
                ResizeD(keys=[label_key], spatial_size=(512, 512), mode='nearest'),
                NormalizeIntensityd(keys=common_keys),
                RandFlipd(keys=common_keys + [label_key], prob=0.5, spatial_axis=1),
                RandRotate90d(keys=common_keys + [label_key], prob=0.5, max_k=3),
                ToTensord(keys=common_keys + [label_key]),
            ])
        else:
            self.transform = Compose([
                ResizeD(keys=common_keys, spatial_size=(512, 512), mode='bilinear'),  # for images
                ResizeD(keys=[label_key], spatial_size=(512, 512), mode='nearest'),
                NormalizeIntensityd(keys=common_keys),
                ToTensord(keys=common_keys + [label_key]),
            ])

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        data = torch.load(self.slice_paths[idx])  # dict with keys: r1, r2, pd, label (optional)
        data['label'].long()
        # Apply MONAI transforms
        transformed = self.transform(data)

        # Stack input channels
        # r2, t1w, psir, t2w = transformed["r2"], transformed["t1w"], transformed["psir"], transformed["t1w"]
        t1w = transformed["t1w"]

        output = torch.cat([t1w])

        if "label" in transformed:
            return output, transformed["label"]
        else:
            return output

class SyMRI3DDataset(Dataset):
    def __init__(self, processed_dir, split='train', test_patients=None,
                 num_classes=1, channels=3):
        assert split in ['train', 'test']
        self.num_classes = num_classes
        self.channels = channels
        self.slice_paths = []

        all_patients = sorted([
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ])

        if test_patients is None:
            selected_patients = all_patients
        elif split == 'train':
            selected_patients = [x for x in all_patients if x not in test_patients]
        else:
            selected_patients = test_patients or []

        for patient in selected_patients:
            patient_dir = os.path.join(processed_dir, patient)
            scan_path = []
            for f in sorted(os.listdir(patient_dir)):
                if f.endswith('.pt'):
                    scan_path.append(os.path.join(patient_dir, f))
            self.slice_paths.append(scan_path)

        # MONAI transforms
        common_keys: KeysCollection = ["r2", "t1w", "psir", "t2w"]
        # common_keys: KeysCollection = ["t1w"]

        label_key = "label"

        if split == 'train':
            self.transform = Compose([
                ResizeD(keys=common_keys, spatial_size=(512, 512), mode='bilinear'),  # for images
                ResizeD(keys=[label_key], spatial_size=(512, 512), mode='nearest'),
                NormalizeIntensityd(keys=common_keys),
                RandFlipd(keys=common_keys + [label_key], prob=0.5, spatial_axis=1),
                RandRotate90d(keys=common_keys + [label_key], prob=0.5, max_k=3),
                ToTensord(keys=common_keys + [label_key]),
            ])
        else:
            self.transform = Compose([
                ResizeD(keys=common_keys, spatial_size=(512, 512), mode='bilinear'),  # for images
                ResizeD(keys=[label_key], spatial_size=(512, 512), mode='nearest'),
                NormalizeIntensityd(keys=common_keys),
                ToTensord(keys=common_keys + [label_key]),
            ])

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        data_list = []
        for slice in self.slice_paths[idx]:
            this_slice = torch.load(slice)
            this_slice['label'].long()
            data_list.append(self.transform(this_slice))  # dict with keys: r1, r2, pd, label (optional)
        data = {}
        data['r2'] = torch.cat([item['r2'] for item in data_list])
        data['t1w'] = torch.cat([item['t1w'] for item in data_list])
        data['psir'] = torch.cat([item['psir'] for item in data_list])
        data['label'] = torch.cat([item['label'] for item in data_list])
        data['t2w'] = torch.cat([item['t2w'] for item in data_list])

        if "label" in data.keys():
            return torch.stack([data['r2'], data["t1w"], data['psir'], data['t2w']], dim=1), data["label"]
            # return torch.stack([data["t1w"]], dim=1), data["label"]
        else:
            return data["t1w"]

def keep_and_strip_base_model(sd_or_ckpt, base_prefix="base_model"):
    """
    sd_or_ckpt: a state_dict OR a checkpoint dict that may contain 'state_dict'/'model'
    returns: filtered state_dict with 'base_model.' prefix removed
    """
    # unwrap to a pure state_dict if a checkpoint dict was passed
    if isinstance(sd_or_ckpt, dict) and not all(torch.is_tensor(v) for v in sd_or_ckpt.values()):
        if "state_dict" in sd_or_ckpt and isinstance(sd_or_ckpt["state_dict"], dict):
            sd = sd_or_ckpt["state_dict"]
        elif "model" in sd_or_ckpt and isinstance(sd_or_ckpt["model"], dict):
            sd = sd_or_ckpt["model"]
        else:
            # assume it's already a state_dict-like
            sd = sd_or_ckpt
    else:
        sd = sd_or_ckpt

    prefixes = [
        f"{base_prefix}.",
        f"module.{base_prefix}.",
        f"model.{base_prefix}.",
    ]

    out = OrderedDict()
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                new_k = k[len(p):]  # strip '...base_model.'
                out[new_k] = v
                break

    if not out:
        raise ValueError(
            f"No parameters found starting with any of {prefixes}. "
            f"Example key: {next(iter(sd.keys())) if len(sd) else '<<empty state_dict>>'}"
        )
    return out

def build_model():
    # sam_model = sam_model_registry['vit_b'](checkpoint='medsam_vit_b.pth')
    # dense_model = ModelEmb()
    # decoder = MaskDecoder(
    #     num_multimask_outputs=4,
    #     transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
    #     transformer_dim=256,
    #     iou_head_depth=3,
    #     iou_head_hidden_dim=256,
    # )
    # medsam = MedSAMAUTOMULTI(
    #     image_encoder=sam_model.image_encoder,
    #     mask_decoder=decoder,
    #     prompt_encoder=sam_model.prompt_encoder,
    #     dense_encoder=dense_model,
    #     image_size=512,
    # )
    model = UNet2d(in_channels=4, out_channels=4)

    return model

def run_kfold_training(data_dir, num_folds=1, batch_size=2, epochs=20, sweep_config=None):
    test_patients = ['HD_1_DL', 'control_3']
    train_dataset = SyMRI2DDataset(data_dir, split='train', test_patients=test_patients)
    test_dataset = SyMRI2DDataset(data_dir, split='test', test_patients=test_patients)

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        wandb.init(project="medsam", config=sweep_config or {}, reinit=True, name=f"fold_{fold}")
        config = wandb.config
        lr = config.get("lr", 1e-4)

        model = build_model()
        optimizer = Adam(model.parameters(), lr=lr)
        loss_fn = Deep_Supervised_Loss(mode='Focal', activation=False)
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        accelerator = Accelerator()
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

        best_val_dice = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for t1, t2, pd, label in train_loader:
                preds = model(t1, t2, pd)
                loss = loss_fn(preds, label)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                for t1, t2, pd, label in val_loader:
                    output = model(t1, t2, pd)
                    pred = torch.argmax(output, dim=1)
                    pred, label = decollate_batch(pred), decollate_batch(label)
                    dice_metric(y_pred=pred, y=label)

                val_dice = dice_metric.aggregate().item()
                dice_metric.reset()

            wandb.log({
                "fold": fold,
                "epoch": epoch,
                "train_loss": total_loss / len(train_loader),
                "val_dice": val_dice
            })

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                accelerator.save_state(output_dir=f"./checkpoints/fold_{fold}_best")

        wandb.finish()


def dice_score(pred: torch.Tensor, y: torch.Tensor, num_classes: int, ignore_index: int = 0, epsilon: float = 1e-6):
    """
    Compute the Dice score between predicted segmentation and ground truth,
    excluding a specific class (e.g., background).

    Args:
        pred (Tensor): shape (B, H, W), predicted class IDs.
        y (Tensor): shape (B, 1, H, W), true class IDs.
        num_classes (int): total number of classes (including background).
        ignore_index (int): class index to exclude from scoring.
        epsilon (float): small constant to avoid division by zero.

    Returns:
        per_class_dice (Tensor): Dice for each non-ignored class, shape (num_classes - 1,)
        mean_dice (float): Mean Dice score excluding the ignored class.
    """
    B, H, W = pred.shape
    pred_onehot = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2)  # (B, C, H, W)
    y_onehot = F.one_hot(y.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2)  # (B, C, H, W)

    # Flatten
    pred_flat = pred_onehot.reshape(B, num_classes, -1).float()
    y_flat = y_onehot.reshape(B, num_classes, -1).float()

    # Compute intersection and union
    intersection = (pred_flat * y_flat).sum(dim=-1)  # (B, C)
    union = pred_flat.sum(dim=-1) + y_flat.sum(dim=-1)  # (B, C)

    dice = (2 * intersection + epsilon) / (union + epsilon)  # (B, C)
    dice = dice.mean(dim=0)  # average over batch → (C,)

    # Exclude the ignored class (e.g., background)
    mask = torch.ones(num_classes, dtype=torch.bool, device=dice.device)
    mask[ignore_index] = False
    per_class_dice = dice[mask]
    mean_dice = per_class_dice.mean()

    return per_class_dice, mean_dice

def dice_score_3d(pred: torch.Tensor, y: torch.Tensor, num_classes: int, ignore_index: int = 0, epsilon: float = 1e-6):
    """
    Compute the Dice score between predicted segmentation and ground truth,
    excluding a specific class (e.g., background).

    Args:
        pred (Tensor): shape (B, H, W), predicted class IDs.
        y (Tensor): shape (B, 1, H, W), true class IDs.
        num_classes (int): total number of classes (including background).
        ignore_index (int): class index to exclude from scoring.
        epsilon (float): small constant to avoid division by zero.

    Returns:
        per_class_dice (Tensor): Dice for each non-ignored class, shape (num_classes - 1,)
        mean_dice (float): Mean Dice score excluding the ignored class.
    """
    B, D, C, H, W = pred.shape
    y = y.long()
    pred_onehot = F.one_hot(pred.squeeze(2), num_classes=num_classes).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
    y_onehot = F.one_hot(y.squeeze(2), num_classes=num_classes).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)

    # Flatten
    pred_flat = pred_onehot.reshape(B, num_classes, -1).float()
    y_flat = y_onehot.reshape(B, num_classes, -1).float()

    # Compute intersection and union
    intersection = (pred_flat * y_flat).sum(dim=-1)  # (B, C)
    union = pred_flat.sum(dim=-1) + y_flat.sum(dim=-1)  # (B, C)

    dice = (2 * intersection + epsilon) / (union + epsilon)  # (B, C)
    dice = dice.mean(dim=0)  # average over batch → (C,)

    # Exclude the ignored class (e.g., background)
    mask = torch.ones(num_classes, dtype=torch.bool, device=dice.device)
    mask[ignore_index] = False
    per_class_dice = dice[mask]
    mean_dice = per_class_dice.mean()

    return per_class_dice, mean_dice

def run_training(data_dir, batch_size=8, epochs=20, sweep_config=None):
    test_patients = ['HD_1_DL', 'control_3']
    train_dataset = SyMRI2DDataset(data_dir, split='train', test_patients=test_patients)
    test_dataset = SyMRI3DDataset(data_dir, split='test', test_patients=test_patients)

    run = wandb.init(project="medsam", config=sweep_config or {}, reinit=True, name=f"SyMRI_train")
    config = wandb.config
    lr = config.get("lr", 1e-4)
    alpha_warmup = 160

    model = build_model()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = Deep_Supervised_Loss(mode='Focal', activation=False, alpha=torch.tensor([0.001, 1., 1., 1.]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    accelerator = Accelerator()
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    plot_train = False
    plot_eval = False
    best_val_dice = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, label in tqdm(train_loader):
            preds = model(data)
            if plot_train:
                plot_segmentation_grid(data[:,3], torch.argmax(preds,dim=1), label.squeeze(1), file_name='train_plot.png')
            if epoch == alpha_warmup:
                loss_fn = Deep_Supervised_Loss(mode='Focal', activation=False, alpha=torch.tensor([1.0,1.0,1.0,1.0]))
            elif epoch == 120:
                loss_fn = Deep_Supervised_Loss(mode='Focal', activation=False, alpha=torch.tensor([0.1, 1.0, 1.0, 1.0]))
            elif epoch == 80:
                loss_fn = Deep_Supervised_Loss(mode='Focal', activation=False, alpha=torch.tensor([0.01, 1.0, 1.0, 1.0]))
            loss = loss_fn(preds, label.long())
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_pred = []
        all_y = []
        with torch.no_grad():
            for data, label in tqdm(test_loader):
                pred_list = []
                label_list = []
                for idx, slice in enumerate(data.squeeze(0)):
                    output = model(slice.unsqueeze(0))
                    pred = torch.argmax(output, dim=1)
                    pred = accelerator.gather_for_metrics(pred)
                    label = accelerator.gather_for_metrics(label)
                    # if accelerator.is_main_process:
                    #     print(f'pred: {pred.shape}')
                    #     print(f'label: {label.shape}')
                    pred_list.append(pred)
                    label_list.append(label[:, idx])
                pred = torch.stack(pred_list)
                label = torch.stack(label_list)
                all_pred.append(pred)
                all_y.append(label)

            per_class_val_dice, mean_val_dice = dice_score_3d(pred=torch.stack(all_pred, dim=0), y=torch.stack(all_y, dim=0), num_classes=4)

        log_dict = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "mean_val_dice": mean_val_dice.item(),
            "caudate_val_dice": per_class_val_dice[0].item(),
            "palidus_val_dice": per_class_val_dice[1].item(),
            "putamen_val_dice": per_class_val_dice[2].item(),
        }
        wandb.log(log_dict)

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            accelerator.save_state(output_dir=f"./brain_segmentation/checkpoints/img_size_512_conventional_T1w_best_80_120_160_220_scheduler")
    if accelerator.is_main_process:
        model.eval()
        os.makedirs("./brain_segmentation/vis", exist_ok=True)

        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):
                pred_list = []
                label_list = []
                for idx, slice in enumerate(data.squeeze(0)):
                    output = model(slice.unsqueeze(0))
                    pred = torch.argmax(output, dim=1)
                    pred = accelerator.gather_for_metrics(pred)
                    label = accelerator.gather_for_metrics(label)
                    # if accelerator.is_main_process:
                    #     print(f'pred: {pred.shape}')
                    #     print(f'label: {label.shape}')
                    pred_list.append(pred)
                    label_list.append(label[:, idx])
                all_pred = torch.stack(pred_list)
                all_label = torch.stack(label_list)
                plot_segmentation_grid(data[0,:,0], all_pred.squeeze(1), all_label.squeeze(1), file_name=f'val_plot_{i}.png')

    wandb.finish()

def tissue_boundary_cnr(contrast_img, mask_img, tissue_labels, shell_size=1):
    """
    Compute CNR between each tissue and its immediate surrounding voxels.

    Args:
        contrast_img: np.ndarray (2D or 3D)
        mask_img: np.ndarray of same shape, integer tissue labels
        tissue_labels: list of labels to evaluate (e.g., [1,2,3,4])
        shell_size: thickness (in voxels) of the neighborhood shell

    Returns:
        cnr_dict: {label: CNR_value}
    """
    cnr_dict = {}
    for label in tissue_labels:
        tissue_mask = mask_img == label
        if not tissue_mask.any():
            continue

        # Dilate mask to get surrounding voxels
        dilated = ndi.binary_dilation(tissue_mask, iterations=shell_size)
        shell_mask = np.logical_and(dilated, np.logical_not(tissue_mask))

        if not shell_mask.any():
            continue

        # compute stats
        mu_tissue, sigma_tissue = contrast_img[tissue_mask].mean(), contrast_img[tissue_mask].std()
        mu_shell, sigma_shell = contrast_img[shell_mask].mean(), contrast_img[shell_mask].std()

        cnr = abs(mu_tissue - mu_shell) / np.sqrt(sigma_tissue ** 2 + sigma_shell ** 2 + 1e-8)
        cnr_dict[label] = cnr

    return cnr_dict

def eval(data_dir, ckpt_path):
    device = 'cuda:1'
    test_patients = ['HD_1_DL', 'control_3']

    test_dataset = SyMRI3DDataset(data_dir, split='test', test_patients=test_patients)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = build_model()
    model.to(device)

    state_dict = load_file(ckpt_path)
    # state_dict = keep_and_strip_base_model(state_dict, base_prefix="base_model")
    model.load_state_dict(state_dict)

    model.eval()
    os.makedirs("./brain_segmentation/vis", exist_ok=True)
    contrast_names = ['r2', 't1w', 'psir', 't2w']
    cnr_dict = {}
    for k, name in enumerate(contrast_names):
        cnr_dict[name] = []

    with torch.no_grad():
        batched_pred = []
        batched_y = []
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            pred_list = []
            label_list = []
            for idx, slice in enumerate(data.squeeze(0)):
                output = model(slice.unsqueeze(0))
                pred = torch.argmax(output, dim=1)
                # if accelerator.is_main_process:
                #     print(f'pred: {pred.shape}')
                #     print(f'label: {label.shape}')
                pred_list.append(pred)
                label_list.append(label[:, idx])
            all_pred = torch.stack(pred_list)
            all_label = torch.stack(label_list)
            for k, name in enumerate(contrast_names):
                cnr_dict[name].append(tissue_boundary_cnr(data.detach().cpu()[0,:,k], all_label.detach().cpu()[:,0], [1,2,3], 6))
            for j in range(data.shape[2]):
                plot_segmentation_grid(data.squeeze(1)[0, :, j], all_pred.squeeze(1), all_label.squeeze(1), slice_interval=1,
                                       file_name=f'val_plot_contrast{j}_{i}.png')
            batched_pred.append(all_pred)
            batched_y.append(all_label)

    per_class_val_dice, mean_val_dice = dice_score_3d(pred=torch.stack(batched_pred, dim=0), y=torch.stack(batched_y, dim=0), num_classes=4)
    print(f'per_class: {per_class_val_dice}')
    print(f'mean: {mean_val_dice}')

    def mean_cnr_across_patients(cnr_dict):
        mean_results = {}
        for contrast, patient_list in cnr_dict.items():
            tissue_vals = {}
            for patient in patient_list:
                for tissue, val in patient.items():
                    tissue_vals.setdefault(tissue, []).append(val)
            mean_results[contrast] = {t: np.mean(v) for t, v in tissue_vals.items()}
        return mean_results

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        else:
            return obj

    mean_cnr = mean_cnr_across_patients(cnr_dict)
    with open(os.path.join(os.path.dirname(ckpt_path), "mean_cnr.json"), "w") as f:
        json.dump(to_serializable(mean_cnr), f, indent=4)

if __name__ == "__main__":
    # wandb.login()
    # sweep_config = {
    #     "lr": 1e-4,
    #     "epochs": 220,
    #     "batch_size": 32,
    # }
    # data_dir = "/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/dataset/convention_T1W"
    # run_training(data_dir, batch_size=sweep_config["batch_size"], epochs=sweep_config["epochs"], sweep_config=sweep_config)

    data_dir = "/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/dataset/SyMRI_contrast_4c"
    ckpt_path = "/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/brain_segmentation/checkpoints/img_size_512_T1_best_40_70_120_200_4c_scheduler/model.safetensors"
    eval(data_dir, ckpt_path)