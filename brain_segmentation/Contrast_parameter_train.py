import json
import math
import os
import torch
import wandb
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
from accelerate import Accelerator
from torch.optim import Adam
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from tqdm import tqdm
import torch.nn.functional as F
from safetensors.torch import load_file
from segmentation_train import plot_segmentation_grid, dice_score_3d

from monai.config import KeysCollection

from brain_segmentation.ParameterLayer import SyMRIPSIRParamLayerSigmoid, SyMRIT1WMP2RAGEParamLayerSigmoid, \
    SyMRIR2ParamLayer, SyMRIPSIRParamLayerLog, SyMRIT1WMP2RAGEParamLayerLog
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

class SyMRISegmentation(nn.Module):
    def __init__(self, psir_layer, t1w_layer, r2_layer, norm_layer, base_model: nn.Module, plot=True):
        super().__init__()
        self.psir_layer = psir_layer
        self.t1w_layer = t1w_layer
        self.r2_layer = r2_layer
        self.norm_layer = norm_layer
        self.base_model = base_model
        self.plot = plot

    def forward(self, t1, t2, pd, *args, **kwargs):
        psir = self.psir_layer(t1, t2)
        t1w = self.t1w_layer(t1)
        r2 = self.r2_layer(t2)
        x = torch.cat([r2, t1w, psir], dim=1)
        x = self.norm_layer(x)
        if self.plot:
            return self.base_model(x, *args, **kwargs), x
        else:
            return self.base_model(x, *args, **kwargs)

class FixedNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W) or (B, 1, H, W)
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        return (x - mean) / (std + self.eps)

class LearnableNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        # Normalize over [C, H, W] per sample
        self.ln = nn.LayerNorm([num_channels, 1, 1], elementwise_affine=True, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W)
        # Flatten spatially to Tensors → same shape
        return self.ln(x)

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
        common_keys: KeysCollection = ["t1", "t2", "pd"]
        label_key = "label"

        if split == 'train':
            self.transform = Compose([
                ResizeWithPadOrCropd(keys=common_keys + [label_key], spatial_size=(512, 512)),
                # NormalizeIntensityd(keys=common_keys),
                RandFlipd(keys=common_keys + [label_key], prob=0.5, spatial_axis=1),
                RandRotate90d(keys=common_keys + [label_key], prob=0.5, max_k=3),
                ToTensord(keys=common_keys + [label_key]),
            ])
        else:
            self.transform = Compose([
                ResizeWithPadOrCropd(keys=common_keys + [label_key], spatial_size=(512, 512)),
                # NormalizeIntensityd(keys=common_keys),
                ToTensord(keys=common_keys + [label_key]),
            ])

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        data = torch.load(self.slice_paths[idx])  # dict with keys: r1, r2, pd, label (optional)
        # Apply MONAI transforms
        transformed = self.transform(data)

        # Stack input channels
        t1, t2, pd = transformed["t1"], transformed["t2"], transformed["pd"]

        if "label" in transformed:
            return t1, t2, pd, transformed["label"]
        else:
            return t1, t2, pd

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
        common_keys: KeysCollection = ["t1", "t2", "pd"]
        label_key = "label"

        if split == 'train':
            self.transform = Compose([
                ResizeD(keys=common_keys, spatial_size=(512, 512), mode='bilinear'),  # for images
                ResizeD(keys=[label_key], spatial_size=(512, 512), mode='nearest'),
                # NormalizeIntensityd(keys=common_keys),
                RandFlipd(keys=common_keys + [label_key], prob=0.5, spatial_axis=1),
                RandRotate90d(keys=common_keys + [label_key], prob=0.5, max_k=3),
                ToTensord(keys=common_keys + [label_key]),
            ])
        else:
            self.transform = Compose([
                ResizeD(keys=common_keys, spatial_size=(512, 512), mode='bilinear'),  # for images
                ResizeD(keys=[label_key], spatial_size=(512, 512), mode='nearest'),
                # NormalizeIntensityd(keys=common_keys),
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
        data['t1'] = torch.cat([item['t1'] for item in data_list])
        data['t2'] = torch.cat([item['t2'] for item in data_list])
        data['pd'] = torch.cat([item['pd'] for item in data_list])
        data['label'] = torch.cat([item['label'] for item in data_list])

        if "label" in data.keys():
            return data['t1'], data["t2"], data['pd'], data["label"].long()
        else:
            return data["t1w"]

def set_phase(model: SyMRISegmentation, phase: str):
    """phase in {'contrast','seg','both'}"""
    train_seg = (phase != "contrast")
    train_contrast = (phase != "seg")
    # segmentation (UNet)
    for p in model.base_model.parameters():
        p.requires_grad = train_seg
    # contrast layers (+ norm if it has params)
    for m in [model.psir_layer, model.t1w_layer, model.r2_layer, model.norm_layer]:
        for p in getattr(m, "parameters", lambda: [])():
            p.requires_grad = train_contrast

def build_model(reparam_type):
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
    base_model = UNet2d(in_channels=3, out_channels=4)

    if reparam_type == 'log':
        PSIR_layer = SyMRIPSIRParamLayerLog()
        T1W_layer = SyMRIT1WMP2RAGEParamLayerLog()
    elif reparam_type == 'sigmoid':
        PSIR_layer = SyMRIPSIRParamLayerSigmoid()
        T1W_layer = SyMRIT1WMP2RAGEParamLayerSigmoid()
    R2_layer = SyMRIR2ParamLayer()
    norm_layer = FixedNorm()

    ckpt_path = "/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/brain_segmentation/checkpoints/img_size_512_T1_best_40_70_120_200/model.safetensors"
    state_dict = load_file(ckpt_path)

    base_model.load_state_dict(state_dict)

    for param in base_model.parameters():
        param.requires_grad = False

    return SyMRISegmentation(psir_layer=PSIR_layer, t1w_layer=T1W_layer, r2_layer=R2_layer, norm_layer=norm_layer, base_model=base_model)


def build_optimizer_and_scheduler(
    model: SyMRISegmentation,
    base_lr: float = 3e-4,
    contrast_lr: float = 1e-2,
    diff_lr_per_param = False,
    weight_decay: float = 0.001,
    cosine_T_max: int = 100,   # epochs for cosine
    exp_gamma: float = 0.9     # exp decay per epoch for contrast params
):
    """
    Returns:
      opt: AdamW with 2 param groups (base vs. contrast)
      sched: LambdaLR that applies cosine to base group, exponential to contrast group
      get_lrs: lambda to read current group LRs as (base_lr, contrast_lr)
    """

    # --- collect parameter groups ---
    contrast_params = []
    contrast_params += list(model.psir_layer.parameters())
    contrast_params += list(model.t1w_layer.parameters())
    contrast_params += list(model.r2_layer.parameters())

    # everything else = base group
    contrast_param_ids = {id(p) for p in contrast_params}
    base_params = [p for p in model.parameters() if id(p) not in contrast_param_ids]

    if diff_lr_per_param:
        opt = torch.optim.AdamW([
            {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": [model.psir_layer.raw_TR], "lr": contrast_lr * 2, "weight_decay": 0.0},
            {"params": [model.psir_layer.raw_TE], "lr": contrast_lr, "weight_decay": 0.0},
            {"params": [model.psir_layer.raw_TI], "lr": contrast_lr / 2, "weight_decay": 0.0},
            {"params": [model.t1w_layer.raw_TI], "lr": contrast_lr, "weight_decay": 0.0},
        ])
    else:
        opt = torch.optim.AdamW([
            {"params": base_params,     "lr": base_lr,     "weight_decay": weight_decay},
            {"params": contrast_params, "lr": contrast_lr, "weight_decay": 0.0},  # usually no WD on tiny scalars
        ])

    # --- per-group LR schedules via LambdaLR ---
    # group 0 (base): cosine schedule in [0,1]
    def lambda_base(epoch):
        # classic cosine anneal: 0.5 * (1 + cos(pi * epoch / T_max))
        # clamp T_max>=1 to avoid div by zero if someone passes 0
        T = max(1, cosine_T_max)
        return 0.5 * (1.0 + math.cos(math.pi * epoch / T))

    # group 1 (contrast): exponential decay gamma**epoch
    def lambda_contrast(epoch):
        return exp_gamma ** epoch

    if diff_lr_per_param:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=[lambda_base, lambda_contrast, lambda_contrast, lambda_contrast, lambda_contrast]
        )
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=[lambda_base, lambda_contrast]
        )

    get_lrs = lambda: (sched.get_last_lr()[0], sched.get_last_lr()[1])
    return opt, sched, get_lrs

def run_kfold_training(data_dir, num_folds=1, batch_size=2, epochs=20, sweep_config=None):
    test_patients = ['HD_1_DL', 'control_3']
    train_dataset = SyMRI2DDataset(data_dir, split='train', test_patients=test_patients)
    test_dataset = SyMRI2DDataset(data_dir, split='test', test_patients=test_patients)

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        run = wandb.init(project="syrmi-medsam", config=sweep_config or {}, reinit=True, name=f"fold_{fold}")
        config = wandb.config
        lr = config.get("lr", 1e-4)

        model = build_model()
        optimizer = torch.optim.Adam([
            {"params": model.base_model.parameters(), "lr": lr},
            {"params": model.syri_layer.parameters(), "lr": lr * 0.1},  # or even lower
        ])
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

def run_training(data_dir, batch_size=8, epochs=20, sweep_config=None):
    test_patients = ['HD_1_DL', 'control_3']
    train_dataset = SyMRI2DDataset(data_dir, split='train', test_patients=test_patients)
    test_dataset = SyMRI3DDataset(data_dir, split='test', test_patients=test_patients)

    run = wandb.init(project="syrmi-medsam", config=sweep_config or {}, reinit=True, name=f"SyMRI_train")
    config = wandb.config
    base_lr = config.get("base_lr", 3e-4)
    contrast_lr = config.get("contrast_lr", 1e-3)
    contrast_period = config.get("contrast_period", 3)  # epochs training contrast
    seg_period      = config.get("seg_period", 3)       # epochs training segmentation
    cycle_len = contrast_period + seg_period
    bgweight = 1.
    reparam_type = 'log'
    exp_gamma = 0.99
    cosine_T_max = 10000
    in_both = 0
    diff_lr_per_param = False

    model = build_model(reparam_type)

    state_dict = load_file("/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/brain_segmentation/checkpoints/Interleave_then_both_diff_lr_img_size_512_Log_reparam_diff_init_bgweight_1_exp_decay_lr/model.safetensors")
    model.load_state_dict(state_dict)
    optimizer, sched, get_lrs = build_optimizer_and_scheduler(
        model,
        base_lr=base_lr,
        diff_lr_per_param=diff_lr_per_param,
        contrast_lr=contrast_lr,
        cosine_T_max=cosine_T_max,
        exp_gamma=exp_gamma
    )
    # loss_fn = Deep_Supervised_Loss(mode='Focal', activation=False, alpha=torch.tensor([bgweight, 1., 1., 1.]))
    loss_fn = Deep_Supervised_Loss(mode='Dice', activation=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    accelerator = Accelerator()
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    plot_train = False

    best_val_dice = 0
    best_epoch = 0
    save_dir = "./brain_segmentation/checkpoints/Interleave_then_both_img_size_512_Log_reparam_diff_init_bgweight_1_exp_decay_lr_dice_finetune"

    for epoch in range(epochs):
        in_contrast = (epoch % cycle_len) < contrast_period
        phase = "contrast" if in_contrast else "seg"
        if epoch >= in_both:
            phase = "both"
        set_phase(model, phase)
        model.train()
        total_loss = 0
        for t1, t2, pd, label in tqdm(train_loader):
            preds, data = model(t1, t2, pd)
            if plot_train:
                plot_segmentation_grid(data[:,0].detach().cpu(), torch.argmax(preds,dim=1), label.squeeze(1), slice_interval=1, file_name='train_plot_0.png')
                plot_segmentation_grid(data[:,1].detach().cpu(), torch.argmax(preds,dim=1), label.squeeze(1), slice_interval=1, file_name='train_plot_1.png')
                plot_segmentation_grid(data[:,2].detach().cpu(), torch.argmax(preds,dim=1), label.squeeze(1), slice_interval=1, file_name='train_plot_2.png')
            loss = loss_fn(preds, label.long())
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        sched.step()
        base_lr_now, contrast_lr_now = get_lrs()
        model.eval()
        all_pred = []
        all_y = []
        with torch.no_grad():
            for t1, t2, pd, label in tqdm(test_loader):
                t1 = t1.unsqueeze(2)
                t2 = t2.unsqueeze(2)
                pd = pd.unsqueeze(2)
                pred_list = []
                label_list = []
                for slice_idx in range(t1.shape[1]):
                    output, data = model(t1[:,slice_idx], t2[:,slice_idx], pd[:,slice_idx])
                    pred = torch.argmax(output, dim=1)
                    pred = accelerator.gather_for_metrics(pred)
                    label = accelerator.gather_for_metrics(label)
                    pred_list.append(pred)
                    label_list.append(label[:, slice_idx])
                case_pred = torch.stack(pred_list)
                case_label = torch.stack(label_list)
                all_pred.append(case_pred)
                all_y.append(case_label)
            per_class_val_dice, mean_val_dice = dice_score_3d(pred=torch.stack(all_pred, dim=0),
                                                                  y=torch.stack(all_y, dim=0), num_classes=4)

        psir_TR = torch.exp(torch.log(torch.tensor(1)) + torch.sigmoid(model.psir_layer.raw_TR.data) * (torch.log(torch.tensor(10000)) - torch.log(torch.tensor(1)))).item()
        psir_TE = torch.exp(torch.log(torch.tensor(1)) + torch.sigmoid(model.psir_layer.raw_TE.data) * (torch.log(torch.tensor(100)) - torch.log(torch.tensor(1)))).item()
        psir_TI = torch.exp(torch.log(torch.tensor(200)) + torch.sigmoid(model.psir_layer.raw_TI.data) * (torch.log(torch.tensor(3000)) - torch.log(torch.tensor(200)))).item()
        t1w_TE = torch.exp(torch.log(torch.tensor(10)) + torch.sigmoid(model.t1w_layer.raw_TI.data) * (torch.log(torch.tensor(3500)) - torch.log(torch.tensor(10)))).item()

        log_dict = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "base_lr": base_lr_now,
            "contrast_lr": contrast_lr_now,
            "mean_val_dice": mean_val_dice.item(),
            "caudate_val_dice": per_class_val_dice[0].item(),
            "palidus_val_dice": per_class_val_dice[1].item(),
            "putamen_val_dice": per_class_val_dice[2].item(),
            "psir_TR": psir_TR,
            "psir_TE": psir_TE,
            "psir_TI": psir_TI,
            "t1w_TE": t1w_TE,
        }
        wandb.log(log_dict)

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            best_epoch = epoch
            accelerator.save_state(output_dir=save_dir)

            metadata = {
                "best_val_dice": float(best_val_dice),
                "best_epoch": best_epoch,
                "base_lr": base_lr,
                "contrast_lr": contrast_lr,
                "contrast_period": contrast_period,
                "seg_period": seg_period,
                "cycle_len": cycle_len,
                "in_both": in_both,
                "bgweight": bgweight,
                "diff_lr_per_param": diff_lr_per_param,
                "reparam_type": reparam_type,
                "exp_gamma": exp_gamma,
                "cosine_T_max": cosine_T_max,
                "psir_TR": psir_TR,
                "psir_TE": psir_TE,
                "psir_TI": psir_TI,
                "t1w_TE": t1w_TE,
            }
            with open(f"{save_dir}/best_metrics.json", "w") as f:
                json.dump(metadata, f, indent=4)

    if accelerator.is_main_process:
        model.eval()
        os.makedirs("./brain_segmentation/vis", exist_ok=True)

        with torch.no_grad():
            for i, (t1, t2, pd, label) in enumerate(test_loader):
                t1 = t1.unsqueeze(2)
                t2 = t2.unsqueeze(2)
                pd = pd.unsqueeze(2)
                pred_list = []
                label_list = []
                data_list = []
                for slice_idx in range(t1.shape[1]):
                    output, data = model(t1[:,slice_idx], t2[:,slice_idx], pd[:,slice_idx])
                    pred = torch.argmax(output, dim=1)
                    pred = accelerator.gather_for_metrics(pred)
                    label = accelerator.gather_for_metrics(label)
                    # if accelerator.is_main_process:
                    #     print(f'pred: {pred.shape}')
                    #     print(f'label: {label.shape}')
                    pred_list.append(pred)
                    label_list.append(label[:, slice_idx])
                    data_list.append(data)
                all_pred = torch.stack(pred_list)
                all_label = torch.stack(label_list)
                all_data = torch.stack(data_list)
                plot_segmentation_grid(all_data.squeeze(1)[:,0], all_pred.squeeze(1), all_label.squeeze(1), file_name=f'val_plot_contrast0_{i}.png')
                plot_segmentation_grid(all_data.squeeze(1)[:,1], all_pred.squeeze(1), all_label.squeeze(1), file_name=f'val_plot_contrast1_{i}.png')
                plot_segmentation_grid(all_data.squeeze(1)[:,2], all_pred.squeeze(1), all_label.squeeze(1), file_name=f'val_plot_contrast2_{i}.png')


    wandb.finish()

def eval(data_dir, ckpt_path):
    test_patients = ['HD_1_DL', 'control_3']

    device = 'cuda'

    test_dataset = SyMRI3DDataset(data_dir, split='test', test_patients=test_patients)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = build_model(reparam_type='log')

    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    os.makedirs("./brain_segmentation/vis", exist_ok=True)

    with torch.no_grad():
        batched_pred = []
        batched_y = []
        for i, (t1, t2, pd, label) in enumerate(test_loader):
            t1 = t1.unsqueeze(2).to(device)
            t2 = t2.unsqueeze(2).to(device)
            pd = pd.unsqueeze(2).to(device)
            label = label.to(device)
            pred_list = []
            label_list = []
            data_list = []
            for slice_idx in range(t1.shape[1]):
                output, data = model(t1[:, slice_idx], t2[:, slice_idx], pd[:, slice_idx])
                pred = torch.argmax(output, dim=1)
                # if accelerator.is_main_process:
                #     print(f'pred: {pred.shape}')
                #     print(f'label: {label.shape}')
                pred_list.append(pred)
                label_list.append(label[:, slice_idx])
                data_list.append(data)
            all_pred = torch.stack(pred_list)
            all_label = torch.stack(label_list)
            all_data = torch.stack(data_list)
            plot_segmentation_grid(all_data.squeeze(1)[:, 0], all_pred.squeeze(1), all_label.squeeze(1), slice_interval=1,
                                   file_name=f'val_plot_contrast0_{i}.png')
            plot_segmentation_grid(all_data.squeeze(1)[:, 1], all_pred.squeeze(1), all_label.squeeze(1), slice_interval=1,
                                   file_name=f'val_plot_contrast1_{i}.png')
            plot_segmentation_grid(all_data.squeeze(1)[:, 2], all_pred.squeeze(1), all_label.squeeze(1), slice_interval=1,
                                   file_name=f'val_plot_contrast2_{i}.png')
            batched_pred.append(all_pred)
            batched_y.append(all_label)
    per_class_val_dice, mean_val_dice = dice_score_3d(pred=torch.stack(batched_pred, dim=0),
                                                      y=torch.stack(batched_y, dim=0), num_classes=4)
    print(f'per_class: {per_class_val_dice}')
    print(f'mean: {mean_val_dice}')

if __name__ == "__main__":
    # wandb.login()
    # sweep_config = {
    #     "base_lr": 3e-4,
    #     "contrast_lr": 1e-2,
    #     "epochs": 140,
    #     "batch_size": 8,
    #     "contrast_period": 10,
    #     "seg_period": 10,
    # }
    # data_dir = "/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/dataset/SyMRI_raw"
    # run_training(data_dir, batch_size=sweep_config["batch_size"], epochs=sweep_config["epochs"], sweep_config=sweep_config)


    data_dir = "/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/dataset/SyMRI_raw"
    ckpt_path = "/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/brain_segmentation/checkpoints/Interleave_then_both_img_size_512_Log_reparam_diff_init_bgweight_1_exp_decay_lr_dice_finetune/model.safetensors"
    eval(data_dir, ckpt_path)
