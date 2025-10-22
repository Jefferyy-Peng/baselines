import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import affine_transform
import torch.nn.functional as F
from nibabel.processing import resample_from_to

from brain_segmentation.segmentation_train import plot_segmentation_grid


def resample_mask_to_target(source_img, target_img):
    """
    Resample integer label mask to target grid with nearest neighbor.
    Uses NiBabel's robust affine logic.
    """
    resampled_img = resample_from_to(source_img, target_img, order=0)  # order=0 = nearest neighbor
    data = np.asanyarray(resampled_img.dataobj).astype(np.int16)
    return nib.Nifti1Image(data, target_img.affine)

# Example usage
source_nii_path = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/new_log/eval/Freesurfer/aseg_HD_1.nii.gz'
target_nii_path = '/home/yxpengcs/Datasets/MRI/CHDI_Multi_Contrast/SyMRI_processed_DL/HD_1_DL/seg.nii.gz'
output_resampled_path = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/new_log/eval/Freesurfer/aseg_HD_1_resampled.nii.gz'

# Load the source and target NIfTI images
source_img = nib.load(source_nii_path)
target_img = nib.load(target_nii_path)

# Resample source to target's space
resampled_img = resample_mask_to_target(source_img, target_img)

# Save the resampled image
nib.save(resampled_img, output_resampled_path)
print(f"Resampled image saved to {output_resampled_path}")


def load_mask(path):
    nii = nib.load(path)
    data = np.asanyarray(nii.dataobj).astype(np.int16)
    return torch.from_numpy(data)

# --- 2. Remap the prediction mask ---
def remap_pred_mask(pred_mask):
    """
    pred_mask: torch.Tensor of shape (D,H,W), label space same as FreeSurfer IDs.
    Returns remapped mask with:
      0 = background
      1 = caudate
      2 = pallidum
      3 = putamen
    """
    remapped = torch.zeros_like(pred_mask, dtype=torch.uint8)

    caudate = torch.isin(pred_mask, torch.tensor([11, 50], dtype=pred_mask.dtype))
    pallidum = torch.isin(pred_mask, torch.tensor([13, 52], dtype=pred_mask.dtype))
    putamen  = torch.isin(pred_mask, torch.tensor([12, 51], dtype=pred_mask.dtype))

    remapped[caudate] = 1
    remapped[pallidum] = 2
    remapped[putamen] = 3

    return remapped

# --- 3. Dice computation ---
def dice_per_class(pred, gt, num_classes=4, eps=1e-6):
    pred_onehot = F.one_hot(pred.long(), num_classes=num_classes).permute(3,0,1,2).float()
    gt_onehot   = F.one_hot(gt.long(),   num_classes=num_classes).permute(3,0,1,2).float()

    intersection = (pred_onehot * gt_onehot).sum(dim=(1,2,3))
    union = pred_onehot.sum(dim=(1,2,3)) + gt_onehot.sum(dim=(1,2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice[1:]  # skip background

pred_mask = load_mask(output_resampled_path)
gt_mask   = load_mask(target_nii_path)

# remap prediction only
pred_remap = remap_pred_mask(pred_mask)

dice_vals = dice_per_class(pred_remap, gt_mask)

plot_segmentation_grid(torch.zeros(106,512,512), pred_remap.permute(2,0,1), gt_mask.permute(2,0,1), slice_interval=1,
                       file_name=f'freesurf_plot_HD1.png')