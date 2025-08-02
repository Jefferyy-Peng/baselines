import os

import nibabel as nib
import numpy as np

def dice_score_per_label(gt, pred, label):
    """
    Compute the Dice score for a specific label.
    """
    gt_binary = (gt == label)  # Binary mask for the label in GT
    pred_binary = (pred == label)  # Binary mask for the label in predicted

    intersection = np.sum(gt_binary * pred_binary)  # Intersection
    gt_sum = np.sum(gt_binary)
    pred_sum = np.sum(pred_binary)

    if gt_sum + pred_sum == 0:  # Avoid division by zero
        return 1.0  # Perfect score if both are empty for this label

    dice = (2.0 * intersection) / (gt_sum + pred_sum)
    return dice

def compute_dice_score(gt_path, pred_path):
    """
    Compute Dice scores for all labels in the mask.
    """
    # Load the ground truth and predicted masks
    gt_nii = nib.load(gt_path)
    pred_nii = nib.load(pred_path)

    gt_data = gt_nii.get_fdata().astype(np.int32)  # Convert to integers
    pred_data = pred_nii.get_fdata().astype(np.int32)

    # Unique labels in the GT and prediction (assumes 0, 1, 2, 3 are the only values)
    labels = [0, 1, 2, 3]

    dice_scores = {}
    for label in labels:
        dice_scores[label] = dice_score_per_label(gt_data, pred_data, label)

    return dice_scores

# Paths to the .nii.gz files
gt_path = "/home/yxpengcs/Datasets/MRI/CHDI_Multi_Contrast/SyMRI_processed_DL/control_3/seg.nii.gz"
# pred_path = "/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/dataset/ucsd_multi_contrast_segdata/FreeSurferResults/Control_3_freesurf.nii.gz"
pred_path = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/new_log/eval/UNetWeightedFocal1000xWeightedFinetune1x/pred/control_3_seg.nii.gz'
# Compute the Dice scores
dice_scores = compute_dice_score(gt_path, pred_path)

# Print the results
for label, dice in dice_scores.items():
    print(f"Dice score for label {label}: {dice:.4f}")
log_dir = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/dataset/ucsd_multi_contrast_segdata/FreeSurferResults'
file_name = 'Control_3_UNet.txt'
os.system(f'cd {log_dir} && touch {file_name} && echo "Caudate: {dice_scores[1]}, Globus:{dice_scores[2]}, putamen: {dice_scores[3]}" >> {file_name}')
