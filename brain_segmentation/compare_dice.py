import torch
import nibabel as nib
import numpy as np
from torch.nn.functional import interpolate

from brain_segmentation.utils import dice_score_per_class


def load_and_transform_nii(nii_file, target_shape=(106, 512, 512)):
    # Load the .nii.gz file
    img = nib.load(nii_file)
    data = img.get_fdata()

    # Convert the NIfTI data into a PyTorch tensor (ensuring it's integer type initially)
    data_tensor = torch.from_numpy(data)

    # Perform value conversions
    transformed_data = torch.zeros_like(data_tensor)  # Initialize with zeros (default value)

    # Map 11 and 50 to 1
    transformed_data[(data_tensor == 11) | (data_tensor == 50)] = 1
    # Map 12 and 51 to 3
    transformed_data[(data_tensor == 12) | (data_tensor == 51)] = 3
    # Map 13 and 52 to 2
    transformed_data[(data_tensor == 13) | (data_tensor == 52)] = 2
    # All other values remain 0

    # Convert back to numpy (integer type)
    transformed_data_np = transformed_data.numpy().astype(np.int16)

    # Print the shape of the resized and transformed data
    print(f"Transformed shape: {transformed_data_np.shape}")

    return transformed_data_np.squeeze(), img.affine

def save_transformed_data(transformed_data, affine, output_file):
    # Create a new Nifti1Image with the transformed data and the original affine
    new_img = nib.Nifti1Image(transformed_data, affine)
    # Save the transformed data to a new .nii.gz file
    nib.save(new_img, output_file)
    print(f"Saved transformed data to {output_file}")


# Example usage
nii_file_path = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/new_log/eval/Freesurfer/aseg_control3_resampled.nii.gz'
output_file_path = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/new_log/eval/Freesurfer/aseg_control3_orig_size.nii.gz'
gt_file_path = '/home/yxpengcs/Datasets/MRI/CHDI_Multi_Contrast/SyMRI_processed_DL/control_3/seg.nii.gz'
model_pred_file_path = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/new_log/eval/UNetWeightedFocal1000xWeightedFinetune1x/pred/control_3_seg.nii.gz'

# Load, transform, and save the data
transformed_data, affine = load_and_transform_nii(nii_file_path)
save_transformed_data(transformed_data, affine, output_file_path)
transformed_data_tensor = torch.from_numpy(transformed_data).long()

gt_img = nib.load(gt_file_path)
gt = gt_img.get_fdata()
gt_tensor = torch.from_numpy(gt).long().permute(2, 1, 0)

model_pred_img = nib.load(model_pred_file_path)
model_pred = model_pred_img.get_fdata()
model_pred_tensor = torch.from_numpy(model_pred).long().permute(2, 1, 0).long()

freesurfer_dice = dice_score_per_class(transformed_data_tensor.unsqueeze(0), gt_tensor.unsqueeze(0), 4, input_logit=False)
model_dice = dice_score_per_class(model_pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0), 4, input_logit=False)
print(f'freesurfer_dice: {freesurfer_dice}, model_dice: {model_dice}')