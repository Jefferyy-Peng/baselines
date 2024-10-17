import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


def resample_mask_to_target(source_img, target_img):
    """
    Resample a mask (integer) source image to match the voxel grid and orientation of a target image.

    Nearest-neighbor interpolation is used to ensure integer values are maintained.

    Args:
    - source_img: nibabel Nifti1Image (the image to be resampled)
    - target_img: nibabel Nifti1Image (the reference image to align to)

    Returns:
    - resampled_img: nibabel Nifti1Image (the resampled image aligned to the target)
    """
    # Get the data and affine of the source and target images
    source_data = source_img.get_fdata()
    source_affine = source_img.affine
    target_affine = target_img.affine
    target_shape = target_img.shape

    # Compute the transformation from source to target space
    transformation = np.linalg.inv(target_affine).dot(source_affine)

    # Resample the source image using nearest-neighbor interpolation
    resampled_data = affine_transform(
        source_data, transformation[:3, :3], offset=transformation[:3, 3],
        output_shape=target_shape, order=0,  # Order 0 is nearest-neighbor interpolation
        mode='constant', cval=0)  # Set background to 0 for out-of-bounds areas

    # Ensure that resampled data is in integer format
    resampled_data = np.rint(resampled_data).astype(np.int16)  # Convert to int16 (or other integer type)

    # Create a new NIfTI image with the resampled data and target's affine
    resampled_img = nib.Nifti1Image(resampled_data, target_affine)

    return resampled_img


# Example usage
source_nii_path = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/new_log/eval/Freesurfer/aseg_control3.nii.gz'
target_nii_path = '/home/yxpengcs/Datasets/MRI/CHDI_Multi_Contrast/SyMRI_processed_DL/control_3/seg.nii.gz'
output_resampled_path = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/brain_segmentation/new_log/eval/Freesurfer/aseg_control3_resampled.nii.gz'

# Load the source and target NIfTI images
source_img = nib.load(source_nii_path)
target_img = nib.load(target_nii_path)

# Resample source to target's space
resampled_img = resample_mask_to_target(source_img, target_img)

# Save the resampled image
nib.save(resampled_img, output_resampled_path)
print(f"Resampled image saved to {output_resampled_path}")