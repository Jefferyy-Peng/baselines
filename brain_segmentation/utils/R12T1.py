import nibabel as nib
import numpy as np
import os

def convert_r_nii_to_t(input_path, output_path=None):
    """
    Convert an R1 NIfTI file (R1 = 1 / T1) to a T1 NIfTI file.
    """
    # Load R1 image
    r1_img = nib.load(input_path)
    r1_data = r1_img.get_fdata()

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        t1_data = np.where(r1_data != 0, 1.0 / r1_data, 0.0)

    # Create new NIfTI image
    t1_img = nib.Nifti1Image(t1_data, affine=r1_img.affine, header=r1_img.header)

    # Determine save path
    if output_path is None:
        output_path = input_path.replace("/R", "/T")

    # Save the T1 image
    nib.save(t1_img, output_path)
    print(f"Saved T map to {output_path}")

# Example:
convert_r_nii_to_t("/home/yxpengcs/Datasets/MRI/CHDI_Multi_Contrast/SyMRI_processed_DL/HD_2_DL/R2.nii.gz")
