import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm


def convert_syMRI_to_2d(input_dir, output_dir, with_label=False):
    os.makedirs(output_dir, exist_ok=True)
    sample_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for sample_idx, name in enumerate(tqdm(sample_names, desc="Processing samples")):
        sample_path = os.path.join(input_dir, name)
        if not os.path.isdir(sample_path):
            continue

        r2 = nib.load(os.path.join(sample_path, "R2.nii.gz")).get_fdata()
        t1w = nib.load(os.path.join(sample_path, "T1W.nii.gz")).get_fdata()
        psir = nib.load(os.path.join(sample_path, "PSIR.nii.gz")).get_fdata()
        t2w = nib.load(os.path.join(sample_path, "T2W.nii.gz")).get_fdata()
        label = nib.load(os.path.join(sample_path, "seg.nii.gz")).get_fdata() if with_label else None

        os.makedirs(os.path.join(output_dir, f"{name}"), exist_ok=True)

        for i in range(r2.shape[2]):
            slice_dict = {
                'r2': torch.tensor(r2[:, :, i], dtype=torch.float32).unsqueeze(0),
                't1w': torch.tensor(t1w[:, :, i], dtype=torch.float32).unsqueeze(0),
                'psir': torch.tensor(psir[:, :, i], dtype=torch.float32).unsqueeze(0),
                't2w': torch.tensor(t2w[:, :, i], dtype=torch.float32).unsqueeze(0),
            }
            if with_label:
                slice_dict['label'] = torch.tensor(label[:, :, i], dtype=torch.long).unsqueeze(0)

            out_path = os.path.join(output_dir, f"{name}", f"slice_{i:03d}.pt")
            torch.save(slice_dict, out_path)

# Example usage:
convert_syMRI_to_2d('/home/yxpengcs/Datasets/MRI/CHDI_Multi_Contrast/SyMRI_processed_DL/', "dataset/SyMRI_contrast_4c", with_label=True)
