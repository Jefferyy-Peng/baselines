import os
import pickle

import SimpleITK as sitk
import pandas as pd
from matplotlib import pyplot as plt
from monai.data.image_reader import nrrd
from picai_prep import atomic_image_write
from tqdm import tqdm
import numpy as np
import h5py
import random
import nibabel as nib
from picai_prep.preprocessing import resample_img, crop_or_pad, PreprocessingSettings
from utils import Sample
import pydicom

import SimpleITK as sitk

settings = {
    "preprocessing": {
        # resample and perform centre crop:
        "matrix_size": [
            24,
            384,
            384
        ],
        "spacing": [
            3.0,
            0.5,
            0.5
        ],
    }
}

df = pd.read_csv('/data/nvme1/meng/Prostate-Diagnosis/metadata.csv')

# relevant_descriptions = [
#     "T2 Weighted Axial",
#     "DWI",
#     "Apparent Diffusion Coefficient"
# ]
#
# # Filter the DataFrame for the relevant series descriptions
# filtered_df = df[df["Series Description"].isin(relevant_descriptions)]
data_path = '/data/nvme1/meng/Prostate-Diagnosis/'
segmentation_paths = ['/data/nvme1/meng/Prostate-Diagnosis/Multi-component-segmentation', '/data/nvme1/meng/Prostate-Diagnosis/NCI-challenge-segmentation']
MC_seg_files = os.listdir(segmentation_paths[0])
NCI_seg_files = os.listdir(segmentation_paths[1])
cols = ['ID', 't2', 'adc', 'dwi'] + ['adc_tumor_reader1'] + ['t2_anatomy_reader1']
grouped = df.groupby(["Subject ID"])
subject_study_dict = {}
for subject_id, group in grouped:
    subject_id = subject_id[0]
    subject_num = subject_id.split('-')[1] + '-' + subject_id.split('-')[2]
    for _, row in group.iterrows():
        print(f"  File Location: {row['File Location']}")
        path = os.path.join(data_path, row['File Location'])
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(path)  # Get all series IDs
        if len(series_ids) != 1:
            print('here')

        if not series_ids:
            continue

        # Select the first series (or iterate if multiple series exist)
        dicom_files = reader.GetGDCMSeriesFileNames(path, series_ids[0])
        reader.SetFileNames(dicom_files)

        # Load the DICOM series as a 3D volume
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)
        if len(image_array.shape) > 3:
            squeezed_array = np.squeeze(image_array)
            squeezed_sitk_image = sitk.GetImageFromArray(squeezed_array)
            # squeezed_sitk_image.CopyInformation(image)
            # Adjust metadata only for dimensions that remain
            if len(image.GetSize()) == len(squeezed_sitk_image.GetSize()):
                squeezed_sitk_image.CopyInformation(image)
            else:
                # Adjust spacing and origin for the remaining dimensions
                original_spacing = image.GetSpacing()
                original_origin = image.GetOrigin()
                original_direction = image.GetDirection()

                original_size = image.GetSize()
                # Determine remaining dimensions
                remaining_dims = [i for i, size in enumerate(original_size) if size > 1]

                # Adjust spacing, origin, and direction
                new_spacing = tuple(original_spacing[i] for i in remaining_dims)
                new_origin = tuple(original_origin[i] for i in remaining_dims)

                # Reshape the direction matrix and reduce dimensions
                direction_matrix = np.array(original_direction).reshape(len(original_size), -1)
                reduced_direction_matrix = direction_matrix[np.ix_(remaining_dims, remaining_dims)]
                new_direction = tuple(reduced_direction_matrix.flatten())

                squeezed_sitk_image.SetSpacing(new_spacing)
                squeezed_sitk_image.SetOrigin(new_origin)
                squeezed_sitk_image.SetDirection(new_direction)

            image = squeezed_sitk_image
        if row['Series Description'] == 'T2WTSEAX':
            t2 = image
        # if row['Series Description'] == 'AX BLISSGAD8':
        #     dwi = image
        # if row['Series Description'] == 'T2 Weighted Axial':
        #     t2 = image
        # if row['Series Description'] == 'T2 Weighted Axial Segmentations':
        #     segmentation = image
        #     size = segmentation.GetSize()
        #     seg_size = int(size[-1] / 4)
        #     seg_1 = segmentation[:, :, :seg_size]
        #     seg_2 = segmentation[:, :, seg_size:2*seg_size]
        #     seg_3 = segmentation[:, :, 2*seg_size:3*seg_size]
        #     seg_4 = segmentation[:, :, 3*seg_size:4*seg_size]
    MC_matches = [file for file in MC_seg_files if subject_id in file]
    NCI_matches = [file for file in NCI_seg_files if subject_id in file]
    if MC_matches:
        seg_path = os.path.join(data_path, segmentation_paths[0], MC_matches[0])
    if NCI_matches:
        if MC_matches:
            print('here')
        seg_path = os.path.join(data_path, segmentation_paths[1], NCI_matches[0])
    if not MC_matches and not NCI_matches:
        continue
    seg_image = sitk.ReadImage(seg_path)
    scans = [t2, t2, t2]

    sample = Sample(
        scans=scans,
        lbl=seg_image,
        settings=PreprocessingSettings(**settings['preprocessing']),
        name=str(subject_id)
    )

    sample.preprocess()
    # write images
    t2_array = sitk.GetArrayFromImage(sample.scans[0])
    segmentation_array = sitk.GetArrayFromImage(sample.lbl)
    
    if MC_matches and not NCI_matches:
        segmentation_array_copy = np.zeros_like(segmentation_array)
        segmentation_array_copy[segmentation_array == 4] = 1
        segmentation_array_copy[segmentation_array == 2] = 2
        segmentation_array = segmentation_array_copy
        sample.lbl = sitk.GetImageFromArray(segmentation_array_copy)
    for i in range(t2_array.shape[0]):
        slice_t2_data = t2_array[i, :, :]  # Extract the i-th slice (shape: 255x255)

        # Plot the slice
        fig, ax = plt.subplots(1, 2, figsize=(8, 8))
        ax[0].imshow(slice_t2_data, cmap="gray")  # Use 'gray' colormap for intensity images
        ax[1].imshow(segmentation_array[i, :, :], cmap="gray")

        # Save the plot to the output directory
        output_path = os.path.join('./plot_diagnosis', f"slice_{i + 1:03d}.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()  # Close the figure to free memory

        print(f"Saved slice {i + 1} to {output_path}")

    sitk.WriteImage(sample.scans[0],
                    f'../output_zone_Diagnosis/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{subject_num}_0000.nii.gz',
                    useCompression=True)
    sitk.WriteImage(sample.scans[1],
                    f'../output_zone_Diagnosis/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{subject_num}_0001.nii.gz',
                    useCompression=True)
    sitk.WriteImage(sample.scans[2],
                    f'../output_zone_Diagnosis/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{subject_num}_0002.nii.gz',
                    useCompression=True)
    sitk.WriteImage(sample.lbl,
                    f'../output_zone_Diagnosis/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/{subject_num}.nii.gz',
                    useCompression=True)
