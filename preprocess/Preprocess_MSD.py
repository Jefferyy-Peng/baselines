import os
import pickle

import SimpleITK as sitk
import pandas as pd
from matplotlib import pyplot as plt
from picai_prep import atomic_image_write
from tqdm import tqdm
import numpy as np
import h5py
import random
import nibabel as nib
from picai_prep.preprocessing import resample_img, crop_or_pad, PreprocessingSettings
from utils import Sample

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


dataset_root = '/data/nvme1/meng/MSD'

image_path = os.path.join(dataset_root, 'imagesTr')

label_path = os.path.join(dataset_root, 'labelsTr')
os.makedirs('../output_lesion_MSD/nnUNet_raw_data/Task2201_picai_baseline/imagesTr', exist_ok=True)
os.makedirs('../output_lesion_MSD/nnUNet_raw_data/Task2201_picai_baseline/labelsTr', exist_ok=True)
for file in os.listdir(image_path):
    if file.startswith('.'):
        continue
    data_path = os.path.join(image_path, file)
    image = sitk.ReadImage(data_path)
    t2 = image[:,:,:,0]
    adc = image[:,:,:,1]
    label = sitk.ReadImage(data_path.replace('imagesTr', 'labelsTr'))
    label_array = sitk.GetArrayFromImage(label)
    label_array_new = np.zeros_like(label_array)
    label = sitk.GetImageFromArray(label_array_new)
    scans = [t2, adc, t2]
    subject_id = file.split('.')[0]

        # set up Sample
    sample = Sample(
        scans=scans,
        lbl=label,
        settings=PreprocessingSettings(**settings['preprocessing']),
        name=subject_id
    )

    sample.preprocess()
    # write images

    # sitk.WriteImage(sample.scans[0], f'../output_lesion_MSD/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{subject_id}_0000.nii.gz', useCompression=True)
    # sitk.WriteImage(sample.scans[1],
    #                 f'../output_lesion_MSD/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{subject_id}_0001.nii.gz',
    #                 useCompression=True)
    # sitk.WriteImage(sample.scans[2],
    #                 f'../output_lesion_MSD/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{subject_id}_0002.nii.gz',
    #                 useCompression=True)
    # sitk.WriteImage(sample.lbl,
    #                 f'../output_lesion_MSD/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/{subject_id}.nii.gz',
    #                 useCompression=True)

