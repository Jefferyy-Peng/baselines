import os
import pickle

import SimpleITK as sitk
import pandas as pd
from matplotlib import pyplot as plt
from picai_prep import atomic_image_write
from pydicom import dcmread
from tqdm import tqdm
import numpy as np
import h5py
import random
import nibabel as nib
from picai_prep.preprocessing import resample_img, crop_or_pad, PreprocessingSettings
from utils import Sample
import pydicom
import pydicom_seg

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

df = pd.read_csv('/data/nvme1/meng/QIN-PROSTATE-Repeatability/metadata.csv')

os.makedirs('../output_gland_QIN/nnUNet_raw_data/Task2201_picai_baseline/imagesTr',exist_ok=True)
os.makedirs('../output_gland_QIN/nnUNet_raw_data/Task2201_picai_baseline/labelsTr', exist_ok=True)

# relevant_descriptions = [
#     "T2 Weighted Axial",
#     "DWI",
#     "Apparent Diffusion Coefficient"
# ]
#
# # Filter the DataFrame for the relevant series descriptions
# filtered_df = df[df["Series Description"].isin(relevant_descriptions)]
data_path = '/data/nvme1/meng/'
cols = ['ID', 't2', 'adc', 'dwi'] + ['adc_tumor_reader1'] + ['t2_anatomy_reader1']
grouped = df.groupby(["Subject ID", "Study UID"])
subject_study_dict = {}
for (subject_id, study_uid), group in grouped:
    if subject_id not in subject_study_dict.keys():
        subject_study_dict[subject_id] = 1
    else:
        subject_study_dict[subject_id] += 1
    subject_num = subject_id.split('-')[1]
    study_num = f'{subject_study_dict[subject_id]:03}'
    data_dict = {}
    print(f"Subject ID: {subject_id}, Study UID: {study_uid}")
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

        # # Extract Rescale Slope and Intercept from metadata (default to 1 and 0 if not found)
        # if 'Segmentation' not in row['Series Description'] and 'Measurements' not in row['Series Description']:
        #     rescale_slope = float(image.GetMetaData("0028|1053")) if image.HasMetaDataKey("0028|1053") else 1
        #     rescale_intercept = float(image.GetMetaData("0028|1052")) if image.HasMetaDataKey("0028|1052") else 0
        #
        #     # Rescale the image intensities
        #     rescaled_image = image * rescale_slope + rescale_intercept

        # if len(image_array.shape) > 3:
        #     # Convert to NumPy array
        #     array = sitk.GetArrayFromImage(image)  # Converts to NumPy array
        #     print("Original Array Shape:", array.shape)  # Shape: (depth, height, width, channels)
        #
        #     # Squeeze the array to remove the size-1 dimension
        #     squeezed_array = np.squeeze(array)
        #     print("Squeezed Array Shape:", squeezed_array.shape)
        #
        #     # Convert back to SimpleITK image
        #     squeezed_image = sitk.GetImageFromArray(squeezed_array)
        #
        #     # Copy metadata (spacing, origin, direction)
        #     squeezed_image.SetSpacing(image.GetSpacing()[:-1])  # Remove last dimension's spacing
        #     squeezed_image.SetOrigin(image.GetOrigin()[:-1])  # Remove last dimension's origin
        #     # Extract the 3x3 part of the direction matrix for the 3D image
        #     original_direction = np.array(image.GetDirection()).reshape(4, 4)
        #     squeezed_direction = original_direction[:3, :3].flatten()
        #     squeezed_image.SetDirection(squeezed_direction)
        #     image = squeezed_image

        if row['Series Description'] == 'Apparent Diffusion Coefficient':
            adc = image
        if row['Series Description'] == 'DWI':
            dwi = image
        if row['Series Description'] == 'T2 Weighted Axial':
            t2 = image
        if row['Series Description'] == 'T2 Weighted Axial Segmentations':
            dicom_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".dcm")]

            reader = pydicom_seg.SegmentReader()
            dcm = dcmread(dicom_files[0])
            result = reader.read(dcm)
            segmentations = {}
            seg_labels = []
            for id, segment_info in result.segment_infos.items():
                seg_labels.append(segment_info.SegmentLabel)
            for segment_number in result.available_segments:
                seg_image = sitk.GetImageFromArray(result.segment_data(segment_number))
                seg_image.SetOrigin(result.origin)
                seg_image.SetDirection(result.direction.flatten())
                seg_image.SetSpacing(result.spacing)
                segmentations[seg_labels[segment_number-1]] = seg_image
    if 'Lesion' not in segmentations.keys() or 'Peripheral zone of the prostate' not in segmentations.keys() or 'Prostate' not in segmentations.keys():
        continue
    scans = [t2, adc, dwi]

    sample = Sample(
        scans=scans,
        lbl=segmentations['Prostate'],
        settings=PreprocessingSettings(**settings['preprocessing']),
        name=str(study_uid)
    )

    sample.preprocess()
    # write images
    # t2_array = sitk.GetArrayFromImage(sample.scans[0])
    # adc_array = sitk.GetArrayFromImage(sample.scans[1])
    # dwi_array = sitk.GetArrayFromImage(sample.scans[2])
    # segmentation_array = sitk.GetArrayFromImage(sample.lbl)
    # for i in range(t2_array.shape[0]):
    #     slice_t2_data = t2_array[i, :, :]  # Extract the i-th slice (shape: 255x255)
    #
    #     # Plot the slice
    #     fig, ax = plt.subplots(1, 4, figsize=(8, 8))
    #     ax[0].imshow(slice_t2_data, cmap="gray")  # Use 'gray' colormap for intensity images
    #     ax[1].imshow(adc_array[i, :, :], cmap="gray")
    #     ax[2].imshow(dwi_array[i, :, :], cmap="gray")
    #     ax[3].imshow(segmentation_array[i, :, :], cmap="gray")
    #
    #     # Save the plot to the output directory
    #     path = './plot_seg_4'
    #     os.makedirs(path, exist_ok=True)
    #     output_path = os.path.join(path, f"slice_{i + 1:03d}.png")
    #     plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    #     plt.close()  # Close the figure to free memory
    #
    #     print(f"Saved slice {i + 1} to {output_path}")

    # sitk.WriteImage(sample.scans[0],
    #                 f'../output_gland_QIN/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{subject_num}_{study_num}_0000.nii.gz',
    #                 useCompression=True)
    # sitk.WriteImage(sample.scans[1],
    #                 f'../output_gland_QIN/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{subject_num}_{study_num}_0001.nii.gz',
    #                 useCompression=True)
    # sitk.WriteImage(sample.scans[2],
    #                 f'../output_gland_QIN/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{subject_num}_{study_num}_0002.nii.gz',
    #                 useCompression=True)
    # sitk.WriteImage(sample.lbl,
    #                 f'../output_gland_QIN/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/{subject_num}_{study_num}.nii.gz',
    #                 useCompression=True)
for files in [train_files, val_files, test_files]:
    for file in files:
        adc_path = file['adc']
        dwi_path = file['dwi']
        t2_path = file['t2']
        t2 = sitk.ReadImage(t2_path)
        adc = sitk.ReadImage(adc_path)
        dwi = sitk.ReadImage(dwi_path)
        anatomy_path = file['t2_anatomy_reader1']
        anatomy = sitk.ReadImage(anatomy_path)
        scans = [t2, adc, dwi]

        # set up Sample
        sample = Sample(
            scans=scans,
            lbl=anatomy,
            settings=PreprocessingSettings(**settings['preprocessing']),
            name=str(file['ID'])
        )

        sample.preprocess()
        # write images

        sitk.WriteImage(sample.scans[0], f'../output_zone_158/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{file["ID"]}_0000.nii.gz', useCompression=True)
        sitk.WriteImage(sample.scans[1],
                        f'../output_zone_158/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{file["ID"]}_0001.nii.gz',
                        useCompression=True)
        sitk.WriteImage(sample.scans[2],
                        f'../output_zone_158/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{file["ID"]}_0002.nii.gz',
                        useCompression=True)
        sitk.WriteImage(sample.lbl,
                        f'../output_zone_158/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/{file["ID"]}.nii.gz',
                        useCompression=True)
for files in [train_files, val_files, test_files]:
    for file in files:
        adc_path = file['adc']
        dwi_path = file['dwi']
        t2_path = file['t2']
        t2 = sitk.ReadImage(t2_path)
        adc = sitk.ReadImage(adc_path)
        dwi = sitk.ReadImage(dwi_path)
        tumor_path = file['adc_tumor_reader1']
        tumor = sitk.ReadImage(tumor_path)
        scans = [t2, adc, dwi]

        # set up Sample
        sample = Sample(
            scans=scans,
            lbl=tumor,
            settings=PreprocessingSettings(**settings['preprocessing']),
            name=str(file['ID'])
        )

        sample.preprocess()
        # write images

        sitk.WriteImage(sample.scans[0], f'../output_lesion_158/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{file["ID"]}_0000.nii.gz', useCompression=True)
        sitk.WriteImage(sample.scans[1],
                        f'../output_lesion_158/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{file["ID"]}_0001.nii.gz',
                        useCompression=True)
        sitk.WriteImage(sample.scans[2],
                        f'../output_lesion_158/nnUNet_raw_data/Task2201_picai_baseline/imagesTr/{file["ID"]}_0002.nii.gz',
                        useCompression=True)
        sitk.WriteImage(sample.lbl,
                        f'../output_lesion_158/nnUNet_raw_data/Task2201_picai_baseline/labelsTr/{file["ID"]}.nii.gz',
                        useCompression=True)

# Load the .nii.gz file
file_path = "/home/yxpengcs/Datasets/MRI/prostate158/prostate158_train/train/020/adc.nii.gz"
nii_image = sitk.ReadImage(file_path)

# Load the .nii.gz file
nii_file = nib.load("~/Datasets/MRI/prostate158/prostate158_train/train/020/adc.nii.gz")

# Get the affine transformation matrix
affine = nii_file.affine

# Extract the voxel spacing from the affine matrix (diagonal elements)
voxel_spacing = affine[:3, :3].diagonal()
print("Voxel spacing:", voxel_spacing)





def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()


def csv_reader_single(csv_file, key_col=None, value_col=None):
    '''
    Extracts the specified single column, return a single level dict.
    The value of specified column as the key of dict.

    Args:
    - csv_file: file path
    - key_col: string, specified column as key, the value of the column must be unique.
    - value_col: string,  specified column as value
    '''
    file_csv = pd.read_csv(csv_file)
    key_list = file_csv[key_col].values.tolist()
    value_list = file_csv[value_col].values.tolist()

    target_dict = {}
    for key_item, value_item in zip(key_list, value_list):
        target_dict[key_item] = value_item

    return target_dict


def store_images_labels_2d(save_path, patient_id, cts, labels):
    for i in range(labels.shape[0]):
        ct = cts[:, i, :, :]
        lab = labels[i, :, :]

        # comment to obtain all data, including slices with no lesion
        # if lab.max() == 0:
        #     continue

        # if 2 not in np.unique(lab) or 1 not in np.unique(lab):
        #     continue
        # else:
        #     new_lab = np.zeros((2, 384, 384)).astype(int)
        #     new_lab[0][lab == 1] = 1
        #     new_lab[1][lab == 2] = 1
        #     lab = new_lab

        hdf5_file = h5py.File(os.path.join(save_path, '%s_%d.hdf5' % (patient_id, i)), 'w')
        hdf5_file.create_dataset('ct', data=ct.astype(np.int16))
        hdf5_file.create_dataset('seg', data=lab.astype(np.uint8))
        hdf5_file.close()


def make_segdata(base_dir, label_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir_2d = os.path.join(output_dir, 'data_2d')
    if not os.path.exists(data_dir_2d):
        os.makedirs(data_dir_2d)
    data_dir_3d = os.path.join(output_dir, 'data_3d')
    if not os.path.exists(data_dir_3d):
        os.makedirs(data_dir_3d)

    count = 0

    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = sorted(list(set(pathlist)))
    print(len(pathlist))
    pid_dict = {}

    for id, path in enumerate(tqdm(pathlist)):
        seg = sitk.ReadImage(os.path.join(label_dir, path + '.nii.gz'))

        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
        # comment for zone segmentations (zone have >= 2 values and is used)
        seg_image[seg_image >= 2] = 1

        # comment for samples that have 1 values
        # if np.max(seg_image) == 0:
        #     count += 1
        #     continue

        in_1 = sitk.ReadImage(os.path.join(base_dir, path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir, path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir, path + '_0002.nii.gz'))

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1, in_2, in_3), axis=0)

        hdf5_path = os.path.join(data_dir_3d, str(count) + '.hdf5')

        save_as_hdf5(img, hdf5_path, 'ct')
        save_as_hdf5(seg_image, hdf5_path, 'seg')

        # count -> path for lesion, path -> count for gland and zone
        pid_dict[count] = path
        store_images_labels_2d(data_dir_2d, count, img, seg_image)

        count += 1

    print(count)
    pickle.dump(pid_dict, open(os.path.join(output_dir, 'lesion_pid.p'), 'wb'))


def make_semidata(base_dir, label_dir, output_dir, test_dir, seg_dir, csv_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir_2d = os.path.join(output_dir, 'data_2d')
    if not os.path.exists(data_dir_2d):
        os.makedirs(data_dir_2d)
    data_dir_3d = os.path.join(output_dir, 'data_3d')
    if not os.path.exists(data_dir_3d):
        os.makedirs(data_dir_3d)

    # label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')

    count = 0

    # collect paths
    pathlist_test_dir = ['_'.join(path.split('_')[:2]) for path in os.listdir(test_dir)]
    pathlist_test_dir = list(set(pathlist_test_dir))

    pathlist_base_dir = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist_base_dir = list(set(pathlist_base_dir))

    # generate random IDs
    rand_list = list(range(len(pathlist_test_dir) + len(pathlist_base_dir)))
    random.shuffle(rand_list)
    print(rand_list)

    for path in tqdm(pathlist_test_dir):
        seg_image = np.load(os.path.join(seg_dir, path + '.npy')).astype(np.uint8)

        seg_image *= int(label_dict[path])

        in_1 = sitk.ReadImage(os.path.join(test_dir, path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(test_dir, path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(test_dir, path + '_0002.nii.gz'))

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1, in_2, in_3), axis=0)

        outc = rand_list[count]

        hdf5_path = os.path.join(data_dir_3d, str(outc) + '.hdf5')

        save_as_hdf5(img, hdf5_path, 'ct')
        save_as_hdf5(seg_image, hdf5_path, 'seg')

        store_images_labels_2d(data_dir_2d, outc, img, seg_image)

        count += 1

    for path in tqdm(pathlist_base_dir):
        seg = sitk.ReadImage(os.path.join(label_dir, path + '.nii.gz'))

        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
        seg_image[seg_image == 2] = 1
        seg_image[seg_image > 2] = 2

        in_1 = sitk.ReadImage(os.path.join(base_dir, path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir, path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir, path + '_0002.nii.gz'))

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1, in_2, in_3), axis=0)

        outc = rand_list[count]

        hdf5_path = os.path.join(data_dir_3d, str(outc) + '.hdf5')

        save_as_hdf5(img, hdf5_path, 'ct')
        save_as_hdf5(seg_image, hdf5_path, 'seg')

        store_images_labels_2d(data_dir_2d, outc, img, seg_image)

        count += 1


if __name__ == "__main__":
    phase = 'seg'
    base_dir = '../output_lesion_combined/nnUNet_raw_data/Task2201_picai_baseline/imagesTr'
    label_dir = '../output_lesion_combined/nnUNet_raw_data/Task2201_picai_baseline/labelsTr'
    output_dir = './dataset/lesion_segdata_combined'
    test_dir = 'path/to/nnUNet_test_data'
    seg_dir = 'path/to/segmentation_result'
    csv_path = 'path/to/classification_result'
    if phase == 'seg':
        make_segdata(base_dir, label_dir, output_dir)
    else:
        make_semidata(base_dir, label_dir, output_dir, test_dir, seg_dir, csv_path)