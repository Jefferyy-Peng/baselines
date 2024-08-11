import os
import pickle

import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import numpy as np
import h5py
import random
import re
import pydicom
from PIL import Image
import nibabel as nib

def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()

def csv_reader_single(csv_file,key_col=None,value_col=None):
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
    for key_item,value_item in zip(key_list,value_list):
        target_dict[key_item] = value_item

    return target_dict

def store_images_labels_2d(save_path, patient_id, cts, labels):

    for i in range(labels.shape[0]):
        ct = cts[:,i,:,:]
        lab = labels[i,:,:]
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


def make_segdata(base_dir,label_dir,output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir_2d = os.path.join(output_dir,'data_2d')
    if not os.path.exists(data_dir_2d):
        os.makedirs(data_dir_2d)
    data_dir_3d = os.path.join(output_dir,'data_3d')
    if not os.path.exists(data_dir_3d):
        os.makedirs(data_dir_3d)

    count = 0

    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = sorted(list(set(pathlist)))
    print(len(pathlist))
    pid_dict = {}


    for id, path in enumerate(tqdm(pathlist)):
        seg = sitk.ReadImage(os.path.join(label_dir,path + '.nii.gz'))

        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
        # comment for zone segmentations (zone have >= 2 values and is used)
        # seg_image[seg_image>=2] = 1

        # comment for samples that have 1 values
        # if np.max(seg_image) == 0:
        #     count += 1
        #     continue

        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))
        in_4 = sitk.ReadImage(os.path.join(base_dir, path + '_0003.nii.gz'))
        in_5 = sitk.ReadImage(os.path.join(base_dir, path + '_0004.nii.gz'))
        in_6 = sitk.ReadImage(os.path.join(base_dir, path + '_0005.nii.gz'))

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        in_4 = sitk.GetArrayFromImage(in_4).astype(np.int16)
        in_5 = sitk.GetArrayFromImage(in_5).astype(np.int16)
        in_6 = sitk.GetArrayFromImage(in_6).astype(np.int16)
        img = np.stack((in_1,in_2,in_3,in_4,in_5,in_6),axis=0)

        hdf5_path = os.path.join(data_dir_3d, str(count) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        # count -> path for lesion, path -> count for gland and zone
        # pid_dict[path] = count
        store_images_labels_2d(data_dir_2d,count,img,seg_image)

        count += 1

    print(count)
    # pickle.dump(pid_dict, open(os.path.join(output_dir ,'lesion_pid.p'), 'wb'))

def make_semidata(base_dir,label_dir,output_dir,test_dir,seg_dir,csv_path):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir_2d = os.path.join(output_dir,'data_2d')
    if not os.path.exists(data_dir_2d):
        os.makedirs(data_dir_2d)
    data_dir_3d = os.path.join(output_dir,'data_3d')
    if not os.path.exists(data_dir_3d):
        os.makedirs(data_dir_3d)

    label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')

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
        seg_image = np.load(os.path.join(seg_dir,path + '.npy')).astype(np.uint8)

        seg_image *= int(label_dict[path])

        in_1 = sitk.ReadImage(os.path.join(test_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(test_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(test_dir,path + '_0002.nii.gz'))
        

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1,in_2,in_3),axis=0)

        outc = rand_list[count]

        hdf5_path = os.path.join(data_dir_3d, str(outc) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        store_images_labels_2d(data_dir_2d,outc,img,seg_image)

        count += 1

    for path in tqdm(pathlist_base_dir):
        seg = sitk.ReadImage(os.path.join(label_dir,path + '.nii.gz'))

        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)
        seg_image[seg_image==2] = 1
        seg_image[seg_image>2] = 2

        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1,in_2,in_3),axis=0)

        outc = rand_list[count]


        hdf5_path = os.path.join(data_dir_3d, str(outc) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        store_images_labels_2d(data_dir_2d,outc,img,seg_image)

        count += 1

def convert_dicom_to_hdf5_3d(subject_path, output_file):
    # Create an HDF5 file
    with h5py.File(output_file, 'w') as hdf5_file:
        for root, dirs, files in os.walk(subject_path):
            if len(files) == 0:
                continue
            if files[0].endswith('.dcm') and ('T1W_AX' in root or 'T1W_SAG' in root or 'T2W_STIR' in root or 'T2W_FLAIR' in root or 'PDW' in root or 'PSIR' in root or 'R2_MAP' in root):

                def extract_number(file_name):
                    match = re.search(r'\d+', file_name)
                    return int(match.group()) if match else 0

                # Sort the file names based on the extracted numeric part
                files = sorted(files, key=extract_number)
                sequence = []
                for file in files:
                    dicom_path = os.path.join(root, file)
                    # Read the DICOM file
                    dicom_data = pydicom.dcmread(dicom_path)
                    # Convert pixel data to a numpy array
                    pixel_array = dicom_data.pixel_array
                    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) if (np.max(pixel_array) - np.min(pixel_array)) != 0 else pixel_array - np.min(pixel_array)
                    sequence.append(pixel_array)
                sequence = np.stack(sequence)
                dataset_name = os.path.relpath(root, subject_path)
                dataset_name = dataset_name.replace(os.sep, '_').split('_')[1]
                # Save the pixel data to the HDF5 file
                hdf5_file.create_dataset(dataset_name, data=sequence)
        seg_path = os.path.join(subject_path, f'seg.nii.gz')
        seg = nib.load(seg_path).get_fdata().transpose(2, 0, 1)
        hdf5_file.create_dataset('seg', data=seg)

def convert_dicom_to_hdf5_2d(subject_path, subject_name, output_path):
    subject_name = subject_path.split('/')[-1]
    seg_path = os.path.join(subject_path, f'seg.nii.gz')
    seg = nib.load(seg_path).get_fdata().transpose(2, 0, 1)
    # Create an HDF5 file
    for root, dirs, files in os.walk(subject_path):
        if len(files) == 0:
            continue
        if files[0].endswith('.dcm') and ('T1W_AX' in root or 'T1W_SAG' in root or 'T2W_STIR' in root or 'T2W_FLAIR' in root or 'PDW' in root or 'PSIR' in root or 'R2_MAP' in root):
            def extract_number(file_name):
                match = re.search(r'\d+', file_name)
                return int(match.group()) if match else 0

            # Sort the file names based on the extracted numeric part
            files = sorted(files, key=extract_number)
            for slice_id, file in enumerate(files):
                with h5py.File(os.path.join(output_path, f'{subject_name}_{slice_id}.h5'), 'a') as hdf5_file:
                    dicom_path = os.path.join(root, file)
                    # Read the DICOM file
                    dicom_data = pydicom.dcmread(dicom_path)
                    # Convert pixel data to a numpy array
                    pixel_array = dicom_data.pixel_array
                    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) if (np.max(pixel_array) - np.min(pixel_array)) != 0 else pixel_array - np.min(pixel_array)
                    dataset_name = os.path.relpath(root, subject_path)
                    dataset_name = dataset_name.replace(os.sep, '_').split('_')[1]
                    if dataset_name in hdf5_file:
                        print(f"Dataset {dataset_name} already exists. Skipping.")
                    # Save the pixel data to the HDF5 file
                    else:
                        hdf5_file.create_dataset(dataset_name, data=pixel_array)
                        print(f"Dataset {dataset_name} added.")
                    if 'seg' in hdf5_file:
                        print(f"seg already exists. Skipping.")
                    else:
                        hdf5_file.create_dataset('seg', data=seg[slice_id])

def process_all_subjects(base_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subjects = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    for subject in subjects:
        subject_path = os.path.join(base_path, subject)
        output_3d_path = os.path.join(output_dir, 'data_3d')
        output_2d_path = os.path.join(output_dir, 'data_2d')
        os.makedirs(output_3d_path, exist_ok=True)
        os.makedirs(output_2d_path, exist_ok=True)
        output_3d_file = os.path.join(output_dir, 'data_3d', f'{subject}.h5')
        convert_dicom_to_hdf5_3d(subject_path, output_3d_file)
        convert_dicom_to_hdf5_2d(subject_path, subject, output_2d_path)

# def make_multi_contrast_data(base_dir, output_dir):

def save_as_png(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    hdf5_files = [f for f in os.listdir(base_dir) if f.endswith('.h5')]
    for hdf5_file in hdf5_files:
        hdf5_file_path = os.path.join(base_dir, hdf5_file)
        dataset_names = []
        with h5py.File(hdf5_file_path, 'r') as hdf5_file:
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    dataset_names.append(name)

            hdf5_file.visititems(collect_datasets)
            t1w_datasets = [name for name in dataset_names if 'T1W' in name]
            slice_image = (hdf5_file[t1w_datasets[0]][:] * 255).astype(np.uint8)
            base_filename = os.path.splitext(os.path.basename(hdf5_file_path))[0]
            img = Image.fromarray(slice_image)
            if img.mode != 'L':  # Ensure image mode is grayscale
                img = img.convert('L')
            output_path = os.path.join(output_dir, f'{base_filename}.png')
            img.save(output_path)

if __name__ == "__main__":
    phase = 'seg'
    base_dir = '/home/yxpengcs/Datasets/MRI/CHDI_Multi_Contrast/SyMRI_processed_DL'
    label_dir = '/home/yxpengcs/Datasets/MRI/CHDI_Multi_Contrast/segs'
    output_dir = './dataset/ucsd_multi_contrast_segdata'
    # base_dir = '../nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset500_CaudatePutamenGlobus/imagesTr'
    # label_dir = '../nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset500_CaudatePutamenGlobus/labelsTr'
    # output_dir = './dataset/multi_contrast_segdata_qiren'
    test_dir = 'path/to/nnUNet_test_data'
    seg_dir = 'path/to/segmentation_result'
    csv_path = 'path/to/classification_result'
    if phase == 'seg':
        process_all_subjects(base_dir, output_dir)
    elif phase == 'make_data':
        make_segdata(base_dir,label_dir,output_dir)
    elif phase == 'viz':
        save_as_png(os.path.join(output_dir, 'data_2d'), os.path.join(output_dir, 'viz'))
    else:
        make_semidata(base_dir,label_dir,output_dir,test_dir,seg_dir,csv_path)