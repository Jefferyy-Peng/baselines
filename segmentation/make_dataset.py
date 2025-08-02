import os
import pickle

import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import numpy as np
import h5py
import random
import nibabel as nib
from glob import glob

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

        seg_image[seg_image == 255] = 1

        # comment for samples that have 1 values
        # if np.max(seg_image) == 0:
        #     count += 1
        #     continue

        in_1 = sitk.ReadImage(os.path.join(base_dir,path + '_0000.nii.gz'))
        in_2 = sitk.ReadImage(os.path.join(base_dir,path + '_0001.nii.gz'))
        in_3 = sitk.ReadImage(os.path.join(base_dir,path + '_0002.nii.gz'))

        in_1 = sitk.GetArrayFromImage(in_1).astype(np.int16)
        in_2 = sitk.GetArrayFromImage(in_2).astype(np.int16)
        in_3 = sitk.GetArrayFromImage(in_3).astype(np.int16)
        img = np.stack((in_1,in_2,in_3),axis=0)

        hdf5_path = os.path.join(data_dir_3d, str(count) + '.hdf5')

        save_as_hdf5(img,hdf5_path,'ct')
        save_as_hdf5(seg_image,hdf5_path,'seg')

        # count -> path for lesion, path -> count for gland and zone
        pid_dict[count] = path
        store_images_labels_2d(data_dir_2d,count,img,seg_image)

        count += 1

    print(count)
    pickle.dump(pid_dict, open(os.path.join(output_dir ,'lesion_pid.p'), 'wb'))

def make_semidata(base_dir,label_dir,output_dir,test_dir,seg_dir,csv_path):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir_2d = os.path.join(output_dir,'data_2d')
    if not os.path.exists(data_dir_2d):
        os.makedirs(data_dir_2d)
    data_dir_3d = os.path.join(output_dir,'data_3d')
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

def make_combined_label(output_dir):
    datasets = [
        # ("/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/output_gland_AI/nnUNet_raw_data/Task2201_picai_baseline/labelsTr", 0),  # offset = 0
        # ("/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/output_zone_AI/nnUNet_raw_data/Task2201_picai_baseline/labelsTr", 1),  # offset = 1
        ("/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/output_lesion_combined/nnUNet_raw_data/Task2201_picai_baseline/labelsTr", 0),  # offset = 2
    ]

    image_folder = "/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/output_lesion_combined/nnUNet_raw_data/Task2201_picai_baseline/imagesTr"
    output_images = "../label_combined/imagesTr"
    output_labels = "../label_combined/labelsTr"

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    # Get list of cases from dataset_1 (assumed to be shared across datasets)
    case_ids = sorted(
        set(f.split("_")[0] + "_" + f.split("_")[1] for f in os.listdir(image_folder) if f.endswith("_0000.nii.gz")))

    for case_id in tqdm(case_ids):
        # Load original image (same across datasets)
        for m in range(3):
            src_img = os.path.join(image_folder, f"{case_id}_000{m}.nii.gz")
            dst_img = os.path.join(output_images, f"{case_id}_000{m}.nii.gz")
            if os.path.exists(src_img):
                os.system(f"cp {src_img} {dst_img}")
            else:
                print(f"Missing modality: {src_img}")

        # 🧠 2. Merge labels from each dataset
        label_merged = None
        reference_img = nib.load(os.path.join(image_folder, f"{case_id}_0000.nii.gz"))

        for label_dir, offset in datasets:
            label_path = os.path.join(label_dir, f"{case_id}.nii.gz")
            if not os.path.exists(label_path):
                label_merged = None
                break
            label = nib.load(label_path).get_fdata().astype(np.uint8)

            if label_merged is None:
                label_merged = np.zeros_like(label, dtype=np.uint8)

            for val in np.unique(label):
                if val == 0:
                    continue
                label_merged[label == val] = val + offset

        if label_merged is not None:
            label_nii = nib.Nifti1Image(label_merged, affine=reference_img.affine)
            nib.save(label_nii, os.path.join(output_labels, f"{case_id}.nii.gz"))
        else:
            print(f"No labels found for {case_id}")

if __name__ == "__main__":
    phase = 'combine_labels'
    base_dir = '../output_lesion_combined/nnUNet_raw_data/Task2201_picai_baseline/imagesTr'
    label_dir = '../output_lesion_combined/nnUNet_raw_data/Task2201_picai_baseline/labelsTr'
    output_dir = './dataset/lesion_segdata_MSD'
    test_dir = 'path/to/nnUNet_test_data'
    seg_dir = 'path/to/segmentation_result'
    csv_path = 'path/to/classification_result'
    if phase == 'seg':
        make_segdata(base_dir,label_dir,output_dir)
    elif phase == 'combine_labels':
        make_combined_label(output_dir)
    else:
        make_semidata(base_dir,label_dir,output_dir,test_dir,seg_dir,csv_path)