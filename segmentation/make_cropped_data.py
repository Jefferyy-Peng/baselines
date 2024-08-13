import os
import pickle
from collections import OrderedDict

import SimpleITK as sitk
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import h5py
import random


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


def crop_with_margin(image, seg, mask, margin):
    # Find the coordinates where the mask is equal to 1
    rows, cols = np.where(mask == 1)

    if rows.size == 0 or cols.size == 0:
        raise ValueError("No mask value of 1 found in the mask.")

    # Determine the bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Add the margin
    min_row = max(min_row - margin, 0)
    max_row = min(max_row + margin, image.shape[0] - 1)
    min_col = max(min_col - margin, 0)
    max_col = min(max_col + margin, image.shape[1] - 1)

    # Crop the image
    cropped_image = image[min_row:max_row + 1, min_col:max_col + 1]
    cropped_seg = seg[min_row:max_row + 1, min_col:max_col + 1]

    return cropped_image, cropped_seg

def store_images_labels_2d(save_path, patient_id, cts, labels):
    for i in range(labels.shape[0]):
        ct = cts[:, i, :, :]
        lab = labels[i, :, :]
        if lab.max() == 0:
            continue
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

    data_dir_2d = os.path.join(output_dir, 'data_2d_crop')
    if not os.path.exists(data_dir_2d):
        os.makedirs(data_dir_2d)
    data_dir_3d = os.path.join(output_dir, 'data_3d_crop')
    if not os.path.exists(data_dir_3d):
        os.makedirs(data_dir_3d)

    count = 0

    pathlist = ['_'.join(path.split('_')[:2]) for path in os.listdir(base_dir)]
    pathlist = sorted(list(set(pathlist)))
    print(len(pathlist))
    pid_dict = {}
    unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                  in_channels=3, out_channels=4 , init_features=32, pretrained=False)
    ckpt_path = './new_ckpt/seg/UNet_Unified_equal_rate_lr_0.0001_weight_decay_0.001/epoch:28-gland_val_dice:0.94080-zone_val_dice:0.88053-lesion_val_dice:0.58659-lesion_val_ap:0.28383-lesion_val_auc:0.78891.pth'
    state_dict = torch.load(ckpt_path, map_location='cuda')['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.` prefix
        new_state_dict[name] = v

    unet.load_state_dict(new_state_dict)
    unet.eval()
    unet = unet.to('cuda')

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
        img_tensor = torch.from_numpy(img).to('cuda').unsqueeze(0)
        output = unet(img_tensor)
        pred = output > 0.5
        pred_gland = pred[0, 0].detach().cpu().numpy()
        if pred_gland.max() == 0:
            count += 1
            continue
        else:
            cropped_img, cropped_seg = crop_with_margin(img, seg, pred_gland, 5)


        hdf5_path = os.path.join(data_dir_3d, str(count) + '.hdf5')

        save_as_hdf5(cropped_img, hdf5_path, 'ct')
        save_as_hdf5(cropped_seg, hdf5_path, 'seg')

        # count -> path for lesion, path -> count for gland and zone
        pid_dict[path] = count
        store_images_labels_2d(data_dir_2d, count, cropped_img, cropped_seg)

        count += 1

    print(count)
    pickle.dump(pid_dict, open(os.path.join(output_dir, 'lesion_pid_cropped.p'), 'wb'))


def make_semidata(base_dir, label_dir, output_dir, test_dir, seg_dir, csv_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir_2d = os.path.join(output_dir, 'data_2d')
    if not os.path.exists(data_dir_2d):
        os.makedirs(data_dir_2d)
    data_dir_3d = os.path.join(output_dir, 'data_3d')
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
    base_dir = '../output_gland_AI/nnUNet_raw_data/Task2201_picai_baseline/imagesTr'
    label_dir = '../output_gland_AI/nnUNet_raw_data/Task2201_picai_baseline/labelsTr'
    output_dir = './dataset/gland_segdata_partial'
    test_dir = 'path/to/nnUNet_test_data'
    seg_dir = 'path/to/segmentation_result'
    csv_path = 'path/to/classification_result'
    if phase == 'seg':
        make_segdata(base_dir, label_dir, output_dir)
    else:
        make_semidata(base_dir, label_dir, output_dir, test_dir, seg_dir, csv_path)