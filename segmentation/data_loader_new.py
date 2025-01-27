import random

import json
import pickle

import os
import sys

sys.path.append('../')

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config_seg_align import PATH_DIR, CHECKPOINT_PATH
from utils import hdf5_reader

from open_clip import get_tokenizer

import re


class Normalize(object):
    def __call__(self, sample):
        for name, data in sample.items():
            if name == 'ct':
                for i in range(data.shape[0]):
                    if np.max(data[i]) != 0:
                        data[i] = data[i] / np.max(data[i])

                data[data < 0] = 0
        return sample


class RandomRotate2D(object):
    """
    Data augmentation method.
    Rotating the image with random degree.
    Args:
    - degree: the rotate degree from (-degree , degree)
    Returns:
    - rotated image and label
    """

    def __init__(self, degree=[-15, -10, -5, 0, 5, 10, 15]):
        self.degree = degree

    def __call__(self, sample):
        new_sample = sample
        rotate_degree = random.choice(self.degree)

        for name, data in sample.items():
            if name == 'ct':
                cts = []
                for i in range(data.shape[0]):
                    cts.append(Image.fromarray(data[i]))
                cts_out = []
                for ct in cts:
                    ct = ct.rotate(rotate_degree, Image.BILINEAR)
                    ct = np.array(ct).astype(np.float32)
                    cts_out.append(ct)
                ct_image = np.asarray(cts_out)
                new_sample['ct'] = ct_image

            elif 'seg' in name:
                label = Image.fromarray(np.uint8(data))
                label = label.rotate(rotate_degree, Image.NEAREST)
                label = np.array(label).astype(np.float32)
                new_sample[name] = label
        # ct_image = sample['ct']
        # lesion_label = sample['lesion_seg']
        #
        # cts = []
        # for i in range(ct_image.shape[0]):
        #     cts.append(Image.fromarray(ct_image[i]))
        # label = Image.fromarray(np.uint8(label))
        #
        # rotate_degree = random.choice(self.degree)
        #
        # cts_out = []
        # for ct in cts:
        #     ct = ct.rotate(rotate_degree, Image.BILINEAR)
        #     ct = np.array(ct).astype(np.float32)
        #     cts_out.append(ct)
        #
        # label = label.rotate(rotate_degree, Image.NEAREST)
        #
        # ct_image = np.asarray(cts_out)
        # label = np.array(label).astype(np.float32)
        return new_sample


class RandomFlip2D(object):
    '''
    Data augmentation method.
    Flipping the image, including horizontal and vertical flipping.
    Args:
    - mode: string, consisting of 'h' and 'v'. Optional methods and 'hv' is default.
            'h'-> horizontal flipping,
            'v'-> vertical flipping,
            'hv'-> random flipping.
    '''

    def __init__(self, mode='hv'):
        self.mode = mode

    def __call__(self, sample):
        new_sample = sample
        random_factor = np.random.uniform(0, 1)

        for name, data in sample.items():
            if name == 'ct':
                if 'h' in self.mode and 'v' in self.mode:
                    if random_factor < 0.3:
                        data = data[:, :, ::-1]
                    elif random_factor < 0.6:
                        data = data[:, ::-1, :]
                elif 'h' in self.mode:
                    if random_factor > 0.5:
                        data = data[:, :, ::-1]
                elif 'v' in self.mode:
                    if random_factor > 0.5:
                        data = data[:, ::-1, :]
                new_sample['ct'] = data.copy()
            elif 'seg' in name:
                if 'h' in self.mode and 'v' in self.mode:
                    if random_factor < 0.3:
                        data = data[:, ::-1]
                    elif random_factor < 0.6:
                        data = data[::-1, :]
                elif 'h' in self.mode:
                    if random_factor > 0.5:
                        data = data[:, ::-1]
                elif 'v' in self.mode:
                    if random_factor > 0.5:
                        data = data[::-1, :]
                new_sample[name] = data.copy()
        return new_sample

        # ct_image = sample['ct']
        # label = sample['seg']
        #
        # if 'h' in self.mode and 'v' in self.mode:
        #     random_factor = np.random.uniform(0, 1)
        #     if random_factor < 0.3:
        #         ct_image = ct_image[:, :, ::-1]
        #         label = label[:, ::-1]
        #     elif random_factor < 0.6:
        #         ct_image = ct_image[:, ::-1, :]
        #         label = label[::-1, :]
        #
        # elif 'h' in self.mode:
        #     if np.random.uniform(0, 1) > 0.5:
        #         ct_image = ct_image[:, :, ::-1]
        #         label = label[:, ::-1]
        #
        # elif 'v' in self.mode:
        #     if np.random.uniform(0, 1) > 0.5:
        #         ct_image = ct_image[:, ::-1, :]
        #         label = label[::-1, :]
        #
        # ct_image = ct_image.copy()
        # label = label.copy()
        # return {'ct': ct_image, 'seg': label}


class To_Tensor(object):
    '''
    Convert the data in sample to torch Tensor.
    Args:
    - n_class: the number of class
    '''

    def __init__(self, num_class=2, input_channel=3):
        self.num_class = num_class
        self.channel = input_channel

    def __call__(self, sample):
        new_sample = sample
        for name, data in sample.items():
            if not isinstance(data, torch.Tensor):
                new_sample[name] = torch.from_numpy(sample[name])

        return new_sample


class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, path_list, num_class=2, transform=None, mode='train'):
        self.path_list = path_list
        self.num_class = num_class
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        ct = torch.Tensor(hdf5_reader(self.path_list[index], 'ct'))
        seg = torch.Tensor(hdf5_reader(self.path_list[index], 'seg')).unsqueeze(0)
        transform = transforms.Resize(size=(1024, 1024))
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=(1024, 1024),
                                          interpolation=transforms.functional.InterpolationMode.NEAREST)
        seg = seg_transform(seg).squeeze(0).numpy()

        sample = {'ct': ct, 'seg': seg}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)
        if self.mode == 'train':
            return sample
        else:
            return sample, self.path_list[index]


class DataGeneratorAlignDummy(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, path_list, num_class=2, transform=None, mode='train'):
        self.path_list = path_list
        self.num_class = num_class
        self.transform = transform
        self.mode = mode

        # self.slice_caption = ['Lesion, lesion, tumor'] * 24  # score: 0.731257
        self.slice_caption = ['prostate gland, transitional zone, peripheral zone, lesion'] * 24
        # self.slice_caption = ['circumscribed homogenous moderate hypointense focus/mass confined to prostate lenticular non-circumscribed homogeneous moderately hypointense'] * 24 # score: 0.731257
        # self.slice_caption = ['This is a benign and normal image. '] * 24
        # self.slice_caption = ['non-relevant anatomical structure'] * 24
        # self.slice_caption = ['benign and normal'] * 24
        # self.slice_caption = ['random text'] * 24
        # self.slice_caption[:7] = ["This image shows non-relevant anatomical structure."] * 7
        # self.slice_caption[-7:] = ["This image shows non-relevant anatomical structure."] * 7

        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        ct = torch.Tensor(hdf5_reader(self.path_list[index], 'ct'))
        seg = torch.Tensor(hdf5_reader(self.path_list[index], 'seg')).unsqueeze(0)
        transform = transforms.Resize(size=(1024, 1024))
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=(1024, 1024),
                                          interpolation=transforms.functional.InterpolationMode.NEAREST)
        seg = seg_transform(seg).squeeze(0).numpy()

        caption_tokens = self.tokenizer(self.slice_caption, context_length=256)

        sample = {'ct': ct, 'seg': seg}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        sample['tokens'] = caption_tokens
        sample['caption'] = self.slice_caption
        sample['score'] = 'PI-RADS notspecified'
        # sample['location'] = 'transition zone'

        if self.mode == 'train':
            return sample
        else:
            return sample, self.path_list[index]

class MultiLevelDataGeneratorAlignDummy(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, lesion_path_list, mode, num_class=2, transform=None, zone_pid=None, lesion_pid=None,
                 gland_pid=None):
        ratio_threshold = 0.001
        if '_158' in lesion_path_list[0] or '_Diagnosis' in lesion_path_list[0] or '_QIN' in lesion_path_list[
            0] or '_MSD' in lesion_path_list[0]:
            new_path_list1 = []
            new_path_list2 = []
            ratios = []
            for idx, path in enumerate(lesion_path_list):
                lesion_seg = torch.Tensor(hdf5_reader(lesion_path_list[idx], 'seg'))
                lesion_count = lesion_path_list[idx].split('/')[-1].split('_')[0]
                this_lesion_pid = lesion_pid[int(lesion_count)]
                if this_lesion_pid == '10705_1000721':
                    continue
                tumor_ratio = lesion_seg.sum() / (lesion_seg.shape[0] * lesion_seg.shape[1])
                if tumor_ratio != 0:
                    ratios.append(tumor_ratio)
                if tumor_ratio > ratio_threshold:
                    new_path_list2.append(path)
                else:
                    new_path_list1.append(path)
        else:
            if os.path.exists(f'./list1_2d_{ratio_threshold}_{mode}.pkl') and os.path.exists(
                    f'./list2_2d_{ratio_threshold}_{mode}.pkl'):
                with open(f'./list1_2d_{ratio_threshold}_{mode}.pkl', 'rb') as file:
                    new_path_list1 = pickle.load(file)
                with open(f'./list2_2d_{ratio_threshold}_{mode}.pkl', 'rb') as file:
                    new_path_list2 = pickle.load(file)
            else:
                new_path_list1 = []
                new_path_list2 = []
                ratios = []
                for idx, path in enumerate(lesion_path_list):
                    lesion_seg = torch.Tensor(hdf5_reader(lesion_path_list[idx], 'seg'))
                    lesion_count = lesion_path_list[idx].split('/')[-1].split('_')[0]
                    this_lesion_pid = lesion_pid[int(lesion_count)]
                    if this_lesion_pid == '10705_1000721':
                        continue
                    tumor_ratio = lesion_seg.sum() / (lesion_seg.shape[0] * lesion_seg.shape[1])
                    if tumor_ratio != 0:
                        ratios.append(tumor_ratio)
                    if tumor_ratio > ratio_threshold:
                        new_path_list2.append(path)
                    else:
                        new_path_list1.append(path)
                with open(f'./list1_2d_{ratio_threshold}_{mode}.pkl', 'wb') as file:
                    pickle.dump(new_path_list1, file)
                with open(f'./list2_2d_{ratio_threshold}_{mode}.pkl', 'wb') as file:
                    pickle.dump(new_path_list2, file)
        self.mode = mode
        self.path_list1 = new_path_list1  # no or small lesion
        self.path_list2 = new_path_list2  # larger lesion
        print(
            f'Got {len(new_path_list1)} slices with no or small lesion and {len(new_path_list2)} with sufficient lesion')
        self.num_class = num_class
        self.transform = transform
        self.zone_pid = zone_pid
        self.gland_pid = gland_pid
        self.lesion_pid = lesion_pid
        # self.slice_caption = ['gland']
        self.slice_caption = ['Lesion, lesion, tumor']  # score: 0.731257
        # self.slice_caption = ['Lesion']  # score: 0.731257
        # self.slice_caption = ['circumscribed homogenous moderate hypointense focus/mass confined to prostate lenticular non-circumscribed homogeneous moderately hypointense'] * 24 # score: 0.731257
        # self.slice_caption = ['This is a benign and normal image. '] * 24
        # self.slice_caption = ['non-relevant anatomical structure'] * 24
        # self.slice_caption = ['benign and normal'] * 24
        # self.slice_caption = ['random text'] * 24
        # self.slice_caption[:7] = ["This image shows non-relevant anatomical structure."] * 7
        # self.slice_caption[-7:] = ["This image shows non-relevant anatomical structure."] * 7

        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    def __len__(self):
        return len(self.path_list1) + len(self.path_list2)

    def __getitem__(self, index):
        if self.mode == 'val':
            if index >= len(self.path_list1):
                path = self.path_list2[index % len(self.path_list1)]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
        else:
            if np.random.choice(2, 1, p=[1 - 0.5, 0.5]) == 0:
                index = index % len(self.path_list1)
                # index = np.random.randint(len(self.img_path1))
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                index = np.random.randint(len(self.path_list2))
                path = self.path_list2[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))

        lesion_count = path.split('/')[-1].split('_')[0]
        lesion_pid = self.lesion_pid[int(lesion_count)]
        lesion_slice = path.split('/')[-1].split('_')[1].split('.')[0]
        zone_count = self.zone_pid[lesion_pid]
        gland_count = None if not self.gland_pid else self.gland_pid[lesion_pid]
        # use zone_segdata_all for all data
        if '_158' in path:
            zone_seg = torch.Tensor(hdf5_reader(
                path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None),
                             'zone_segdata_158').replace('/' + str(lesion_count), '/' + str(zone_count)), 'seg'))
        elif '_QIN' in path:
            zone_seg = torch.Tensor(hdf5_reader(
                path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None),
                             'zone_segdata_QIN').replace('/' + str(lesion_count), '/' + str(zone_count)), 'seg'))
        elif '_MSD' in path:
            zone_seg = torch.Tensor(hdf5_reader(
                path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None),
                             'zone_segdata_MSD').replace('/' + str(lesion_count), '/' + str(zone_count)), 'seg'))
        else:
            zone_seg = torch.Tensor(hdf5_reader(
                path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None),
                             'zone_segdata_all').replace('/' + str(lesion_count), '/' + str(zone_count)), 'seg'))
        if not self.gland_pid:
            gland_seg = torch.zeros_like(lesion_seg)
        elif '_QIN' in path:
            gland_seg = torch.Tensor(hdf5_reader(
                path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None),
                             'gland_segdata_QIN').replace('/' + str(lesion_count), '/' + str(gland_count)), 'seg'))
        else:
            gland_seg = torch.Tensor(hdf5_reader(
                path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None),
                             'gland_segdata').replace('/' + str(lesion_count), '/' + str(gland_count)), 'seg'))
        lesion_seg = create_binary_masks(lesion_seg)
        zone_seg = create_binary_masks_zone(zone_seg)
        gland_seg = create_binary_masks(gland_seg)
        transform = transforms.Resize(size=(1024, 1024))
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=(1024, 1024),
                                          interpolation=transforms.functional.InterpolationMode.NEAREST)
        lesion_seg = seg_transform(lesion_seg)
        zone_seg = seg_transform(zone_seg)
        gland_seg = seg_transform(gland_seg)
        caption_tokens = self.tokenizer(self.slice_caption, context_length=256)

        sample = {'ct': ct}
        for i in range(lesion_seg.shape[0]):
            sample[f'lesion_seg_{i}'] = lesion_seg[i]
        for i in range(zone_seg.shape[0]):
            sample[f'zone_seg_{i}'] = zone_seg[i]
        for i in range(gland_seg.shape[0]):
            sample[f'gland_seg_{i}'] = gland_seg[i]
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)
        sample['tokens'] = caption_tokens
        sample['caption'] = self.slice_caption
        sample['score'] = 'PI-RADS notspecified'

        return sample, lesion_pid, lesion_slice


class DataGeneratorAlignPseudo(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, path_list, num_class=2, transform=None, mode='train'):
        self.path_list = path_list
        self.num_class = num_class
        self.transform = transform
        self.mode = mode

        self.slice_caption = ['Lesion, lesion, tumor'] * 24  # score: 0.731257
        # self.slice_caption = []
        # self.slice_caption = ['This image shows non-relevant anatomical structure.'] *24
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        with open(CHECKPOINT_PATH + '/pdt_tensor.pkl', 'r') as file:
            self.pdt_text = json.load(file)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        ct = torch.Tensor(hdf5_reader(self.path_list[index], 'ct'))
        seg = torch.Tensor(hdf5_reader(self.path_list[index], 'seg')).unsqueeze(0)
        transform = transforms.Resize(size=(1024, 1024))
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=(1024, 1024),
                                          interpolation=transforms.functional.InterpolationMode.NEAREST)
        seg = seg_transform(seg).squeeze(0).numpy()

        caption_tokens = self.tokenizer(self.slice_caption, context_length=256)

        sample = {'ct': ct, 'seg': seg}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        sample['tokens'] = caption_tokens
        sample['caption'] = self.slice_caption

        if self.mode == 'train':
            return sample
        else:
            return sample, self.path_list[index]


class DataGeneratorAlign(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, path_list, num_class=2, transform=None, mode='train'):

        self.path_list = [path for path in path_list if '721' not in path]
        # self.path_list = path_list

        self.num_class = num_class
        self.transform = transform
        self.mode = mode

        # self.slice_caption = ['Prostate tumor'] * 24

        # with open( CHECKPOINT_PATH + '/patient_text_prealign.json', 'r') as file:
        # with open( CHECKPOINT_PATH + '/patient_text_with_classes_updated.json', 'r') as file:
        with open(CHECKPOINT_PATH + '/patient_text_with_classes_updated_v2.json', 'r') as file:
            self.text_describtions = json.load(file)

        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        ct = torch.Tensor(hdf5_reader(self.path_list[index], 'ct'))
        seg = torch.Tensor(hdf5_reader(self.path_list[index], 'seg')).unsqueeze(0)
        transform = transforms.Resize(size=(1024, 1024))
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=(1024, 1024),
                                          interpolation=transforms.functional.InterpolationMode.NEAREST)
        seg = seg_transform(seg).squeeze(0).numpy()

        # caption_tokens = self.tokenizer(self.slice_caption, context_length=256)

        patient_caption = self.text_describtions[str(self.path_list[index].split('/')[-1])][2:]

        caption_tokens = self.tokenizer(patient_caption, context_length=256)

        sample = {'ct': ct, 'seg': seg}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        sample['caption'] = patient_caption
        sample['tokens'] = caption_tokens
        sample['location'] = self.text_describtions[str(self.path_list[index].split('/')[-1])][0]
        sample['score'] = self.text_describtions[str(self.path_list[index].split('/')[-1])][1]
        sample['name'] = self.path_list[index].split('/')[-1]

        if self.mode == 'train':
            return sample
        else:
            return sample, self.path_list[index]


class DataGenerator_no_resize(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, path_list, num_class=2, transform=None, mode='train'):
        self.path_list = path_list
        self.num_class = num_class
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        ct = torch.Tensor(hdf5_reader(self.path_list[index], 'ct')).numpy()
        seg = torch.Tensor(hdf5_reader(self.path_list[index], 'seg')).numpy()

        sample = {'ct': ct, 'seg': seg}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)
        if self.mode == 'train':
            return sample
        else:
            return sample, self.path_list[index]


def create_binary_masks(mask):
    """
    Create binary masks from a mask with multiple integer values.

    Parameters:
    mask (torch.Tensor): The original mask tensor with shape (1, width, height) containing integer values.

    Returns:
    dict: A dictionary where keys are unique values from the original mask and values are binary masks.
    """
    unique_values = torch.unique(mask)
    binary_masks = []

    for value in unique_values:
        if value.item() != 0:
            binary_mask = (mask == value).type(torch.uint8)
            binary_masks.append(binary_mask)
    if len(binary_masks) == 0:
        binary_masks = [mask.to(torch.uint8)]

    return torch.stack(binary_masks, dim=0)


def create_binary_masks_zone(mask):
    """
    Create binary masks from a mask with multiple integer values.

    Parameters:
    mask (torch.Tensor): The original mask tensor with shape (1, width, height) containing integer values.

    Returns:
    dict: A dictionary where keys are unique values from the original mask and values are binary masks.
    """
    binary_masks = []
    for value in [1, 2]:  # 1 --> pz? 2--> tz?
        binary_mask = (mask == value).type(torch.uint8)
        binary_masks.append(binary_mask)
    if len(binary_masks) == 0:
        binary_masks = [mask.to(torch.uint8)]

    return torch.stack(binary_masks, dim=0)


class MultiLevelDataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, lesion_path_list, mode, num_class=2, transform=None, zone_pid=None, lesion_pid=None,
                 gland_pid=None):
        new_path_list1 = []
        new_path_list2 = []
        ratios = []

        for idx, path in enumerate(lesion_path_list):
            lesion_seg = torch.Tensor(hdf5_reader(lesion_path_list[idx], 'seg'))
            lesion_count = lesion_path_list[idx].split('/')[-1].split('_')[0]
            this_lesion_pid = lesion_pid[int(lesion_count)]
            if this_lesion_pid == '10705_1000721':
                continue
            lesion_slice = lesion_path_list[idx].split('/')[-1].split('_')[1].split('.')[0]
            this_zone_count = zone_pid[this_lesion_pid]
            this_gland_count = gland_pid[this_lesion_pid]
            # print(lesion_path_list[idx].replace(PATH_DIR.split('/')[5], 'zone_segdata_all').replace(
            #             '/' + str(lesion_count), '/' + str(this_zone_count)))
            try:
                zone_seg = torch.Tensor(hdf5_reader(
                    lesion_path_list[idx].replace(PATH_DIR.split('/')[7], 'zone_segdata_all').replace(
                        '/' + str(lesion_count), '/' + str(this_zone_count)), 'seg'))
                gland_seg = torch.Tensor(hdf5_reader(
                    lesion_path_list[idx].replace(PATH_DIR.split('/')[7], 'gland_segdata').replace(
                        '/' + str(lesion_count), '/' + str(this_gland_count)), 'seg'))
            except:
                # print('here1')
                continue
            # if len(np.unique(zone_seg)) != 3:
            #     print('here2')
            #     continue
            tumor_ratio = lesion_seg.sum() / (lesion_seg.shape[0] * lesion_seg.shape[1])
            if tumor_ratio != 0:
                ratios.append(tumor_ratio)
            if tumor_ratio > 0.001:
                new_path_list2.append(path)
            else:
                new_path_list1.append(path)
        self.mode = mode
        self.path_list1 = new_path_list1  # no or small lesion
        self.path_list2 = new_path_list2  # larger lesion
        print(
            f'Got {len(new_path_list1)} slices with no or small lesion and {len(new_path_list2)} with sufficient lesion')
        self.num_class = num_class
        self.transform = transform
        self.zone_pid = zone_pid
        self.gland_pid = gland_pid
        self.lesion_pid = lesion_pid

    def __len__(self):
        return len(self.path_list1) + len(self.path_list2)

    def __getitem__(self, index):
        if self.mode == 'val':
            if index >= len(self.path_list1):
                path = self.path_list2[index % len(self.path_list1)]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
        else:
            if np.random.choice(2, 1, p=[1 - 0.5, 0.5]) == 0:
                index = index % len(self.path_list1)
                # index = np.random.randint(len(self.img_path1))
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                index = np.random.randint(len(self.path_list2))
                path = self.path_list2[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))

        lesion_count = path.split('/')[-1].split('_')[0]
        lesion_pid = self.lesion_pid[int(lesion_count)]
        lesion_slice = path.split('/')[-1].split('_')[1].split('.')[0]
        zone_count = self.zone_pid[lesion_pid]
        gland_count = self.gland_pid[lesion_pid]
        # use zone_segdata_all for all data
        zone_seg = torch.Tensor(hdf5_reader(
            path.replace(PATH_DIR.split('/')[7], 'zone_segdata_all').replace('/' + str(lesion_count),
                                                                             '/' + str(zone_count)), 'seg'))
        gland_seg = torch.Tensor(hdf5_reader(
            path.replace(PATH_DIR.split('/')[7], 'gland_segdata').replace('/' + str(lesion_count),
                                                                          '/' + str(gland_count)), 'seg'))
        lesion_seg = create_binary_masks(lesion_seg)
        zone_seg = create_binary_masks_zone(zone_seg)
        gland_seg = create_binary_masks(gland_seg)
        transform = transforms.Resize(size=(1024, 1024))
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=(1024, 1024),
                                          interpolation=transforms.functional.InterpolationMode.NEAREST)
        lesion_seg = seg_transform(lesion_seg)
        zone_seg = seg_transform(zone_seg)
        gland_seg = seg_transform(gland_seg)

        sample = {'ct': ct}
        for i in range(lesion_seg.shape[0]):
            sample[f'lesion_seg_{i}'] = lesion_seg[i]
        for i in range(zone_seg.shape[0]):
            sample[f'zone_seg_{i}'] = zone_seg[i]
        for i in range(gland_seg.shape[0]):
            sample[f'gland_seg_{i}'] = gland_seg[i]
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, lesion_pid, lesion_slice


class MultiLevelImgTxtDataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, lesion_path_list, mode, num_class=2, transform=None, zone_pid=None, lesion_pid=None,
                 gland_pid=None, tr_with_dummpy=False, txt_file_name='/slice_text_pre_align.json'):
        ratio_threshold = 0.001
        if os.path.exists(f'./list1_2d_{ratio_threshold}_{mode}.pkl') and os.path.exists(
                f'./list2_2d_{ratio_threshold}_{mode}.pkl'):
            with open(f'./list1_2d_{ratio_threshold}_{mode}.pkl', 'rb') as file:
                new_path_list1 = pickle.load(file)
            with open(f'./list2_2d_{ratio_threshold}_{mode}.pkl', 'rb') as file:
                new_path_list2 = pickle.load(file)
        else:
            new_path_list1 = []
            new_path_list2 = []
            ratios = []
            for idx, path in enumerate(lesion_path_list):
                lesion_seg = torch.Tensor(hdf5_reader(lesion_path_list[idx], 'seg'))
                lesion_count = lesion_path_list[idx].split('/')[-1].split('_')[0]
                this_lesion_pid = lesion_pid[int(lesion_count)]
                if this_lesion_pid == '10705_1000721':
                    continue
                # lesion_slice = lesion_path_list[idx].split('/')[-1].split('_')[1].split('.')[0]
                # this_zone_count = zone_pid[this_lesion_pid]
                # this_gland_count = gland_pid[this_lesion_pid]
                # try:
                #     zone_seg = torch.Tensor(hdf5_reader(
                #         lesion_path_list[idx].replace(PATH_DIR.split('/')[5], 'zone_segdata_all').replace(
                #             '/' + str(lesion_count), '/' + str(this_zone_count)), 'seg'))
                #     gland_seg = torch.Tensor(hdf5_reader(
                #         lesion_path_list[idx].replace(PATH_DIR.split('/')[5], 'gland_segdata').replace(
                #             '/' + str(lesion_count), '/' + str(this_gland_count)), 'seg'))
                # except:
                #     # print('here1')
                #     continue
                # if len(np.unique(zone_seg)) != 3:
                #     print('here2')
                #     continue
                tumor_ratio = lesion_seg.sum() / (lesion_seg.shape[0] * lesion_seg.shape[1])
                if tumor_ratio != 0:
                    ratios.append(tumor_ratio)
                if tumor_ratio > ratio_threshold:
                    new_path_list2.append(path)
                else:
                    new_path_list1.append(path)
            with open(f'./list1_2d_{ratio_threshold}_{mode}.pkl', 'wb') as file:
                pickle.dump(new_path_list1, file)
            with open(f'./list2_2d_{ratio_threshold}_{mode}.pkl', 'wb') as file:
                pickle.dump(new_path_list2, file)

        # read the textual dscribtion for each slices
        with open(CHECKPOINT_PATH + txt_file_name, 'r') as file:
            self.text_describtions = json.load(file)
            print(f'load text from {txt_file_name}')

        self.mode = mode
        self.path_list1 = new_path_list1  # no or small lesion
        self.path_list2 = new_path_list2  # larger lesion
        print(
            f'Got {len(new_path_list1)} slices with no or small lesion and {len(new_path_list2)} with sufficient lesion')
        self.num_class = num_class
        self.transform = transform
        self.zone_pid = zone_pid
        self.gland_pid = gland_pid
        self.lesion_pid = lesion_pid
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tr_dummy = tr_with_dummpy

        if self.tr_dummy:
            print('==> Training with dummy text!!!')

    def __len__(self):
        return len(self.path_list1) + len(self.path_list2)

    def __getitem__(self, index):
        if self.mode == 'val':
            if index >= len(self.path_list1):
                path = self.path_list2[index % len(self.path_list1)]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
        else:
            if np.random.choice(2, 1, p=[1 - 0.5, 0.5]) == 0:
                index = index % len(self.path_list1)
                # index = np.random.randint(len(self.img_path1))
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                index = np.random.randint(len(self.path_list2))
                path = self.path_list2[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))

        lesion_count = path.split('/')[-1].split('_')[0]
        lesion_pid = self.lesion_pid[int(lesion_count)]
        lesion_slice = path.split('/')[-1].split('_')[1].split('.')[0]
        zone_count = self.zone_pid[lesion_pid]
        gland_count = self.gland_pid[lesion_pid]
        # use zone_segdata_all for all data
        zone_seg = torch.Tensor(hdf5_reader(
            path.replace(PATH_DIR.split('/')[7], 'zone_segdata_all').replace('/' + str(lesion_count),
                                                                             '/' + str(zone_count)), 'seg'))
        gland_seg = torch.Tensor(hdf5_reader(
            path.replace(PATH_DIR.split('/')[7], 'gland_segdata').replace('/' + str(lesion_count),
                                                                          '/' + str(gland_count)), 'seg'))
        lesion_seg = create_binary_masks(lesion_seg)
        zone_seg = create_binary_masks_zone(zone_seg)
        gland_seg = create_binary_masks(gland_seg)
        transform = transforms.Resize(size=(1024, 1024))
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=(1024, 1024),
                                          interpolation=transforms.functional.InterpolationMode.NEAREST)
        lesion_seg = seg_transform(lesion_seg)
        zone_seg = seg_transform(zone_seg)
        gland_seg = seg_transform(gland_seg)

        # # if self.tr_dummy and self.mode != 'val':
        # if self.tr_dummy:
        #     slice_caption = random.choice(self.text_describtions[str(path.split('/')[-1])])
        # else:

        slice_caption = self.text_describtions[str(path.split('/')[-1])][0]

        caption_tokens = self.tokenizer(slice_caption, context_length=256)

        sample = {'ct': ct}

        for i in range(lesion_seg.shape[0]):
            sample[f'lesion_seg_{i}'] = lesion_seg[i]
        for i in range(zone_seg.shape[0]):
            sample[f'zone_seg_{i}'] = zone_seg[i]
        for i in range(gland_seg.shape[0]):
            sample[f'gland_seg_{i}'] = gland_seg[i]

        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        sample['caption'] = slice_caption

        # text_lower = slice_caption.lower()
        # # Ensure the text contains "lesion"
        # if "lesion" in text_lower:
        #     # Regex pattern to capture the location phrase after "lesion is located"
        #     location_pattern = (
        #         r"(?:involving|located(?: in| within| at)?)? "
        #         r"(peripheral and transition zones|peripheral zone|transition zone)"
        #     )
        #     location_match = re.search(location_pattern, text_lower)
        #     if location_match:
        #         location = location_match.group(1)
        #     else:
        #         location = "NotSpecified"

        #     # Regex to extract PI-RADS score with diverse patterns
        #     # pi_rads_match = re.search(r"PI-RADS(?: score(?: is| of)?| is| of)? (\d)", slice_caption, re.IGNORECASE)
        #     # pi_rads_score = int(pi_rads_match.group(1)) if pi_rads_match else "Not specified"
        #     pi_rads_pattern = r"PI-RADS(?: score)?(?: is| of| assessed(?: to be| as| of)?|:)*? (\d)"
        #     pi_rads_match = re.search(pi_rads_pattern, slice_caption)
        #     if pi_rads_match:
        #         pi_rads_score = int(pi_rads_match.group(1))
        #     else:
        #         pi_rads_score = "NotSpecified"
        # else:
        #     location = "NoLesion"
        #     pi_rads_score = "NotSpecified"

        location = self.text_describtions[str(path.split('/')[-1])][1]
        pi_rads_score = self.text_describtions[str(path.split('/')[-1])][2]

        if location == 'peripheral and transition zones' or location == 'peripheral and transition zone':
            location = 'transition zone'

        sample['tokens'] = caption_tokens
        sample['location'] = location
        sample['score'] = pi_rads_score

        return sample, lesion_pid, lesion_slice


if __name__ == "__main__":
    from config import PATH_DIR, PATH_LIST, FOLD_NUM, AP_LIST
    from run import get_cross_validation_by_sample
    import time
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    train_path, val_path = get_cross_validation_by_sample(AP_LIST, 5, 5)
    print('tr:', len(train_path), 'val:', len(val_path))

    with open('patien_list.txt', 'w') as file:
        for item in AP_LIST:
            file.write(f"{item}\n")

    lesion_pid = pickle.load(open(os.path.join(PATH_DIR, '../lesion_pid.p'), 'rb'))
    zone_pid = pickle.load(open(os.path.join(PATH_DIR, '../../zone_segdata_all/zone_pid.p'), 'rb'))
    gland_pid = pickle.load(open(os.path.join(PATH_DIR, '../../gland_segdata/gland_pid.p'), 'rb'))

    # print('start time...')

    start_time = time.time()

    val_transformer = transforms.Compose([
        Normalize(),
        # tio.Resize(target_shape=(24, 128, 128)),
        # tio.CropOrPad(target_shape=(32, 128, 128)),
        To_Tensor(num_class=2, input_channel=3)
    ])
    val_dataset = MultiLevelImgTxtDataGenerator(val_path, 'val',
                                                num_class=2,
                                                transform=val_transformer,
                                                zone_pid=zone_pid,
                                                gland_pid=gland_pid,
                                                lesion_pid=lesion_pid,
                                                tr_with_dummpy=False,
                                                txt_file_name='/slice_text_with_classes.json')

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    idx = 0
    for step, (sample, pid, slice) in enumerate(tqdm(val_loader)):
        if idx == 5:
            break
        # print(step, sample['caption'], 'score:', sample['score'], 'location:', sample['location'], '\n')
        print(step, 'score:', sample['score'], 'location:', sample['location'], '\n')
        idx += 1

    # # jj

    # for idx in range(len(val_dataset)):
    #     sample, path = val_dataset[idx]

    #     print(idx, len(sample['caption']))
    #     print(idx, sample['seg'].shape)
    #     print(idx, sample['ct'].shape)
    #     print(idx, len(sample['tokens']), sample['tokens'][0].shape)
    # # print(path)

    # print('run time:%.4f' % (time.time() - start_time))