import os.path
import pickle
import random
from typing import Any, Callable, Optional, Sequence
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from monai.transforms import Randomizable
from monai.utils import MAX_SEED

from config import PATH_DIR
from utils import hdf5_reader


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
                if len(data.shape) > 2:
                    segs = []
                    for i in range(data.shape[0]):
                        segs.append(Image.fromarray(data[i]))
                    segs_out = []
                    for seg in segs:
                        seg = seg.rotate(rotate_degree, Image.BILINEAR)
                        seg = np.array(seg).astype(np.float32)
                        segs_out.append(seg)
                    label = np.asarray(segs_out).astype(np.float32)
                else:
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
                        data = data[:, ::-1] if len(data.shape) == 2 else data[:, :, ::-1]
                    elif random_factor < 0.6:
                        data = data[::-1, :] if len(data.shape) == 2 else data[:, ::-1, :]
                elif 'h' in self.mode:
                    if random_factor > 0.5:
                        data = data[:, ::-1] if len(data.shape) == 2 else data[:, :, ::-1]
                elif 'v' in self.mode:
                    if random_factor > 0.5:
                        data = data[::-1, :] if len(data.shape) == 2 else data[:, ::-1, :]
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

    def __init__(self, path_list, num_class=2, transform=None, mode='train', image_size=1024):
        self.path_list = path_list
        self.num_class = num_class
        self.transform = transform
        self.mode=mode
        self.image_size = image_size

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        ct = torch.Tensor(hdf5_reader(self.path_list[index], 'ct'))
        seg = torch.Tensor(hdf5_reader(self.path_list[index], 'seg')).unsqueeze(0)
        transform = transforms.Resize(size=(self.image_size, self.image_size))
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=(self.image_size, self.image_size), interpolation=transforms.functional.InterpolationMode.NEAREST)
        seg = seg_transform(seg).squeeze(0).numpy()

        sample = {'ct': ct, 'seg': seg}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)
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
        self.mode=mode

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        ct = torch.Tensor(hdf5_reader(self.path_list[index], 'ct')).numpy()
        seg = torch.Tensor(hdf5_reader(self.path_list[index], 'seg')).numpy()

        sample = {'ct': ct, 'seg': np.expand_dims(seg, 0)}
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
    for value in [1, 2]:
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

    def __init__(self, lesion_path_list, mode, image_size, num_class=2, transform=None, zone_pid=None, lesion_pid=None, gland_pid=None):
        ratio_threshold = 0.001
        if '_158' in lesion_path_list[0] or '_Diagnosis' in lesion_path_list[0] or '_QIN' in lesion_path_list[0] or '_MSD' in lesion_path_list[0]:
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
            if os.path.exists(f'./list1_2d_{ratio_threshold}_{mode}.pkl') and os.path.exists(f'./list2_2d_{ratio_threshold}_{mode}.pkl'):
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
        self.image_size = image_size
        self.mode = mode
        self.path_list1 = new_path_list1
        self.path_list2 = new_path_list2
        print(f'Got {len(new_path_list1)} slices with no or small lesion and {len(new_path_list2)} with sufficient lesion')
        self.num_class = num_class
        self.transform = transform
        self.zone_pid = zone_pid
        self.gland_pid = gland_pid
        self.lesion_pid = lesion_pid

    def __len__(self):
        return len(self.path_list1) + len(self.path_list2)

    def __getitem__(self, index):
        if self.mode == 'val' or self.mode == 'split':
            if index >= len(self.path_list1):
                path = self.path_list2[index % len(self.path_list1)]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
        else:
            if np.random.choice(2, 1, p=[1-0.5, 0.5]) == 0:
                index = index % len(self.path_list1)
                #index = np.random.randint(len(self.img_path1))
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
        # use zone_segdata_all for all dataq
        if '_158' in path:
            zone_seg = torch.Tensor(hdf5_reader(path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'zone_segdata_158').replace('/'+str(lesion_count), '/'+str(zone_count)), 'seg'))
        elif '_QIN' in path:
            zone_seg = torch.Tensor(hdf5_reader(
                path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None),
                             'zone_segdata_QIN').replace('/' + str(lesion_count), '/' + str(zone_count)), 'seg'))
        elif '_MSD' in path:
            zone_seg = torch.Tensor(hdf5_reader(
                path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None),
                             'zone_segdata_MSD').replace('/' + str(lesion_count), '/' + str(zone_count)), 'seg'))
        else:
            zone_seg = torch.Tensor(hdf5_reader(path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'zone_segdata_all').replace('/'+str(lesion_count), '/'+str(zone_count)), 'seg'))
        if not self.gland_pid:
            gland_seg = torch.zeros_like(lesion_seg)
        elif '_QIN' in path:
            gland_seg = torch.Tensor(hdf5_reader(path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'gland_segdata_QIN').replace('/'+str(lesion_count), '/'+str(gland_count)), 'seg'))
        else:
            gland_seg = torch.Tensor(hdf5_reader(path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'gland_segdata').replace('/'+str(lesion_count), '/'+str(gland_count)), 'seg'))
        lesion_seg = create_binary_masks(lesion_seg)
        zone_seg = create_binary_masks_zone(zone_seg)
        gland_seg = create_binary_masks(gland_seg)
        transform = transforms.Resize(size=self.image_size)
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=self.image_size, interpolation=transforms.functional.InterpolationMode.NEAREST)
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
        if self.mode == 'split':
            return sample, lesion_pid, lesion_slice, path
        else:
            return sample, lesion_pid, lesion_slice


class MultiLevelDataGeneratorSeg(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, path_list, mode, image_size, num_class=2, transform=None, lesion_pid=None):
        new_path_list1 = []
        new_path_list2 = []
        ratios = []
        for idx, path in enumerate(path_list):
            seg = torch.Tensor(hdf5_reader(path_list[idx], 'seg'))
            count = path_list[idx].split('/')[-1].split('_')[0]
            this_lesion_pid = lesion_pid[int(count)]

            # if len(np.unique(zone_seg)) != 3:
            #     print('here2')
            #     continue
            lesion_seg = seg[-1]
            tumor_ratio = lesion_seg.sum() / (lesion_seg.shape[0] * lesion_seg.shape[1])
            if tumor_ratio != 0:
                ratios.append(tumor_ratio)
            if tumor_ratio > 0.001:
                new_path_list2.append(path)
            else:
                new_path_list1.append(path)
        self.image_size = image_size
        self.mode = mode
        self.path_list1 = new_path_list1
        self.path_list2 = new_path_list2
        print(f'Got {len(new_path_list1)} slices with no or small lesion and {len(new_path_list2)} with sufficient lesion')
        self.num_class = num_class
        self.transform = transform
        self.lesion_pid = lesion_pid

    def __len__(self):
        return len(self.path_list1) + len(self.path_list2)

    def __getitem__(self, index):
        if self.mode == 'val' or self.mode == 'split':
            if index >= len(self.path_list1):
                path = self.path_list2[index % len(self.path_list1)]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                seg = torch.Tensor(hdf5_reader(path, 'seg'))
        else:
            if np.random.choice(2, 1, p=[1-0.5, 0.5]) == 0:
                index = index % len(self.path_list1)
                #index = np.random.randint(len(self.img_path1))
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                index = np.random.randint(len(self.path_list2))
                path = self.path_list2[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                seg = torch.Tensor(hdf5_reader(path, 'seg'))

        lesion_count = path.split('/')[-1].split('_')[0]
        lesion_pid = self.lesion_pid[int(lesion_count)]
        lesion_slice = path.split('/')[-1].split('_')[1].split('.')[0]

        transform = transforms.Resize(size=self.image_size)
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=self.image_size, interpolation=transforms.functional.InterpolationMode.NEAREST)
        seg = seg_transform(seg).numpy()

        sample = {'ct': ct, 'seg': seg}

        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, lesion_pid, lesion_slice

class MultiLevel3DDataGenerator(Dataset, Randomizable):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''

    def __init__(self, lesion_path_list, mode, image_size, num_class=2, transform=None, zone_pid=None, lesion_pid=None, gland_pid=None):
        if '_158' in lesion_path_list[0] or '_Diagnosis' in lesion_path_list[0] or '_QIN' in lesion_path_list[0] or '_MSD' in lesion_path_list[0]:
            ratio_threshold = 0.001
            new_path_list1 = []
            new_path_list2 = []
            new_path_list = []
            ratios = []
            for idx, path in enumerate(lesion_path_list):
                lesion_seg = torch.Tensor(hdf5_reader(lesion_path_list[idx], 'seg'))
                lesion_count = lesion_path_list[idx].split('/')[-1].split('.')[0]
                this_lesion_pid = lesion_pid[int(lesion_count)]
                if this_lesion_pid == '10705_1000721':
                    continue
                tumor_ratio = lesion_seg.sum() / (lesion_seg.shape[0] * lesion_seg.shape[1] * lesion_seg.shape[2])
                new_path_list.append(path)
                if tumor_ratio != 0:
                    ratios.append(tumor_ratio)
                if tumor_ratio > ratio_threshold:
                    new_path_list2.append(path)
                else:
                    new_path_list1.append(path)
        else:
            ratio_threshold = 0.001
            if os.path.exists(f'./list1_3d_{ratio_threshold}_{mode}.pkl') and os.path.exists(f'./list_3d_{ratio_threshold}_{mode}.pkl') and os.path.exists(f'./list2_3d_{ratio_threshold}_{mode}.pkl'):
                with open(f'./list_3d_{ratio_threshold}_{mode}.pkl', 'rb') as file:
                    new_path_list = pickle.load(file)
                with open(f'./list1_3d_{ratio_threshold}_{mode}.pkl', 'rb') as file:
                    new_path_list1 = pickle.load(file)
                with open(f'./list2_3d_{ratio_threshold}_{mode}.pkl', 'rb') as file:
                    new_path_list2 = pickle.load(file)
            else:
                new_path_list1 = []
                new_path_list2 = []
                new_path_list = []
                ratios = []
                for idx, path in enumerate(lesion_path_list):
                    lesion_seg = torch.Tensor(hdf5_reader(lesion_path_list[idx], 'seg'))
                    lesion_count = lesion_path_list[idx].split('/')[-1].split('.')[0]
                    this_lesion_pid = lesion_pid[int(lesion_count)]
                    if this_lesion_pid == '10705_1000721':
                        continue
                    tumor_ratio = lesion_seg.sum() / (lesion_seg.shape[0] * lesion_seg.shape[1]  * lesion_seg.shape[2])
                    new_path_list.append(path)
                    if tumor_ratio != 0:
                        ratios.append(tumor_ratio)
                    if tumor_ratio > ratio_threshold:
                        new_path_list2.append(path)
                    else:
                        new_path_list1.append(path)
                with open(f'./list_3d_{ratio_threshold}_{mode}.pkl', 'wb') as file:
                    pickle.dump(new_path_list, file)
                with open(f'./list1_3d_{ratio_threshold}_{mode}.pkl', 'wb') as file:
                    pickle.dump(new_path_list1, file)
                with open(f'./list2_3d_{ratio_threshold}_{mode}.pkl', 'wb') as file:
                    pickle.dump(new_path_list2, file)
        self.path_list = new_path_list
        self.mode = mode
        self.path_list1 = new_path_list1
        self.path_list2 = new_path_list2
        self.image_size = image_size
        self.num_class = num_class
        self.transform = transform
        self.zone_pid = zone_pid
        self.gland_pid = gland_pid
        self.lesion_pid = lesion_pid
        self.weight = np.array([len(new_path_list2), len(new_path_list1)]) / (len(self.path_list1) + len(self.path_list2))

    def __len__(self):
        return len(self.path_list1) + len(self.path_list2)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index):
        if self.mode == 'random':
            path = self.path_list[index]
            ct = torch.Tensor(hdf5_reader(path, 'ct'))
            lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
        elif self.mode == 'val' or self.mode == 'split':
            if index >= len(self.path_list1):
                path = self.path_list2[index % len(self.path_list1)]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
        else:
            if np.random.choice(2, 1, p=[1-0.5, 0.5]) == 0:
                index = index % len(self.path_list1)
                #index = np.random.randint(len(self.img_path1))
                path = self.path_list1[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
            else:
                index = np.random.randint(len(self.path_list2))
                path = self.path_list2[index]
                ct = torch.Tensor(hdf5_reader(path, 'ct'))
                lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))

        lesion_count = path.split('/')[-1].split('.')[0]
        lesion_pid = self.lesion_pid[int(lesion_count)]
        zone_count = self.zone_pid[lesion_pid]
        gland_count = None if not self.gland_pid else self.gland_pid[lesion_pid]
        # use zone_segdata_all for all data
        if '_158' in path:
            zone_seg = torch.Tensor(hdf5_reader(path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'zone_segdata_158').replace('/'+str(lesion_count), '/'+str(zone_count)), 'seg'))
        elif '_MSD' in path:
            zone_seg = torch.Tensor(hdf5_reader(path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'zone_segdata_MSD').replace('/'+str(lesion_count), '/'+str(zone_count)), 'seg'))
        else:
            zone_seg = torch.Tensor(hdf5_reader(path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'zone_segdata_all').replace('/'+str(lesion_count), '/'+str(zone_count)), 'seg'))
        gland_seg = torch.zeros_like(lesion_seg) if not self.gland_pid else torch.Tensor(hdf5_reader(path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'gland_segdata').replace('/'+str(lesion_count), '/'+str(gland_count)), 'seg'))
        lesion_seg = create_binary_masks(lesion_seg)
        zone_seg = create_binary_masks_zone(zone_seg)
        gland_seg = create_binary_masks(gland_seg)
        # Convert to TorchIO Images
        # image = tio.ScalarImage(tensor=ct)
        # label = tio.LabelMap(tensor=segs)

        sample = {'ct': ct}
        for i in range(lesion_seg.shape[0]):
            sample[f'lesion_seg_{i}'] = lesion_seg[i].unsqueeze(0)
        for i in range(zone_seg.shape[0]):
            sample[f'zone_seg_{i}'] = zone_seg[i].unsqueeze(0)
        for i in range(gland_seg.shape[0]):
            sample[f'gland_seg_{i}'] = gland_seg[i].unsqueeze(0)

        sample = self.transform(sample)
        if self.mode == 'split':
            return sample, lesion_pid, 0, path
        else:
            return sample, lesion_pid, 0