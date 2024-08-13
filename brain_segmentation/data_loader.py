import random

import monai
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import hdf5_reader

def create_binary_masks(mask):
    """
    Create binary masks from a mask with multiple integer values.

    Parameters:
    mask (torch.Tensor): The original mask tensor with shape (1, width, height) containing integer values.

    Returns:
    dict: A dictionary where keys are unique values from the original mask and values are binary masks.
    """
    binary_masks = []
    for value in [0, 1, 2, 3]:
        binary_mask = (mask == value).type(torch.uint8)
        binary_masks.append(binary_mask)
    if len(binary_masks) == 0:
        binary_masks = [mask.to(torch.uint8)]

    return torch.stack(binary_masks, dim=0)

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
                label = Image.fromarray(data)
                label = label.rotate(rotate_degree, Image.NEAREST)
                label = np.array(label).astype(np.float32)
                new_sample[name] = label
        # ct_image = sample['ct']
        # lesion_label = sample['seg']
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

class RandomRotate3D(object):
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
                c, s, h, w = data.shape
                data = data.reshape(-1, h, w)
                cts = []
                for i in range(data.shape[0]):
                    cts.append(Image.fromarray(data[i]))
                cts_out = []
                for ct in cts:
                    ct = ct.rotate(rotate_degree, Image.BILINEAR)
                    ct = np.array(ct).astype(np.float32)
                    cts_out.append(ct)
                ct_image = np.asarray(cts_out)
                ct_image = ct_image.reshape(c, s, h, w)
                new_sample['ct'] = ct_image

            elif 'seg' in name:
                labels = []
                for i in range(data.shape[0]):
                    labels.append(Image.fromarray(data[i]))
                labels_out = []
                for label in labels:
                    label = label.rotate(rotate_degree, Image.NEAREST)
                    label = np.array(label).astype(np.float32)
                    labels_out.append(label)
                label = np.asarray(labels_out)
                new_sample[name] = label
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

class RandomFlip3D(object):
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
                        data = data[:, :, :, ::-1]
                    elif random_factor < 0.6:
                        data = data[:, :, ::-1, :]
                elif 'h' in self.mode:
                    if random_factor > 0.5:
                        data = data[:, :, :, ::-1]
                elif 'v' in self.mode:
                    if random_factor > 0.5:
                        data = data[:, :, ::-1, :]
                new_sample['ct'] = data.copy()
            elif 'seg' in name:
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
                new_sample[name] = data.copy()
        return new_sample

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

class DataGenerator3D(Dataset):
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
        PSIR = torch.Tensor(hdf5_reader(self.path_list[index], 'PSIR'))
        T1W = torch.Tensor(hdf5_reader(self.path_list[index], 'T1W'))
        T2W = torch.Tensor(hdf5_reader(self.path_list[index], 'T2W'))
        R2 = torch.Tensor(hdf5_reader(self.path_list[index], 'R2'))
        seg = torch.Tensor(hdf5_reader(self.path_list[index], 'seg')).permute(0, 2, 1)
        transform = monai.transforms.Compose([
            monai.transforms.SpatialPad(spatial_size=(128,512,512), method='symmetric'),  # Pad if smaller than target size
            monai.transforms.CenterSpatialCrop(roi_size=(128,512,512))  # Crop to target size
        ])
        ct = torch.stack([PSIR, T1W, T2W, R2])
        ct = transform(ct).numpy()
        # seg_transform = monai.transforms.Resize(size=(1024, 1024), interpolation=transforms.functional.InterpolationMode.NEAREST)
        # seg = seg_transform(seg).numpy()
        seg = transform(seg.unsqueeze(0)).numpy().squeeze(0)

        sample = {'ct': ct, 'seg': seg}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        # sample['seg'] = create_binary_masks(sample['seg'])
        if self.mode == 'train':
            return sample
        elif self.mode == 'val':
            return sample, self.path_list[index]


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
        PSIR = torch.Tensor(hdf5_reader(self.path_list[index], 'PSIR'))
        T1W = torch.Tensor(hdf5_reader(self.path_list[index], 'T1W'))
        T2W = torch.Tensor(hdf5_reader(self.path_list[index], 'T2W'))
        R2 = torch.Tensor(hdf5_reader(self.path_list[index], 'R2'))
        seg = torch.Tensor(hdf5_reader(self.path_list[index], 'seg')).unsqueeze(0).permute(0, 2, 1)
        transform = transforms.Resize(size=(1024, 1024))
        ct = torch.stack([PSIR, T1W, T2W, R2])
        ct = transform(ct).numpy()
        seg_transform = transforms.Resize(size=(1024, 1024), interpolation=transforms.functional.InterpolationMode.NEAREST)
        seg = seg_transform(seg).squeeze(0).numpy()

        sample = {'ct': ct, 'seg': seg}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        # sample['seg'] = create_binary_masks(sample['seg'])
        if self.mode == 'train':
            return sample
        elif self.mode == 'val':
            return sample, self.path_list[index]


