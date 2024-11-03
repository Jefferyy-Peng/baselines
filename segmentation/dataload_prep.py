import os
import pickle

import h5py
import torch

from segmentation.data_loader import create_binary_masks, create_binary_masks_zone
from utils import hdf5_reader

root = './dataset/lesion_segdata_combined'
gland_path = './dataset/gland_segdata'
zone_path = './dataset/zone_segdata_all'
output_path = './dataset/combined/data_2d'
os.makedirs(output_path, exist_ok=True)

lesion_pids = pickle.load(open(os.path.join(root, 'lesion_pid.p'), 'rb'))
gland_pids = pickle.load(open(os.path.join(gland_path, 'gland_pid.p'), 'rb'))
zone_pids = pickle.load(open(os.path.join(zone_path, 'zone_pid.p'), 'rb'))

path_list = os.listdir(os.path.join(root, 'data_2d'))

for path in path_list:
    lesion_path = os.path.join(root, 'data_2d', path)
    ct = torch.Tensor(hdf5_reader(lesion_path, 'ct'))
    lesion_seg = torch.Tensor(hdf5_reader(lesion_path, 'seg'))

    lesion_count = path.split('/')[-1].split('_')[0]
    lesion_pid = lesion_pids[int(lesion_count)]
    if lesion_pid == '10705_1000721':
        continue
    lesion_slice = path.split('/')[-1].split('_')[1].split('.')[0]
    zone_count = zone_pids[lesion_pid]
    gland_count = gland_pids[lesion_pid]
    # use zone_segdata_all for all dataq
    zone_seg = torch.Tensor(hdf5_reader(lesion_path.replace(lesion_path.split('/')[2], 'zone_segdata_all').replace('/'+str(lesion_count), '/'+str(zone_count)), 'seg'))
    gland_seg = torch.Tensor(hdf5_reader(lesion_path.replace(lesion_path.split('/')[2], 'gland_segdata').replace('/'+str(lesion_count), '/'+str(gland_count)), 'seg'))
    lesion_seg = create_binary_masks(lesion_seg)
    zone_seg = create_binary_masks_zone(zone_seg)
    gland_seg = create_binary_masks(gland_seg)
    multilevel = torch.cat([gland_seg, zone_seg, lesion_seg])
    with h5py.File(os.path.join(output_path, path), 'w') as h5_file:
        # Save image and segmentation arrays as datasets
        h5_file.create_dataset("ct", data=ct, compression="gzip")
        h5_file.create_dataset("seg", data=multilevel, compression="gzip")