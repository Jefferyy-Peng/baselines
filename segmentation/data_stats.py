import glob
import os
import torch
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

from segmentation.utils import hdf5_reader, get_cross_validation_by_sample

PATH_DIR = '/data/nvme1/meng/picai/lesion_segdata_combined/data_2d'
PATH_LIST = glob.glob(os.path.join(PATH_DIR,'*.hdf5'))

sizes_2d = []
sizes_3d = []
train_path, val_path = get_cross_validation_by_sample(PATH_LIST, 5, 1)
lesion_pids = pickle.load(open(os.path.join(PATH_DIR, '../lesion_pid.p'), 'rb'))
zone_pids = pickle.load(open(os.path.join(PATH_DIR, '../../zone_segdata_all/zone_pid.p'), 'rb'))
gland_pids = pickle.load(open(os.path.join(PATH_DIR, '../../gland_segdata/lesion_pid.p'), 'rb'))
lesion_slice_count = 0
gland_slice_count = 0
nothing_slice_count = 0
for i in tqdm(range(len(val_path))):
    path = val_path[i]
    lesion_seg = torch.Tensor(hdf5_reader(path, 'seg'))
    lesion_count = path.split('/')[-1].split('_')[0]
    lesion_pid = lesion_pids[int(lesion_count)]
    lesion_slice = path.split('/')[-1].split('_')[1].split('.')[0]
    zone_count = zone_pids[lesion_pid]
    gland_count = None if not gland_pids else gland_pids[lesion_pid]
    # use zone_segdata_all for all dataq
    zone_seg = torch.Tensor(hdf5_reader(
        path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'zone_segdata_158').replace(
            '/' + str(lesion_count), '/' + str(zone_count)), 'seg')) if '_158' in path else torch.Tensor(hdf5_reader(
        path.replace(next((x for x in path.split('/') if 'lesion_segdata' in x), None), 'zone_segdata_all').replace(
            '/' + str(lesion_count), '/' + str(zone_count)), 'seg'))
    gland_seg = torch.zeros_like(lesion_seg) if not gland_pids else torch.Tensor(hdf5_reader(
        path.replace(PATH_DIR.split('/')[5], 'gland_segdata').replace('/' + str(lesion_count), '/' + str(gland_count)),
        'seg'))

    lesion_size_2d = lesion_seg.sum().item()
    zone_size_2d = zone_seg.sum().item()
    gland_size_2d = gland_seg.sum().item()
    if lesion_size_2d != 0:
        lesion_slice_count += 1
    elif gland_size_2d != 0 or zone_size_2d != 0:
        gland_slice_count += 1
    else:
        nothing_slice_count += 1

for i in range(len(PATH_LIST)):
    lesion_seg = torch.Tensor(hdf5_reader(PATH_LIST[i], 'seg'))
    size_3d = lesion_seg.sum().item()
    if size_3d != 0:
        sizes_3d.append(size_3d)

    for slice in lesion_seg:
        size_2d = slice.sum().item()
        if size_2d != 0:
            sizes_2d.append(size_2d)

plt.figure(figsize=(18,16))  # Set figure size
plt.hist(sizes_2d, bins=200, edgecolor='black')  # 'bins' adjusts the number of bars

# Adding labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f'Histogram of size2d, min: {torch.Tensor(sizes_2d).min()}, max: {torch.Tensor(sizes_2d).max()}')

# Display the histogram
plt.savefig('./stats/2d_size.png')

plt.figure(figsize=(18,16))  # Set figure size
plt.hist(sizes_3d, bins=200, edgecolor='black')  # 'bins' adjusts the number of bars

# Adding labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f'Histogram of size3d, min: {torch.Tensor(sizes_3d).min()}, max: {torch.Tensor(sizes_3d).max()}')

# Display the histogram
plt.savefig('./stats/3d_size.png')
