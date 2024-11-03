import glob
import os
import torch
from matplotlib import pyplot as plt

from segmentation.utils import hdf5_reader

PATH_DIR = './dataset/lesion_segdata_combined/data_3d'
PATH_LIST = glob.glob(os.path.join(PATH_DIR,'*.hdf5'))

sizes_2d = []
sizes_3d = []

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
