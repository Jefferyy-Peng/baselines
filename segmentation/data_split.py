import copy
import glob
import os
import pickle
import random
import re
from collections import OrderedDict
import seaborn as sns
from monai.transforms import Resize

from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Sized,
                    Tuple, Union)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from data_loader import (DataGenerator, Normalize, RandomFlip2D,
                         RandomRotate2D, To_Tensor, MultiLevelDataGenerator, DataGenerator_no_resize)
from config import FOLD_NUM, CURRENT_FOLD
from utils import get_cross_validation_by_sample, Normalize_2d

from utils import compute_results_detect, post_process, calculate_max_tumor_distance

PATH_AP = './dataset/lesion_segdata_combined/data_3d'
AP_LIST = glob.glob(os.path.join(PATH_AP, '*.hdf5'))
train_path, val_path = get_cross_validation_by_sample(AP_LIST, FOLD_NUM, 1)



val_transformer = transforms.Compose(
        [Normalize_2d(), To_Tensor()])

val_dataset = DataGenerator_no_resize(val_path, transform=val_transformer, mode='val')

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=32,
    pin_memory=True
)
small_list = []
large_list = []

for step, (sample, path) in enumerate(tqdm(val_loader)):
    data = sample['ct'].squeeze().numpy()
    target = sample['seg'].squeeze().numpy()

    max_dist = calculate_max_tumor_distance(target, [3.0, 0.5, 0.5])
    if max_dist <= 10:
        small_list.append(path[0])
    else:
        large_list.append(path[0])

dict = {'small': small_list, 'large': large_list}

with open('./dataset/lesion_segdata_combined/data_split.p', 'wb') as f:
    pickle.dump(dict, f)

    # data = data.squeeze().transpose(1, 0)
    # data = data.to(device)
    # target = target.to(device)