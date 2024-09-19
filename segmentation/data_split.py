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
from segmentation.MedSAMAuto import MedSAMAUTO, MedSAMAUTOZONE, MedSAMAUTOMULTI, MedSAMAUTOCNN
from segmentation.config import FOLD_NUM, CURRENT_FOLD
from segmentation.model_single import ModelEmb, SegDecoderCNN
from segmentation.segment_anything import sam_model_registry
from segmentation.run import get_cross_validation_by_sample
from segmentation.segment_anything.modeling import TwoWayTransformer, MaskDecoder
from picai_eval import Metrics
from picai_eval.eval import evaluate_case

from segmentation.utils import compute_results_detect, post_process, calculate_max_tumor_distance
from segmentation.eval_utils import erode_dilate

PATH_AP = './dataset/lesion_segdata_combined/data_3d'
AP_LIST = glob.glob(os.path.join(PATH_AP, '*.hdf5'))
train_path, val_path = get_cross_validation_by_sample(AP_LIST, FOLD_NUM, 1)

class Normalize_2d(object):
    def __call__(self, sample):
        ct = sample['ct']
        seg = sample['seg']
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if np.max(ct[i, j]) != 0:
                    ct[i, j] = ct[i, j] / np.max(ct[i, j])

        new_sample = {'ct': ct, 'seg': seg}
        return new_sample

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