import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tr
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

from classification.config import CSV_PATH, TASK, VERSION
from classification.data_utils.csv_reader import csv_reader_single
from classification.data_utils.data_loader import DataGenerator
from classification.data_utils.transforms import RandomRotate
from classification.run import get_cross_validation
from classification.utils import dfs_remove_weight
from efficientnet_pytorch import EfficientNet

torch.manual_seed(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = 'cuda:0'

output_dir = './ckpt/{}/{}'.format(TASK,VERSION)
log_dir = ./log/{}/{}'.format(TASK,VERSION)
ckpt_file = os.path.join(output_dir, 'classification/ckpt/picai_all_class_weight/v0/fold1/epoch:1-train_loss:0.69738-val_loss:0.57193-train_acc:0.78560-val_ap:0.93620.pth')

output_dir = os.path.join(output_dir, "fold"+str(1))
log_dir = os.path.join(log_dir, "fold"+str(1))

state_dict = torch.load(ckpt_file, map_location=device)['state_dict']
net = EfficientNet.from_name(model_name='efficientnet-b5')
num_ftrs = net._fc.in_features
net._fc = nn.Linear(num_ftrs, 2)
net.load_state_dict(state_dict)
net.eval()
net.to(device)
lr = 1e-3
weight_decay = 0

label_dict = {}
# Set data path & classifier

pre_csv_path = CSV_PATH
pre_label_dict = csv_reader_single(pre_csv_path, key_col='id', value_col='label')
label_dict.update(pre_label_dict)
path_list = list(label_dict.keys())
train_path, val_path = get_cross_validation(
                path_list, 5, 1)

label_dict.update(pre_label_dict)
transform = [
            tr.Resize(size=(384,384)),  #2
            tr.RandomAffine(0,(0.05,0.05),(0.8,1.2)),  #3
            tr.ColorJitter(brightness=.3, hue=.3, contrast=.3),  #4
            tr.RandomPerspective(distortion_scale=0.6, p=0.5),  #5
            RandomRotate((-12,12)),
            tr.RandomHorizontalFlip(p=0.5),   #  7
            tr.RandomVerticalFlip(p=0.5),   #8
            tr.ToTensor(),   #9
        ]
val_transformer = transforms.Compose(transform)

val_dataset = DataGenerator(
    val_path, label_dict, channels=3, transform=val_transformer)