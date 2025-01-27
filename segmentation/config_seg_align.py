import glob
import os

from utils import get_weight_path
from time import gmtime, process_time_ns, strftime


TRANSFORMER_DEPTH = 24
# VERSION = 'MedSAMAuto_Finetune_Encoder_Focal_Unified_equal_rate_0.8_weighted_loss_combined_label_lr_0.0001_weight_decay_0.001'
VERSION  ='Seg_Align'


PHASE = 'seg'   # 'seg' or 'detect'
NUM_CLASSES = 2 if 'seg' in PHASE else 3

DEVICE = 'cuda:0'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = True
# True if use resume model
CKPT_POINT = False

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))

#--------------------------------- mode and data path setting
PATH_DIR = '/usa/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/segmentation/dataset/lesion_segdata_combined/data_2d'
PATH_LIST = glob.glob(os.path.join(PATH_DIR,'*.hdf5'))
PATH_AP = '/usa/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/segmentation/dataset/lesion_segdata_combined/data_3d'
AP_LIST = glob.glob(os.path.join(PATH_AP,'*.hdf5'))
#---------------------------------

CHECKPOINT_PATH = '/usa/mengma/myproject/cvpr25/baselines/segmentation/checkpoints'

CKPT_PATH = './new_ckpt/{}/{}/fold1'.format(PHASE,VERSION)
# WEIGHT_PATH = '/usa/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/segmentation/new_ckpt/seg/MedSAMAuto_Focal_Unified_equal_rate_high_weighted_loss_combined_label_lr_0.0001_weight_decay_0.001/fold1/epoch:11-gland_val_dice:0.83997-zone_val_dice:0.84296-lesion_val_dice:0.73409-lesion_val_ap:0.49188-lesion_val_auc:0.90977.pth'
WEIGHT_PATH = '/usa/mengma/myproject/cvpr25/baselines/segmentation/new_ckpt/seg/Seg_Align/2024-11-12T03:06:44_lr_0.0001_weight_decay_0.001/fold1/epoch:2-gland_val_dice:0.89853-zone_val_dice:0.86082-lesion_val_dice:0.92010-lesion_val_ap:0.76636-lesion_val_auc:0.97264.pth'
# WEIGHT_PATH = '/usa/mengma/myproject/cvpr25/baselines/segmentation/new_ckpt/seg/MedSAMAuto_Finetune_Encoder_Focal_0.97_weighted_loss_Seg_Align_lr_0.0001_weight_decay_0.001/fold1/2024-11-06T14:12:46-epoch:6-gland_val_dice:0.90087-zone_val_dice:0.87288-lesion_val_dice:0.91010-lesion_val_ap:0.76170-lesion_val_auc:0.96527.pth'
# print(WEIGHT_PATH)

#you could set it for your device
INIT_TRAINER = {
  'num_classes':NUM_CLASSES,
  'n_epoch':160,
  'batch_size':110,
  'num_workers':16,
  'device':'cuda',
  'pre_trained':PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'use_fp16':False,
  'transformer_depth': TRANSFORMER_DEPTH
 }
#---------------------------------

SETUP_TRAINER = {
  'output_dir':'./new_ckpt/{}/{}/{}'.format(PHASE,VERSION,strftime("%Y-%m-%dT%H:%M:%S")),
  'log_dir':'./new_log/{}/{}/{}'.format(PHASE,VERSION,strftime("%Y-%m-%dT%H:%M:%S")),
  'phase':PHASE,
  'activation': True
  }
