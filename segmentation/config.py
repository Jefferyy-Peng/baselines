import glob
import os
import json

from utils import get_weight_path, ModelName

TRANSFORMER_DEPTH = 24
MODEL_NAME = ModelName.medsam
batch_size = 32
VERSION = f'{MODEL_NAME.value}_Focal_Unified_equal_rate_batch_{batch_size}_tumorsplit_0.001_0.97_weighted_loss_image_1024_combined_label_test'

PHASE = 'seg'   # 'seg' or 'detect'
NUM_CLASSES = 2 if 'seg' in PHASE else 3
# resume = '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/segmentation/new_ckpt/seg/UNet_Focal_Unified_equal_rate_batch_70_tumorsplit_0.001_0.97_weighted_loss_image_256_combined_label__valmode_3d_lr_0.0001_weight_decay_0.001/fold1'
resume = None
DEVICE = 'cuda:0'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = True if resume else False
# True if use resume model
CKPT_POINT = False
finetune = False

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))

#--------------------------------- mode and data path setting
PATH_DIR = './dataset/combined/data_3d' if MODEL_NAME == ModelName.swin_unetr else './dataset/combined/data_2d'
PATH_LIST = glob.glob(os.path.join(PATH_DIR,'*.hdf5'))
PATH_AP = './dataset/combined/data_3d'
AP_LIST = glob.glob(os.path.join(PATH_AP,'*.hdf5'))
#--------------------------------- 

CKPT_PATH = './new_ckpt/{}/{}/fold{}'.format(PHASE,VERSION,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

#you could set it for your device
INIT_TRAINER = {
  'num_classes':NUM_CLASSES, 
  'n_epoch':160,
  'batch_size':batch_size,
  'num_workers':12,
  'device':'cuda',
  'pre_trained':PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'use_fp16':False,
  'transformer_depth': TRANSFORMER_DEPTH,
  'model_name': MODEL_NAME,
  # 'load_ckpt': '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/segmentation/new_ckpt/seg/MedSAMAuto_Focal_Unified_equal_rate_high_weighted_loss_combined_label_lr_0.0001_weight_decay_0.001/fold1/epoch:11-gland_val_dice:0.83997-zone_val_dice:0.84296-lesion_val_dice:0.73409-lesion_val_ap:0.49188-lesion_val_auc:0.90977.pth',
  'load_ckpt': None,
  'image_size': (1024, 1024),
  'finetune': finetune
}
#---------------------------------

SETUP_TRAINER = {
  'output_dir':'./new_ckpt/{}/{}'.format(PHASE,VERSION),
  'log_dir':'./new_log/{}/{}'.format(PHASE,VERSION),
  'phase':PHASE,
  'activation': True if MODEL_NAME == ModelName.medsam or MODEL_NAME == ModelName.itunet or MODEL_NAME == ModelName.swin_unetr else False,
  'val_mode': '2d',
  'resume': resume
  }