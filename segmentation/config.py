import glob
import os
import json

from utils import get_weight_path, ModelName

TRANSFORMER_DEPTH = 24
MODEL_NAME = ModelName.unet
batch_size = 70
# dataset = '158'
dataset = 'picai'
loss_mode = 'Focal'
image_size = 256
VERSION = f'{MODEL_NAME.value}_{loss_mode}_0.8_Unified_equal_rate_batch_{batch_size}_tumorsplit_0.001_image_{str(image_size)}_dataset_{dataset}'

PHASE = 'seg'   # 'seg' or 'detect'
NUM_CLASSES = 2 if 'seg' in PHASE else 3
# resume = '/usa/yxpengcs/PycharmProjects/baselines/segmentation/new_ckpt/seg/UNet_Focal_0.97_Unified_equal_rate_batch_70_tumorsplit_0.001_image_256_dataset_picai_seed_1_valmode_3d_lr_0.0001_weight_decay_0.001/fold1'
resume = None
DEVICE = 'cuda:0'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = True if resume else False
# True if use resume model
CKPT_POINT = False
finetune_stepwise = False
CHECKPOINT_PATH = '/data/nvme1/meng/checkpoints'

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))

#--------------------------------- mode and data path setting
if dataset == 'picai':
  PATH_DIR = '/data/nvme1/meng/picai/lesion_segdata_combined/data_3d' if MODEL_NAME == ModelName.swin_unetr or MODEL_NAME == ModelName.masam else '/data/nvme1/meng/picai/lesion_segdata_combined/data_2d'
  PATH_AP = '/data/nvme1/meng/picai/lesion_segdata_combined/data_3d'
elif dataset == '158':
  PATH_DIR = '/data/nvme1/meng/picai/lesion_segdata_158/data_3d' if MODEL_NAME == ModelName.swin_unetr or MODEL_NAME == ModelName.masam else '/data/nvme1/meng/picai/lesion_segdata_158/data_2d'
  PATH_AP = '/data/nvme1/meng/picai/lesion_segdata_158/data_3d'
PATH_LIST = glob.glob(os.path.join(PATH_DIR,'*.hdf5'))
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
  'device':'cuda:5',
  'pre_trained':PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'use_fp16':False,
  'transformer_depth': TRANSFORMER_DEPTH,
  'model_name': MODEL_NAME,
  # 'load_ckpt': '/home/yxpengcs/PycharmProjects/ITUNet-for-PICAI-2022-Challenge/segmentation/new_ckpt/seg/MedSAMAuto_Focal_Unified_equal_rate_high_weighted_loss_combined_label_lr_0.0001_weight_decay_0.001/fold1/epoch:11-gland_val_dice:0.83997-zone_val_dice:0.84296-lesion_val_dice:0.73409-lesion_val_ap:0.49188-lesion_val_auc:0.90977.pth',
  'load_ckpt': None,
  'image_size': (image_size, image_size),
  'finetune_stepwise': finetune_stepwise
}
#---------------------------------

SETUP_TRAINER = {
  'output_dir':'./new_ckpt/{}/{}'.format(PHASE,VERSION),
  'log_dir':'./new_log/{}/{}'.format(PHASE,VERSION),
  'phase':PHASE,
  'activation': False if MODEL_NAME == ModelName.unet else True,
  'val_mode': '2d',
  'resume': resume,
  'loss_mode': loss_mode
  }