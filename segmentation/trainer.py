import argparse
import copy
import math
import os
import pickle
import shutil
import warnings
from importlib import import_module

import numpy as np
import torch
from matplotlib import pyplot as plt
from picai_eval import evaluate, Metrics
from picai_eval.eval import evaluate_case
from report_guided_annotation import extract_lesion_candidates
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sam_fact_tt_image_encoder import Fact_tt_Sam

from segment_anything import sam_model_registry
from model_single import ModelEmb, SegDecoderCNN
from peft import get_peft_model, LoraConfig, TaskType

from data_loader import (DataGenerator, Normalize, RandomFlip2D,
                         RandomRotate2D, To_Tensor, MultiLevelDataGenerator, MultiLevel3DDataGenerator,
                         MultiLevelDataGeneratorSeg)
from loss import Deep_Supervised_Loss
from model import itunet_2d
from MedSAMAuto import MedSAMAUTO, MedSAMAUTOMULTI, MedSAMAUTOCNN
from config import PATH_DIR
from segment_anything.modeling import MaskDecoder, TwoWayTransformer
from segmentation.lora_image_encoder import LoRA_Sam
from segmentation.segment_anything_from_MASAM.build_sam import sam_model_registry_MASAM
from utils import Normalize_2d
from eval_utils import search_ckpt_path
from utils import dfs_remove_weight, poly_lr, compute_results_detect, plot_segmentation2D, ModelName
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    LoadImageD,
    ScaleIntensityD,
    ResizeD,
    RandFlipD,
    RandRotateD,
    RandZoomD,
    ToTensorD,
)
from TransUNet import VisionTransformer, CONFIGS

warnings.filterwarnings('ignore')

def issue_warning():
    print("Warning: The log path already exists, continue will overwrite the log and ckpt")
    input("Press Enter to continue...")

def compute_results(logits, target, results, val_mode='2d'):
    preds = []
    logits = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
    for slices in logits:
        if val_mode == '2d':
            preds.append(extract_lesion_candidates(np.expand_dims(slices, axis=-1), threshold=0.5)[0])
        else:
            preds.append(extract_lesion_candidates(slices, min_voxels_detection=int((slices.shape[1]/256)**2*(slices.shape[0]/24)*10), threshold=0.5)[0])
    for y_det, y_true in zip(preds,
                             [target[:, i, :, :] for i in range(target.shape[1])] if val_mode=='2d' else [target[i, :, :, :] for i in range(target.shape[0])]):
        y_list, *_ = evaluate_case(
            y_det=y_det,
            y_true=y_true,
        )

        # aggregate all validation evaluations
        results.append(y_list)
    return results

class SemanticSeg(object):
    def __init__(self,lr=1e-3,n_epoch=1,channels=3,num_classes=2, input_shape=(384,384),batch_size=6,num_workers=0,
                  device=None,pre_trained=False,ckpt_point=True,weight_path=None,weight_decay=0.0001,
                  use_fp16=False,transformer_depth = 18, model_name = 'MedSAMAuto', image_size=(384,384),load_ckpt=None, finetune=False):
        super(SemanticSeg,self).__init__()
        self.lr = lr
        self.finetune = finetune
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.image_size = image_size

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.use_fp16 = use_fp16

        self.pre_trained = pre_trained
        self.ckpt_point = ckpt_point
        self.weight_path = weight_path
        self.weight_decay = weight_decay

        self.start_epoch = 0
        self.global_step = 0
        self.metrics_threshold = 0.

        self.transformer_depth = transformer_depth
        self.model_name = model_name

        # os.environ['CUDA_VISIBLE_DEVICES'] = self.device
        # using UNet need to disable all sigmoid activation function


        # # Create LoRA configuration
        # lora_config = LoraConfig(
        #     task_type=TaskType.SEGMENTATION,  # Specify the task type
        #     inference_mode=False,
        #     r=4,  # Rank
        #     lora_alpha=32,  # Scaling factor
        #     lora_dropout=0.1,  # Dropout rate
        # )
        #
        # # Apply LoRA to the image encoder
        # sam_model.image_encoder = get_peft_model(sam_model.image_encoder, lora_config)

        if model_name == ModelName.medsam:
            sam_model = sam_model_registry['vit_b'](checkpoint='/data/nvme1/meng/cvpr25_results/medsam_vit_b.pth')
            dense_model = ModelEmb()
            multi_mask_decoder = MaskDecoder(
                num_multimask_outputs=4,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=256,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
            self.net = DataParallel(MedSAMAUTOMULTI(
                    image_encoder=sam_model.image_encoder,
                    mask_decoder=multi_mask_decoder,
                    prompt_encoder=sam_model.prompt_encoder,
                    dense_encoder=dense_model,
                    image_size=512
                ), device_ids=[0, 1, 2, 3])
            if finetune:
                state_dict = torch.load(load_ckpt, map_location=device)['state_dict']
                self.net.load_state_dict(state_dict)
        elif model_name == ModelName.samcnn:
            sam_model = sam_model_registry['vit_b'](checkpoint='/data/nvme1/meng/cvpr25_results/sam_vit_b_01ec64.pth')
            lora_sam_model = LoRA_Sam(sam_model, 4)
            multi_mask_decoder = SegDecoderCNN(num_classes=4, num_depth=4)
            self.net = DataParallel(MedSAMAUTOCNN(
                    image_encoder=lora_sam_model.sam.image_encoder,
                    mask_decoder=multi_mask_decoder,
                ), device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        elif model_name == ModelName.swin_unetr:
            self.net = DataParallel(
                SwinUNETR(img_size=(32, image_size[0], image_size[1]),
                          in_channels=3,
                          out_channels=4,
                          feature_size=48,
                          use_checkpoint=True,
                          )
            )
        elif model_name == ModelName.unet:
            self.net = DataParallel(torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                      in_channels=3, out_channels=4 , init_features=32, pretrained=False))
        elif model_name == ModelName.itunet:
            self.net = DataParallel(itunet_2d(n_channels=self.channels, n_classes=4,
                                 image_size=self.image_size, transformer_depth=self.transformer_depth))
        elif model_name == ModelName.masam:
            sam, img_embedding_size = sam_model_registry_MASAM['vit_b'](image_size=self.img_size,
                                                                        num_classes=4,
                                                                        checkpoint='/data/nvme1/meng/cvpr25_results/sam_vit_b_01ec64.pth', pixel_mean=[0., 0., 0.],
                                                                        pixel_std=[1., 1., 1.])

            self.net = DataParallel(Fact_tt_Sam(sam, 32, s=1.0), device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        elif model_name == ModelName.transunet:
            self.net = DataParallel(VisionTransformer(con))
        # mask_decoder_model = SegDecoderCNN(num_classes=4, num_depth=4)
        #
        # self.net = DataParallel(MedSAMAUTOCNN(
        #     image_encoder=sam_model.image_encoder,
        #     mask_decoder=mask_decoder_model,
        #     prompt_encoder=sam_model.prompt_encoder,
        #     dense_encoder=None,
        #     image_size=512
        # ).to(device))



    def plot_eval(self, number_plots, val_path, ckpt_path, log_dir, device):
        net = copy.deepcopy(self.net)
        files = os.listdir(ckpt_path)
        sorted_files = sorted(files)
        ckpt_file = os.path.join(ckpt_path, sorted_files[-1])
        state_dict = torch.load(ckpt_file, map_location=device)['state_dict']
        net.load_state_dict(state_dict)
        net.eval()
        net.to(self.device)
        plot_path = os.path.join(log_dir, 'plots')
        os.makedirs(plot_path, exist_ok=True)
        val_transformer = transforms.Compose([
            Normalize(),
            # tio.CropOrPad(target_shape=(32, 128, 128)),
            To_Tensor(num_class=self.num_classes, input_channel=self.channels)
        ])

        val_dataset = DataGenerator(val_path, num_class=self.num_classes, transform=val_transformer)

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        count = 0
        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.to(self.device)
                target = target.to(self.device)
                with autocast(self.use_fp16):
                    output = net(data)
                    if isinstance(output, tuple):
                        output = output[0]
                plot_segmentation2D(data.squeeze(0).permute(1, 2, 0).detach().cpu(), output.squeeze(0)[0].detach().cpu(), target.squeeze(0)[0].detach().cpu(), plot_path, count)
                count += 1

    def trainer(self, train_path,val_path,val_ap, cur_fold,output_dir=None,log_dir=None,phase = 'seg', activation=True, val_mode='2d', resume=None):

        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold"+str(cur_fold))
        log_dir = os.path.join(log_dir, "fold"+str(cur_fold))

        if os.path.exists(log_dir):
            issue_warning()
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)
            
        self.step_pre_epoch = len(train_path) // self.batch_size
        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * math.ceil(len(train_path)/self.batch_size)

        net = self.net
        lr = self.lr
        loss = Deep_Supervised_Loss(mode='Focal', activation=activation, model_name=self.model_name)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        lesion_pid = pickle.load(open(os.path.join(PATH_DIR, '../lesion_pid.p'), 'rb'))
        zone_pid = pickle.load(open(os.path.join(PATH_DIR, '../../zone_segdata_all/zone_pid.p'), 'rb'))
        gland_pid = pickle.load(open(os.path.join(PATH_DIR, '../../gland_segdata/gland_pid.p'), 'rb'))
        if self.model_name == ModelName.swin_unetr:
            train_transformer = transforms.Compose([
                ScaleIntensityD(keys=["ct"]),
                ResizeD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"], spatial_size=(32, self.image_size[0], self.image_size[1]), mode=("trilinear", "nearest", "nearest", "nearest", "nearest")),  # Resize the ct to 128x128x64
                RandFlipD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"], spatial_axis=0, prob=0.5),  # Random flip along x-axis
                RandRotateD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"], range_x=0.4, range_y=0.4, range_z=0.4, prob=0.5, mode=("trilinear", "nearest", "nearest", "nearest", "nearest")),  # Random rotate
                ToTensorD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"])
            ])
            train_dataset = MultiLevel3DDataGenerator(train_path, 'train', self.image_size, num_class=self.num_classes,
                                                    transform=train_transformer, zone_pid=zone_pid, gland_pid=gland_pid,
                                                    lesion_pid=lesion_pid)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            val_transformer = transforms.Compose([
                ScaleIntensityD(keys=["ct"]),
                ResizeD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"],
                        spatial_size=(32, self.image_size[0], self.image_size[1]),
                        mode=("trilinear", "nearest", "nearest", "nearest", "nearest")),  # Resize the ct to 128x128x64
                ToTensorD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"])
            ])
            val_dataset = MultiLevel3DDataGenerator(val_path, 'val', self.image_size, num_class=self.num_classes,
                                                  transform=val_transformer, zone_pid=zone_pid, gland_pid=gland_pid,
                                                  lesion_pid=lesion_pid)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )

        else:
            train_transformer = transforms.Compose([
                Normalize(),  # 1
                # tio.CropOrPad(target_shape=(32, 128, 128)),
                RandomRotate2D(),  # 6
                RandomFlip2D(mode='hv'),  # 7
                To_Tensor(num_class=self.num_classes, input_channel=self.channels)  # 10
            ])
            train_dataset = MultiLevelDataGenerator(train_path, 'train', self.image_size, num_class=self.num_classes,transform=train_transformer, zone_pid=zone_pid, gland_pid=gland_pid, lesion_pid=lesion_pid)
            # train_dataset = MultiLevelDataGeneratorSeg(train_path, 'train', self.image_size, num_class=self.num_classes,
            #                                         transform=train_transformer,lesion_pid=lesion_pid)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=1
            )
            if val_mode == '2d':
                val_transformer = transforms.Compose([
                    Normalize(),
                    # tio.Resize(target_shape=(24, 128, 128)),
                    # tio.CropOrPad(target_shape=(32, 128, 128)),
                    To_Tensor(num_class=self.num_classes, input_channel=self.channels)
                ])
                val_dataset = MultiLevelDataGenerator(val_path, 'val', self.image_size, num_class=self.num_classes,transform=val_transformer, zone_pid=zone_pid, gland_pid=gland_pid, lesion_pid=lesion_pid)
                # val_dataset = MultiLevelDataGeneratorSeg(val_path, 'val', self.image_size, num_class=self.num_classes,
                #                                       transform=val_transformer, lesion_pid=lesion_pid)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            else:
                if self.model_name == ModelName.swin_unetr:
                    val_transformer = transforms.Compose([
                        ScaleIntensityD(keys=["ct"]),
                        ResizeD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"],
                                spatial_size=(32 if self.model_name == ModelName.swin_unetr else 24, self.image_size[0], self.image_size[1]),
                                mode=("trilinear", "nearest", "nearest", "nearest", "nearest")),
                        # Resize the ct to 128x128x64
                        ToTensorD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"])
                    ])
                else:
                    val_transformer = transforms.Compose(
                        [Normalize_2d(), To_Tensor()])
                val_dataset = MultiLevel3DDataGenerator(val_ap, 'random', self.image_size, num_class=self.num_classes,
                                                        transform=val_transformer, zone_pid=zone_pid,
                                                        gland_pid=gland_pid,
                                                        lesion_pid=lesion_pid)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    prefetch_factor=1
                )
        if resume:
            ckpt_file = search_ckpt_path(resume)
            start_epoch = int(ckpt_file.split('epoch:')[1].split('-')[0])
            state_dict = torch.load(os.path.join(resume, ckpt_file), map_location=self.device)['state_dict']
            net.load_state_dict(state_dict)
            resume_gland_dice = float(ckpt_file.split('gland_val_dice:')[1].split('-')[0])
            resume_zone_dice = float(ckpt_file.split('zone_val_dice:')[1].split('-')[0])
            resume_lesion_ap = float(ckpt_file.split('lesion_val_ap:')[1].split('-lesion')[0])
            resume_lesion_auc = float(ckpt_file.split('lesion_val_auc:')[1].split('.pth')[0])
            resume_score = 0.2 * resume_gland_dice + 0.2 * resume_zone_dice + 0.3 * resume_lesion_ap + 0.3 * resume_lesion_auc
            self.metrics_threshold = resume_score
        # copy to gpu
        net = net.to(self.device)
        loss = loss.to(self.device)

        # optimizer setting
        optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=self.weight_decay)

        scaler = GradScaler()

        early_stopping = EarlyStopping(patience=20,verbose=True,monitor='val_score',op_type='max')

        epoch = self.start_epoch
        optimizer.param_groups[0]['lr'] = poly_lr(epoch, self.n_epoch, initial_lr = lr)

        while epoch < self.n_epoch:
            if resume:
                if epoch <= start_epoch:
                    epoch += 1
                    continue
            train_loss,gland_train_dice,zone_train_dice,lesion_train_dice = self._train_on_epoch(output_dir, epoch,net,loss,optimizer,train_loader,val_loader, scaler, activation=activation, finetune=self.finetune, val_mode=val_mode)
            self.writer.add_scalar(
                'data/train_loss', train_loss, epoch
            )
            self.writer.add_scalar('data/gland_train_dice', gland_train_dice, epoch)
            self.writer.add_scalar('data/zone_train_dice', zone_train_dice, epoch)
            self.writer.add_scalar('data/lesion_train_dice', lesion_train_dice, epoch)

            self.writer.add_scalar(
                'data/train_loss_epochs', train_loss, epoch
            )

            if phase == 'seg':
                val_loss,gland_val_dice,zone_val_dice,lesion_val_dice, lesion_ap, lesion_auc, lesion_positive_dice = self._val_on_epoch(epoch,net,loss,val_loader, activation=activation, val_mode=val_mode)
                self.writer.add_scalar(
                    'data/eval_loss_epochs', val_loss, epoch
                )
                self.writer.add_scalar(
                    'data/eval_gland_dice_epochs', gland_val_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_zone_dice_epochs', zone_val_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_lesion_dice_epochs', lesion_val_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_lesion_positive_dice_epochs', lesion_positive_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_lesion_ap_epochs', lesion_ap, epoch
                )
                self.writer.add_scalar(
                    'data/eval_lesion_auc_epochs', lesion_auc, epoch
                )
                score = (gland_val_dice * 0.2 + zone_val_dice * 0.2 + lesion_ap * 0.3 + lesion_auc * 0.3)
            else:
                auc, ap = self.val(epoch,val_ap,net,mode = 'train')
                score = (ap + auc) / 2

            optimizer.param_groups[0]['lr'] = poly_lr(epoch, self.n_epoch, initial_lr = lr)

            torch.cuda.empty_cache()
                
            self.writer.add_scalar(
              'data/lr',optimizer.param_groups[0]['lr'],epoch
            )

            early_stopping(score)

            #save
            if score > self.metrics_threshold:
                self.metrics_threshold = score

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                  'epoch':epoch,
                  'save_dir':output_dir,
                  'state_dict':state_dict,
                }
                
                if phase == 'seg':
                    file_name = 'epoch:{}-gland_val_dice:{:.5f}-zone_val_dice:{:.5f}-lesion_val_dice:{:.5f}-lesion_val_ap:{:.5f}-lesion_val_auc:{:.5f}.pth'.format(
                    epoch,gland_val_dice,zone_val_dice,lesion_val_dice, lesion_ap, lesion_auc)
                else:
                    file_name = 'epoch:{}-train_loss:{:.5f}-train_dice:{:.5f}-train_run_dice:{:.5f}-val_auroc:{:.5f}-val_ap:{:.5f}-val_score:{:.5f}.pth'.format(
                    epoch,train_loss,train_dice,train_run_dice,ap.auroc,ap.AP,ap.score)
                save_path = os.path.join(output_dir,file_name)
                print("Save as: %s" % file_name)

                torch.save(saver,save_path)

            epoch += 1
            
            # early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.plot_eval(100, val_path, output_dir, log_dir, device='cuda')
        self.writer.close()
        dfs_remove_weight(output_dir,retain=3)

    def _train_on_epoch(self,output_dir, epoch,net,criterion,optimizer,train_loader, val_loader, scaler, activation=True, plot=False, finetune=False, val_mode='2d'):
        net.train()

        train_loss = AverageMeter()
        gland_train_dice = AverageMeter()
        zone_train_dice = AverageMeter()
        lesion_train_dice = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes),ignore_label=-1)

        for step, (sample, pid, slice) in enumerate(tqdm(train_loader)):
            if finetune:
                net.train()
            lesion_targets = []
            zone_targets = []
            gland_targets = []
            for name, value in sample.items():
                order_correct = True
                if name == 'zone_seg_0':
                    var_a_assigned = True
                elif name == 'zone_seg_1':
                    var_b_assigned = True
                    if var_a_assigned:
                        order_correct = True
                    else: order_correct = False
                if name == 'ct':
                    data = value
                elif 'lesion' in name:
                    lesion_targets.append(value)
                elif 'gland' in name:
                    gland_targets.append(value)
                elif 'zone' in name:
                    zone_targets.append(value)
            lesion_target = torch.stack(lesion_targets)
            zone_target = torch.stack(zone_targets)
            gland_target = torch.stack(gland_targets)
            if self.model_name == ModelName.swin_unetr:
                multi_level_targets = torch.cat([gland_target, zone_target, lesion_target]).squeeze(2).permute(1, 0, 2, 3, 4)
            else:
                multi_level_targets = torch.cat([gland_target, zone_target, lesion_target]).permute(1, 0, 2, 3)
            # data = sample['ct']
            # multi_level_targets = sample['seg']
            data = data.to(self.device)
            multi_level_targets = multi_level_targets.to(self.device)

            with autocast(self.use_fp16):
                output = net(data)
                if isinstance(output,tuple):
                    output = output[0]
                loss = criterion(output, multi_level_targets)
                if isinstance(output,list):
                    output = output[0]
                # loss = criterion(output,multi_level_targets[:, -1].unsqueeze(1))

            if plot:
                pred = torch.sigmoid(output)
                for id, img in enumerate(data):
                    plot_segmentation2D(img.permute(1, 2, 0)[..., 0].unsqueeze(-1).expand(-1,-1,3).detach().cpu().numpy(),
                                            (pred[id, -1, ...] > 0.5).detach().cpu(),
                                            gland_target[0, id, ...].detach().cpu().numpy(), f'./train_plot',
                                            f'{id}', image_dice=None)

            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            output = output.float()
            loss = loss.float()

            # dice = compute_dice(output.detach(),multi_level_targets[:, -1].unsqueeze(1), activation=activation)
            dice = compute_dice(output.detach(), multi_level_targets, activation=activation)
            average_dice = torch.mean(dice, dim=1)
            train_loss.update(loss.item(),data.size(0))
            gland_train_dice.update(average_dice[0],data.size(0))
            zone_train_dice.update(sum(average_dice[1:3]) / 2, data.size(0))
            lesion_train_dice.update(average_dice[3], data.size(0))
            # lesion_train_dice.update(average_dice[0], data.size(0))

            # output = (torch.sigmoid(output) > 0.5).int().detach().cpu().numpy()  #N*H*W
            # multi_level_targets = multi_level_targets.detach().cpu().numpy()
            # run_dice.update_matrix(multi_level_targets,output)

            torch.cuda.empty_cache()

            if self.global_step%1==0:
                # rundice, dice_list = run_dice.compute_dice()
                # print("Category Dice: ", dice_list)
                print('epoch:{}/{},step:{},train_loss:{:.5f},gland_train_dice:{:.5f},zone_train_dice:{:.5f},lesion_train_dice:{:.5f},lr:{}'.format(epoch,self.n_epoch, step, loss.item(), gland_train_dice.avg, zone_train_dice.avg, lesion_train_dice.avg, optimizer.param_groups[0]['lr']))
                # run_dice.init_op()
                # self.writer.add_scalar(
                #   'data/train_loss',loss.item(),self.global_step
                # )
                # self.writer.add_scalar('data/gland_train_dice', gland_train_dice.avg,self.global_step)
                # self.writer.add_scalar('data/zone_train_dice', zone_train_dice.avg, self.global_step)
                # self.writer.add_scalar('data/lesion_train_dice', lesion_train_dice.avg, self.global_step)

            self.global_step += 1
            if finetune:
                if step % 50 == 0 and step != 0:
                    val_loss,gland_val_dice,zone_val_dice,lesion_val_dice, lesion_ap, lesion_auc, lesion_positive_dice = self._val_on_epoch(epoch,net,criterion,val_loader, activation=activation, val_mode=val_mode)
                    print(f'eval epoch,step:{epoch},{step}-gland_dice:{gland_val_dice}-zone_dice:{zone_val_dice}-lesion_ap:{lesion_ap}-lesion_auc:{lesion_auc}')
                    self.writer.add_scalar(
                        'data/eval_loss_steps', val_loss, step
                    )
                    self.writer.add_scalar(
                        'data/eval_gland_dice_steps', gland_val_dice, step
                    )
                    self.writer.add_scalar(
                        'data/eval_zone_dice_steps', zone_val_dice, step
                    )
                    self.writer.add_scalar(
                        'data/eval_lesion_positive_dice_steps', lesion_positive_dice, step
                    )
                    self.writer.add_scalar(
                        'data/eval_ap_steps', lesion_ap, step
                    )
                    self.writer.add_scalar(
                        'data/eval_auc_steps', lesion_auc, step
                    )
                    score = (gland_val_dice * 0.2 + zone_val_dice * 0.2 + lesion_ap * 0.3 + lesion_auc * 0.3)
                    if score > self.metrics_threshold:
                        self.metrics_threshold = score

                        if len(self.device.split(',')) > 1:
                            state_dict = net.module.state_dict()
                        else:
                            state_dict = net.state_dict()

                        saver = {
                            'epoch': epoch,
                            'step': step,
                            'save_dir': output_dir,
                            'state_dict': state_dict,
                        }

                        file_name = 'epoch,step:{},{}-gland_val_dice:{:.5f}-zone_val_dice:{:.5f}-lesion_val_dice:{:.5f}-lesion_val_ap:{:.5f}-lesion_val_auc:{:.5f}.pth'.format(
                                epoch, step, gland_val_dice, zone_val_dice, lesion_val_dice, lesion_ap, lesion_auc)
                        save_path = os.path.join(output_dir, file_name)
                        print("Save as: %s" % file_name)

                        torch.save(saver, save_path)


        return train_loss.avg,gland_train_dice.avg,zone_train_dice.avg,lesion_train_dice.avg


    def _val_on_epoch(self,epoch,net,criterion,val_loader,val_transformer=None, activation=True, plot=False, val_mode='2d'):
        net.eval()

        val_loss = AverageMeter()
        gland_val_dice = AverageMeter()
        zone_val_dice = AverageMeter()
        lesion_val_dice = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes),ignore_label=-1)
        lesion_results = []
        gland_results = []
        cz_results = []
        pz_results = []
        positive_dice = []
        with torch.no_grad():
            for step,(sample, pid, slice) in enumerate(tqdm(val_loader)):
                lesion_targets = []
                zone_targets = []
                gland_targets = []
                for name, value in sample.items():
                    if name == 'ct':
                        data = value
                    elif 'lesion' in name:
                        lesion_targets.append(value)
                    elif 'gland' in name:
                        gland_targets.append(value)
                    elif 'zone' in name:
                        zone_targets.append(value)
                lesion_target = torch.stack(lesion_targets) if val_mode == '2d' else torch.stack(lesion_targets).squeeze(1).squeeze(1)
                zone_target = torch.stack(zone_targets) if val_mode == '2d' else torch.stack(zone_targets).squeeze(1).squeeze(1)
                gland_target = torch.stack(gland_targets) if val_mode == '2d' else torch.stack(gland_targets).squeeze(1).squeeze(1)
                if self.model_name == ModelName.swin_unetr:
                    multi_level_targets = torch.cat([gland_target, zone_target, lesion_target]).squeeze(2).permute(1, 0, 2, 3, 4)
                else:
                    multi_level_targets = torch.cat([gland_target, zone_target, lesion_target]).permute(1, 0, 2, 3)

                data = data.to(self.device) if val_mode == '2d' else data.squeeze(0).permute(1,0,2,3)
                multi_level_targets = multi_level_targets.to(self.device)
                with autocast(self.use_fp16):
                    output = net(data)
                    if isinstance(output,tuple):
                        output = output[0]
                # loss = criterion(output,multi_level_targets[:, -1].unsqueeze(1).float())
                loss = criterion(output, multi_level_targets.float())
                if isinstance(output,list):
                    output = output[0]

                output = output.float()
                loss = loss.float()

                # dice = compute_dice(output.detach(),multi_level_targets[:, -1].unsqueeze(1),activation=activation)
                dice = compute_dice(output.detach(), multi_level_targets, activation=activation)
                average_dice = torch.mean(dice, dim=1)
                for id, target in enumerate(lesion_target[0]):
                    if target.max() > 0:
                        positive_dice.append(dice[-1, id])
                val_loss.update(loss.item(),data.size(0))
                gland_val_dice.update(average_dice[0], data.size(0))
                zone_val_dice.update(sum(average_dice[1:3]) / 2, data.size(0))
                lesion_val_dice.update(average_dice[3], data.size(0))
                # lesion_val_dice.update(average_dice[0], data.size(0))

                if activation:
                    logits = torch.sigmoid(output)
                else:
                    logits = output
                if plot:
                    for id, img in enumerate(data):
                        plot_segmentation2D(img.permute(1, 2, 0).detach().cpu().numpy(),
                                            (logits[id, -1, ...] > 0.5).detach().cpu(),
                                            lesion_target[0, id, ...].detach().cpu().numpy(), f'./test',
                                            f'{id}', image_dice=None)
                output = (logits > 0.5).int().detach().cpu().numpy()  # N*H*W
                # target = target.detach().cpu().numpy()
                # run_dice.update_matrix(target,output)
                if self.model_name == ModelName.swin_unetr:
                    logits = logits[:, -1, :, :, :]
                    target = lesion_target.squeeze(0).squeeze(1).detach().cpu().numpy()
                    preds = []
                    logits = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
                    for sample in logits:
                        preds.append(extract_lesion_candidates(sample, min_voxels_detection=10, threshold=0.5)[0])
                    for y_det, y_true in zip(preds,
                                             [target[i, :, :, :] for i in range(target.shape[0])]):
                        y_list, *_ = evaluate_case(
                            y_det=y_det,
                            y_true=y_true,
                        )

                        # aggregate all validation evaluations
                        lesion_results.append(y_list)
                else:
                    lesion_results = compute_results(logits[:, -1, :, :], lesion_target.detach().cpu().numpy(), lesion_results, val_mode=val_mode) if val_mode == '2d' else compute_results(logits[:, -1, :, :].unsqueeze(0), lesion_target.detach().cpu().numpy(), lesion_results, val_mode=val_mode)

                torch.cuda.empty_cache()

                if step % 1 == 0:
                    # rundice, dice_list = run_dice.compute_dice()
                    # print("Category Dice: ", dice_list)
                    print('Eval epoch:{}/{},step:{},val_loss:{:.5f},gland_val_dice:{:.5f},zone_val_dice:{:.5f},lesion_val_dice:{:.5f}'.format(epoch,self.n_epoch, step, loss.item(), gland_val_dice.avg, zone_val_dice.avg, lesion_val_dice.avg))
                    # run_dice.init_op()
                    # self.writer.add_scalar(
                    #     'data/eval_loss', loss.item(), self.global_step
                    # )
                    # self.writer.add_scalar('data/eval_dice', rundice, self.global_step)
        lesion_results = {idx: result for idx, result in enumerate(lesion_results)}
        lesion_valid_metrics = Metrics(lesion_results)
        lesion_auc = lesion_valid_metrics.auroc
        lesion_ap = lesion_valid_metrics.AP

        return val_loss.avg,gland_val_dice.avg,zone_val_dice.avg,lesion_val_dice.avg,lesion_ap, lesion_auc, torch.mean(torch.stack(positive_dice))


    def val(self,epoch, val_path,net = None,val_transformer=None,mode = 'val', image_size=(1024,1024)):
        if net is None:
            net = self.net
            net = net.to(self.device)
        net.eval()

        val_transformer = transforms.Compose([
            ScaleIntensityD(keys=["ct"]),
            ResizeD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"],
                    spatial_size=(32, self.image_size[0], self.image_size[1]),
                    mode=("trilinear", "nearest", "nearest", "nearest", "nearest")),  # Resize the ct to 128x128x64
            ToTensorD(keys=["ct", "lesion_seg_0", "zone_seg_0", "zone_seg_1", "gland_seg_0"])
        ])
        val_dataset = MultiLevel3DDataGenerator(val_path, 'val', self.image_size, num_class=self.num_classes,
                                                transform=val_transformer, zone_pid=zone_pid, gland_pid=gland_pid,
                                                lesion_pid=lesion_pid)

        val_loader = DataLoader(
          val_dataset,
          batch_size=1,
          shuffle=False,
          num_workers=self.num_workers,
          pin_memory=True
        )

        y_pred = []
        y_true = []
        lesion_results = []

        with torch.no_grad():
            for step,sample in enumerate(val_loader):
                data = sample['ct']
                target = sample['seg']
                
                data = data.squeeze().transpose(1,0) 
                data = data.to(self.device)
                target = target.to(self.device)
                with autocast(self.use_fp16):
                    output = net(data)
                    if isinstance(output,tuple):
                        output = output[0]

                output = output.float()
                output = torch.sigmoid(output)  #N*H*W
                output = output.detach().cpu().numpy()

                lesion_results = compute_results_detect(output[:, -1, :, :], target.detach().cpu().numpy()[0],
                                                 lesion_results)

                print(
                    'Eval epoch:{}/{},step:{}'.format(
                        epoch, self.n_epoch, step))


        lesion_results = {idx: result for idx, result in enumerate(lesion_results)}
        lesion_valid_metrics = Metrics(lesion_results)
        lesion_auc = lesion_valid_metrics.auroc
        lesion_ap = lesion_valid_metrics.AP
        print('Eval epoch:{}/{}, ap:{:.5f}, auc:{:.5f}', epoch, self.n_epoch, lesion_ap, lesion_auc)
        return lesion_auc, lesion_ap


    def _get_pre_trained(self,weight_path, ckpt_point=True):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        if ckpt_point:
            self.start_epoch = checkpoint['epoch'] + 1


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True, delta=0, monitor='val_loss',op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
           print(self.monitor, f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        self.val_score_min = val_score

class AverageMeter(object):
    '''
  Computes and stores the average and current value
  '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def binary_dice(predict, target, smooth=1e-5):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1) #N，H*W
    target = target.contiguous().view(target.shape[0], -1) #N，H*W

    inter = torch.sum(torch.mul(predict, target), dim=1) #N
    union = torch.sum(predict + target, dim=1) #N

    dice = (2*inter + smooth) / (union + smooth ) #N

    return dice

def compute_dice(predict,target,ignore_index=0, activation=True):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'
    if activation:
        predict = (F.sigmoid(predict) > 0.5).int()
    else:
        predict = (predict > 0.5).int()
    dice_list = []
    for i in range(predict.shape[1]):
        dice = binary_dice((predict[:,i]==1).float(), (target[:, i]==1).float())
        dice_list.append(dice)
    return torch.stack(dice_list)