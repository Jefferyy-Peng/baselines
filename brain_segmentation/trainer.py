import copy
import math
import os
import pickle
import shutil
import warnings

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
from collections import OrderedDict

import torch.nn as nn

from segment_anything import sam_model_registry
from model_single import ModelEmb, SegDecoderCNN
from peft import get_peft_model, LoraConfig, TaskType
from monai.transforms import RandRotate, RandFlip

from data_loader import (DataGenerator, Normalize, RandomFlip2D,
                         RandomRotate2D, To_Tensor, RandomRotate3D, RandomFlip3D, DataGenerator3D)
from loss import Deep_Supervised_Loss
from utils import plot_segmentation2D, transform_mask_to_single_channel, one_hot_encode, \
    dice_score_per_class, plot_segmentation3D
from utils import dfs_remove_weight, poly_lr, compute_results_detect
from picai_baseline.unet.training_setup.neural_networks.unets import UNet

warnings.filterwarnings('ignore')
def issue_warning():
    print("Warning: The log path already exists, continue will overwrite the log and ckpt")
    input("Press Enter to continue...")

class UNet2d(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def compute_results(logits, target, results):
    preds = []
    logits = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
    for slices in logits:
        preds.append(extract_lesion_candidates(np.expand_dims(slices, axis=-1), threshold=0.5)[0])
    for y_det, y_true in zip(preds,
                             [target[:, i, :, :] for i in range(target.shape[1])]):
        y_list, *_ = evaluate_case(
            y_det=y_det,
            y_true=y_true.transpose(1, 2, 0),
        )

        # aggregate all validation evaluations
        results.append(y_list)
    return results

class SemanticSeg(object):
    def __init__(self,lr=1e-3,n_epoch=1,channels=4,num_classes=2, input_shape=(384,384),batch_size=6,num_workers=0,
                  device=None,pre_trained=False,ckpt_point=True,weight_path=None,weight_decay=0.0001,
                  use_fp16=False,transformer_depth = 18, mode='2d'):
        super(SemanticSeg,self).__init__()
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape

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

        # os.environ['CUDA_VISIBLE_DEVICES'] = self.device
        # using UNet need to disable all sigmoid activation function
        # self.net = DataParallel(UNet(in_channels=4, out_channels=4, init_features=32))
        # self.net = DataParallel(UNet2d(in_channels=4, out_channels=4, init_features=32), device_ids=[0,1,2,3,4,5, 6, 7])
        self.net = UNet(spatial_dims=3,
            in_channels=4,
            out_channels=4,
            strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
            channels=[32, 64, 128, 256, 512, 1024])


        # sam_model = sam_model_registry['vit_b'](checkpoint='medsam_vit_b.pth')

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

        # dense_model = ModelEmb()
        # multi_mask_decoder = MaskDecoder(
        #     num_multimask_outputs=4,
        #     transformer=TwoWayTransformer(
        #         depth=2,
        #         embedding_dim=256,
        #         mlp_dim=2048,
        #         num_heads=8,
        #     ),
        #     transformer_dim=256,
        #     iou_head_depth=3,
        #     iou_head_hidden_dim=256,
        # )
        # self.net = DataParallel(MedSAMAUTOMULTI(
        #         image_encoder=sam_model.image_encoder,
        #         mask_decoder=multi_mask_decoder,
        #         prompt_encoder=sam_model.prompt_encoder,
        #         dense_encoder=dense_model,
        #         image_size=512
        #     ))

        # mask_decoder_model = SegDecoderCNN(num_classes=4, num_depth=4)
        #
        # self.net = DataParallel(MedSAMAUTOCNN(
        #     image_encoder=sam_model.image_encoder,
        #     mask_decoder=mask_decoder_model,
        #     prompt_encoder=sam_model.prompt_encoder,
        #     dense_encoder=None,
        #     image_size=512
        # ).to(device))


        if self.pre_trained:
            self._get_pre_trained(self.weight_path,ckpt_point)
        if mode == '2d':
            self.train_transform = [
                Normalize(),   #1
                # tio.CropOrPad(target_shape=(32, 128, 128)),
                RandomRotate2D(),  #6
                RandomFlip2D(mode='hv'),  #7
                To_Tensor(num_class=self.num_classes, input_channel = self.channels)   # 10
            ]
        elif mode == '3d':
            self.train_transform = [
                Normalize(),  # 1
                # tio.CropOrPad(target_shape=(32, 128, 128)),
                # RandRotate(range_x=0, range_y=(-20, 20), range_z=(-20, 20), prob=1.0, keep_size=True),  # 6
                # RandFlip(prob=0.3, spatial_axis=(1, 2)),  # 7
                RandomRotate3D(),
                RandomFlip3D(mode='hv'),
                To_Tensor(num_class=self.num_classes, input_channel=self.channels)  # 10
            ]

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

    def trainer(self,train_path,val_path,val_ap, cur_fold,output_dir=None,log_dir=None,phase = 'seg', activation=True):

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
        loss = Deep_Supervised_Loss(mode='Focal', activation=activation)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # dataloader setting
        train_transformer = transforms.Compose(self.train_transform)


        train_dataset = DataGenerator(train_path, transform=train_transformer)

        train_loader = DataLoader(
          train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=self.num_workers,
          pin_memory=True
        )
        val_transformer = transforms.Compose([
            Normalize(),
            # tio.Resize(target_shape=(24, 128, 128)),
            # tio.CropOrPad(target_shape=(32, 128, 128)),
            To_Tensor(num_class=self.num_classes, input_channel=self.channels)
        ])
        val_dataset = DataGenerator(val_path, transform=val_transformer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # copy to gpu
        net = net.to(self.device)
        loss = loss.to(self.device)

        # optimizer setting
        optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=self.weight_decay)

        scaler = GradScaler()

        early_stopping = EarlyStopping(patience=50,verbose=True,monitor='val_score',op_type='max')

        epoch = self.start_epoch
        optimizer.param_groups[0]['lr'] = poly_lr(epoch, self.n_epoch, initial_lr = lr)

        while epoch < self.n_epoch:
            train_loss,caudate_train_dice,putamen_train_dice,globus_train_dice = self._train_on_epoch(epoch,net,loss,optimizer,train_loader,scaler, activation=activation)
            self.writer.add_scalar(
                'data/train_loss', train_loss, epoch
            )
            self.writer.add_scalar('data/caudate_train_dice', caudate_train_dice, epoch)
            self.writer.add_scalar('data/putamen_train_dice', putamen_train_dice, epoch)
            self.writer.add_scalar('data/globus_train_dice', globus_train_dice, epoch)

            self.writer.add_scalar(
                'data/train_loss_epochs', train_loss, epoch
            )

            if phase == 'seg':
                val_loss,caudate_val_dice,putamen_val_dice,globus_val_dice, caudate_positive_dice, putamen_positive_dice, globus_positive_dice = self._val_on_epoch(epoch,net,loss,val_loader, activation=activation)
                self.writer.add_scalar(
                    'data/eval_loss_epochs', val_loss, epoch
                )
                self.writer.add_scalar(
                    'data/eval_caudate_dice_epochs', caudate_val_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_putamen_dice_epochs', putamen_val_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_globus_dice_epochs', globus_val_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_caudate_positive_dice_epochs', caudate_positive_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_putamen_positive_dice_epochs', putamen_positive_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_globus_positive_dice_epochs', globus_positive_dice, epoch
                )

                score = (caudate_val_dice + putamen_val_dice + globus_val_dice) / 3
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
                    file_name = 'epoch:{}-caudate_val_dice:{:.5f}-putamen_val_dice:{:.5f}-globus_val_dice:{:.5f}.pth'.format(
                    epoch,caudate_val_dice,putamen_val_dice,globus_val_dice)
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

    def trainer_3d(self, train_path, val_path, val_ap, cur_fold, output_dir=None, log_dir=None, phase='seg',
                activation=True):

        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold" + str(cur_fold))
        log_dir = os.path.join(log_dir, "fold" + str(cur_fold))

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
        self.global_step = self.start_epoch * math.ceil(len(train_path) / self.batch_size)

        net = self.net
        lr = self.lr
        loss = Deep_Supervised_Loss(mode='Focal', activation=activation)

        # dataloader setting
        train_transformer = transforms.Compose(self.train_transform)

        train_dataset = DataGenerator3D(train_path, transform=train_transformer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        val_transformer = transforms.Compose([
            Normalize(),
            # tio.Resize(target_shape=(24, 128, 128)),
            # tio.CropOrPad(target_shape=(32, 128, 128)),
            To_Tensor(num_class=self.num_classes, input_channel=self.channels)
        ])
        val_dataset = DataGenerator3D(val_path, transform=val_transformer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # copy to gpu
        net = net.to(self.device)
        loss = loss.to(self.device)

        # optimizer setting
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=self.weight_decay)

        scaler = GradScaler()

        early_stopping = EarlyStopping(patience=50, verbose=True, monitor='val_score', op_type='max')

        epoch = self.start_epoch
        optimizer.param_groups[0]['lr'] = poly_lr(epoch, self.n_epoch, initial_lr=lr)

        while epoch < self.n_epoch:
            train_loss, caudate_train_dice, putamen_train_dice, globus_train_dice = self._train_on_epoch(epoch, net,
                                                                                                         loss,
                                                                                                         optimizer,
                                                                                                         train_loader,
                                                                                                         scaler,
                                                                                                         activation=activation)
            self.writer.add_scalar(
                'data/train_loss', train_loss, epoch
            )
            self.writer.add_scalar('data/caudate_train_dice', caudate_train_dice, epoch)
            self.writer.add_scalar('data/putamen_train_dice', putamen_train_dice, epoch)
            self.writer.add_scalar('data/globus_train_dice', globus_train_dice, epoch)

            self.writer.add_scalar(
                'data/train_loss_epochs', train_loss, epoch
            )

            if phase == 'seg':
                val_loss, caudate_val_dice, putamen_val_dice, globus_val_dice, caudate_positive_dice, putamen_positive_dice, globus_positive_dice = self._val_on_epoch(
                    epoch, net, loss, val_loader, activation=activation)
                self.writer.add_scalar(
                    'data/eval_loss_epochs', val_loss, epoch
                )
                self.writer.add_scalar(
                    'data/eval_caudate_dice_epochs', caudate_val_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_putamen_dice_epochs', putamen_val_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_globus_dice_epochs', globus_val_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_caudate_positive_dice_epochs', caudate_positive_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_putamen_positive_dice_epochs', putamen_positive_dice, epoch
                )
                self.writer.add_scalar(
                    'data/eval_globus_positive_dice_epochs', globus_positive_dice, epoch
                )

                score = (caudate_val_dice + putamen_val_dice + globus_val_dice) / 3
            else:
                auc, ap = self.val(epoch, val_ap, net, mode='train')
                score = (ap + auc) / 2

            optimizer.param_groups[0]['lr'] = poly_lr(epoch, self.n_epoch, initial_lr=lr)

            torch.cuda.empty_cache()

            self.writer.add_scalar(
                'data/lr', optimizer.param_groups[0]['lr'], epoch
            )

            early_stopping(score)

            # save
            if score > self.metrics_threshold:
                self.metrics_threshold = score

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                }

                if phase == 'seg':
                    file_name = 'epoch:{}-caudate_val_dice:{:.5f}-putamen_val_dice:{:.5f}-globus_val_dice:{:.5f}.pth'.format(
                        epoch, caudate_val_dice, putamen_val_dice, globus_val_dice)
                else:
                    file_name = 'epoch:{}-train_loss:{:.5f}-train_dice:{:.5f}-train_run_dice:{:.5f}-val_auroc:{:.5f}-val_ap:{:.5f}-val_score:{:.5f}.pth'.format(
                        epoch, train_loss, train_dice, train_run_dice, ap.auroc, ap.AP, ap.score)
                save_path = os.path.join(output_dir, file_name)
                print("Save as: %s" % file_name)

                torch.save(saver, save_path)

            epoch += 1

            # early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.plot_eval(100, val_path, output_dir, log_dir, device='cuda')
        self.writer.close()
        dfs_remove_weight(output_dir, retain=3)

    def _train_on_epoch(self,epoch,net,criterion,optimizer,train_loader,scaler, activation=True, plot=False):
        net.train()

        train_loss = AverageMeter()
        caudate_train_dice = AverageMeter()
        putamen_train_dice = AverageMeter()
        globus_train_dice = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes),ignore_label=-1)

        for step, sample in enumerate(tqdm(train_loader)):
            data = sample['ct']
            targets = sample['seg']
            data = data.to(self.device)
            targets = targets.to(self.device)

            with autocast(self.use_fp16):
                output = net(data)
                if isinstance(output,tuple):
                    output = output[0]
                loss = criterion(output, targets.long())
                # loss = criterion(output,multi_level_targets[:, -1].unsqueeze(1))

            if plot:
                for id, img in enumerate(data):
                    # plot_segmentation2D(img[:3].permute(1, 2, 0).detach().cpu().numpy(),
                    #                     torch.argmax(torch.softmax(output[id, ...], dim=1), dim=1).detach().cpu(),
                    #                     targets[id, ...].detach().cpu().numpy(), f'./train_plot',
                    #                     f'{id}', image_dice=None)
                    plot_segmentation3D(img[:3].permute(1,2,3,0).detach().cpu().numpy(), torch.argmax(torch.softmax(output[id, ...], dim=0), dim=0).detach().cpu(),
                                            targets[id, ...].detach().cpu().numpy(), f'./train_plot',
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
            dice = compute_dice(torch.softmax(output, dim=1).detach(), targets, activation=activation)
            average_dice = torch.mean(dice, dim=0)
            train_loss.update(loss.item(),data.size(0))
            caudate_train_dice.update(average_dice[1],data.size(0))
            putamen_train_dice.update(average_dice[3], data.size(0))
            globus_train_dice.update(average_dice[2], data.size(0))
            # lesion_train_dice.update(average_dice[0], data.size(0))

            # output = (torch.sigmoid(output) > 0.5).int().detach().cpu().numpy()  #N*H*W
            # multi_level_targets = multi_level_targets.detach().cpu().numpy()
            # run_dice.update_matrix(multi_level_targets,output)

            torch.cuda.empty_cache()

            if self.global_step%1==0:
                # rundice, dice_list = run_dice.compute_dice()
                # print("Category Dice: ", dice_list)
                print('epoch:{}/{},step:{},train_loss:{:.5f},caudate_train_dice:{:.5f},putamen_train_dice:{:.5f},globus_train_dice:{:.5f},lr:{}'.format(epoch,self.n_epoch, step, loss.item(), caudate_train_dice.avg, putamen_train_dice.avg, globus_train_dice.avg, optimizer.param_groups[0]['lr']))
                # run_dice.init_op()
                # self.writer.add_scalar(
                #   'data/train_loss',loss.item(),self.global_step
                # )
                # self.writer.add_scalar('data/gland_train_dice', gland_train_dice.avg,self.global_step)
                # self.writer.add_scalar('data/zone_train_dice', zone_train_dice.avg, self.global_step)
                # self.writer.add_scalar('data/lesion_train_dice', lesion_train_dice.avg, self.global_step)

            self.global_step += 1

        return train_loss.avg,caudate_train_dice.avg,putamen_train_dice.avg,globus_train_dice.avg


    def _val_on_epoch(self,epoch,net,criterion,val_loader,val_transformer=None, activation=True, plot=False):
        net.eval()

        val_loss = AverageMeter()
        caudate_val_dice = AverageMeter()
        putamen_val_dice = AverageMeter()
        globus_val_dice = AverageMeter()

        from metrics import RunningDice
        caudate_positive_dice = []
        putamen_positive_dice = []
        globus_positive_dice = []
        with torch.no_grad():
            for step,sample in enumerate(tqdm(val_loader)):
                data = sample['ct']
                targets = sample['seg']

                data = data.to(self.device)
                targets = targets.to(self.device)
                with autocast(self.use_fp16):
                    output = net(data)
                    if isinstance(output,tuple):
                        output = output[0]

                if plot:
                    for id, img in enumerate(data):
                        # plot_segmentation2D(img[:3].permute(1, 2, 0).detach().cpu().numpy(),
                        #                     torch.argmax(torch.softmax(output[id, ...], dim=1), dim=1).detach().cpu(),
                        #                     targets[id, ...].detach().cpu().numpy(), f'./val_plot',
                        #                     f'{id}', image_dice=None)
                        plot_segmentation3D(img[:3].permute(1, 2, 3, 0).detach().cpu().numpy(),
                                            torch.argmax(torch.softmax(output[id, ...], dim=0), dim=0).detach().cpu(),
                                            targets[id, ...].detach().cpu().numpy(), f'./val_plot',
                                            f'{id}', image_dice=None)
                loss = criterion(output, targets.long())

                output = output.float()
                loss = loss.float()

                # dice = compute_dice(output.detach(),multi_level_targets[:, -1].unsqueeze(1),activation=activation)
                dice = compute_dice(torch.softmax(output, dim=1).detach(), targets, activation=activation)
                average_dice = torch.mean(dice, dim=0)
                for id, target in enumerate(targets):
                    if 1 in torch.unique(target):
                        caudate_positive_dice.append(dice[id, 1])
                    if 3 in torch.unique(target):
                        putamen_positive_dice.append(dice[id, 3])
                    if 2 in torch.unique(target):
                        globus_positive_dice.append(dice[id, 2])
                val_loss.update(loss.item(),data.size(0))
                caudate_val_dice.update(average_dice[1], data.size(0))
                putamen_val_dice.update(average_dice[3], data.size(0))
                globus_val_dice.update(average_dice[2], data.size(0))
                # lesion_val_dice.update(average_dice[0], data.size(0))

                # if activation:
                #     logits = torch.sigmoid(output)
                # else:
                #     logits = output
                # output = (logits > 0.5).int().detach().cpu().numpy()  # N*H*W
                # # target = target.detach().cpu().numpy()
                # # run_dice.update_matrix(target,output)
                #
                # lesion_results = compute_results(logits[:, -1, :, :], lesion_target.detach().cpu().numpy(), lesion_results)

                torch.cuda.empty_cache()

                if step % 1 == 0:
                    # rundice, dice_list = run_dice.compute_dice()
                    # print("Category Dice: ", dice_list)
                    print('Eval epoch:{}/{},step:{},val_loss:{:.5f},caudate_val_dice:{:.5f},putamen_val_dice:{:.5f},globus_val_dice:{:.5f}'.format(epoch,self.n_epoch, step, loss.item(), caudate_val_dice.avg, putamen_val_dice.avg, globus_val_dice.avg))
                    # run_dice.init_op()
                    # self.writer.add_scalar(
                    #     'data/eval_loss', loss.item(), self.global_step
                    # )
                    # self.writer.add_scalar('data/eval_dice', rundice, self.global_step)
        # lesion_results = {idx: result for idx, result in enumerate(lesion_results)}
        # lesion_valid_metrics = Metrics(lesion_results)
        # lesion_auc = lesion_valid_metrics.auroc
        # lesion_ap = lesion_valid_metrics.AP

        return val_loss.avg,caudate_val_dice.avg,putamen_val_dice.avg,globus_val_dice.avg, torch.mean(torch.stack(caudate_positive_dice)), torch.mean(torch.stack(putamen_positive_dice)), torch.mean(torch.stack(globus_positive_dice))


    def val(self,epoch, val_path,net = None,val_transformer=None,mode = 'val'):
        if net is None:
            net = self.net
            net = net.to(self.device)
        net.eval()

        class Normalize_2d(object):
            def __call__(self,sample):
                ct = sample['ct']
                seg = sample['seg']
                for i in range(ct.shape[0]):
                    for j in range(ct.shape[1]):
                        if np.max(ct[i,j])!=0:
                            ct[i,j] = ct[i,j]/np.max(ct[i,j])
                    
                new_sample = {'ct':ct, 'seg':seg}
                return new_sample

        val_transformer = transforms.Compose([Normalize_2d(),To_Tensor(num_class=self.num_classes,input_channel = self.channels)])

        val_dataset = DataGenerator(val_path,num_class=self.num_classes,transform=val_transformer)

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

                # from report_guided_annotation import extract_lesion_candidates
                #
                # # process softmax prediction to detection map
                # if mode == 'train':
                #     cspca_det_map_npy = extract_lesion_candidates(
                #         output, threshold='dynamic-fast')[0]
                # else:
                #     cspca_det_map_npy = extract_lesion_candidates(
                #         output, threshold='dynamic',num_lesions_to_extract=5,min_voxels_detection=10,dynamic_threshold_factor = 2.5)[0]
                #
                # # remove (some) secondary concentric/ring detections
                # cspca_det_map_npy[cspca_det_map_npy<(np.max(cspca_det_map_npy)/2)] = 0
                #
                # y_pred.append(cspca_det_map_npy)
                # target = torch.argmax(target,1).detach().cpu().numpy().squeeze()
                # target[target>0] = 1
                # y_true.append(target)
                #
                # print(np.sum(target)>0,np.max(cspca_det_map_npy))
                #
                # torch.cuda.empty_cache()
                # break
        # m = evaluate(y_pred,y_true)
        # print(m)
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
    target = target.long().detach().cpu()
    scores = dice_score_per_class(predict.detach().cpu(), target, num_classes=4)
    return scores