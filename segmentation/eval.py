import glob
import os
import re

from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_loader import (DataGenerator, Normalize, RandomFlip2D,
                                      RandomRotate2D, To_Tensor)
from segmentation.MedSAMAuto import MedSAMAUTO
from segmentation.model_single import ModelEmb
from segmentation.segment_anything import sam_model_registry
from segmentation.run import get_cross_validation_by_sample


def plot_segmentation2D(img2D, prev_masks, gt2D, save_path, count, image_dice=None):
    """
        Plot each slice of a 3D image, its corresponding previous mask, and ground truth mask.

        Parameters:
        img3D (numpy.ndarray): The 3D image array of shape (depth, height, width).
        prev_masks (numpy.ndarray): The 3D array of previous masks of shape (depth, height, width).
        gt3D (numpy.ndarray): The 3D array of ground truth masks of shape (depth, height, width).
        slice_axis (int): The axis along which to slice the image (0=depth, 1=height, 2=width).
        """
    os.makedirs(save_path, exist_ok=True)
    # Determine the number of slices based on the selected axis

    # Iterate over each slice
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))

        # Plot image slice
    ax = axes[0]
    ax.imshow(img2D, cmap='gray')
    ax.set_title(f'Image')
    ax.axis('off')

        # Plot previous mask slice
    cmap = plt.cm.get_cmap('viridis', 2)
    cmap.colors[0, 3] = 0
    ax = axes[1]
    ax.imshow(img2D, cmap='gray')
    ax.imshow(prev_masks, cmap=cmap, alpha=0.5)
    ax.set_title(f'Predict Mask')
    ax.axis('off')

        # Plot ground truth slice
    cmap = plt.cm.get_cmap('viridis', 2)
    cmap.colors[0, 3] = 0
    ax = axes[2]
    ax.imshow(img2D, cmap='gray')
    ax.imshow(gt2D, cmap=cmap, alpha=0.5)
    ax.set_title(f'Ground Truth')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'slice_{count}'))
    plt.close()

def plot_eval(net, val_path, ckpt_path, log_dir, device):
    epoch_pattern = re.compile(r'epoch:(\d+)-')

    # Initialize variables to keep track of the largest epoch and the corresponding file
    largest_epoch = -1
    ckpt_file = None

    # Iterate over all files in the directory
    for filename in os.listdir(ckpt_path):
        # Match the pattern to find the epoch number
        match = epoch_pattern.search(filename)
        if match:
            epoch = int(match.group(1))
            # Update the largest epoch and file if the current epoch is larger
            if epoch > largest_epoch:
                largest_epoch = epoch
                ckpt_file = filename

    ckpt_file = os.path.join(ckpt_path, ckpt_file)
    state_dict = torch.load(ckpt_file, map_location=device)['state_dict']
    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    plot_path = os.path.join(log_dir, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    val_transformer = transforms.Compose([
        Normalize(),
        # tio.CropOrPad(target_shape=(32, 128, 128)),
        To_Tensor(num_class=2, input_channel=3)
    ])

    val_dataset = DataGenerator(val_path, num_class=2, transform=val_transformer)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    count = 0
    with torch.no_grad():
        for step, sample in enumerate(val_loader):
            data = sample['image']
            target = sample['label'][:, 1].unsqueeze(1)

            data = data.cuda()
            target = target.cuda()
            with autocast(False):
                output = net(data)
                if isinstance(output, tuple):
                    output = output[0]
            plot_segmentation2D(data.squeeze(0).permute(1, 2, 0).detach().cpu(), output.squeeze(0)[0].detach().cpu(), target.squeeze(0)[0].detach().cpu(), plot_path, count)
            count += 1

if __name__ == '__main__':
    PATH_DIR = './dataset/segdata/data_2d'
    PATH_LIST = glob.glob(os.path.join(PATH_DIR, '*.hdf5'))
    train_path, val_path = get_cross_validation_by_sample(PATH_LIST, 5, 1)
    sam_model = sam_model_registry['vit_b'](checkpoint='medsam_vit_b.pth')
    dense_model = ModelEmb()
    net = MedSAMAUTO(
                image_encoder=sam_model.image_encoder,
                mask_decoder=sam_model.mask_decoder,
                prompt_encoder=sam_model.prompt_encoder,
                dense_encoder=dense_model,
                image_size=512
            )
    ckpt_path = './new_ckpt/{}/{}/fold1'.format('seg','MedSAMAuto_no_empty_freeze_image_encoder_adam_lr_0.0001_weight_decay_0.001')
    log_dir = './new_log/{}/{}/fold1'.format('seg','MedSAMAuto_no_empty_freeze_image_encoder_adam_lr_0.0001_weight_decay_0.001')
    plot_eval(net, val_path, ckpt_path, log_dir, 'cuda:7')