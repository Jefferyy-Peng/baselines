from collections.abc import Hashable, Mapping, Sequence

import torch
from matplotlib import pyplot as plt
from monai.apps import TciaDataset
from monai.apps.tcia import TCIA_LABEL_DICT
from monai.transforms import (
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    EnsureTyped,
    Compose, EnsureChannelFirstD, ResampleToMatchd, Resized, MapTransform,
)
from monai.data import DataLoader
import os

# Specify dataset name and download directory
tcia_collection = "QIN-PROSTATE-Repeatability"
data_dir = "./tcia_data"


class HandleBatchDim(MapTransform):
    """
    Custom transform to remove and reintroduce batch dimensions for dictionary-based pipelines.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Remove batch dimension
            if d[key].ndim == 4:  # Image: (B, H, W, D)
                d[key] = d[key]  # Remove batch dim
            if d[key].ndim == 5:  # Segmentation: (B, H, W, D, C)
                d[key] = d[key][0].permute(3, 0, 1, 2)  # Remove batch dim, permute to (C, H, W, D)
        return d


class ReintroduceBatchDim(MapTransform):
    """
    Custom transform to reintroduce batch dimensions after spatial processing.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Reintroduce batch dimension
            if key == 'image':  # Image: (H, W, D)
                d[key] = d[key].permute(0, 3, 1, 2).squeeze(0)  # Add batch dim
            elif key == 'seg':  # Segmentation: (C, H, W, D)
                d[key] = d[key].permute(0, 3, 1, 2)  # Permute back to (H, W, D, C) and add batch dim
        return d

# Define transforms for the dataset
transforms = Compose([
    LoadImaged(keys=["image", "seg"], reader="PydicomReader", label_dict=TCIA_LABEL_DICT[tcia_collection]),
    EnsureChannelFirstD(channel_dim='no_channel',keys=["image", "seg"]),
    HandleBatchDim(keys=["image", "seg"]),
    Spacingd(
        keys=["image", "seg"],
        pixdim=(0.5, 0.5, 3.0),
        mode=("bilinear", "nearest"),  # Use bilinear for image, nearest for segmentation
    ),
    # Resize to target size (24, 384, 384)
    Resized(
        keys=["image", "seg"],
        spatial_size=(384, 384, 24),
        mode=("trilinear", "nearest"),  # Use trilinear for image, nearest for segmentation
    ),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    ReintroduceBatchDim(keys=["image", "seg"]),
])

# Initialize the TCIA dataset
dataset = TciaDataset(
    collection=tcia_collection,
    section="training",  # Options: "training", "validation", or "testing"
    root_dir=data_dir,
    transform=transforms,
    download=False,  # Set to True to download the dataset
)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate over the dataloader and visualize the data
for data in dataloader:
    image = data["image"]
    seg = data["seg"]
    # Remove batch dimension for easier processing
    image = image[0]  # Shape: (24, 384, 384)
    seg = seg[0]  # Shape: (4, 24, 384, 384)

    # Define a colormap for the segmentation overlay
    colormaps = ['Reds', 'Greens', 'Blues', 'Oranges']

    # Loop through slices
    for slice_idx in range(image.shape[0]):  # Iterate over slices (24 in this case)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Show the image slice
        ax.imshow(image[slice_idx], cmap='gray', alpha=1.0)  # Alpha = 1.0 for full image intensity

        # Overlay each segmentation channel
        for channel_idx in range(seg.shape[0]):  # Iterate over channels (4 in this case)
            seg_slice = seg[channel_idx, slice_idx]  # Extract segmentation slice for this channel
            ax.imshow(seg_slice, cmap=colormaps[channel_idx], alpha=0.4)  # Make segmentation semi-transparent

        # Set title and axis off
        ax.set_title(f"Slice {slice_idx + 1}")
        ax.axis('off')

        # Show the plot
        plt.savefig(f'./plot/{slice_idx}.png')