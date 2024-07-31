import copy

import torch
from torch.nn import functional as F
import torch.nn as nn

class MedSAMAUTO(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
            dense_encoder,
            image_size
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.dense_encoder = dense_encoder
        self.image_size = image_size

        # freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # # freeze mask decoder
        # for param in self.mask_decoder.parameters():
        #     param.requires_grad = False

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):

        image_small = F.interpolate(image, (self.image_size, self.image_size), mode='bilinear', align_corners=True)

        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        dense_embeddings = self.dense_encoder(image_small)  # (B, 256, 64, 64)

        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings_none, dense_embeddings_none = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings_none,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        # print(image_embedding.shape, dense_embeddings.shape, low_res_masks.shape)

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # print(ori_res_masks.shape)

        return ori_res_masks

class MedSAMAUTOMULTI(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
            dense_encoder,
            image_size,
            mode='normal'
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.dense_encoder = dense_encoder
        self.image_size = image_size
        self.mode = mode

        # freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # # freeze mask decoder
        # for param in self.mask_decoder.parameters():
        #     param.requires_grad = False

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):

        image_small = F.interpolate(image, (self.image_size, self.image_size), mode='bilinear', align_corners=True)

        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        dense_embeddings = self.dense_encoder(image_small)  # (B, 256, 64, 64)

        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings_none, dense_embeddings_none = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings_none,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
        )
        # print(image_embedding.shape, dense_embeddings.shape, low_res_masks.shape)

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # print(ori_res_masks.shape)
        if self.mode == 'normal':
            return ori_res_masks
        elif self.mode == 'viz_representation':
            return ori_res_masks, image_embedding, dense_embeddings


def get_bounding_box(label_mask, margin=0):
    """
    Compute the bounding box from a label mask with an added margin.

    :param label_mask: A 2D tensor where the object mask is non-zero.
    :param margin: An integer representing the number of pixels to add as a margin around the bounding box.
    :return: A list representing the bounding box: [x_min, y_min, x_max, y_max]
    """
    # Ensure the input is a 2D tensor
    assert label_mask.dim() == 2, "label_mask should be a 2D tensor"

    # Get the indices of non-zero elements
    y_indices, x_indices = torch.nonzero(label_mask, as_tuple=True)
    if not y_indices.numel() or not x_indices.numel():
        return None  # No non-zero elements in the mask

    # Compute the bounding box with margin
    x_min = max(torch.min(x_indices).item() - margin, 0)  # Ensure x_min is not negative
    x_max = min(torch.max(x_indices).item() + margin, label_mask.shape[1] - 1)  # Ensure x_max is within the image width
    y_min = max(torch.min(y_indices).item() - margin, 0)  # Ensure y_min is not negative
    y_max = min(torch.max(y_indices).item() + margin,
                label_mask.shape[0] - 1)  # Ensure y_max is within the image height

    bounding_box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.int32)

    return bounding_box
class TwoStageMedSAM(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
            seg_model,
            image_size,
            mode='normal'
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.seg_model = seg_model
        self.image_size = image_size
        self.mode = mode

        # freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # # freeze one of the mask decoder
        # for param in self.mask_decoder_freeze.parameters():
        #     param.requires_grad = False

        # # freeze prompt encoder
        # for param in self.prompt_encoder.parameters():
        #     param.requires_grad = False

        # freeze seg model
        for param in self.seg_model.parameters():
            param.requires_grad = False

    def forward(self, image):

        # image_small = F.interpolate(image, (self.image_size, self.image_size), mode='bilinear', align_corners=True)

        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        prior_masks = self.seg_model(image)

        # gland_mask = (prior_masks[:, 0].unsqueeze(1) > 0.5)
        gland_mask = prior_masks[:, 0].unsqueeze(1)
        low_res_gland_mask = F.interpolate(
            gland_mask,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        # bbox = torch.stack([get_bounding_box(mask) for mask in gland_mask])

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            # boxes=bbox,
            boxes=None,
            masks=low_res_gland_mask,
        )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
        )
        # print(image_embedding.shape, dense_embeddings.shape, low_res_masks.shape)

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )



        # print(ori_res_masks.shape)
        if self.mode == 'normal':
            return ori_res_masks
        elif self.mode == 'viz_representation':
            return ori_res_masks, image_embedding, dense_embeddings


class MedSAMAUTOCNN(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
            dense_encoder,
            image_size
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder

        self.prompt_encoder = prompt_encoder
        self.dense_encoder = dense_encoder
        self.image_size = image_size

        # freeze image encoder
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False

        # # freeze mask decoder
        # for param in self.mask_decoder.parameters():
        #     param.requires_grad = False

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        # image_small = F.interpolate(image, (self.image_size, self.image_size), mode='bilinear', align_corners=True)

        image_embeddings = self.image_encoder(image)  # (B, 256, 64, 64)

        low_res_masks = self.mask_decoder(image_embeddings)
        # print(image_embedding.shape, dense_embeddings.shape, low_res_masks.shape)

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # print(ori_res_masks.shape)

        return ori_res_masks

class MedSAMAUTOZONE(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
            dense_encoder,
            image_size
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.dense_encoder = dense_encoder
        self.image_size = image_size

        # # freeze image encoder
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False

        # # freeze mask decoder
        # for param in self.mask_decoder.parameters():
        #     param.requires_grad = False

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):

        image_small = F.interpolate(image, (self.image_size, self.image_size), mode='bilinear', align_corners=True)

        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        dense_embeddings = self.dense_encoder(image_small)  # (B, 256, 64, 64)

        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings_none, dense_embeddings_none = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings_none,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
        )
        # print(image_embedding.shape, dense_embeddings.shape, low_res_masks.shape)

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # print(ori_res_masks.shape)

        return ori_res_masks[:, :2]