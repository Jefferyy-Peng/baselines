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