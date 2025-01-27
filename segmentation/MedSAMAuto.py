import json
import os

import torch
from einops import einsum, rearrange
from torch.nn import functional as F
import torch.nn as nn
from open_clip.model import _build_text_tower
import numpy as np


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


class MedSAMAUTOMULTIALIGNTYPE1(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
            text_encoder,
            dense_encoder,
            image_size,
            attention_encoder,
            mode='align',

    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.text_encoder = text_encoder
        self.dense_encoder = dense_encoder
        self.image_size = image_size
        self.mode = mode
        # self.img_self_attention = attention_encoder

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # print('init logit scale', self.logit_scale)

        # self.img_projector1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.img_projector2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.relu = nn.ReLU(inplace=True)

        # self.cls = nn.Sequential(
        #                         nn.AdaptiveAvgPool2d(1),  # Averages across spatial dimensions [64, 64] to [1, 1]
        #                         nn.Flatten(),             # Flattens [batch, 256, 1, 1] to [batch, 256]
        #                         nn.Linear(256, 256),    # Fully connected layer for 3 classes
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(256, 128),    # Hidden layer to introduce more capacity
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(128, 3)
        #                     )

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # freeze dense encoder
        for param in self.dense_encoder.parameters():
            param.requires_grad = False

        # freeze image encoder; Not updating the image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # # freeze mask decoder
        # for param in self.mask_decoder.parameters():
        #     param.requires_grad = False

    def forward(self, image, tokens):

        image_small = F.interpolate(image, (self.image_size, self.image_size), mode='bilinear', align_corners=True)

        with torch.no_grad():
            # image_embedding_hidden = self.image_encoder(image)  # (B, 256, 64, 64)
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        # image_embedding = self.img_projector2(self.relu(self.img_projector1(image_embedding_hidden)))  # (B, 256, 64, 64)
        # image_embedding = self.img_self_attention(image_embedding_hidden)
        # cls_logits  = self.cls(image_embedding)

        text_embedding = self.text_encoder(tokens)

        image_text_fusion = image_embedding + text_embedding.view(text_embedding.shape[0], text_embedding.shape[1], 1,
                                                                  1)

        dense_embeddings = self.dense_encoder(image_small)  # (B, 256, 64, 64)

        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings_none, dense_embeddings_none = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            # image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_embeddings=image_text_fusion,  # (B, 256, 64, 64)
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
            # print('normal')
            return ori_res_masks
        elif self.mode == 'align':
            # print('align')
            return ori_res_masks, image_embedding, text_embedding, self.logit_scale.exp()
        elif self.mode == 'viz_representation':
            return ori_res_masks, image_embedding, dense_embeddings


class MedSAMAUTOMULTIALIGNTYPE2FINE(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
            text_encoder,
            dense_encoder,
            image_size,
            mode='align',

    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.text_encoder = text_encoder
        self.dense_encoder = dense_encoder
        self.image_size = image_size
        self.mode = mode

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.img_projector1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.img_projector2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
        # self.relu = nn.ReLU(inplace=True)

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # #freeze dense encoder
        # for param in self.dense_encoder.parameters():
        #     param.requires_grad = False

        # # freeze image encoder; Not updating the image encoder
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False

        # # freeze mask decoder
        # for param in self.mask_decoder.parameters():
        #     param.requires_grad = False

    def forward(self, image, tokens, concept_embeding):

        image_small = F.interpolate(image, (self.image_size, self.image_size), mode='bilinear', align_corners=True)

        # with torch.no_grad():
        #     image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        text_embedding = self.text_encoder(tokens)

        image_embed_normalized = F.normalize(image_embedding, dim=1)  # Model output image shape: [B, 256, 64, 64]
        concept_embed_normalized = F.normalize(concept_embeding, dim=-1)  # shape [B, # of concepts, 256]
        image_reshaped = rearrange(image_embed_normalized, 'b c h w -> b (h w) c')  # Shape: [B, 4096, 256]

        # print(concept_embed_normalized.shape)

        out_heatmaps = []
        out_it = []

        for i in range(concept_embed_normalized.shape[1]):  # iterate numebr of concepts
            it_sim = einsum(image_reshaped, concept_embed_normalized[:, i, :], 'b n c, b c -> b n')  # Shape: [B, 4096]
            it_sim = rearrange(it_sim, 'b (h w) -> b h w', h=64, w=64)  # Shape: [B, 64, 64]
            it_sim = F.relu(it_sim, inplace=False)  # remove negative
            it_sim = it_sim.unsqueeze(1)
            it_heatmap = F.interpolate(it_sim, size=(1024, 1024), mode='bilinear', align_corners=True)
            # it_heatmap = it_sim
            out_heatmaps.append(it_heatmap)
            out_it.append(it_sim)

        final_heatmaps = torch.cat(out_heatmaps, dim=1)
        final_it_ori = torch.cat(out_it, dim=1)
        # print(final_it_ori.shape)
        # print(f'image shape: {image.shape} | text shape: {tokens.shape} | cencept shape: {concept_embeding.shape} | computed heatmaps shape:{final_heatmaps.shape}')

        # image_text_fusion = image_embedding + text_embedding.view(text_embedding.shape[0], text_embedding.shape[1], 1, 1)

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
            # image_embeddings=image_text_fusion,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings_none,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
        )
        # print(low_res_masks.shape, out_heatmaps[0].shape)
        # jj

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        if self.mode == 'normal':
            return ori_res_masks
        elif self.mode == 'align':
            return ori_res_masks, image_embedding, text_embedding, self.logit_scale.exp(), final_heatmaps
        elif self.mode == 'align_eval':
            return ori_res_masks, image_embedding, text_embedding, self.logit_scale.exp(), final_heatmaps, final_it_ori
        elif self.mode == 'viz_representation':
            return ori_res_masks, image_embedding, dense_embeddings


class TextEncoder(nn.Module):
    def __init__(
            self,
            embed_dim,
            text_cfg_path,
    ):
        super().__init__()

        with open(os.path.join(text_cfg_path, 'open_clip_config.json'), "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]

        text_encoder = _build_text_tower(embed_dim=model_cfg['embed_dim'], text_cfg=model_cfg['text_cfg'])

        text_encoder.load_state_dict(torch.load(os.path.join(text_cfg_path, 'textencoder.pth')))
        # text_encoder.requires_grad_(False)

        self.text_encoder = text_encoder
        self.text_encoder_head = nn.Linear(512, embed_dim)

    def forward(
            self, tokens
    ):
        # with torch.no_grad():
        #     encoder_hidden_states = self.text_encoder(tokens)

        encoder_hidden_states = self.text_encoder(tokens)

        text_embeddings = self.text_encoder_head(encoder_hidden_states)

        return text_embeddings

class MedSAMAUTOCNN(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder


    def forward(self, image):
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