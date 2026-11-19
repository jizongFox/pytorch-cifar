"""Vision Transformers for 32-by-32 CIFAR images."""

import torch
from torch import nn


class VisionTransformer(nn.Module):
    """Apply `depth` encoder blocks with independent or shared weights."""

    def __init__(self, *, depth=6, looped=False):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")

        self.depth = depth
        self.looped = looped
        width = 128
        patch_size = 4
        num_patches = (32 // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, width, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.pos_embed = nn.Parameter(torch.empty(1, num_patches + 1, width))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 10)

        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=4 * width,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(1 if looped else depth)
        )

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
            raise ValueError("expected images of shape [batch, 3, 32, 32]")

        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(images.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1) + self.pos_embed
        for step in range(self.depth):
            tokens = self.blocks[0 if self.looped else step](tokens)

        return self.head(self.norm(tokens[:, 0]))
