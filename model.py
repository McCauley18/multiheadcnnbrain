import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from config import (
    NUM_CLASSES, FEATURE_DIM, NUM_HEADS,
    ATTENTION_DROPOUT, HEAD_DROPOUT
)


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, embed_dim=FEATURE_DIM, num_heads=NUM_HEADS, dropout=ATTENTION_DROPOUT):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class BrainTumorModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.attention = MultiHeadAttentionModule(
            embed_dim=self.feature_dim,
            num_heads=NUM_HEADS,
            dropout=ATTENTION_DROPOUT
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(512, num_classes)
        )

        self.freeze_backbone()

    def freeze_backbone(self, unfreeze_last_n=0):
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0:
            total_blocks = len(self.backbone.features)
            start_idx = total_blocks - unfreeze_last_n
            for i in range(start_idx, total_blocks):
                for param in self.backbone.features[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        features = self.backbone.features(x)
        B, C, H, W = features.shape
        tokens = features.flatten(2).permute(0, 2, 1)
        attn_tokens = self.attention(tokens)
        pooled = attn_tokens.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, pooled