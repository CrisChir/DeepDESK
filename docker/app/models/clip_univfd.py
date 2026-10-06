"""CLIP-based universal fake image detector.

Faithful re-implementation of models/clip_models.py::CLIPModel from
https://github.com/WisconsinAIVision/UniversalFakeDetect (MIT License, CVPR 2023).
"""
import torch
import torch.nn as nn
import clip


class CLIPFakeDetector(nn.Module):
    """Frozen CLIP encoder + single linear layer (ViT-L/14, feature dim 768)."""

    CHANNELS = {"RN50": 1024, "ViT-L/14": 768}

    def __init__(self, name="ViT-L/14", num_classes=1, device="cpu"):
        super().__init__()
        self.model, _ = clip.load(name, device=device)
        self.fc = nn.Linear(self.CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)
