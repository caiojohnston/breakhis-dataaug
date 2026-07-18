"""EfficientNet-B0 wrapper para classificação 8-classes do BreakHis."""

import timm
import torch
import torch.nn as nn


class BreakHisClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        model_name: str = "efficientnet_b0",
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
