"""LDM condicionado por subtipo histológico via cross-attention."""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class ConditionedLDM(nn.Module):
    # Índice reservado para condicionamento nulo (classifier-free guidance)
    NULL_CLASS = 8

    def __init__(self, num_classes: int = 8, embedding_dim: int = 512) -> None:
        super().__init__()
        # +1 para null class (CFG)
        self.class_embedding = nn.Embedding(num_classes + 1, embedding_dim)
        # std=0.02 evita overflow FP16 em cross-attention (mesmo range que BERT/GPT)
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)
        self.unet = UNet2DConditionModel(
            sample_size=32,          # latente 256/8
            in_channels=4,
            out_channels=4,
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(128, 256, 256, 256),
            layers_per_block=2,
            attention_head_dim=8,
            cross_attention_dim=embedding_dim,
        )

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
        cfg_dropout_prob: float = 0.1,
    ) -> torch.Tensor:
        """Prediz ruído. Durante treino aplica dropout de condicionamento para CFG."""
        if self.training and cfg_dropout_prob > 0:
            mask = torch.rand(len(class_labels), device=class_labels.device) < cfg_dropout_prob
            class_labels = class_labels.clone()
            class_labels[mask] = self.NULL_CLASS

        encoder_hidden_states = self.class_embedding(class_labels).unsqueeze(1)
        return self.unet(latents, timesteps, encoder_hidden_states).sample
