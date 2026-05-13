"""VAE wrapper para fine-tuning no domínio histopatológico BreakHis."""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class BreakHisVAE(nn.Module):
    # Fator de escala padrão do sd-vae-ft-mse (mantém variância dos latentes ~1)
    SCALE_FACTOR = 0.18215

    def __init__(self, pretrained: str = "stabilityai/sd-vae-ft-mse") -> None:
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained)
        # Gradient checkpointing reduz uso de VRAM durante fine-tuning
        self.vae.enable_gradient_checkpointing()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) em [-1,1] → latente escalado (B,4,H/8,W/8)."""
        return self.vae.encode(x).latent_dist.sample() * self.SCALE_FACTOR

    def encode_dist(self, x: torch.Tensor):
        """Retorna DiagonalGaussianDistribution para cálculo de KL."""
        return self.vae.encode(x).latent_dist

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: latente escalado → imagem (B,3,H,W) em [-1,1]."""
        return self.vae.decode(z / self.SCALE_FACTOR).sample

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (reconstrução, kl_loss) para fine-tuning."""
        dist = self.vae.encode(x).latent_dist
        z    = dist.sample()
        recon = self.vae.decode(z).sample
        kl    = dist.kl().mean()
        return recon, kl
