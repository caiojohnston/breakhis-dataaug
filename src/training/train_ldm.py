"""
Fase 4: Treino do VAE e LDM condicionado.

Uso:
    python src/training/train_ldm.py --stage vae --config configs/ldm.yaml
    python src/training/train_ldm.py --stage ldm --config configs/ldm.yaml
    python src/training/train_ldm.py --stage all --config configs/ldm.yaml
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from diffusers import DDPMScheduler
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.ldm import ConditionedLDM
from src.models.vae import BreakHisVAE


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class VAEDataset(Dataset):
    """Imagens em [-1,1] para treino/avaliação do VAE."""

    def __init__(self, csv_path: Path, image_size: int = 256) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        return self.transform(img), int(row["label_8"])


class LatentDataset(Dataset):
    """Latentes pré-codificados + labels para treino do LDM."""

    def __init__(self, latents: torch.Tensor, labels: torch.Tensor) -> None:
        self.latents = latents
        self.labels  = labels

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# VAE stage
# ---------------------------------------------------------------------------

def train_vae(config: dict, splits_dir: Path, checkpoints_dir: Path) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("VAE fine-tuning | device=%s", device)

    lr         = float(config.get("vae_lr", 1e-5))
    epochs     = int(config.get("vae_epochs", 10))
    batch_size = int(config.get("vae_batch_size", 8))
    kl_weight  = float(config.get("kl_weight", 1e-4))
    image_size = int(config["image_size"])

    train_ds = VAEDataset(splits_dir / "train.csv", image_size)
    val_ds   = VAEDataset(splits_dir / "val.csv",   image_size)
    num_workers = min(4, torch.get_num_threads())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model     = BreakHisVAE().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ckpt_dir = checkpoints_dir / "vae"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_path     = ckpt_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for imgs, _ in tqdm(train_loader, desc=f"VAE epoch {epoch}/{epochs}", leave=False):
            imgs = imgs.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                recon, kl = model(imgs)
                loss = nn.functional.mse_loss(recon, imgs) + kl_weight * kl
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(imgs)
        train_loss = total_loss / len(train_ds)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    recon, kl = model(imgs)
                    loss = nn.functional.mse_loss(recon, imgs) + kl_weight * kl
                val_loss += loss.item() * len(imgs)
        val_loss /= len(val_ds)
        scheduler.step()

        log.info("VAE epoch %02d/%02d | train=%.4f val=%.4f", epoch, epochs, train_loss, val_loss)
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:02d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            log.info("  → Melhor VAE val_loss=%.4f, checkpoint salvo.", best_val_loss)

    log.info("VAE fine-tuning concluído. Checkpoint: %s", best_path)
    return best_path


# ---------------------------------------------------------------------------
# LDM stage
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_latents(
    vae: BreakHisVAE,
    csv_path: Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pré-codifica todas as imagens de treino para latentes. Feito uma vez."""
    log.info("Pré-codificando latentes de %s …", csv_path)
    ds = VAEDataset(csv_path, image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=min(4, torch.get_num_threads()), pin_memory=True)
    vae.eval()
    all_latents, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Encoding latents"):
        imgs = imgs.to(device)
        latents = vae.encode(imgs).cpu()
        all_latents.append(latents)
        all_labels.append(labels)
    all_latents = torch.cat(all_latents, dim=0)
    all_labels  = torch.cat(all_labels,  dim=0)
    n_nan = torch.isnan(all_latents).sum().item()
    if n_nan > 0:
        log.warning("%d NaN(s) nos latentes — substituindo por 0.", n_nan)
        all_latents = torch.nan_to_num(all_latents, nan=0.0)
    log.info("Latentes pré-codificados: %s | min=%.3f max=%.3f std=%.3f",
             tuple(all_latents.shape),
             all_latents.min().item(), all_latents.max().item(), all_latents.std().item())
    return all_latents, all_labels


def train_ldm(
    config: dict,
    splits_dir: Path,
    checkpoints_dir: Path,
    vae_ckpt: Path | None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("LDM training | device=%s", device)

    timesteps   = int(config["timesteps"])
    lr          = float(config["ldm_lr"])
    epochs      = int(config["ldm_epochs"])
    batch_size  = int(config.get("ldm_batch_size", 8))
    image_size  = int(config["image_size"])
    num_classes = int(config["num_classes"])
    emb_dim     = int(config["embedding_dim"])

    # Carrega VAE (fine-tunado ou base)
    vae = BreakHisVAE().to(device)
    if vae_ckpt and vae_ckpt.exists():
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        log.info("VAE fine-tunado carregado: %s", vae_ckpt)
    else:
        log.info("VAE base (sem fine-tuning) carregado.")
    for p in vae.parameters():
        p.requires_grad_(False)

    # Pré-computa latentes (uma vez)
    all_latents, all_labels = precompute_latents(
        vae, splits_dir / "train.csv", image_size, batch_size, device
    )

    train_ds     = LatentDataset(all_latents, all_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    # Modelo + scheduler DDPM
    model     = ConditionedLDM(num_classes=num_classes, embedding_dim=emb_dim).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=timesteps)
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_sched  = CosineAnnealingLR(optimizer, T_max=epochs)
    # AMP desativado: FP16 causa NaN em cross-attention com pesos aleatórios do UNet
    scaler    = None

    ckpt_dir  = checkpoints_dir / "ldm"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss    = float("inf")
    best_path    = ckpt_dir / "best.pt"
    patience     = int(config.get("early_stopping_patience", 15))
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss   = 0.0
        n_valid      = 0
        for latents, labels in tqdm(train_loader, desc=f"LDM epoch {epoch}/{epochs}", leave=False):
            latents = latents.to(device)
            labels  = labels.to(device)

            t     = torch.randint(0, timesteps, (len(latents),), device=device).long()
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, t)

            pred = model(noisy, t, labels, cfg_dropout_prob=0.1)
            loss = nn.functional.mse_loss(pred, noise)

            if not torch.isfinite(loss):
                log.warning("Loss não-finita detectada — pulando batch.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(latents)
            n_valid    += len(latents)

        epoch_loss = epoch_loss / n_valid if n_valid > 0 else float("nan")
        lr_sched.step()
        log.info("LDM epoch %03d/%03d | loss=%.4f | no_improve=%d/%d",
                 epoch, epochs, epoch_loss, no_improve, patience)
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

        if n_valid > 0 and epoch_loss < best_loss:
            best_loss  = epoch_loss
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping: %d épocas sem melhora. Melhor loss=%.4f", patience, best_loss)
                break

    log.info("LDM training concluído. Checkpoint: %s", best_path)
    return best_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino VAE + LDM para BreakHis")
    parser.add_argument("--stage",           required=True, choices=["vae", "ldm", "all"])
    parser.add_argument("--config",          type=Path, default=Path("configs/ldm.yaml"))
    parser.add_argument("--splits-dir",      type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--vae-ckpt",        type=Path, default=None,
                        help="Checkpoint do VAE para pular stage vae")
    args = parser.parse_args()

    set_seeds(42)
    config = load_config(args.config)

    vae_ckpt = args.vae_ckpt

    if args.stage in ("vae", "all"):
        vae_ckpt = train_vae(config, args.splits_dir, args.checkpoints_dir)

    if args.stage in ("ldm", "all"):
        if vae_ckpt is None:
            vae_ckpt = args.checkpoints_dir / "vae" / "best.pt"
        ldm_best = train_ldm(config, args.splits_dir, args.checkpoints_dir, vae_ckpt)
        marker = args.checkpoints_dir / "ldm" / "training_complete.txt"
        marker.write_text("ok")
        log.info("Marcador de conclusão salvo: %s", marker)


if __name__ == "__main__":
    main()
