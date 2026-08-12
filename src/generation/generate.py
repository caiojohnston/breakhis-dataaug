"""
Fase 4.3: Geração de imagens sintéticas por subtipo usando LDM treinado.

Equaliza o dataset de treino: para cada subtipo gera
    N = max_count_treino - count_subtipo_treino
imagens sintéticas e salva em data/synthetic/{subtipo}/.

Uso:
    python src/generation/generate.py --config configs/ldm.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import DDIMScheduler
from PIL import Image
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
from src.data.breakhis_setup import SUBTYPE_MAP
from src.models.ldm import ConditionedLDM
from src.models.vae import BreakHisVAE

SUBTYPES = [
    "adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",
    "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma",
]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_generation_counts(train_csv: Path) -> dict[str, int]:
    """Calcula quantas imagens gerar por subtipo para equalizar o treino."""
    df = pd.read_csv(train_csv)
    counts = df["subtype"].value_counts().to_dict()
    max_count = max(counts.values())
    log.info("Distribuição treino (max=%d):", max_count)
    to_generate = {}
    for sub in SUBTYPES:
        n = counts.get(sub, 0)
        needed = max(0, max_count - n)
        to_generate[sub] = needed
        log.info("  %-25s  atual=%4d  gerar=%4d", sub, n, needed)
    return to_generate


@torch.no_grad()
def generate_for_subtype(
    model: ConditionedLDM,
    vae: BreakHisVAE,
    scheduler: DDIMScheduler,
    class_label: int,
    n_images: int,
    guidance_scale: float,
    latent_size: int,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    start_idx: int = 0,
) -> None:
    """Gera n_images para um subtipo e salva em out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    vae.eval()

    label_tensor      = torch.tensor([class_label], device=device)
    null_label_tensor = torch.tensor([ConditionedLDM.NULL_CLASS], device=device)

    generated = 0
    pbar = tqdm(total=n_images, desc=f"  {SUBTYPES[class_label]}", leave=False)

    while generated < n_images:
        bs = min(batch_size, n_images - generated)

        # Ruído inicial
        latents = torch.randn(bs, 4, latent_size, latent_size, device=device)

        labels      = label_tensor.expand(bs)
        null_labels = null_label_tensor.expand(bs)

        scheduler.set_timesteps(50)

        for t in scheduler.timesteps:
            ts = t.expand(bs).to(device)

            # Classifier-free guidance: 2 passes
            noise_cond   = model.unet(
                latents,
                ts,
                model.class_embedding(labels).unsqueeze(1),
            ).sample
            noise_uncond = model.unet(
                latents,
                ts,
                model.class_embedding(null_labels).unsqueeze(1),
            ).sample
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode para imagens
        images = vae.decode(latents)                        # [-1, 1]
        images = (images.clamp(-1, 1) + 1) / 2             # [0, 1]
        images = (images * 255).byte().permute(0, 2, 3, 1) # (B, H, W, 3)

        for i in range(bs):
            img_np  = images[i].cpu().numpy()
            img_pil = Image.fromarray(img_np)
            fname   = out_dir / f"synthetic_{start_idx + generated + i:05d}.png"
            img_pil.save(fname)

        generated += bs
        pbar.update(bs)

    pbar.close()


def generate(
    config: dict,
    splits_dir: Path,
    checkpoints_dir: Path,
    synthetic_dir: Path,
    vae_ckpt: Path,
    ldm_ckpt: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Geração de sintéticas | device=%s", device)

    num_classes    = int(config["num_classes"])
    emb_dim        = int(config["embedding_dim"])
    guidance_scale = float(config["guidance_scale"])
    image_size     = int(config["image_size"])
    latent_size    = image_size // 8
    batch_size     = int(config.get("gen_batch_size", config.get("ldm_batch_size", 8)))

    # Carrega modelos
    vae = BreakHisVAE().to(device)
    if vae_ckpt.exists():
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
        log.info("VAE fine-tunado carregado: %s", vae_ckpt)
    else:
        log.info("VAE base (sd-vae-ft-mse) — sem fine-tuning.")

    unet_channels = tuple(config.get("unet_block_out_channels", [128, 256, 256, 256]))
    ldm = ConditionedLDM(
        num_classes=num_classes,
        embedding_dim=emb_dim,
        block_out_channels=unet_channels,
        layers_per_block=int(config.get("unet_layers_per_block", 2)),
        attention_head_dim=int(config.get("unet_attention_head_dim", 8)),
    ).to(device)
    ldm.load_state_dict(torch.load(ldm_ckpt, map_location=device))
    log.info("LDM carregado (block_out_channels=%s): %s", unet_channels, ldm_ckpt)

    ddim = DDIMScheduler(
        num_train_timesteps=int(config["timesteps"]),
        beta_schedule="linear",
        clip_sample=False,
    )

    to_generate = compute_generation_counts(splits_dir / "train.csv")
    total = sum(to_generate.values())
    log.info("Total a gerar: %d imagens", total)

    for subtype, n_needed in to_generate.items():
        if n_needed == 0:
            log.info("%-25s — já balanceado, pulando.", subtype)
            continue

        # Encontra label_8 pelo nome do subtipo
        label_8 = next(
            v[2] for k, v in SUBTYPE_MAP.items() if v[0] == subtype
        )
        out_dir = synthetic_dir / subtype
        log.info("Gerando %d imagens para '%s' (label=%d) …", n_needed, subtype, label_8)

        generate_for_subtype(
            model=ldm,
            vae=vae,
            scheduler=ddim,
            class_label=label_8,
            n_images=n_needed,
            guidance_scale=guidance_scale,
            latent_size=latent_size,
            batch_size=batch_size,
            device=device,
            out_dir=out_dir,
        )
        log.info("  '%s' concluído → %s", subtype, out_dir)

    log.info("Geração de sintéticas finalizada.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera imagens sintéticas com LDM treinado")
    parser.add_argument("--config",          type=Path, default=Path("configs/ldm.yaml"))
    parser.add_argument("--splits-dir",      type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--synthetic-dir",   type=Path, default=Path("data/synthetic"))
    parser.add_argument("--vae-ckpt",        type=Path, default=None)
    parser.add_argument("--ldm-ckpt",        type=Path, default=None)
    args = parser.parse_args()

    config   = load_config(args.config)
    vae_ckpt = args.vae_ckpt or (args.checkpoints_dir / "vae" / "best.pt")
    ldm_ckpt = args.ldm_ckpt or (args.checkpoints_dir / "ldm" / "best.pt")

    generate(
        config=config,
        splits_dir=args.splits_dir,
        checkpoints_dir=args.checkpoints_dir,
        synthetic_dir=args.synthetic_dir,
        vae_ckpt=vae_ckpt,
        ldm_ckpt=ldm_ckpt,
    )


if __name__ == "__main__":
    main()
