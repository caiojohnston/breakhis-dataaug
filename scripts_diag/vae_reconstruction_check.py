"""Diagnostico Fase 0: reconstrucao do VAE congelado (sd-vae-ft-mse) em imagens reais do BreakHis.

Nao treina nada. So mede se o VAE consegue preservar estrutura (nucleo, glandula)
ao fazer encode->decode de imagens reais. Decide se o gargalo de qualidade das
sinteticas esta no VAE congelado ou no LDM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import piq
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.vae import BreakHisVAE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scripts_diag" / "vae_recon_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBTYPES = ["fibroadenoma", "mucinous_carcinoma", "ductal_carcinoma", "adenosis"]
N_PER_SUBTYPE = 2
IMAGE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

to_tensor = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),  # [0,1]
    ]
)


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = to_tensor(img)  # (3,H,W) in [0,1]
    return t * 2.0 - 1.0  # -> [-1,1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", type=Path, default=None,
                        help="Checkpoint fine-tunado. Sem isso, usa o VAE base congelado.")
    parser.add_argument("--tag", type=str, default="baseline",
                        help="Rotulo pra essa rodada (ex. epoch_03) — usado no nome do arquivo de saida.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="val (default) evita circularidade — o fine-tuning otimiza no train.")
    args = parser.parse_args()

    df = pd.read_csv(ROOT / "data" / "splits" / f"{args.split}.csv")

    vae = BreakHisVAE().to(device).eval()
    if args.vae_ckpt is not None:
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
        print(f"VAE fine-tunado carregado: {args.vae_ckpt}")
    else:
        print("VAE base congelado (sd-vae-ft-mse), sem fine-tuning.")

    rows = []
    for subtype in SUBTYPES:
        sub_df = df[df["subtype"] == subtype].sample(
            n=min(N_PER_SUBTYPE, len(df[df["subtype"] == subtype])), random_state=42
        )
        for i, row in enumerate(sub_df.itertuples()):
            img_path = ROOT / row.filepath
            x = load_image(img_path).unsqueeze(0).to(device)

            with torch.no_grad():
                z = vae.encode(x)
                recon = vae.decode(z)

            recon_01 = ((recon.clamp(-1, 1) + 1) / 2).cpu()
            orig_01 = ((x + 1) / 2).cpu()

            ssim_val = piq.ssim(recon_01, orig_01, data_range=1.0).item()
            psnr_val = piq.psnr(recon_01, orig_01, data_range=1.0).item()

            out_name = f"{subtype}_{i}_{args.tag}"
            transforms.ToPILImage()(orig_01[0]).save(OUT_DIR / f"{out_name}_real.png")
            transforms.ToPILImage()(recon_01[0]).save(OUT_DIR / f"{out_name}_recon.png")

            rows.append({"subtype": subtype, "idx": i, "ssim": ssim_val, "psnr": psnr_val})
            print(f"{out_name}: SSIM={ssim_val:.4f} PSNR={psnr_val:.2f}")

    result_df = pd.DataFrame(rows)
    summary = {
        "tag": args.tag,
        "split": args.split,
        "vae_ckpt": str(args.vae_ckpt) if args.vae_ckpt else "base_frozen",
        "ssim_mean": result_df["ssim"].mean(),
        "psnr_mean": result_df["psnr"].mean(),
    }
    metrics_path = OUT_DIR / f"vae_reconstruction_metrics_{args.tag}.csv"
    result_df.to_csv(metrics_path, index=False)

    # Acumula um historico simples pra comparar entre tags/epocas
    history_path = OUT_DIR / "vae_reconstruction_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(summary)
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))

    print("\nResumo:")
    print(result_df.groupby("subtype")[["ssim", "psnr"]].mean())
    print(f"\n[{args.tag}] Media geral SSIM={summary['ssim_mean']:.4f} PSNR={summary['psnr_mean']:.2f} (split={args.split})")
    print(f"Historico acumulado: {history_path}")


if __name__ == "__main__":
    main()
