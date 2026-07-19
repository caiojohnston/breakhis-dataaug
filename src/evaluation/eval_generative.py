"""Avalia qualidade das imagens sinteticas geradas pelo LDM.

O BreakHis real esta espalhado em multiplas pastas e a geracao sintetica so
existe para subtipos minoritarios. Para manter a avaliacao reprodutivel e leve,
este script cria diretorios temporarios com hardlinks de amostras deterministicos
e calcula:

- FID via clean-fid;
- SSIM, PSNR e LPIPS via piq em pares real/sintetico amostrados.

Saida padrao: results/metricas_gerativas.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from cleanfid import fid
from PIL import Image
from torchvision import transforms

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SUBTYPES = [
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma",
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
]


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sample_paths(paths: Iterable[Path], limit: int | None, rng: random.Random) -> list[Path]:
    items = sorted(Path(p) for p in paths)
    if limit is None or len(items) <= limit:
        return items
    return sorted(rng.sample(items, limit))


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_flat_dir(paths: list[Path], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(paths):
        suffix = src.suffix.lower() or ".png"
        safe_link_or_copy(src, out_dir / f"img_{idx:06d}{suffix}")


def load_image_tensor(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def compute_pair_metrics(
    real_paths: list[Path],
    synthetic_paths: list[Path],
    *,
    image_size: int,
    max_pairs: int,
    rng: random.Random,
    device: torch.device,
) -> dict:
    import piq

    n_pairs = min(len(real_paths), len(synthetic_paths), max_pairs)
    if n_pairs == 0:
        return {"SSIM": None, "LPIPS": None, "PSNR": None, "n_pairs": 0, "lpips_error": None}

    sampled_real = sample_paths(real_paths, n_pairs, rng)
    sampled_synthetic = sample_paths(synthetic_paths, n_pairs, rng)
    rng.shuffle(sampled_synthetic)

    lpips_metric = None
    lpips_error = None
    try:
        lpips_metric = piq.LPIPS(reduction="none").to(device)
        lpips_metric.eval()
    except Exception as exc:  # pragma: no cover - depends on local weights/cache
        lpips_error = str(exc)
        log.warning("LPIPS indisponivel: %s", exc)

    ssim_values: list[float] = []
    psnr_values: list[float] = []
    lpips_values: list[float] = []

    with torch.no_grad():
        for real_path, synthetic_path in zip(sampled_real, sampled_synthetic):
            real = load_image_tensor(real_path, image_size, device)
            synthetic = load_image_tensor(synthetic_path, image_size, device)
            try:
                ssim_values.append(float(piq.ssim(real, synthetic, data_range=1.0).item()))
            except Exception as exc:
                log.warning("SSIM falhou para %s vs %s: %s", real_path, synthetic_path, exc)
            try:
                psnr_values.append(float(piq.psnr(real, synthetic, data_range=1.0).item()))
            except Exception as exc:
                log.warning("PSNR falhou para %s vs %s: %s", real_path, synthetic_path, exc)
            if lpips_metric is not None:
                try:
                    # PIQ LPIPS espera imagens em [-1, 1].
                    lpips = lpips_metric(real * 2 - 1, synthetic * 2 - 1)
                    lpips_values.append(float(lpips.mean().item()))
                except Exception as exc:
                    lpips_error = str(exc)
                    log.warning("LPIPS falhou para %s vs %s: %s", real_path, synthetic_path, exc)
                    lpips_metric = None

    return {
        "SSIM": mean_or_none(ssim_values),
        "LPIPS": mean_or_none(lpips_values),
        "PSNR": mean_or_none(psnr_values),
        "n_pairs": n_pairs,
        "lpips_error": lpips_error,
    }


def compute_fid_for_dirs(
    real_dir: Path,
    synthetic_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    value = fid.compute_fid(
        fdir1=str(real_dir),
        fdir2=str(synthetic_dir),
        mode="clean",
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        verbose=False,
        use_dataparallel=False,
    )
    return round(float(value), 4)


def subtype_real_paths(train_df: pd.DataFrame, subtype: str) -> list[Path]:
    return [Path(p) for p in train_df.loc[train_df["subtype"] == subtype, "filepath"].tolist()]


def subtype_synthetic_paths(synthetic_dir: Path, subtype: str) -> list[Path]:
    subtype_dir = synthetic_dir / subtype
    if not subtype_dir.exists():
        return []
    return sorted(subtype_dir.glob("*.png"))


def evaluate(
    *,
    splits_dir: Path,
    synthetic_dir: Path,
    output_path: Path,
    temp_dir: Path,
    image_size: int,
    max_fid_images: int,
    max_pairs: int,
    fid_batch_size: int,
    num_workers: int,
    seed: int,
    keep_temp: bool,
) -> dict:
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Avaliacao gerativa | device=%s", device)

    train_df = pd.read_csv(splits_dir / "train.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_dir.exists() and not keep_temp:
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    per_subtype: dict[str, dict] = {}
    global_real: list[Path] = []
    global_synthetic: list[Path] = []

    for subtype in SUBTYPES:
        real_paths = subtype_real_paths(train_df, subtype)
        synthetic_paths = subtype_synthetic_paths(synthetic_dir, subtype)
        log.info("%s | real=%d synthetic=%d", subtype, len(real_paths), len(synthetic_paths))

        if not synthetic_paths:
            per_subtype[subtype] = {
                "FID": None,
                "SSIM": None,
                "LPIPS": None,
                "PSNR": None,
                "n_real": len(real_paths),
                "n_synthetic": 0,
                "n_fid_real": 0,
                "n_fid_synthetic": 0,
                "n_pairs": 0,
                "status": "skipped_no_synthetic_images",
            }
            continue

        fid_n = min(len(real_paths), len(synthetic_paths), max_fid_images)
        real_fid_paths = sample_paths(real_paths, fid_n, rng)
        synthetic_fid_paths = sample_paths(synthetic_paths, fid_n, rng)
        real_tmp = temp_dir / subtype / "real"
        synthetic_tmp = temp_dir / subtype / "synthetic"
        prepare_flat_dir(real_fid_paths, real_tmp)
        prepare_flat_dir(synthetic_fid_paths, synthetic_tmp)

        try:
            fid_value = compute_fid_for_dirs(
                real_tmp,
                synthetic_tmp,
                batch_size=fid_batch_size,
                num_workers=num_workers,
                device=device,
            )
        except Exception as exc:
            fid_value = None
            log.warning("FID falhou para %s: %s", subtype, exc)

        pair_metrics = compute_pair_metrics(
            real_paths,
            synthetic_paths,
            image_size=image_size,
            max_pairs=max_pairs,
            rng=rng,
            device=device,
        )
        per_subtype[subtype] = {
            "FID": fid_value,
            "SSIM": pair_metrics["SSIM"],
            "LPIPS": pair_metrics["LPIPS"],
            "PSNR": pair_metrics["PSNR"],
            "n_real": len(real_paths),
            "n_synthetic": len(synthetic_paths),
            "n_fid_real": fid_n,
            "n_fid_synthetic": fid_n,
            "n_pairs": pair_metrics["n_pairs"],
            "lpips_error": pair_metrics["lpips_error"],
            "status": "ok" if fid_value is not None else "partial_fid_failed",
        }
        global_real.extend(real_fid_paths)
        global_synthetic.extend(synthetic_fid_paths)

    global_fid = None
    if global_real and global_synthetic:
        global_n = min(len(global_real), len(global_synthetic), max_fid_images * 2)
        real_global_paths = sample_paths(global_real, global_n, rng)
        synthetic_global_paths = sample_paths(global_synthetic, global_n, rng)
        real_global_tmp = temp_dir / "global" / "real"
        synthetic_global_tmp = temp_dir / "global" / "synthetic"
        prepare_flat_dir(real_global_paths, real_global_tmp)
        prepare_flat_dir(synthetic_global_paths, synthetic_global_tmp)
        try:
            global_fid = compute_fid_for_dirs(
                real_global_tmp,
                synthetic_global_tmp,
                batch_size=fid_batch_size,
                num_workers=num_workers,
                device=device,
            )
        except Exception as exc:
            log.warning("FID global falhou: %s", exc)

    payload = {
        "FID_global": global_fid,
        "global_scope": "generated_subtypes_only_sampled",
        "seed": seed,
        "image_size": image_size,
        "max_fid_images_per_subtype": max_fid_images,
        "max_pairs_per_subtype": max_pairs,
        "per_subtype": per_subtype,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Metricas gerativas salvas em %s", output_path)

    if not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia imagens sinteticas BreakHis")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output", type=Path, default=Path("results/metricas_gerativas.json"))
    parser.add_argument("--temp-dir", type=Path, default=Path("results/generative_eval_tmp"))
    parser.add_argument("--config", type=Path, default=Path("configs/ldm.yaml"))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-fid-images", type=int, default=500)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    image_size = args.image_size or int(config.get("image_size", 256))
    evaluate(
        splits_dir=args.splits_dir,
        synthetic_dir=args.synthetic_dir,
        output_path=args.output,
        temp_dir=args.temp_dir,
        image_size=image_size,
        max_fid_images=args.max_fid_images,
        max_pairs=args.max_pairs,
        fid_batch_size=args.fid_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()
