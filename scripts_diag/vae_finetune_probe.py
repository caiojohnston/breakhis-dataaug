"""Calibra tempo/passo e VRAM do fine-tuning do VAE em batches variados.

Usa imagens reais (nao ruido) porque decode de VAE tem custo dependente de
conteudo em menor grau, mas principalmente pra ja aquecer o dataloader real.
So mede — nao decide nada sozinho.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.vae import BreakHisVAE  # noqa: E402
from src.training.train_ldm import VAEDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def probe(batch_size: int, n_steps: int = 3) -> dict:
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    log(f"[batch={batch_size}] montando dataloader...")
    ds = VAEDataset(Path("data/splits/train.csv"), image_size=256)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    log(f"[batch={batch_size}] carregando modelo...")
    model = BreakHisVAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    it = iter(loader)
    step_times = []
    try:
        for i in range(n_steps):
            t_step = time.time()
            imgs, _ = next(it)
            imgs = imgs.to(device)
            recon, kl = model(imgs)
            loss = nn.functional.mse_loss(recon, imgs) + 1e-4 * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            dt = time.time() - t_step
            step_times.append(dt)
            log(f"[batch={batch_size}] passo {i+1}/{n_steps}: {dt:.2f}s")
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return {"ok": True, "peak_vram_MB": peak_mb, "sec_per_step": sum(step_times) / len(step_times)}
    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "peak_vram_MB": None, "sec_per_step": None}
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    n_train_images = 4892
    for bs in (2, 4, 8):
        r = probe(bs)
        if r["ok"]:
            sec_epoch = r["sec_per_step"] * (n_train_images / bs)
            log(f"RESUMO batch={bs:3d} -> {r['sec_per_step']:.3f}s/passo | "
                f"VRAM={r['peak_vram_MB']:.0f}MB | ~epoca={sec_epoch/60:.1f}min")
        else:
            log(f"RESUMO batch={bs:3d} -> OOM")
            break
