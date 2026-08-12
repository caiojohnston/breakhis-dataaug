"""Mede VRAM de pico do ConditionedLDM pra uma dada arquitetura/batch, sem dataset real.

Usa latentes aleatorios (mesma forma dos latentes reais: 4x32x32) so pra
cronometrar/medir memoria de forward+backward. Serve pra calibrar o quanto
uma GPU maior (ex. 16GB) aguenta antes de rodar o treino de verdade la.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import DDPMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.ldm import ConditionedLDM  # noqa: E402


def probe(
    block_out_channels: tuple[int, ...],
    batch_size: int,
    layers_per_block: int = 2,
    attention_head_dim: int = 8,
    gradient_checkpointing: bool = False,
    n_steps: int = 5,
    embedding_dim: int = 512,
    num_classes: int = 8,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    model = ConditionedLDM(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        attention_head_dim=attention_head_dim,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    t0 = time.time()
    try:
        for _ in range(n_steps):
            latents = torch.randn(batch_size, 4, 32, 32, device=device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            t = torch.randint(0, 1000, (batch_size,), device=device).long()
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, t)

            pred = model(noisy, t, labels, cfg_dropout_prob=0.1)
            loss = nn.functional.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return {
            "ok": True,
            "params_M": n_params / 1e6,
            "peak_vram_MB": peak_mb,
            "sec_per_step": elapsed / n_steps,
        }
    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "params_M": n_params / 1e6, "peak_vram_MB": None, "sec_per_step": None}
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def find_max_batch(
    block_out_channels: tuple[int, ...],
    gradient_checkpointing: bool,
    headroom_frac: float = 0.85,
    start_batch: int = 8,
) -> int:
    """Busca binaria pelo maior batch que roda sem OOM, com folga de seguranca."""
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    budget_mb = total_vram_mb * headroom_frac

    batch = start_batch
    last_ok = None
    # Fase 1: dobra ate estourar
    while True:
        r = probe(block_out_channels, batch, gradient_checkpointing=gradient_checkpointing, n_steps=2)
        print(f"  tentando batch={batch} -> ok={r['ok']} peak={r['peak_vram_MB']}")
        if r["ok"] and r["peak_vram_MB"] <= budget_mb:
            last_ok = batch
            batch *= 2
        else:
            break
    if last_ok is None:
        return 0
    # Fase 2: busca binaria entre last_ok e batch
    lo, hi = last_ok, batch
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = probe(block_out_channels, mid, gradient_checkpointing=gradient_checkpointing, n_steps=2)
        print(f"  bisseccao batch={mid} -> ok={r['ok']} peak={r['peak_vram_MB']}")
        if r["ok"] and r["peak_vram_MB"] <= budget_mb:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=str, default="128,256,256,256")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--find-max-batch", action="store_true",
                        help="Busca o maior batch seguro pra essa arquitetura na GPU atual")
    args = parser.parse_args()

    channels = tuple(int(c) for c in args.channels.split(","))

    if args.find_max_batch:
        best = find_max_batch(channels, args.grad_checkpoint)
        print(f"\nMaior batch seguro (85% da VRAM) pra channels={channels}: {best}")
        return

    result = probe(
        block_out_channels=channels,
        batch_size=args.batch,
        gradient_checkpointing=args.grad_checkpoint,
        n_steps=args.steps,
    )
    print(f"channels={channels} batch={args.batch} grad_ckpt={args.grad_checkpoint} -> {result}")


if __name__ == "__main__":
    main()
