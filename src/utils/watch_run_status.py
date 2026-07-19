"""Painel textual de progresso para runs monitoradas."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acompanha uma run monitorada em loop.")
    parser.add_argument("--status", type=Path, default=Path("results/run_status.json"))
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--tail", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    return parser


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def load_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fmt_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def fmt_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def progress_bar(percent: float | int | None, width: int = 30) -> str:
    if percent is None:
        return "[" + "-" * width + "]"
    percent = max(0.0, min(100.0, float(percent)))
    filled = int(round(width * percent / 100))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {percent:5.1f}%"


def metric_line(metrics: dict[str, Any]) -> str:
    preferred = [
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_f1_macro",
        "val_balanced_accuracy",
        "val_auc_macro",
        "loss",
    ]
    parts = []
    for key in preferred:
        if key in metrics:
            parts.append(f"{key}={fmt_float(metrics[key])}")
    for key, value in metrics.items():
        if key not in preferred:
            parts.append(f"{key}={fmt_float(value)}")
    return " | ".join(parts)


def render(status: dict[str, Any] | None, status_path: Path, tail: int) -> str:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["BreakHis Run Monitor", "=" * 22, f"Agora: {now}", f"Status file: {status_path}"]

    if status is None:
        lines.extend([
            "",
            "Nenhum status valido encontrado ainda.",
            "Assim que uma run monitorada iniciar, este painel atualiza sozinho.",
        ])
        return "\n".join(lines)

    lines.extend([
        "",
        f"Run: {status.get('name', '-')}",
        f"Estado: {status.get('state', '-')}",
        f"PID filho: {status.get('child_pid', '-')}",
        f"Inicio: {status.get('started_at', '-')}",
        f"Ultima atualizacao: {status.get('last_update_at', '-')}",
        f"Tempo decorrido: {fmt_seconds(status.get('elapsed_seconds'))}",
        f"Log: {status.get('log_path', '-')}",
    ])

    progress = status.get("progress") or {}
    if progress:
        percent = progress.get("percent")
        lines.extend([
            "",
            f"Progresso: {progress.get('kind', '-')} {progress.get('current', '-')}/{progress.get('total', '-')}",
            progress_bar(percent),
        ])
        metrics = progress.get("metrics") or {}
        if metrics:
            lines.append("Metricas: " + metric_line(metrics))
        if progress.get("safe_pause_hint"):
            lines.append("Pausa segura: " + progress["safe_pause_hint"])

    best = status.get("best")
    if best:
        lines.extend(["", f"Melhor ate agora: {best.get('metric')}={fmt_float(best.get('value'))}"])

    checkpoint = status.get("latest_checkpoint")
    if checkpoint:
        lines.extend([
            "",
            "Checkpoint recente:",
            f"  {checkpoint.get('path')}",
            f"  modificado: {checkpoint.get('modified_at')} | tamanho: {checkpoint.get('size_mb')} MB",
        ])

    if "result_exists" in status:
        lines.append(f"Resultado final existe: {status.get('result_exists')}")

    test_metrics = status.get("test_metrics")
    if test_metrics:
        lines.extend(["", "Teste: " + metric_line(test_metrics)])

    last_lines = (status.get("last_lines") or [])[-tail:]
    if last_lines:
        lines.extend(["", "Ultimas linhas:"])
        lines.extend(f"  {line}" for line in last_lines)

    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    try:
        while True:
            status = load_status(args.status)
            clear_screen()
            print(render(status, args.status, args.tail))
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor encerrado.")


if __name__ == "__main__":
    main()
