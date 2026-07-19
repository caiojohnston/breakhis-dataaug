"""Mostra o status resumido da ultima run monitorada."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mostra status de run monitorada.")
    parser.add_argument("--status", type=Path, default=Path("results/run_status.json"))
    parser.add_argument("--tail", type=int, default=8, help="Quantidade de linhas finais do log.")
    parser.add_argument("--json", action="store_true", help="Imprime o JSON bruto.")
    return parser


def metric_text(metrics: dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def main() -> None:
    args = build_parser().parse_args()
    if not args.status.exists():
        print(f"Nenhuma run monitorada encontrada em {args.status}.")
        return

    status = json.loads(args.status.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return

    print(f"Run: {status.get('name', '-')}")
    print(f"Estado: {status.get('state', '-')}")
    print(f"PID filho: {status.get('child_pid', '-')}")
    print(f"Inicio: {status.get('started_at', '-')}")
    print(f"Ultima atualizacao: {status.get('last_update_at', '-')}")
    print(f"Tempo decorrido: {status.get('elapsed_seconds', 0)}s")
    print(f"Log: {status.get('log_path', '-')}")

    progress = status.get("progress")
    if progress:
        print(
            "Progresso: "
            f"{progress.get('kind')} {progress.get('current')}/{progress.get('total')} "
            f"({progress.get('percent')}%)"
        )
        metrics = progress.get("metrics")
        if metrics:
            print(f"Metricas atuais: {metric_text(metrics)}")
        if progress.get("safe_pause_hint"):
            print(f"Pausa segura: {progress['safe_pause_hint']}")

    best = status.get("best")
    if best:
        print(f"Melhor ate agora: {best.get('metric')}={best.get('value')}")

    checkpoint = status.get("latest_checkpoint")
    if checkpoint:
        print(f"Checkpoint recente: {checkpoint.get('path')} ({checkpoint.get('modified_at')})")

    if "result_exists" in status:
        print(f"Resultado final existe: {status.get('result_exists')}")

    test_metrics = status.get("test_metrics")
    if test_metrics:
        print(f"Teste: {metric_text(test_metrics)}")

    last_lines = status.get("last_lines", [])[-args.tail:]
    if last_lines:
        print("\nUltimas linhas:")
        for line in last_lines:
            print(f"  {line}")


if __name__ == "__main__":
    main()
