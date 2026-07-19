"""Solicita parada da run monitorada atual.

No Windows, usa taskkill no PID filho registrado em results/run_status.json.
O objetivo e parar depois de um checkpoint, nao remover arquivos.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Para a run monitorada atual.")
    parser.add_argument("--status", type=Path, default=Path("results/run_status.json"))
    parser.add_argument("--force", action="store_true", help="Usa taskkill /F no Windows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.status.exists():
        raise SystemExit(f"Status nao encontrado: {args.status}")

    status = json.loads(args.status.read_text(encoding="utf-8"))
    child_pid = status.get("child_pid")
    if not child_pid:
        raise SystemExit("Status nao contem child_pid.")

    command = ["taskkill", "/PID", str(child_pid), "/T"]
    if args.force:
        command.append("/F")
    subprocess.run(command, check=False)

    status["state"] = "stop_requested"
    status["stop_requested_at"] = local_now()
    args.status.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pedido de parada enviado para PID {child_pid}.")


if __name__ == "__main__":
    main()
