"""Inicia uma run monitorada em background.

Exemplo:
    python src/utils/start_monitored.py --name c50 --log logs/c50.log -- \
        .venv/Scripts/python -u src/training/train_classifier.py --scenario C \
        --config configs/classifier_binary_c50.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def command_from_args(raw_command: list[str]) -> list[str]:
    command = list(raw_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Informe o comando depois de --")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inicia run monitorada sem prender o terminal.")
    parser.add_argument("--name", required=True, help="Nome curto da run.")
    parser.add_argument("--log", required=True, type=Path, help="Arquivo de log vivo.")
    parser.add_argument("--status", type=Path, default=Path("results/run_status.json"))
    parser.add_argument("--cwd", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--status-interval", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = command_from_args(args.command)

    monitor_command = [
        sys.executable,
        "-u",
        "src/utils/run_monitored.py",
        "--name",
        args.name,
        "--log",
        str(args.log),
        "--status",
        str(args.status),
        "--cwd",
        str(args.cwd),
        "--status-interval",
        str(args.status_interval),
    ]
    if args.checkpoint_dir:
        monitor_command.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
    if args.results_path:
        monitor_command.extend(["--results-path", str(args.results_path)])
    monitor_command.append("--")
    monitor_command.extend(command)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    args.log.parent.mkdir(parents=True, exist_ok=True)
    monitor_log = args.log.parent / f"{args.name}_monitor.log"

    creationflags = 0
    if os.name == "nt":
        # CREATE_BREAKAWAY_FROM_JOB: sem isso, o Windows Terminal/ConHost pode
        # matar o processo "desanexado" mesmo assim quando a janela fecha,
        # porque o processo ainda pertence ao Job Object do terminal.
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_BREAKAWAY_FROM_JOB
        )

    with monitor_log.open("a", encoding="utf-8") as monitor_output:
        process = subprocess.Popen(
            monitor_command,
            cwd=args.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=monitor_output,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    print(f"Run monitorada iniciada: {args.name}")
    print(f"Monitor PID: {process.pid}")
    print(f"Status: {args.status}")
    print(f"Log: {args.log}")
    print("Acompanhar:")
    print(f"  {sys.executable} src/utils/show_run_status.py --status {args.status}")


if __name__ == "__main__":
    main()
