"""Executa comandos longos com log vivo e status em JSON.

Este runner foi criado para treinos/geracoes que levam horas no Windows. Ele
mantem `results/run_status.json` atualizado enquanto o processo filho roda, de
modo que o progresso possa ser consultado sem esperar o comando terminar.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


CLASSIFIER_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+).*?"
    r"train_loss=([0-9.eE+-]+)\s+acc=([0-9.eE+-]+).*?"
    r"val_loss=([0-9.eE+-]+)\s+acc=([0-9.eE+-]+).*?"
    r"val_f1=([0-9.eE+-]+)\s+bal_acc=([0-9.eE+-]+)\s+auc=([0-9.eE+-]+)"
)
LDM_RE = re.compile(r"LDM epoch\s+(\d+)/(\d+)\s+\|\s+loss=([0-9.eE+-]+)")
VAE_RE = re.compile(r"VAE epoch\s+(\d+)/(\d+)\s+\|\s+train=([0-9.eE+-]+)\s+val=([0-9.eE+-]+)")
BEST_RE = re.compile(r"Melhor\s+([^=]+)=([0-9.eE+-]+)")
TEST_RE = re.compile(
    r"Test accuracy:\s+([0-9.eE+-]+).*?"
    r"balanced acc:\s+([0-9.eE+-]+).*?"
    r"F1 macro:\s+([0-9.eE+-]+).*?"
    r"AUC macro:\s+([0-9.eE+-]+)"
)
TOTAL_GENERATION_RE = re.compile(r"Total a gerar:\s+(\d+)\s+imagens")
RESULT_RE = re.compile(r"Resultado salvo em\s+(.+)$")


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_float(value: str) -> float:
    return float(value) if value.lower() != "nan" else float("nan")


def command_from_args(raw_command: list[str]) -> list[str]:
    command = list(raw_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Informe o comando depois de --")
    return command


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def newest_checkpoint(checkpoint_dir: Path | None) -> dict[str, Any] | None:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        return None
    latest = checkpoints[0]
    stat = latest.stat()
    return {
        "path": str(latest),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
    }


def update_progress_from_line(status: dict[str, Any], line: str) -> None:
    if match := CLASSIFIER_RE.search(line):
        epoch = int(match.group(1))
        total = int(match.group(2))
        status["progress"] = {
            "kind": "classifier",
            "current": epoch,
            "total": total,
            "percent": round(epoch * 100 / total, 1),
            "metrics": {
                "train_loss": parse_float(match.group(3)),
                "train_accuracy": parse_float(match.group(4)),
                "val_loss": parse_float(match.group(5)),
                "val_accuracy": parse_float(match.group(6)),
                "val_f1_macro": parse_float(match.group(7)),
                "val_balanced_accuracy": parse_float(match.group(8)),
                "val_auc_macro": parse_float(match.group(9)),
            },
            "safe_pause_hint": (
                "Checkpoint salvo ao fim de cada epoca; pause depois de uma linha Epoch."
            ),
        }
        return

    if match := LDM_RE.search(line):
        epoch = int(match.group(1))
        total = int(match.group(2))
        status["progress"] = {
            "kind": "ldm",
            "current": epoch,
            "total": total,
            "percent": round(epoch * 100 / total, 1),
            "metrics": {"loss": parse_float(match.group(3))},
            "safe_pause_hint": (
                "Checkpoint salvo ao fim de cada epoca; pause depois de uma linha LDM epoch."
            ),
        }
        return

    if match := VAE_RE.search(line):
        epoch = int(match.group(1))
        total = int(match.group(2))
        status["progress"] = {
            "kind": "vae",
            "current": epoch,
            "total": total,
            "percent": round(epoch * 100 / total, 1),
            "metrics": {
                "train_loss": parse_float(match.group(3)),
                "val_loss": parse_float(match.group(4)),
            },
            "safe_pause_hint": (
                "Checkpoint salvo ao fim de cada epoca; pause depois de uma linha VAE epoch."
            ),
        }
        return

    if match := BEST_RE.search(line):
        status["best"] = {
            "metric": match.group(1).strip(),
            "value": parse_float(match.group(2)),
            "updated_at": local_now(),
        }
        return

    if match := TEST_RE.search(line):
        status["test_metrics"] = {
            "accuracy": parse_float(match.group(1)),
            "balanced_accuracy": parse_float(match.group(2)),
            "f1_macro": parse_float(match.group(3)),
            "auc_macro": parse_float(match.group(4)),
        }
        return

    if match := TOTAL_GENERATION_RE.search(line):
        status["generation"] = {"total_images": int(match.group(1))}
        return

    if match := RESULT_RE.search(line):
        status["result_path_from_log"] = match.group(1).strip()


def refresh_status(
    status: dict[str, Any],
    *,
    started_at_seconds: float,
    last_lines: list[str],
    checkpoint_dir: Path | None,
    results_path: Path | None,
) -> None:
    status["last_update_at"] = local_now()
    status["elapsed_seconds"] = round(time.time() - started_at_seconds, 1)
    status["last_lines"] = last_lines[-20:]
    status["latest_checkpoint"] = newest_checkpoint(checkpoint_dir)
    if results_path is not None:
        status["result_path"] = str(results_path)
        status["result_exists"] = results_path.exists()


def reader_thread(pipe: Any, output_queue: queue.Queue[str | None]) -> None:
    try:
        for line in pipe:
            output_queue.put(line)
    finally:
        output_queue.put(None)


def run(args: argparse.Namespace) -> int:
    command = command_from_args(args.command)
    cwd = Path(args.cwd).resolve()
    log_path = Path(args.log)
    status_path = Path(args.status)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    results_path = Path(args.results_path) if args.results_path else None

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_queue: queue.Queue[str | None] = queue.Queue()
    started = time.time()
    last_lines: list[str] = []

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None

    status: dict[str, Any] = {
        "name": args.name,
        "state": "running",
        "monitor_pid": os.getpid(),
        "child_pid": process.pid,
        "command": command,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "status_path": str(status_path),
        "started_at": local_now(),
    }
    refresh_status(
        status,
        started_at_seconds=started,
        last_lines=last_lines,
        checkpoint_dir=checkpoint_dir,
        results_path=results_path,
    )
    atomic_write_json(status_path, status)

    thread = threading.Thread(target=reader_thread, args=(process.stdout, output_queue), daemon=True)
    thread.start()

    exit_code: int | None = None
    last_status_write = 0.0
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"=== START {local_now()} | pid={process.pid} ===\n")
        log_file.write("COMMAND: " + " ".join(command) + "\n")
        try:
            while True:
                try:
                    line = output_queue.get(timeout=args.status_interval)
                except queue.Empty:
                    line = None

                if line is None:
                    exit_code = process.poll()
                    if exit_code is not None:
                        break
                else:
                    log_file.write(line)
                    stripped = line.rstrip()
                    if stripped:
                        last_lines.append(stripped)
                        update_progress_from_line(status, stripped)

                should_write = (
                    line is not None
                    or time.time() - last_status_write >= args.status_interval
                )
                if should_write:
                    refresh_status(
                        status,
                        started_at_seconds=started,
                        last_lines=last_lines,
                        checkpoint_dir=checkpoint_dir,
                        results_path=results_path,
                    )
                    atomic_write_json(status_path, status)
                    last_status_write = time.time()

            while not output_queue.empty():
                line = output_queue.get_nowait()
                if line:
                    log_file.write(line)
                    stripped = line.rstrip()
                    if stripped:
                        last_lines.append(stripped)
                        update_progress_from_line(status, stripped)
        except KeyboardInterrupt:
            status["state"] = "interrupted"
            status["stop_requested_at"] = local_now()
            process.terminate()
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=30)
        finally:
            if exit_code is None:
                exit_code = process.wait()
            status["state"] = "succeeded" if exit_code == 0 else "failed"
            status["exit_code"] = exit_code
            status["finished_at"] = local_now()
            refresh_status(
                status,
                started_at_seconds=started,
                last_lines=last_lines,
                checkpoint_dir=checkpoint_dir,
                results_path=results_path,
            )
            atomic_write_json(status_path, status)
            log_file.write(f"=== END {local_now()} | exit={exit_code} ===\n")

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa comando longo com status JSON.")
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
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
