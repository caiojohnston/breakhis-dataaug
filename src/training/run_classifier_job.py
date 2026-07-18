"""Executa train_classifier.py com stdout/stderr redirecionados.

Uso interno para rodar treinos longos via pythonw.exe no Windows sem prender o terminal.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "Uso: run_classifier_job.py <stdout_log> <stderr_log> <train_classifier_args...>"
        )

    stdout_path = Path(sys.argv[1])
    stderr_path = Path(sys.argv[2])
    train_args = sys.argv[3:]

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_file, stderr_path.open(
        "w", encoding="utf-8", buffering=1
    ) as stderr_file:
        sys.stdout = stdout_file
        sys.stderr = stderr_file
        sys.argv = ["src/training/train_classifier.py", *train_args]
        runpy.run_path("src/training/train_classifier.py", run_name="__main__")


if __name__ == "__main__":
    main()
