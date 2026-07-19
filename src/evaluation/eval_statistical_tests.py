"""Testes estatisticos pareados para classificadores binarios.

Compara dois modelos no mesmo conjunto de teste usando:

- teste de McNemar com correcao exata binomial para acuracia pareada;
- bootstrap por paciente para IC 95% das diferencas de metricas.

Saidas:
    results/statistical_tests.json
    results/statistical_tests.md
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import BreakHisDataset
from src.models.classifier import BreakHisClassifier


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def collect_predictions(
    *,
    name: str,
    scenario: str,
    config_path: Path,
    checkpoint_path: Path,
    threshold: float,
    splits_dir: Path,
    device: torch.device,
) -> dict:
    config = load_config(config_path)
    if int(config["num_classes"]) != 2 or str(config.get("label_column")) != "label_2":
        raise ValueError(f"Config nao binaria: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")

    ds = BreakHisDataset(
        splits_dir / "test.csv",
        scenario,
        "test",
        label_column=str(config.get("label_column", "label_2")),
    )
    loader = DataLoader(
        ds,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=True,
    )
    model = BreakHisClassifier(
        num_classes=int(config["num_classes"]),
        pretrained=False,
        model_name=str(config.get("model", "efficientnet_b0")),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    labels: list[int] = []
    probs: list[float] = []
    for imgs, batch_labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels.extend(int(x) for x in batch_labels.numpy())
        probs.extend(float(x) for x in batch_probs)

    labels_arr = np.asarray(labels, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)
    preds_arr = (probs_arr >= threshold).astype(int)
    metrics = compute_metrics(labels_arr, preds_arr, probs_arr)
    log.info(
        "%s | threshold=%.4f | acc=%.4f bal=%.4f f1=%.4f auc=%.4f",
        name,
        threshold,
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["f1_macro"],
        metrics["auc"],
    )
    return {
        "name": name,
        "scenario": scenario,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "threshold": threshold,
        "labels": labels_arr,
        "probs": probs_arr,
        "preds": preds_arr,
        "metrics": metrics,
    }


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray | None = None) -> dict:
    out = {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, preds)), 4),
        "f1_macro": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
    }
    if probs is not None and len(np.unique(labels)) == 2:
        out["auc"] = round(float(roc_auc_score(labels, probs)), 4)
    return out


def mcnemar_exact(labels: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    correct_a = preds_a == labels
    correct_b = preds_b == labels
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    discordant = a_correct_b_wrong + a_wrong_b_correct
    p_value = 1.0
    if discordant > 0:
        p_value = float(binomtest(min(a_correct_b_wrong, a_wrong_b_correct), discordant, 0.5).pvalue)
    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_pairs": discordant,
        "p_value_exact_two_sided": round(p_value, 6),
        "significant_at_0_05": bool(p_value < 0.05),
    }


def patient_bootstrap(
    *,
    test_df: pd.DataFrame,
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    patients = sorted(test_df["patient_id"].unique().tolist())
    patient_to_indices = {
        patient: np.asarray(test_df.index[test_df["patient_id"] == patient].tolist(), dtype=int)
        for patient in patients
    }
    diffs: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "f1_macro": [],
        "auc": [],
    }

    for _ in range(n_bootstrap):
        sampled_patients = [rng.choice(patients) for _ in patients]
        indices = np.concatenate([patient_to_indices[p] for p in sampled_patients])
        y = labels[indices]
        pa = preds_a[indices]
        pb = preds_b[indices]
        proba = probs_a[indices]
        probb = probs_b[indices]
        if len(np.unique(y)) < 2:
            continue
        metrics_a = compute_metrics(y, pa, proba)
        metrics_b = compute_metrics(y, pb, probb)
        for metric in diffs:
            if metric in metrics_a and metric in metrics_b:
                diffs[metric].append(float(metrics_b[metric]) - float(metrics_a[metric]))

    out = {}
    for metric, values in diffs.items():
        arr = np.asarray(values, dtype=float)
        out[metric] = {
            "n": int(len(arr)),
            "mean_diff_b_minus_a": round(float(np.mean(arr)), 4) if len(arr) else None,
            "ci95_low": round(float(np.percentile(arr, 2.5)), 4) if len(arr) else None,
            "ci95_high": round(float(np.percentile(arr, 97.5)), 4) if len(arr) else None,
            "p_bootstrap_two_sided_around_zero": bootstrap_p_value(arr) if len(arr) else None,
        }
    return out


def bootstrap_p_value(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    p_le_zero = float(np.mean(values <= 0))
    p_ge_zero = float(np.mean(values >= 0))
    return round(min(1.0, 2.0 * min(p_le_zero, p_ge_zero)), 6)


def write_markdown(payload: dict, out_path: Path) -> None:
    a = payload["models"][0]
    b = payload["models"][1]
    mcnemar = payload["mcnemar_exact"]
    boot = payload["bootstrap_patient_level"]

    lines = [
        "# Testes Estatisticos - Downstream Binario",
        "",
        "Comparacao pareada no conjunto de teste entre baseline binario A e C50_full calibrado.",
        "O bootstrap foi feito por paciente para respeitar a estrutura patient-wise do BreakHis.",
        "",
        "## Modelos Comparados",
        "",
        "| Modelo | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {a['name']} | {a['threshold']:.4f} | {a['metrics']['accuracy']:.4f} | {a['metrics']['balanced_accuracy']:.4f} | {a['metrics']['f1_macro']:.4f} | {a['metrics']['auc']:.4f} |",
        f"| {b['name']} | {b['threshold']:.4f} | {b['metrics']['accuracy']:.4f} | {b['metrics']['balanced_accuracy']:.4f} | {b['metrics']['f1_macro']:.4f} | {b['metrics']['auc']:.4f} |",
        "",
        "## McNemar Exato",
        "",
        f"- A correto / C50_full errado: `{mcnemar['a_correct_b_wrong']}`",
        f"- A errado / C50_full correto: `{mcnemar['a_wrong_b_correct']}`",
        f"- Pares discordantes: `{mcnemar['discordant_pairs']}`",
        f"- p-valor exato bicaudal: `{mcnemar['p_value_exact_two_sided']}`",
        f"- Significativo a 5%: `{mcnemar['significant_at_0_05']}`",
        "",
        "## Bootstrap por Paciente - Diferenca C50_full menos A",
        "",
        "| Metrica | Media diff | IC 95% baixo | IC 95% alto | p bootstrap |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, item in boot.items():
        lines.append(
            f"| {metric} | {fmt(item['mean_diff_b_minus_a'])} | {fmt(item['ci95_low'])} | "
            f"{fmt(item['ci95_high'])} | {fmt(item['p_bootstrap_two_sided_around_zero'])} |"
        )
    lines.extend([
        "",
        "## Leitura",
        "",
        payload["interpretation"],
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def serializable_model(item: dict) -> dict:
    return {
        "name": item["name"],
        "scenario": item["scenario"],
        "config_path": item["config_path"],
        "checkpoint_path": item["checkpoint_path"],
        "threshold": item["threshold"],
        "metrics": item["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Testes estatisticos para A vs C50_full")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-json", type=Path, default=Path("results/statistical_tests.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/statistical_tests.md"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--a-threshold", type=float, default=0.5)
    parser.add_argument("--c50-threshold", type=float, default=0.97)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    baseline = collect_predictions(
        name="A_binary_argmax",
        scenario="A",
        config_path=Path("configs/classifier_binary.yaml"),
        checkpoint_path=Path("checkpoints/cenario_A_binary/best.pt"),
        threshold=args.a_threshold,
        splits_dir=args.splits_dir,
        device=device,
    )
    c50 = collect_predictions(
        name="C50_full_calibrated",
        scenario="C",
        config_path=Path("configs/classifier_binary_c50_full.yaml"),
        checkpoint_path=Path("checkpoints/cenario_C_binary_c50_full/best.pt"),
        threshold=args.c50_threshold,
        splits_dir=args.splits_dir,
        device=device,
    )

    if not np.array_equal(baseline["labels"], c50["labels"]):
        raise RuntimeError("Labels desalinhados entre modelos")

    test_df = pd.read_csv(args.splits_dir / "test.csv").reset_index(drop=True)
    labels = baseline["labels"]
    mcnemar = mcnemar_exact(labels, baseline["preds"], c50["preds"])
    boot = patient_bootstrap(
        test_df=test_df,
        labels=labels,
        preds_a=baseline["preds"],
        preds_b=c50["preds"],
        probs_a=baseline["probs"],
        probs_b=c50["probs"],
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    interpretation = (
        "O ganho de C50_full calibrado sobre A e positivo nas metricas principais. "
        "A significancia deve ser interpretada com cautela porque ha apenas 17 pacientes no teste; "
        "por isso o bootstrap por paciente e mais conservador do que uma analise por imagem."
    )
    payload = {
        "comparison": "A_binary_argmax_vs_C50_full_calibrated",
        "test_set": {"images": int(len(labels)), "patients": int(test_df["patient_id"].nunique())},
        "bootstrap": {"n_resamples": args.bootstrap, "seed": args.seed, "unit": "patient_id"},
        "models": [serializable_model(baseline), serializable_model(c50)],
        "mcnemar_exact": mcnemar,
        "bootstrap_patient_level": boot,
        "interpretation": interpretation,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, args.output_md)
    log.info("Resultados salvos em %s e %s", args.output_json, args.output_md)


if __name__ == "__main__":
    main()
