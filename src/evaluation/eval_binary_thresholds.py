"""Avalia classificadores binarios com threshold calibrado na validacao.

Usa o checkpoint best.pt de cada cenario, escolhe o threshold da probabilidade
maligna que maximiza F1 macro na validacao e reporta as metricas no teste.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
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
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def collect_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    probs_all = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        labels_all.extend(labels.numpy())
        probs_all.extend(probs)
    return np.asarray(labels_all), np.asarray(probs_all)


def metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float, class_names: list[str]) -> dict:
    malignant_probs = probs[:, 1]
    preds = (malignant_probs >= threshold).astype(int)
    auc = roc_auc_score(labels, malignant_probs)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, preds)), 4),
        "f1_macro": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "auc": round(float(auc), 4),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels,
            preds,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def choose_threshold(labels: np.ndarray, probs: np.ndarray, metric: str) -> tuple[float, dict]:
    class_names = ["benign", "malignant"]
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.array([0.5])]))
    best_threshold = 0.5
    best_result = None
    best_score = -float("inf")

    for threshold in candidates:
        result = metrics_at_threshold(labels, probs, float(threshold), class_names)
        score = float(result[metric])
        if score > best_score or (score == best_score and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)
            best_result = result

    if best_result is None:
        raise RuntimeError("Nenhum threshold avaliado")
    return best_threshold, best_result


def evaluate_scenario(
    scenario: str,
    config: dict,
    splits_dir: Path,
    checkpoints_dir: Path,
    metric: str,
    device: torch.device,
) -> dict:
    suffix = str(config.get("result_suffix", ""))
    ckpt_path = checkpoints_dir / f"cenario_{scenario}{suffix}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {ckpt_path}")

    class_names = list(config.get("class_names", ["benign", "malignant"]))
    label_column = str(config.get("label_column", "label_2"))
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))

    val_ds = BreakHisDataset(splits_dir / "val.csv", scenario, "val", label_column=label_column)
    test_ds = BreakHisDataset(splits_dir / "test.csv", scenario, "test", label_column=label_column)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = BreakHisClassifier(
        num_classes=int(config["num_classes"]),
        pretrained=False,
        model_name=str(config.get("model", "efficientnet_b0")),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val_labels, val_probs = collect_probs(model, val_loader, device)
    threshold, val_result = choose_threshold(val_labels, val_probs, metric)
    test_labels, test_probs = collect_probs(model, test_loader, device)
    test_result = metrics_at_threshold(test_labels, test_probs, threshold, class_names)
    default_test_result = metrics_at_threshold(test_labels, test_probs, 0.5, class_names)

    log.info(
        "Scenario %s | threshold=%.4f | test acc=%.4f f1=%.4f bal=%.4f auc=%.4f",
        scenario,
        threshold,
        test_result["accuracy"],
        test_result["f1_macro"],
        test_result["balanced_accuracy"],
        test_result["auc"],
    )

    return {
        "scenario": scenario,
        "checkpoint": str(ckpt_path),
        "selection_metric": metric,
        "chosen_threshold": round(float(threshold), 4),
        "validation_at_chosen_threshold": val_result,
        "test_at_chosen_threshold": test_result,
        "test_at_default_threshold_0_5": default_test_result,
    }


def write_markdown(results: list[dict], out_path: Path, metric: str) -> None:
    lines = [
        "# Comparativo Binario com Threshold Calibrado",
        "",
        f"Threshold escolhido em validacao para maximizar `{metric}`; metricas reportadas no teste.",
        "",
        "| Cenario | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        test = item["test_at_chosen_threshold"]
        lines.append(
            f"| {item['scenario']} | {item['chosen_threshold']:.4f} | {test['accuracy']:.4f} | "
            f"{test['balanced_accuracy']:.4f} | {test['f1_macro']:.4f} | {test['auc']:.4f} |"
        )
    lines.append("")
    lines.append("## Matrizes de Confusao")
    lines.append("")
    for item in results:
        lines.append(f"### Cenario {item['scenario']}")
        lines.append("")
        lines.append("```text")
        lines.append(str(item["test_at_chosen_threshold"]["confusion_matrix"]))
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia thresholds binarios A/B/C")
    parser.add_argument("--config", type=Path, default=Path("configs/classifier_binary.yaml"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--scenarios", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--metric", default="f1_macro", choices=["f1_macro", "balanced_accuracy", "accuracy"])
    args = parser.parse_args()

    config = load_config(args.config)
    if int(config["num_classes"]) != 2 or str(config.get("label_column")) != "label_2":
        raise ValueError("Este avaliador exige config binaria com num_classes=2 e label_column=label_2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    results = [
        evaluate_scenario(s, config, args.splits_dir, args.checkpoints_dir, args.metric, device)
        for s in args.scenarios
    ]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metric_suffix = args.metric.replace("_", "-")
    json_path = args.results_dir / f"comparativo_binary_thresholds_{metric_suffix}.json"
    md_path = args.results_dir / f"comparativo_binary_thresholds_{metric_suffix}.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"metric": args.metric, "results": results}, f, indent=2, ensure_ascii=False)
    write_markdown(results, md_path, args.metric)
    log.info("Resultados salvos em %s e %s", json_path, md_path)


if __name__ == "__main__":
    main()
