"""
Treino do classificador EfficientNet-B0 — Cenários A, B e C.

Uso:
    python src/training/train_classifier.py --scenario A --config configs/classifier.yaml
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Importações internas
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import BreakHisDataset
from src.models.classifier import BreakHisClassifier

SUBTYPES = [
    "adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",
    "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma",
]


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class FocalLoss(nn.Module):
    """Focal Loss para reduzir dominancia das classes majoritarias."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def compute_class_weights(labels: list[int], num_classes: int) -> tuple[torch.Tensor, np.ndarray]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
    safe_counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * safe_counts)
    return torch.tensor(weights, dtype=torch.float32), counts


def build_criterion(config: dict, train_labels: list[int], device: torch.device) -> nn.Module:
    loss_name = str(config.get("loss", "cross_entropy")).lower()
    use_class_weights = bool(config.get("class_weights", False))
    weights = None

    if use_class_weights:
        weights, counts = compute_class_weights(train_labels, int(config["num_classes"]))
        weights = weights.to(device)
        log.info("Class counts: %s", counts.tolist())
        log.info("Class weights: %s", [round(float(w), 4) for w in weights.cpu()])

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weights)
    if loss_name == "focal":
        return FocalLoss(gamma=float(config.get("focal_gamma", 2.0)), weight=weights)

    raise ValueError(f"Loss nao suportada: {loss_name}")


def selection_score(selection_metric: str, val_loss: float, val_metrics: dict) -> float:
    if selection_metric == "val_loss":
        return -val_loss
    if selection_metric not in val_metrics:
        raise ValueError(f"Metrica de selecao nao encontrada: {selection_metric}")
    return float(val_metrics[selection_metric])


def build_dataloaders(
    splits_dir: Path,
    scenario: str,
    batch_size: int,
    synthetic_dir: Path | None,
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, BreakHisDataset]:
    label_column = str(config.get("label_column", "label_8"))
    synthetic_fraction = float(config.get("synthetic_fraction", 1.0))
    synthetic_seed = int(config.get("synthetic_seed", 42))
    train_ds = BreakHisDataset(
        splits_dir / "train.csv",
        scenario,
        "train",
        synthetic_dir,
        label_column=label_column,
        synthetic_fraction=synthetic_fraction,
        synthetic_seed=synthetic_seed,
    )
    val_ds = BreakHisDataset(
        splits_dir / "val.csv", scenario, "val", label_column=label_column
    )
    test_ds = BreakHisDataset(
        splits_dir / "test.csv", scenario, "test", label_column=label_column
    )
    log.info("Label column: %s", label_column)
    if scenario == "C":
        log.info("Synthetic fraction: %.2f | seed=%d", synthetic_fraction, synthetic_seed)

    num_workers = int(config.get("num_workers", min(4, torch.get_num_threads())))
    log.info("DataLoader num_workers: %d", num_workers)

    sampler = None
    shuffle_train = True
    if bool(config.get("weighted_sampler", False)):
        weights, _ = compute_class_weights(train_ds.labels, int(config["num_classes"]))
        sample_weights = [float(weights[int(label)]) for label in train_ds.labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False
        log.info("WeightedRandomSampler ativo no treino.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    log.info("Train: %d imagens | Val: %d | Test: %d",
             len(train_ds), len(val_ds), len(test_ds))
    return train_loader, val_loader, test_loader, train_ds


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    acc     = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_mac  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm      = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    report  = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    # AUC-ROC por subtipo (one-vs-rest)
    auc_per_class = {}
    auc_values = []
    present_classes = np.unique(all_labels)
    for c in range(num_classes):
        if c not in present_classes:
            auc_per_class[class_names[c]] = None
            continue
        try:
            auc = roc_auc_score(
                (all_labels == c).astype(int),
                all_probs[:, c],
            )
            auc_per_class[class_names[c]] = round(float(auc), 4)
            auc_values.append(float(auc))
        except ValueError:
            auc_per_class[class_names[c]] = None

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "f1_macro": round(float(f1_mac), 4),
        "auc_macro": round(float(np.mean(auc_values)), 4) if auc_values else None,
        "auc_roc_per_subtype": auc_per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def train(
    scenario: str,
    config: dict,
    splits_dir: Path,
    results_dir: Path,
    checkpoints_dir: Path,
    synthetic_dir: Path | None,
) -> None:
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | Cenário: %s", device, scenario)

    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        splits_dir, scenario, config["batch_size"], synthetic_dir, config
    )
    class_names = list(config.get("class_names", SUBTYPES[: int(config["num_classes"])]))
    if len(class_names) != int(config["num_classes"]):
        raise ValueError("class_names deve ter o mesmo tamanho de num_classes")

    model = BreakHisClassifier(
        num_classes=config["num_classes"],
        pretrained=config["pretrained"],
        model_name=config.get("model", "efficientnet_b0"),
    ).to(device)
    log.info("Modelo: %s", config.get("model", "efficientnet_b0"))
    criterion = build_criterion(config, train_ds.labels, device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    result_suffix = str(config.get("result_suffix", ""))
    ckpt_dir = checkpoints_dir / f"cenario_{scenario}{result_suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    selection_metric = str(config.get("selection_metric", "val_loss"))
    best_score = -float("inf")
    best_epoch = 0
    patience_count = 0
    best_ckpt_path = ckpt_dir / "best.pt"
    history = []
    log.info("Selecao de checkpoint por: %s", selection_metric)

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, device)
        val_metrics = evaluate(model, val_loader, device, config["num_classes"], class_names)
        val_auc_macro = val_metrics["auc_macro"] if val_metrics["auc_macro"] is not None else float("nan")
        scheduler.step()

        log.info(
            "Epoch %02d/%02d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f | val_f1=%.4f bal_acc=%.4f auc=%.4f",
            epoch, config["epochs"], train_loss, train_acc, val_loss, val_acc,
            val_metrics["f1_macro"], val_metrics["balanced_accuracy"], val_auc_macro,
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_accuracy": round(float(train_acc), 4),
            "val_loss": round(float(val_loss), 4),
            "val_accuracy": round(float(val_acc), 4),
            "val_f1_macro": val_metrics["f1_macro"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_auc_macro": val_metrics["auc_macro"],
        })

        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:02d}.pt")

        score = selection_score(selection_metric, val_loss, val_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
            log.info("  -> Melhor %s=%.4f, checkpoint salvo.", selection_metric, best_score)
        else:
            patience_count += 1
            if patience_count >= config["early_stopping_patience"]:
                log.info("Early stopping na época %d.", epoch)
                break

    # Avaliação final no test com melhor checkpoint
    log.info("Carregando melhor checkpoint para avaliação no test.")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    metrics = evaluate(model, test_loader, device, config["num_classes"], class_names)

    test_auc_macro = metrics["auc_macro"] if metrics["auc_macro"] is not None else float("nan")
    log.info(
        "Test accuracy: %.4f | balanced acc: %.4f | F1 macro: %.4f | AUC macro: %.4f",
        metrics["accuracy"], metrics["balanced_accuracy"], metrics["f1_macro"], test_auc_macro,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"cenario_{scenario}{result_suffix}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "scenario": scenario,
            "best_epoch": best_epoch,
            "selection_metric": selection_metric,
            "best_selection_score": round(float(best_score), 4),
            "config": config,
            "history": history,
            **metrics,
        }, f, indent=2, ensure_ascii=False)
    log.info("Resultado salvo em %s", result_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino do classificador BreakHis")
    parser.add_argument("--scenario",      required=True, choices=["A", "B", "C"])
    parser.add_argument("--config",        type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument("--splits-dir",    type=Path, default=Path("data/splits"))
    parser.add_argument("--results-dir",   type=Path, default=Path("results"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    config = load_config(args.config)
    synthetic_dir = args.synthetic_dir if args.scenario == "C" else None

    train(
        scenario=args.scenario,
        config=config,
        splits_dir=args.splits_dir,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        synthetic_dir=synthetic_dir,
    )


if __name__ == "__main__":
    main()
