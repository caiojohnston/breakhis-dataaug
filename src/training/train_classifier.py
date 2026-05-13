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
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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


def build_dataloaders(
    splits_dir: Path,
    scenario: str,
    batch_size: int,
    synthetic_dir: Path | None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = BreakHisDataset(splits_dir / "train.csv", scenario, "train", synthetic_dir)
    val_ds   = BreakHisDataset(splits_dir / "val.csv",   scenario, "val")
    test_ds  = BreakHisDataset(splits_dir / "test.csv",  scenario, "test")

    num_workers = min(4, torch.get_num_threads())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    log.info("Train: %d imagens | Val: %d | Test: %d",
             len(train_ds), len(val_ds), len(test_ds))
    return train_loader, val_loader, test_loader


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
    f1_mac  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm      = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    report  = classification_report(
        all_labels, all_preds,
        target_names=SUBTYPES[:num_classes],
        zero_division=0,
        output_dict=True,
    )

    # AUC-ROC por subtipo (one-vs-rest)
    auc_per_class = {}
    present_classes = np.unique(all_labels)
    for c in range(num_classes):
        if c not in present_classes:
            auc_per_class[SUBTYPES[c]] = None
            continue
        try:
            auc = roc_auc_score(
                (all_labels == c).astype(int),
                all_probs[:, c],
            )
            auc_per_class[SUBTYPES[c]] = round(float(auc), 4)
        except ValueError:
            auc_per_class[SUBTYPES[c]] = None

    return {
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_mac), 4),
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

    train_loader, val_loader, test_loader = build_dataloaders(
        splits_dir, scenario, config["batch_size"], synthetic_dir
    )

    model     = BreakHisClassifier(config["num_classes"], config["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    ckpt_dir = checkpoints_dir / f"cenario_{scenario}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss  = float("inf")
    patience_count = 0
    best_ckpt_path = ckpt_dir / "best.pt"

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        log.info(
            "Epoch %02d/%02d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f",
            epoch, config["epochs"], train_loss, train_acc, val_loss, val_acc,
        )

        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:02d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
            log.info("  → Melhor val_loss=%.4f, checkpoint salvo.", best_val_loss)
        else:
            patience_count += 1
            if patience_count >= config["early_stopping_patience"]:
                log.info("Early stopping na época %d.", epoch)
                break

    # Avaliação final no test com melhor checkpoint
    log.info("Carregando melhor checkpoint para avaliação no test.")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    metrics = evaluate(model, test_loader, device, config["num_classes"])

    log.info("Test accuracy: %.4f | F1 macro: %.4f", metrics["accuracy"], metrics["f1_macro"])

    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"cenario_{scenario}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"scenario": scenario, **metrics}, f, indent=2, ensure_ascii=False)
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
