"""PyTorch Dataset para os três cenários experimentais do TCC."""

import logging
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import get_transforms

log = logging.getLogger(__name__)


class BreakHisDataset(Dataset):
    """
    Carrega imagens BreakHis a partir de um CSV de split.

    Cenário C: passa synthetic_dir para concatenar sintéticas ao train.
    """

    def __init__(
        self,
        csv_path: Path,
        scenario: str,
        split: str,
        synthetic_dir: Path | None = None,
        label_column: str = "label_8",
        synthetic_fraction: float = 1.0,
        synthetic_seed: int = 42,
    ) -> None:
        self.transform = get_transforms(scenario, split)
        self.label_column = label_column
        df = pd.read_csv(csv_path)

        if split == "train" and scenario == "C" and synthetic_dir is not None:
            synthetic_records = self._load_synthetic(synthetic_dir)
            synthetic_records = self._sample_synthetic_records(
                synthetic_records, synthetic_fraction, synthetic_seed
            )
            if synthetic_records:
                syn_df = pd.DataFrame(synthetic_records)
                df = pd.concat([df, syn_df], ignore_index=True)
                log.info(
                    "Cenário C: %d reais + %d sintéticas = %d total",
                    len(df) - len(syn_df), len(syn_df), len(df),
                )

        if label_column not in df.columns:
            raise ValueError(f"Coluna de label nao encontrada: {label_column}")

        self.filepaths = df["filepath"].tolist()
        self.labels    = df[label_column].astype(int).tolist()

    def _load_synthetic(self, synthetic_dir: Path) -> list[dict]:
        """Varre synthetic_dir/{subtipo}/*.png e retorna registros com label_8."""
        from src.data.breakhis_setup import FOLDER_TO_ABBR, SUBTYPE_MAP

        records = []
        for subtype_dir in sorted(synthetic_dir.iterdir()):
            if not subtype_dir.is_dir():
                continue
            abbr = FOLDER_TO_ABBR.get(subtype_dir.name)
            if abbr is None:
                continue
            _, class_type, label_8 = SUBTYPE_MAP[abbr]
            for img_path in sorted(subtype_dir.glob("*.png")):
                records.append({
                    "filepath":   str(img_path),
                    "label_8":    label_8,
                    "label_2":    0 if class_type == "benign" else 1,
                    "class_type": class_type,
                    "subtype":    subtype_dir.name,
                })
        return records

    def _sample_synthetic_records(
        self,
        records: list[dict],
        fraction: float,
        seed: int,
    ) -> list[dict]:
        """Amostra uma fracao deterministica das sinteticas por subtipo."""
        if fraction >= 1.0:
            return records
        if fraction <= 0.0:
            return []

        rng = random.Random(seed)
        grouped: dict[str, list[dict]] = {}
        for record in records:
            grouped.setdefault(record["subtype"], []).append(record)

        sampled: list[dict] = []
        for subtype in sorted(grouped):
            subtype_records = grouped[subtype]
            n_keep = max(1, int(round(len(subtype_records) * fraction)))
            n_keep = min(n_keep, len(subtype_records))
            sampled.extend(
                sorted(
                    rng.sample(subtype_records, n_keep),
                    key=lambda item: item["filepath"],
                )
            )
        return sampled

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple:
        img = Image.open(self.filepaths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]
