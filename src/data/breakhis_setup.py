"""
Fase 1: organização e split estratificado por patient_id do dataset BreakHis.

Saída:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
    data/splits/distribution_report.txt
"""

import argparse
import logging
import random
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# --- Constantes -----------------------------------------------------------

# Filename format: SOB_B_A-14-22549AB-100-001.png (no X suffix in filename)
PATIENT_RE = re.compile(r"SOB_[BM]_[A-Za-z]+-(\d+-[A-Za-z0-9]+)-(?:40|100|200|400)-\d+")

SUBTYPE_MAP = {
    "A":  ("adenosis",          "benign",    0),
    "F":  ("fibroadenoma",      "benign",    1),
    "PT": ("phyllodes_tumor",   "benign",    2),
    "TA": ("tubular_adenoma",   "benign",    3),
    "DC": ("ductal_carcinoma",  "malignant", 4),
    "LC": ("lobular_carcinoma", "malignant", 5),
    "MC": ("mucinous_carcinoma","malignant", 6),
    "PC": ("papillary_carcinoma","malignant",7),
}

# Mapeamento do nome da pasta para a abreviação usada em SUBTYPE_MAP
FOLDER_TO_ABBR = {
    "adenosis":           "A",
    "fibroadenoma":       "F",
    "phyllodes_tumor":    "PT",
    "tubular_adenoma":    "TA",
    "ductal_carcinoma":   "DC",
    "lobular_carcinoma":  "LC",
    "mucinous_carcinoma": "MC",
    "papillary_carcinoma":"PC",
}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20


# --- Utilitários ----------------------------------------------------------

def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def extract_patient_id(filename: str) -> str | None:
    """Extrai patient_id do nome do arquivo BreakHis."""
    m = PATIENT_RE.search(filename)
    return m.group(1) if m else None


def infer_subtype_from_path(filepath: Path) -> tuple[str, str, str, int] | None:
    """Infere subtipo a partir da estrutura de diretórios."""
    for part in filepath.parts:
        part_lower = part.lower()
        for folder, abbr in FOLDER_TO_ABBR.items():
            if folder in part_lower:
                name, class_type, label_8 = SUBTYPE_MAP[abbr]
                return name, class_type, abbr, label_8
    return None


def extract_magnification(filepath: Path) -> str | None:
    """Extrai magnificação (40, 100, 200, 400) — a pasta pai é sempre '100X', '40X', etc."""
    for part in filepath.parts:
        m = re.fullmatch(r"(40|100|200|400)X", part)
        if m:
            return m.group(1)
    # fallback: extrair do filename (formato -100- entre patient_id e seq)
    m = re.search(r"-(40|100|200|400)-\d+\.", filepath.name)
    return m.group(1) if m else None


# --- Coleta de metadados --------------------------------------------------

def collect_metadata(raw_root: Path) -> pd.DataFrame:
    """Percorre raw_root, extrai metadados de cada imagem .png/.jpg."""
    log.info("Coletando metadados em %s", raw_root)

    records = []
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    for img_path in sorted(raw_root.rglob("*")):
        if img_path.suffix.lower() not in image_extensions:
            continue

        fname = img_path.name
        patient_id = extract_patient_id(fname)
        if patient_id is None:
            log.warning("patient_id não extraído: %s", img_path)
            continue

        subtype_info = infer_subtype_from_path(img_path)
        if subtype_info is None:
            log.warning("subtipo não inferido: %s", img_path)
            continue

        subtype, class_type, label_abbr, label_8 = subtype_info
        magnification = extract_magnification(img_path)
        label_2 = 0 if class_type == "benign" else 1

        records.append({
            "filepath":      str(img_path),
            "class_type":    class_type,
            "subtype":       subtype,
            "label_abbr":    label_abbr,
            "magnification": magnification,
            "patient_id":    patient_id,
            "label_8":       label_8,
            "label_2":       label_2,
        })

    df = pd.DataFrame(records)
    log.info("Total de imagens coletadas: %d", len(df))
    return df


# --- Split estratificado por patient_id -----------------------------------

def stratified_patient_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split 70/10/20 estratificado por patient_id.

    Garante que cada paciente aparece em apenas um split.
    A estratificação usa o subtipo majoritário do paciente.
    """
    patients = (
        df.groupby("patient_id")["subtype"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={"subtype": "dominant_subtype"})
    )

    # Primeira divisão: train+val vs test
    train_val_pts, test_pts = train_test_split(
        patients,
        test_size=TEST_RATIO,
        stratify=patients["dominant_subtype"],
        random_state=SEED,
    )

    # Segunda divisão: train vs val
    # val_ratio relativo ao train+val
    relative_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_pts, val_pts = train_test_split(
        train_val_pts,
        test_size=relative_val,
        stratify=train_val_pts["dominant_subtype"],
        random_state=SEED,
    )

    train_ids = set(train_pts["patient_id"])
    val_ids   = set(val_pts["patient_id"])
    test_ids  = set(test_pts["patient_id"])

    train_df = df[df["patient_id"].isin(train_ids)].copy()
    val_df   = df[df["patient_id"].isin(val_ids)].copy()
    test_df  = df[df["patient_id"].isin(test_ids)].copy()

    return train_df, val_df, test_df


# --- Verificação de overlap -----------------------------------------------

def verify_no_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Aborta se qualquer paciente aparece em mais de um split."""
    train_ids = set(train_df["patient_id"])
    val_ids   = set(val_df["patient_id"])
    test_ids  = set(test_df["patient_id"])

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids   & test_ids

    if tv or tt or vt:
        log.error("OVERLAP DETECTADO!")
        if tv: log.error("  train ∩ val  : %s", tv)
        if tt: log.error("  train ∩ test : %s", tt)
        if vt: log.error("  val   ∩ test : %s", vt)
        raise RuntimeError("Split contém pacientes em múltiplos conjuntos. Abortando.")

    log.info("Verificação de overlap: OK — zero pacientes duplicados entre splits.")


# --- Relatório de distribuição --------------------------------------------

def build_distribution_report(
    df_all: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> str:
    subtypes = sorted(df_all["subtype"].unique())
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("RELATÓRIO DE DISTRIBUIÇÃO — BreakHis Split Fase 1")
    lines.append("=" * 70)
    lines.append("")

    # Totais gerais
    lines.append(f"Total de imagens : {len(df_all):>6}")
    lines.append(f"  Train          : {len(train_df):>6} ({len(train_df)/len(df_all)*100:.1f}%)")
    lines.append(f"  Val            : {len(val_df):>6} ({len(val_df)/len(df_all)*100:.1f}%)")
    lines.append(f"  Test           : {len(test_df):>6} ({len(test_df)/len(df_all)*100:.1f}%)")
    lines.append("")

    # Pacientes únicos
    all_pts   = df_all["patient_id"].nunique()
    train_pts = train_df["patient_id"].nunique()
    val_pts   = val_df["patient_id"].nunique()
    test_pts  = test_df["patient_id"].nunique()
    lines.append(f"Pacientes únicos : {all_pts:>6}")
    lines.append(f"  Train          : {train_pts:>6}")
    lines.append(f"  Val            : {val_pts:>6}")
    lines.append(f"  Test           : {test_pts:>6}")
    lines.append("")

    # Por magnificação
    lines.append("── Por Magnificação ──────────────────────────────────────────────")
    for mag in sorted(df_all["magnification"].dropna().unique(), key=int):
        t = (train_df["magnification"] == mag).sum()
        v = (val_df["magnification"]   == mag).sum()
        e = (test_df["magnification"]  == mag).sum()
        lines.append(f"  {mag}×  train={t:>4}  val={v:>4}  test={e:>4}")
    lines.append("")

    # Por subtipo
    lines.append("── Por Subtipo ───────────────────────────────────────────────────")
    header = f"  {'Subtipo':<25} {'label':>5} {'Classe':<10} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}"
    lines.append(header)
    lines.append("  " + "-" * 65)

    for sub in subtypes:
        row = df_all[df_all["subtype"] == sub].iloc[0]
        total = (df_all["subtype"]  == sub).sum()
        tr    = (train_df["subtype"] == sub).sum()
        vl    = (val_df["subtype"]   == sub).sum()
        te    = (test_df["subtype"]  == sub).sum()
        lines.append(
            f"  {sub:<25} {row['label_8']:>5} {row['class_type']:<10} {total:>6} {tr:>6} {vl:>5} {te:>5}"
        )

    lines.append("")
    lines.append("── Classe Binária ────────────────────────────────────────────────")
    for cls, label in [("benign", 0), ("malignant", 1)]:
        total = (df_all["class_type"]  == cls).sum()
        tr    = (train_df["class_type"] == cls).sum()
        vl    = (val_df["class_type"]   == cls).sum()
        te    = (test_df["class_type"]  == cls).sum()
        lines.append(f"  {cls:<12} label_2={label}  total={total:>5}  train={tr:>5}  val={vl:>4}  test={te:>4}")

    lines.append("")
    lines.append("── Overlap de Pacientes ──────────────────────────────────────────")
    tv = set(train_df["patient_id"]) & set(val_df["patient_id"])
    tt = set(train_df["patient_id"]) & set(test_df["patient_id"])
    vt = set(val_df["patient_id"])   & set(test_df["patient_id"])
    lines.append(f"  train ∩ val  : {len(tv)} pacientes")
    lines.append(f"  train ∩ test : {len(tt)} pacientes")
    lines.append(f"  val   ∩ test : {len(vt)} pacientes")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# --- Main -----------------------------------------------------------------

def main(raw_root: Path, splits_dir: Path) -> None:
    set_seeds()

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Diretório do dataset não encontrado: {raw_root}\n"
            "Extraia o BreaKHis_v1 em data/raw/ antes de rodar este script."
        )

    df = collect_metadata(raw_root)

    if df.empty:
        raise RuntimeError(f"Nenhuma imagem encontrada em {raw_root}")

    log.info("Gerando splits estratificados por patient_id (seed=%d)…", SEED)
    train_df, val_df, test_df = stratified_patient_split(df)

    verify_no_overlap(train_df, val_df, test_df)

    splits_dir.mkdir(parents=True, exist_ok=True)
    cols = ["filepath", "class_type", "subtype", "label_abbr",
            "magnification", "patient_id", "label_8", "label_2"]

    train_df[cols].to_csv(splits_dir / "train.csv", index=False)
    val_df[cols].to_csv(splits_dir / "val.csv",   index=False)
    test_df[cols].to_csv(splits_dir / "test.csv",  index=False)
    log.info("CSVs salvos em %s", splits_dir)

    report = build_distribution_report(df, train_df, val_df, test_df)
    report_path = splits_dir / "distribution_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    log.info("Relatório salvo em %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 1 — Setup BreakHis")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/BreaKHis_v1/histology_slides/breast"),
        help="Raiz das imagens BreakHis",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Diretório de saída dos CSVs e relatório",
    )
    args = parser.parse_args()
    main(args.raw_root, args.splits_dir)
