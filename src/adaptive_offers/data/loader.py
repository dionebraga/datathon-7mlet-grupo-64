"""Load the factual Bank Marketing base and record its provenance.

Two modes, selected automatically:

1. **Real Kaggle base** — if a raw CSV is present under ``data/kaggle/raw/``
   (e.g. ``bank-additional-full.csv``), it is loaded and provenance is marked
   ``real``.
2. **Deterministic facsimile** — otherwise a reproducible synthetic base with
   the *same schema and domains* is generated (provenance ``facsimile``) so the
   whole pipeline runs offline and in CI. Conclusions of business value must use
   the real base; this is documented in the README and reports.

Either way we register source / version / license in a provenance record so the
banca can trace the dataset's origin (Stage 1 acceptance evidence).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_offers.config import get_settings
from adaptive_offers.data.schema import CATEGORY_LEVELS, all_raw_columns
from adaptive_offers.logging_utils import get_logger

logger = get_logger("data.loader")

KAGGLE_DATASET = "henriqueyamahata/bank-marketing"
KAGGLE_URL = "https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing"
KAGGLE_FILE = "bank-additional-full.csv"
DATASET_LICENSE = "CC BY 4.0 (UCI Machine Learning Repository / Moro et al., 2014)"
DATASET_VERSION = "UCI bank-additional-full (2014)"


@dataclass(frozen=True)
class DatasetProvenance:
    """Traceable origin of the loaded base (written to ``data/processed``)."""

    source: str  # "real" | "facsimile"
    kaggle_dataset: str
    kaggle_url: str
    file_name: str
    version: str
    license: str
    n_rows: int
    n_cols: int
    random_seed: int | None
    loaded_on: str

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


def _raw_csv_path() -> Path:
    return get_settings().paths.kaggle / "raw" / KAGGLE_FILE


def _generate_facsimile(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate a deterministic Bank-Marketing-shaped dataset.

    The target is produced by a documented logistic model over a few drivers so
    that downstream EDA, bandit simulation and evaluation are meaningful and
    reproducible. This is a *stand-in*, never a substitute for the real base.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    def pick(col: str, probs: list[float]) -> np.ndarray:
        levels = CATEGORY_LEVELS[col]
        weights = np.asarray(probs, dtype=float)
        weights /= weights.sum()  # tolerate rounding so probabilities sum to 1
        return rng.choice(levels, size=n, p=weights)

    age = np.clip(rng.normal(40, 11, n).round(), 18, 95).astype(int)
    job = pick("job", [0.25, 0.21, 0.16, 0.09, 0.07, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01])
    marital = pick("marital", [0.60, 0.28, 0.11, 0.01])
    education = pick("education", [0.29, 0.23, 0.15, 0.13, 0.10, 0.05, 0.02, 0.03])
    default = pick("default", [0.79, 0.20, 0.01])
    housing = pick("housing", [0.52, 0.45, 0.03])
    loan = pick("loan", [0.82, 0.15, 0.03])
    contact = pick("contact", [0.63, 0.37])
    month = pick("month", [0.01, 0.06, 0.33, 0.05, 0.17, 0.15, 0.01, 0.02, 0.18, 0.02])
    day_of_week = pick("day_of_week", [0.21, 0.20, 0.20, 0.20, 0.19])
    poutcome = pick("poutcome", [0.86, 0.11, 0.03])

    campaign = (rng.gamma(2.0, 1.2, n) + 1).round().clip(1, 43).astype(int)
    previous = rng.poisson(0.17, n).clip(0, 7).astype(int)
    pdays = np.where(previous == 0, 999, rng.integers(1, 27, n))
    duration = rng.gamma(2.0, 130, n).round().clip(0, 4900).astype(int)

    emp_var_rate = rng.choice([-3.4, -1.8, -1.1, -0.1, 1.1, 1.4], n,
                              p=[0.06, 0.10, 0.16, 0.18, 0.40, 0.10]).round(1)
    cons_price_idx = (93.5 + emp_var_rate * 0.25 + rng.normal(0, 0.25, n)).round(3)
    cons_conf_idx = (-40.0 + emp_var_rate * 1.5 + rng.normal(0, 4.0, n)).round(1)
    euribor3m = np.clip(1.5 + emp_var_rate * 0.9 + rng.normal(0, 0.25, n), 0.6, 5.1).round(3)
    nr_employed = (5176 + emp_var_rate * 25 + rng.normal(0, 30, n)).round(1)

    # --- documented logit for the subscription target -----------------------
    success_prev = (poutcome == "success").astype(float)
    retired_student = np.isin(job, ["retired", "student"]).astype(float)
    cellular = (contact == "cellular").astype(float)
    z = (
        -2.55
        + 1.85 * success_prev
        + 0.45 * retired_student
        + 0.35 * cellular
        + 0.012 * (age - 40)
        - 0.06 * campaign
        - 0.55 * euribor3m
        + 0.20 * previous
        + 0.0009 * duration  # mirrors real leakage; dropped in preprocessing
        + rng.normal(0, 0.35, n)
    )
    p = 1.0 / (1.0 + np.exp(-z))
    y = np.where(rng.random(n) < p, "yes", "no")

    df = pd.DataFrame({
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan, "contact": contact,
        "month": month, "day_of_week": day_of_week, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome, "emp_var_rate": emp_var_rate,
        "cons_price_idx": cons_price_idx, "cons_conf_idx": cons_conf_idx,
        "euribor3m": euribor3m, "nr_employed": nr_employed, "y": y,
    })
    return df[all_raw_columns()]


def _read_real_csv(path: Path) -> pd.DataFrame:
    """Read the real Bank Marketing CSV (semicolon-separated, dotted names)."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().replace(".", "_").replace('"', "") for c in df.columns]
    return df


def load_raw(n_rows: int = 20_000, seed: int | None = None) -> tuple[pd.DataFrame, DatasetProvenance]:
    """Return the raw base and its provenance record.

    Args:
        n_rows: rows for the facsimile generator (ignored when a real CSV exists).
        seed: reproducibility seed (defaults to the global settings seed).
    """
    settings = get_settings()
    seed = settings.random_seed if seed is None else seed
    real_path = _raw_csv_path()

    if real_path.exists():
        logger.info(json.dumps({"event": "load_real", "path": str(real_path)}))
        df = _read_real_csv(real_path)
        source, used_seed = "real", None
    else:
        logger.info(json.dumps({"event": "load_facsimile", "n_rows": n_rows, "seed": seed}))
        df = _generate_facsimile(n_rows=n_rows, seed=seed)
        source, used_seed = "facsimile", seed

    provenance = DatasetProvenance(
        source=source,
        kaggle_dataset=KAGGLE_DATASET,
        kaggle_url=KAGGLE_URL,
        file_name=KAGGLE_FILE,
        version=DATASET_VERSION,
        license=DATASET_LICENSE,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        random_seed=used_seed,
        loaded_on=date.today().isoformat(),
    )
    return df, provenance
