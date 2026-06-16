"""Distribution-drift detection: PSI and KS.

* **PSI** (Population Stability Index) — standard credit/marketing metric for
  feature drift. Bands: <0.10 estável, 0.10-0.25 moderado, >0.25 significativo.
* **KS** (Kolmogorov-Smirnov) — non-parametric test for distribution change of a
  continuous feature or the policy's score.

Used by the MLOps lifecycle to trigger review/retraining when the relationship
between context and conversion shifts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and current sample."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:  # near-constant reference; widen edges
        edges = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), bins + 1)
    eps = 1e-6
    ref_pct = np.histogram(ref, bins=edges)[0] / max(len(ref), 1) + eps
    cur_pct = np.histogram(cur, bins=edges)[0] / max(len(cur), 1) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_band(value: float) -> str:
    if value < 0.10:
        return "estável"
    if value < 0.25:
        return "moderado"
    return "significativo"


def ks_drift(reference: np.ndarray, current: np.ndarray) -> dict[str, float]:
    """Two-sample KS test; ``drift`` True if p < 0.05."""
    stat, p = stats.ks_2samp(np.asarray(reference), np.asarray(current))
    return {"ks_stat": round(float(stat), 4), "p_value": round(float(p), 4),
            "drift": bool(p < 0.05)}


def drift_report(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    psi_alert: float = 0.25,
) -> dict[str, Any]:
    """PSI + KS per feature; overall flag if any feature drifts significantly."""
    features: dict[str, Any] = {}
    triggered = False
    for name in reference:
        if name not in current:
            continue
        p = psi(reference[name], current[name])
        ks = ks_drift(reference[name], current[name])
        feat_alert = p >= psi_alert
        triggered = triggered or feat_alert
        features[name] = {"psi": round(p, 4), "psi_band": psi_band(p), **ks,
                          "alert": feat_alert}
    return {
        "features": features,
        "retrain_recommended": triggered,
        "summary": ("Drift significativo detectado — revisar/retreinar política."
                    if triggered else "Sem drift significativo."),
    }
