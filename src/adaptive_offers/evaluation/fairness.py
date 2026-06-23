"""Exposure-fairness analysis across synthetic segments (Stage 4).

We never use protected attributes to decide. Instead we audit whether the
policy's **exposure** (which offers, and any-offer rate) is balanced across
segments. We cover both the **protected attributes** registered in
``responsible.PROTECTED_ATTRIBUTES`` (age band, marital, education) and the
synthetic segments (prior-success, channel). Large gaps flag that some group is
systematically denied value, a documented risk.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from adaptive_offers.bandits.base import Policy
from adaptive_offers.data.synthetic import (
    OfferArm,
    build_context_vector,
    eligible_arms,
)
from adaptive_offers.responsible import _AGE_BINS, _AGE_LABELS

CONTROL_ARM = "OFF_NONE"

# Segments audited: protected attributes first, then synthetic segments.
PROTECTED_SEGMENT_COLS = ("age_band", "marital", "education")
DEFAULT_SEGMENT_COLS = PROTECTED_SEGMENT_COLS + ("prior_success", "channel")


def _segment_frame(processed: pd.DataFrame) -> pd.DataFrame:
    seg = pd.DataFrame(index=processed.index)
    seg["age_band"] = pd.cut(
        processed["age"], bins=_AGE_BINS, labels=_AGE_LABELS,
    ).astype(str)
    # Protected proxies — audited for exposure parity, never used to decide.
    seg["marital"] = processed.get("marital", "unknown").astype(str)
    seg["education"] = processed.get("education", "unknown").astype(str)
    seg["prior_success"] = np.where(
        processed["poutcome"].astype(str).str.lower() == "success", "yes", "no"
    )
    seg["channel"] = processed["contact"].astype(str)
    return seg


def exposure_report(
    policy: Policy,
    processed: pd.DataFrame,
    catalog: list[OfferArm],
    rate_median: float,
    segment_cols: tuple[str, ...] = DEFAULT_SEGMENT_COLS,
) -> dict[str, Any]:
    """Per-segment exposure fairness: any-offer rate, offer mix, *and value parity*.

    ``any_offer_rate`` can saturate (everyone gets some offer). The discriminating
    signal is **value parity**: whether a protected group systematically receives
    *lower-margin* offers. We report mean chosen-offer margin per group and its
    relative gap (``value_disparity`` = (max−min)/max), which drives the flag.
    """
    seg = _segment_frame(processed)
    margin_by_id = {a.offer_id: float(a.margin) for a in catalog}
    chosen: list[str] = []
    for i in range(len(processed)):
        row = processed.iloc[i]
        elig = [a.offer_id for a in eligible_arms(row, catalog)]
        ctx = build_context_vector(row, rate_median)
        chosen.append(policy.select(ctx, elig).arm_id)
    df = seg.copy()
    df["chosen"] = chosen
    df["got_offer"] = (df["chosen"] != CONTROL_ARM).astype(int)
    df["offer_margin"] = df["chosen"].map(margin_by_id).fillna(0.0)

    report: dict[str, Any] = {"policy": policy.name, "segments": {}}
    for col in segment_cols:
        grp = df.groupby(col)
        any_offer = grp["got_offer"].mean().round(4)
        mean_margin = grp["offer_margin"].mean().round(2)
        mix = (
            df.groupby([col, "chosen"]).size()
            .groupby(level=0).apply(lambda s: (s / s.sum()).round(3))
        )
        disparity = float(any_offer.max() - any_offer.min())
        hi = float(mean_margin.max())
        value_disparity = float((hi - float(mean_margin.min())) / hi) if hi > 0 else 0.0
        report["segments"][col] = {
            "protected": col in PROTECTED_SEGMENT_COLS,
            "any_offer_rate": {str(k): float(v) for k, v in any_offer.items()},
            "mean_offer_margin": {str(k): float(v) for k, v in mean_margin.items()},
            "exposure_disparity": round(disparity, 4),  # demographic-parity-like gap
            "value_disparity": round(value_disparity, 4),  # relative margin gap
            "offer_mix": {str(k): float(v) for k, v in mix.items()},
        }
    # Overall fairness flag: worst gap across segment dimensions.
    worst = max(
        report["segments"][c]["exposure_disparity"] for c in segment_cols
    )
    protected_cols = [c for c in segment_cols if c in PROTECTED_SEGMENT_COLS]
    worst_protected = max(
        (report["segments"][c]["exposure_disparity"] for c in protected_cols),
        default=0.0,
    )
    worst_value_protected = max(
        (report["segments"][c]["value_disparity"] for c in protected_cols),
        default=0.0,
    )
    report["max_exposure_disparity"] = round(worst, 4)
    report["max_protected_disparity"] = round(worst_protected, 4)
    report["max_protected_value_disparity"] = round(worst_value_protected, 4)
    # Flag review if a protected group is denied contact (>0.25) or gets
    # systematically lower-value offers (relative margin gap >0.30).
    report["fairness_flag"] = (
        "review" if (worst_protected > 0.25 or worst_value_protected > 0.30) else "ok"
    )
    return report
