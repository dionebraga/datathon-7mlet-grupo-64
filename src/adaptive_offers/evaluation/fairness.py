"""Exposure-fairness analysis across synthetic segments (Stage 4).

We never use protected attributes to decide. Instead we audit whether the
policy's **exposure** (which offers, and any-offer rate) is balanced across
synthetic segments — age band, prior-success and channel. Large gaps would flag
that some segments are systematically denied value, a documented risk.
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

CONTROL_ARM = "OFF_NONE"


def _segment_frame(processed: pd.DataFrame) -> pd.DataFrame:
    seg = pd.DataFrame(index=processed.index)
    seg["age_band"] = pd.cut(
        processed["age"], bins=[17, 30, 45, 60, 100],
        labels=["<=30", "31-45", "46-60", "60+"],
    ).astype(str)
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
    segment_cols: tuple[str, ...] = ("age_band", "prior_success", "channel"),
) -> dict[str, Any]:
    """Compute per-segment any-offer rate + offer mix and exposure disparity."""
    seg = _segment_frame(processed)
    chosen: list[str] = []
    for i in range(len(processed)):
        row = processed.iloc[i]
        elig = [a.offer_id for a in eligible_arms(row, catalog)]
        ctx = build_context_vector(row, rate_median)
        chosen.append(policy.select(ctx, elig).arm_id)
    df = seg.copy()
    df["chosen"] = chosen
    df["got_offer"] = (df["chosen"] != CONTROL_ARM).astype(int)

    report: dict[str, Any] = {"policy": policy.name, "segments": {}}
    for col in segment_cols:
        grp = df.groupby(col)
        any_offer = grp["got_offer"].mean().round(4)
        mix = (
            df.groupby([col, "chosen"]).size()
            .groupby(level=0).apply(lambda s: (s / s.sum()).round(3))
        )
        disparity = float(any_offer.max() - any_offer.min())
        report["segments"][col] = {
            "any_offer_rate": {str(k): float(v) for k, v in any_offer.items()},
            "exposure_disparity": round(disparity, 4),  # demographic-parity-like gap
            "offer_mix": {str(k): float(v) for k, v in mix.items()},
        }
    # Overall fairness flag: worst gap across segment dimensions.
    worst = max(
        report["segments"][c]["exposure_disparity"] for c in segment_cols
    )
    report["max_exposure_disparity"] = round(worst, 4)
    report["fairness_flag"] = "review" if worst > 0.25 else "ok"
    return report
