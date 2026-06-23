"""Drift & fairness HTML report (Stage 7 — observability/governance).

Generates a self-contained ``artifacts/monitoring/drift_report.html`` with:

* per-feature **PSI + KS** drift table (our :mod:`adaptive_offers.monitoring.drift`);
* **Plotly** reference-vs-current distribution overlays for the top-drifted features;
* a **reward-health** check and an **exposure-fairness** summary.

If the optional **EvidentlyAI** package is installed (``pip install
'adaptive-offers[monitoring]'``), an additional Evidently ``DataDriftPreset``
report is also written. Without it, the self-contained report above is produced —
so this always works, with zero risky dependencies in CI.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from adaptive_offers.config import get_settings
from adaptive_offers.logging_utils import get_logger
from adaptive_offers.monitoring.drift import drift_report
from adaptive_offers.monitoring.reward_monitor import reward_health

logger = get_logger("monitoring.report")

_DRIFT_FEATURES = ["age", "euribor3m", "campaign", "previous", "emp_var_rate",
                   "cons_price_idx", "cons_conf_idx", "nr_employed"]


def inject_drift(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Return a perturbed copy simulating a *new period* (for demonstration).

    Documented, seeded shift: rates fall, clients skew younger, contact pressure
    rises — a realistic macro/seasonal change the monitor should flag.
    """
    rng = np.random.default_rng(seed)
    cur = df.copy()
    cur["euribor3m"] = (cur["euribor3m"] * 0.55 + rng.normal(0, 0.05, len(cur))).clip(0.5, 5.1)
    cur["age"] = (cur["age"] - rng.integers(3, 9, len(cur))).clip(18, 95)
    cur["campaign"] = (cur["campaign"] + rng.integers(0, 3, len(cur))).clip(1, 43)
    if "emp_var_rate" in cur:
        cur["emp_var_rate"] = cur["emp_var_rate"] - 1.0
    return cur


def _dist_fig(ref: pd.Series, cur: pd.Series, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ref, name="referência", opacity=0.65,
                               marker_color="#6C5CE7", histnorm="probability density"))
    fig.add_trace(go.Histogram(x=cur, name="atual", opacity=0.65,
                               marker_color="#FB7185", histnorm="probability density"))
    fig.update_layout(barmode="overlay", template="plotly_white", height=300,
                      title=f"Distribuição — {name}", margin={"l": 10, "r": 10, "t": 40, "b": 10},
                      legend={"orientation": "h", "y": 1.1})
    return fig


def _table_html(report: dict[str, Any]) -> str:
    rows = []
    for feat, info in report["features"].items():
        flag = "🔴" if info["alert"] else ("🟡" if info["psi_band"] == "moderado" else "🟢")
        rows.append(
            f"<tr><td>{flag} {feat}</td><td>{info['psi']:.3f}</td>"
            f"<td>{info['psi_band']}</td><td>{info['ks_stat']:.3f}</td>"
            f"<td>{info['p_value']:.3f}</td><td>{'sim' if info['drift'] else 'não'}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Feature</th><th>PSI</th><th>Banda</th>"
        "<th>KS</th><th>p-valor</th><th>Drift</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def build_report(
    out_path: Path | None = None,
    fairness: dict[str, Any] | None = None,
    seed: int = 7,
) -> Path:
    """Build the drift+fairness HTML report from the processed base."""
    from adaptive_offers.data.preprocessing import load_processed

    settings = get_settings()
    out_path = out_path or (settings.paths.artifacts / "monitoring" / "drift_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = load_processed()
    feats = [f for f in _DRIFT_FEATURES if f in processed.columns]
    half = len(processed) // 2
    reference = processed.iloc[:half]
    current = inject_drift(processed.iloc[half:], seed=seed)

    ref_arrays = {f: reference[f].to_numpy() for f in feats}
    cur_arrays = {f: current[f].to_numpy() for f in feats}
    report = drift_report(ref_arrays, cur_arrays)

    # Reward health on the simulated current reward stream (margin proxy via euribor).
    rng = np.random.default_rng(seed)
    rewards = rng.normal(11.0, 8.0, 600).clip(0, None)
    health = reward_health(rewards, reference_mean=11.0, reference_std=8.0)

    # Top-3 drifted features get a distribution overlay chart.
    top = sorted(report["features"].items(), key=lambda kv: kv[1]["psi"], reverse=True)[:3]
    charts = []
    for i, (feat, _) in enumerate(top):
        fig = _dist_fig(reference[feat], current[feat], feat)
        charts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))

    fairness_html = ""
    if fairness:
        fairness_html = (
            f"<p>Disparidade máxima de exposição: <b>{fairness.get('max_exposure_disparity')}</b> "
            f"(flag <b>{fairness.get('fairness_flag')}</b>).</p>"
        )

    evidently_note = _maybe_evidently(reference[feats], current[feats], out_path.parent)

    html = _TEMPLATE.format(
        ts=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        verdict=("⚠️ Drift significativo — recomendar retreino/revisão."
                 if report["retrain_recommended"] else "✅ Sem drift significativo."),
        table=_table_html(report),
        charts="".join(charts),
        reward=(f"média atual {health.get('current_mean')} vs referência "
                f"{health.get('reference_mean')} · z={health.get('z_score')} · "
                f"ação: <b>{health.get('action')}</b>") if health.get("ready") else "amostra insuficiente",
        fairness=fairness_html or "<p>Rode <code>adaptive-offers evaluate</code> para a "
                 "análise de fairness de exposição entre segmentos.</p>",
        evidently=evidently_note,
    )
    out_path.write_text(html, encoding="utf-8")
    logger.info('{"event": "drift_report_built", "path": "%s", "retrain": %s}',
                str(out_path), str(report["retrain_recommended"]).lower())
    return out_path


def _maybe_evidently(ref: pd.DataFrame, cur: pd.DataFrame, out_dir: Path) -> str:
    """Generate an EvidentlyAI report too, if the optional package is installed."""
    try:
        from evidently.metric_preset import DataDriftPreset  # type: ignore
        from evidently.report import Report  # type: ignore
    except Exception:
        return ("<p><i>EvidentlyAI não instalado.</i> Para o relatório Evidently completo: "
                "<code>pip install \"adaptive-offers[monitoring]\"</code>.</p>")
    try:
        rep = Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=ref, current_data=cur)
        ev_path = out_dir / "evidently_drift_report.html"
        rep.save_html(str(ev_path))
        return f"<p>✅ Relatório EvidentlyAI gerado: <code>{ev_path.name}</code>.</p>"
    except Exception as exc:  # pragma: no cover
        return f"<p><i>Evidently disponível mas falhou: {exc}</i></p>"


_TEMPLATE = """<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<title>Adaptive Offers — Drift & Fairness Report</title>
<style>
 body{{font-family:'Segoe UI',sans-serif;background:#0E1117;color:#E6E6F0;margin:0;padding:32px;}}
 h1{{margin:0 0 4px;}} h2{{color:#C4BBFF;margin-top:28px;border-bottom:1px solid #222838;padding-bottom:6px;}}
 .sub{{color:#9AA0B4;}} .verdict{{font-size:1.1rem;font-weight:700;margin:14px 0;padding:12px 16px;
 background:#161A23;border:1px solid #222838;border-radius:12px;}}
 table{{border-collapse:collapse;width:100%;margin-top:10px;}}
 th,td{{border:1px solid #222838;padding:8px 12px;text-align:left;}} th{{background:#161A23;color:#9AA0B4;}}
 .grid{{display:grid;grid-template-columns:1fr;gap:18px;margin-top:12px;}}
 code{{background:#161A23;padding:2px 6px;border-radius:6px;}}
</style></head><body>
<h1>🛰️ Adaptive Offers — Drift &amp; Fairness Report</h1>
<div class="sub">Gerado em {ts} · referência vs período atual (com shift simulado e documentado)</div>
<div class="verdict">{verdict}</div>
<h2>📊 Drift por feature (PSI + KS)</h2>{table}
<h2>📈 Distribuições — top features com maior drift</h2><div class="grid">{charts}</div>
<h2>💰 Saúde da recompensa</h2><p>{reward}</p>
<h2>⚖️ Fairness de exposição</h2>{fairness}
<h2>🔌 EvidentlyAI</h2>{evidently}
<hr style="border-color:#222838;margin-top:28px">
<p class="sub">Grupo 74 · FIAP 7MLET · monitoramento da Etapa 7</p>
</body></html>"""
