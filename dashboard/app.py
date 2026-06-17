"""Observability-style BI dashboard for the Adaptive Offers Platform.

Course: ML Hands-On — "BI com análise". A dark, modern operations board:
big KPI tiles with sparklines, gauges, experiment panels, a live decision feed
(from the audit log) and an interactive decision explorer with reason codes and
an LLM/RAG explanation.

Run (PowerShell):  streamlit run dashboard\app.py   ->  http://localhost:8501
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_offers.bandits.registry import build_policy  # noqa: E402
from adaptive_offers.data.preprocessing import build_processed, load_processed  # noqa: E402
from adaptive_offers.data.quality import quality_report, target_rate_by  # noqa: E402
from adaptive_offers.data.synthetic import CONTEXT_FEATURES, generate  # noqa: E402
from adaptive_offers.simulation.environment import build_arms, run_simulation  # noqa: E402
from adaptive_offers.simulation.metrics import compare_results, regret_curve  # noqa: E402

# --------------------------------------------------------------------------- #
# Compat + helpers
# --------------------------------------------------------------------------- #
_ST_VER = tuple(int(p) for p in (st.__version__.split(".") + ["0", "0"])[:2])


def fill() -> dict:
    return {"width": "stretch"} if _ST_VER >= (1, 49) else {"use_container_width": True}


NO_BAR = {"displayModeBar": False}


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- #
# Theme constants
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Adaptive Offers · Observability", page_icon="🛰️", layout="wide")

BG = "#0E1117"
PANEL = "#161A23"
GRID = "#222838"
TEXT = "#E6E6F0"
MUTED = "#9AA0B4"
VIOLET, CYAN, GREEN, AMBER, RED = "#7C6CFF", "#22D3EE", "#34D399", "#FBBF24", "#FB7185"
POLICY_COLORS = {"linucb": VIOLET, "thompson": GREEN, "nilos_ucb": CYAN, "baseline": "#64748B"}
POLICY_LABEL = {"linucb": "LinUCB", "thompson": "Thompson", "nilos_ucb": "Nilos-UCB",
                "baseline": "Baseline"}

st.markdown(
    f"""
    <style>
      #MainMenu, footer, [data-testid="stToolbar"] {{visibility: hidden;}}
      .stApp {{background: {BG};}}
      .block-container {{padding-top: 1rem; padding-bottom: 2rem; max-width: 1360px;}}
      .topbar {{display:flex; align-items:center; justify-content:space-between;
        border:1px solid {GRID}; background:linear-gradient(90deg,#171B26,#12141C);
        border-radius:14px; padding:14px 20px; margin-bottom:14px;}}
      .topbar h1 {{margin:0;font-size:1.35rem;font-weight:800;color:{TEXT};letter-spacing:-.01em;}}
      .topbar .sub {{color:{MUTED};font-size:.82rem;margin-top:2px;}}
      .stat {{display:inline-block;padding:5px 12px;border-radius:999px;font-size:.76rem;
        font-weight:700;margin-left:6px;border:1px solid {GRID};}}
      .on {{color:{GREEN};background:rgba(52,211,153,.10);}}
      .off {{color:#64748B;background:rgba(100,116,139,.10);}}
      .sect {{color:{MUTED};font-size:.82rem;font-weight:700;text-transform:uppercase;
        letter-spacing:.08em;margin:26px 0 10px;border-left:3px solid {VIOLET};padding-left:10px;}}
      div[data-testid="column"] {{padding:0 5px;}}
      div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {{
        background:{PANEL}; border:1px solid {GRID}; border-radius:16px; padding:8px 10px;
        overflow:hidden; box-shadow:0 6px 22px rgba(0,0,0,.28);}}
      div[data-testid="stPlotlyChart"] > div, .js-plotly-plot, .plot-container {{
        overflow:hidden !important;}}
      div[data-testid="stPlotlyChart"]:hover {{border-color:#2E3650;}}
      .pill {{display:inline-block;padding:4px 11px;border-radius:999px;
        background:rgba(124,108,255,.14);color:#C4BBFF;font-size:.74rem;font-weight:700;margin:2px 5px 2px 0;}}
      .result {{background:linear-gradient(135deg,#171B26,#12141C);border:1px solid {GRID};
        border-radius:16px;padding:20px 24px;}}
      .result .arm {{font-size:1.5rem;font-weight:800;color:#C4BBFF;}}
      .svc {{font-size:.82rem;padding:5px 0;color:{TEXT};}}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Experiment (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Treinando políticas e simulando o experimento…")
def load_experiment(horizon: int, seed: int):
    proc_path = ROOT / "data" / "processed" / "bank_marketing_processed.parquet"
    needs_build = not proc_path.exists()
    if not needs_build:
        try:
            needs_build = len(pd.read_parquet(proc_path, columns=["client_event_id"])) < 20_000
        except Exception:
            needs_build = True
    if needs_build:
        build_processed(n_rows=20_000, seed=42)
    processed = load_processed()
    bundle = generate(processed=processed, seed=42)
    arms = build_arms(bundle.catalog)
    results = {}
    for name in ("baseline", "thompson", "nilos_ucb", "linucb"):
        pol = build_policy(name, arms, context_dim=len(CONTEXT_FEATURES), seed=seed)
        results[name] = run_simulation(pol, processed, bundle, horizon=horizon, seed=seed)
    return processed, bundle, results


def downsample(arr, points: int = 48):
    a = np.asarray(arr, dtype=float)
    if len(a) <= points:
        return a
    return a[np.linspace(0, len(a) - 1, points).astype(int)]


def tile(col, label: str, value: str, series, color: str) -> None:
    s = downsample(series)
    fig = go.Figure(go.Scatter(
        y=s, mode="lines", line={"color": color, "width": 2.4, "shape": "spline", "smoothing": 0.7},
        fill="tozeroy", fillcolor=hex_rgba(color, 0.12), hoverinfo="skip"))
    fig.update_layout(
        height=172, margin={"l": 10, "r": 10, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[
            {"text": label.upper(), "x": 0.02, "y": 1.20, "xref": "paper", "yref": "paper",
             "showarrow": False, "xanchor": "left",
             "font": {"size": 12, "color": MUTED, "family": "Inter"}},
            {"text": value, "x": 0.02, "y": 0.60, "xref": "paper", "yref": "paper",
             "showarrow": False, "xanchor": "left",
             "font": {"size": 38, "color": color, "family": "Inter"}},
        ],
    )
    col.plotly_chart(fig, config=NO_BAR, **fill())


def gauge(value: float, title: str, vmax: float, color: str, suffix: str = "") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": suffix, "font": {"size": 36, "color": TEXT, "family": "Inter"}},
        title={"text": title, "font": {"size": 13, "color": MUTED}},
        gauge={
            "axis": {"range": [0, vmax], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)",
                     "nticks": 4, "tickfont": {"color": MUTED, "size": 9}},
            "bar": {"color": color, "thickness": 0.30, "line": {"width": 0}},
            "bgcolor": "rgba(255,255,255,.05)", "borderwidth": 0,
            "threshold": {"line": {"color": "#FFFFFF", "width": 2}, "thickness": 0.8, "value": value},
        },
    ))
    fig.update_layout(height=236, margin={"l": 26, "r": 26, "t": 46, "b": 10},
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT})
    return fig


def style_panel(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        title={"text": title, "font": {"size": 14, "color": TEXT}, "x": 0.01, "xanchor": "left"},
        margin={"l": 14, "r": 14, "t": 46, "b": 12}, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font={"family": "Inter", "color": TEXT},
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"size": 11}},
        hoverlabel={"bgcolor": PANEL, "bordercolor": GRID, "font_size": 12},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", zeroline=False, showline=False)
    return fig


def recent_decisions(n: int = 8) -> pd.DataFrame:
    p = ROOT / "artifacts" / "decisions" / "audit.jsonl"
    if not p.exists():
        return pd.DataFrame()
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()][-n:]
    rows = [json.loads(ln) for ln in lines]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).iloc[::-1]
    return pd.DataFrame({
        "Hora": df["ts"].str.slice(11, 19) if "ts" in df else "",
        "Oferta": df.get("arm_name", ""),
        "Modo": df["explored"].map({True: "🔍 expl.", False: "🎯 explot."}) if "explored" in df else "",
        "Valor": df["expected_reward"].map(lambda v: f"R$ {float(v):.1f}") if "expected_reward" in df else "",
    })


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🛰️ Adaptive Offers")
    st.caption("FIAP 7MLET · Grupo 64 · Observability")
    st.divider()
    horizon = st.select_slider("Horizonte de simulação",
                               options=[2000, 4000, 6000, 8000, 12000, 20000], value=6000)
    seed = st.number_input("Seed", value=123, step=1)
    st.divider()
    st.markdown("##### 🔌 Serviços")
    api_up, mlf_up = port_open(8000), port_open(5000)

    def badge(up):  # noqa: ANN001
        return '<span class="stat on">● online</span>' if up else '<span class="stat off">● offline</span>'

    st.markdown('<div class="svc">API REST ' + badge(api_up)
                + '<br/><code>adaptive-offers serve</code></div>', unsafe_allow_html=True)
    st.markdown('<div class="svc">MLflow ' + badge(mlf_up)
                + '<br/><code>mlflow ui</code></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("##### 📌 Como ler o board")
    st.markdown(
        f'<div style="font-size:.82rem;color:{MUTED};line-height:1.9">'
        f'↑ <b style="color:{TEXT}">Reward</b> — valor capturado (margem×conversão)<br/>'
        f'↓ <b style="color:{TEXT}">Regret</b> — distância do ótimo<br/>'
        f'<b style="color:{TEXT}">Lift</b> — ganho vs política fixa (baseline)<br/>'
        '🔍 exploração · 🎯 explotação</div>', unsafe_allow_html=True)

processed, bundle, results = load_experiment(int(horizon), int(seed))
summary = compare_results(list(results.values()))
sdf = pd.DataFrame(summary)
best = summary[0]
best_res = results[best["policy"]]
base_res = results["baseline"]
qrep = quality_report(processed)

# --------------------------------------------------------------------------- #
# Topbar
# --------------------------------------------------------------------------- #
api_up, mlf_up = port_open(8000), port_open(5000)
st.markdown(
    '<div class="topbar"><div><h1>🛰️ Adaptive Offers — Observability Board</h1>'
    '<div class="sub">Multi-armed bandit · decisão de ofertas financeiras · '
    f'política ativa <b>{POLICY_LABEL.get(best["policy"], best["policy"])}</b></div></div>'
    f'<div><span class="stat {"on" if api_up else "off"}">API {"●" if api_up else "○"}</span>'
    f'<span class="stat {"on" if mlf_up else "off"}">MLflow {"●" if mlf_up else "○"}</span>'
    '<span class="stat on">BI ●</span></div></div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Row 1 — big KPI tiles with sparklines
# --------------------------------------------------------------------------- #
st.markdown('<div class="sect">⚡ Métricas em tempo real</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
conv_curve = np.cumsum(best_res.converted) / (np.arange(len(best_res.converted)) + 1)
lift_curve = best_res.cumulative_reward - base_res.cumulative_reward
tile(k1, "Reward / 1k impressões", f"R$ {best['reward_per_1k']:,.0f}", best_res.cumulative_reward, VIOLET)
tile(k2, "Regret ratio", f"{best['regret_ratio']:.1%}", best_res.cumulative_regret, RED)
tile(k3, "Conversão", f"{best['conversion_rate']:.1%}", conv_curve, CYAN)
tile(k4, "Lift vs baseline", f"+{best.get('lift_vs_baseline_pct', 0):.0f}%", lift_curve, GREEN)

# --------------------------------------------------------------------------- #
# Dense panel grid (New Relic style) — compact builders
# --------------------------------------------------------------------------- #
GRID_H = 250
PLABELS = [POLICY_LABEL.get(p, p) for p in sdf["policy"]]
PCOLORS = [POLICY_COLORS.get(p, VIOLET) for p in sdf["policy"]]


def p_hbar(value_col: str, title: str, money: bool = False) -> go.Figure:
    o = sdf.sort_values(value_col)
    txt = [f"R$ {v:,.0f}" if money else (f"{v:.1%}" if v < 1 else f"{v:,.1f}") for v in o[value_col]]
    fig = go.Figure(go.Bar(x=o[value_col], y=[POLICY_LABEL.get(p, p) for p in o["policy"]],
                           orientation="h", marker_color=[POLICY_COLORS.get(p, VIOLET) for p in o["policy"]],
                           text=txt, textposition="auto"))
    return style_panel(fig, title, height=GRID_H)


def p_vbar(x, y, title: str, color: str, pct: bool = True) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker_color=color,
                           text=[f"{v:.1%}" if pct else f"{v:,.0f}" for v in y], textposition="outside"))
    return style_panel(fig, title, height=GRID_H)


def p_donut(labels, values, title: str, colors) -> go.Figure:
    fig = go.Figure(go.Pie(labels=list(labels), values=list(values), hole=.62,
                           marker={"colors": colors}, textinfo="percent"))
    return style_panel(fig, title, height=GRID_H).update_layout(showlegend=False)


def p_stat(col, title: str, rows: list[tuple[str, str]]) -> None:
    items = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
        f'border-bottom:1px solid {GRID}"><span style="color:{MUTED};font-size:.85rem">{lab}</span>'
        f'<span style="font-weight:700;color:{TEXT}">{val}</span></div>' for lab, val in rows)
    col.markdown(
        f'<div style="background:{PANEL};border:1px solid {GRID};border-radius:16px;'
        f'padding:14px 16px;height:{GRID_H}px;box-shadow:0 6px 22px rgba(0,0,0,.28)">'
        f'<div style="color:{TEXT};font-weight:700;font-size:14px;margin-bottom:6px">{title}</div>'
        f'{items}</div>', unsafe_allow_html=True)


# pre-computed series
seg = processed.copy()
seg["age_band"] = pd.cut(seg["age"], [17, 30, 45, 60, 100], labels=["≤30", "31-45", "46-60", "60+"])
by_age = seg.groupby("age_band", observed=True)["subscribed"].mean()
by_pout = target_rate_by(processed, "poutcome")["subscription_rate"]
by_contact = target_rate_by(processed, "contact")["subscription_rate"]
pulls = pd.Series(best_res.arm_pulls)
pulls = pulls[pulls > 0].sort_values(ascending=False)
regret_fig = go.Figure()
for name, res in results.items():
    idx, cum = regret_curve(res, points=70)
    regret_fig.add_trace(go.Scatter(x=idx, y=cum, mode="lines", name=POLICY_LABEL.get(name, name),
                                    line={"width": 2.4, "color": POLICY_COLORS.get(name, VIOLET),
                                          "shape": "spline", "smoothing": 0.5}))

# --- Grid row A: gauges ---------------------------------------------------- #
st.markdown('<div class="sect">🎛️ Indicadores</div>', unsafe_allow_html=True)
a1, a2, a3, a4 = st.columns(4)
a1.plotly_chart(gauge(best["regret_ratio"] * 100, "Regret ratio", 50, RED, "%"), config=NO_BAR, **fill())
a2.plotly_chart(gauge(best["exploration_rate"] * 100, "Exploração", 30, VIOLET, "%"), config=NO_BAR, **fill())
a3.plotly_chart(gauge(best["conversion_rate"] * 100, "Conversão", 20, CYAN, "%"), config=NO_BAR, **fill())
a4.plotly_chart(gauge(best.get("lift_vs_baseline_pct", 0), "Lift vs baseline", 100, GREEN, "%"),
                config=NO_BAR, **fill())

# --- Grid row B: experiment ------------------------------------------------ #
st.markdown('<div class="sect">📊 Experimento</div>', unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)
b1.plotly_chart(p_hbar("cumulative_reward", "💰 Valor por política", money=True), config=NO_BAR, **fill())
b2.plotly_chart(style_panel(regret_fig, "📉 Regret acumulado", height=GRID_H), config=NO_BAR, **fill())
b3.plotly_chart(p_donut(pulls.index, pulls.values, "🎯 Mix de ofertas", px.colors.sequential.Purp[::-1]),
                config=NO_BAR, **fill())
b4.plotly_chart(p_hbar("reward_per_1k", "⚡ Reward / 1k", money=True), config=NO_BAR, **fill())

# --- Grid row C: dataset signals ------------------------------------------- #
st.markdown('<div class="sect">🧬 Sinais da base</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.plotly_chart(p_vbar(by_pout.index, by_pout.values, "Conversão · poutcome", CYAN), config=NO_BAR, **fill())
c2.plotly_chart(p_vbar(by_contact.index, by_contact.values, "Conversão · canal", GREEN), config=NO_BAR, **fill())
c3.plotly_chart(p_vbar(by_age.index.astype(str), by_age.values, "Conversão · faixa etária", AMBER),
                config=NO_BAR, **fill())
c4.plotly_chart(p_donut([POLICY_LABEL.get(p, p) for p in sdf["policy"]], sdf["conversion_rate"],
                        "Conversão por política", PCOLORS), config=NO_BAR, **fill())

# --- Grid row D: comparison + ops ------------------------------------------ #
st.markdown('<div class="sect">🧭 Comparação & operação</div>', unsafe_allow_html=True)
d1, d2, d3, d4 = st.columns(4)
d1.plotly_chart(p_hbar("exploration_rate", "🔍 Exploração por política"), config=NO_BAR, **fill())
d2.plotly_chart(p_hbar("regret_ratio", "🎯 Regret ratio por política"), config=NO_BAR, **fill())
p_stat(d3, "📦 Base factual", [
    ("Registros", f"{qrep['n_rows']:,}"),
    ("Conversão", f"{qrep['target']['positive_rate']:.1%}"),
    ("Desbalanceamento", f"{qrep['target']['imbalance_ratio']:.0f}:1"),
    ("Duplicatas", str(qrep["n_duplicates"])),
    ("Ofertas (braços)", str(len(bundle.catalog))),
])
with d4:
    st.markdown('<div class="sect" style="margin-top:0">📡 Feed de decisões</div>', unsafe_allow_html=True)
    feed = recent_decisions(6)
    if feed.empty:
        st.caption("Sem decisões ainda — use o explorador abaixo.")
    else:
        st.dataframe(feed, hide_index=True, height=GRID_H - 40, **fill())

# --------------------------------------------------------------------------- #
# Decision explorer
# --------------------------------------------------------------------------- #
st.markdown('<div class="sect">🧪 Explorador de decisão</div>', unsafe_allow_html=True)
col = st.columns(3)
age = col[0].slider("Idade", 18, 95, 66)
contact = col[0].selectbox("Canal", ["cellular", "telephone"])
poutcome = col[1].selectbox("Resultado anterior", ["nonexistent", "failure", "success"], index=2)
euribor = col[1].slider("Euribor 3m (juros)", 0.6, 5.1, 0.8)
default = col[2].selectbox("Em default?", ["no", "yes", "unknown"])
loan = col[2].selectbox("Tem empréstimo?", ["no", "yes", "unknown"])
prev = 1 if poutcome != "nonexistent" else 0

if st.button("🚀 Decidir oferta", type="primary", **fill()):
    with st.spinner("Decidindo…"):
        from adaptive_offers.assistant import Assistant
        from adaptive_offers.bootstrap import ensure_service
        from adaptive_offers.data.synthetic import (
            build_context_vector,
            eligible_arms,
            expected_reward,
            is_eligible,
            latent_conversion_prob,
            offer_catalog,
        )

        ctx = {"age": age, "contact": contact, "poutcome": poutcome, "euribor3m": euribor,
               "default": default, "loan": loan, "previously_contacted": prev}
        svc = ensure_service(train_if_missing=True)
        rec = svc.decide(context=ctx, log=True)
        exp = Assistant().explain_decision(rec.to_dict())

        cat = offer_catalog()
        by_id = {a.offer_id: a for a in cat}
        rate_median = float(svc.fs.get_metadata("rate_median", 2.5)) if svc.fs.is_materialized() else 2.5
        ctxv = build_context_vector(ctx, rate_median)
        elig_ids = {a.offer_id for a in eligible_arms(ctx, cat)}
        rows = []
        for a in cat:
            p = latent_conversion_prob(a, ctxv)
            rows.append({"id": a.offer_id, "Oferta": a.name, "p": p, "Margem": a.margin,
                         "Valor": expected_reward(a, ctxv), "Elegível": a.offer_id in elig_ids,
                         "Escolhida": a.offer_id == rec.arm_id})
        bdf = pd.DataFrame(rows).sort_values("Valor", ascending=False)
        bdf_e = bdf[bdf["Elegível"]]

    # --- headline card -----------------------------------------------------
    mode_txt = "🔍 Exploração (testa alternativa)" if rec.explored else "🎯 Explotação (melhor estimativa)"
    pills = " ".join(f'<span class="pill">{c}</span>' for c in rec.reason_codes)
    st.markdown(
        f'<div class="result"><div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div class="arm">🎁 {rec.arm_name}</div>'
        f'<div style="text-align:right"><div style="font-size:1.6rem;font-weight:800;color:{GREEN}">'
        f'R$ {rec.expected_reward:.1f}</div><div style="color:{MUTED};font-size:.78rem">valor esperado</div></div></div>'
        f'<div style="color:{MUTED};margin:6px 0 12px">{mode_txt} · '
        f'política <b style="color:{TEXT}">{rec.policy_name}@{rec.policy_version}</b> · '
        f'{len(elig_ids)} de {len(cat)} ofertas elegíveis</div>{pills}</div>',
        unsafe_allow_html=True)
    st.write("")

    # --- value breakdown chart + detail table ------------------------------
    e1, e2 = st.columns([3, 2])
    colors = [GREEN if c else (VIOLET if el else "#3A3F52")
              for c, el in zip(bdf_e["Escolhida"], bdf_e["Elegível"], strict=False)]
    fig = go.Figure(go.Bar(
        x=bdf_e["Valor"], y=bdf_e["Oferta"], orientation="h", marker_color=colors,
        text=[f"R$ {v:.1f}{' ✅' if c else ''}" for v, c in zip(bdf_e["Valor"], bdf_e["Escolhida"], strict=False)],
        textposition="auto", hovertemplate="%{y}: R$ %{x:.1f}<extra></extra>"))
    e1.plotly_chart(style_panel(fig, "📊 Valor esperado por oferta elegível (margem × P conversão)",
                                height=300), config=NO_BAR, **fill())
    with e2:
        st.markdown("**🧾 Detalhe por oferta**")
        show = bdf_e[["Oferta", "p", "Margem", "Valor", "Escolhida"]].copy()
        show["P(conv)"] = (show["p"] * 100).round(1).astype(str) + "%"
        show["Margem"] = "R$ " + show["Margem"].astype(int).astype(str)
        show["Valor"] = "R$ " + show["Valor"].round(1).astype(str)
        show["✓"] = show["Escolhida"].map({True: "✅", False: ""})
        st.dataframe(show[["Oferta", "P(conv)", "Margem", "Valor", "✓"]], hide_index=True, **fill())

    # --- client profile + chosen offer details -----------------------------
    f1, f2, f3 = st.columns(3)
    p_stat(f1, "👤 Perfil do cliente", [
        ("Idade", str(age)), ("Canal", contact), ("Resultado anterior", poutcome),
        ("Em default", default), ("Tem empréstimo", loan), ("Euribor 3m", f"{euribor}")])
    arm = by_id[rec.arm_id]
    rules = []
    if arm.requires_no_default:
        rules.append("sem default")
    if arm.requires_no_loan:
        rules.append("sem empréstimo")
    if arm.min_age:
        rules.append(f"idade ≥ {arm.min_age}")
    p_stat(f2, f"🎯 {rec.arm_name}", [
        ("ID", rec.arm_id), ("Categoria", arm.category), ("Margem", f"R$ {arm.margin:.0f}"),
        ("Suitability", arm.suitability_tier), ("Regras", ", ".join(rules) or "nenhuma"),
        ("Elegível agora", "✅" if is_eligible(arm, ctx) else "❌")])
    with f3:
        st.markdown("**🧠 Por que esta decisão**")
        for r in rec.reasons:
            st.markdown(f"<div style='font-size:.86rem;margin:3px 0'><b style='color:#C4BBFF'>"
                        f"{r['code']}</b> — {r['description']}</div>", unsafe_allow_html=True)

    # --- assistant (RAG) ---------------------------------------------------
    st.markdown("##### 🤖 Explicação do assistente (LLM + RAG)")
    st.markdown(f'<div class="result">{exp["answer"]}</div>', unsafe_allow_html=True)
    with st.expander("📄 Citações de política comercial (RAG)"):
        for c in exp["citations"]:
            st.markdown(f"- **`{c['source']}`** · relevância {c.get('score', 0)} — {c['text']}")

st.divider()
st.markdown(
    f'<div style="text-align:center;color:{MUTED};font-size:.82rem;padding:8px 0 4px">'
    f'<b style="color:{TEXT}">Adaptive Offers Platform</b> · © 2026 '
    f'<b style="color:#C4BBFF">Dione Braga</b> — Grupo 64 · FIAP Pós-Tech 7MLET'
    '<br/><span style="font-size:.76rem">Licença MIT · '
    '<a href="https://github.com/dionebraga/datathon-7mlet-grupo-64" '
    'style="color:#9AA0B4;text-decoration:none">github.com/dionebraga/datathon-7mlet-grupo-64</a>'
    '</span></div>', unsafe_allow_html=True)
