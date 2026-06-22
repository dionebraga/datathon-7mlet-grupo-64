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
from adaptive_offers.bandits.thompson_linear import LinThompson  # noqa: E402
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


NO_BAR = {"displayModeBar": False, "scrollZoom": False}


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _md2html(text: str) -> str:
    """Convert subset of markdown to HTML for inline display inside st.markdown divs."""
    import re as _re
    # ### Section headers → refined accent header (dot + label + fading divider)
    text = _re.sub(
        r"^#{1,4}\s+(.+)$",
        r'<div style="display:flex;align-items:center;gap:9px;margin:18px 0 9px;'
        r'break-inside:avoid;-webkit-column-break-inside:avoid">'
        r'<span style="width:4px;height:13px;border-radius:2px;flex-shrink:0;'
        r'background:linear-gradient(180deg,#1A6FFF,#0033CC);'
        r'box-shadow:0 0 7px rgba(26,111,255,.55)"></span>'
        r'<span style="color:#1A6FFF;font-size:.71rem;font-weight:800;letter-spacing:.09em;'
        r'text-transform:uppercase;white-space:nowrap">\1</span>'
        r'<span style="flex:1;height:1px;'
        r'background:linear-gradient(90deg,rgba(26,111,255,.35),rgba(255,255,255,0))"></span>'
        r'</div>',
        text, flags=_re.MULTILINE,
    )
    # checkmark bullets ("- ✓ x" / "• ✓ x") → green-check row
    text = _re.sub(
        r"^[•\-]\s*✓\s*(.+)$",
        r'<div style="display:flex;align-items:flex-start;gap:9px;margin:5px 0 5px 3px;'
        r'break-inside:avoid;-webkit-column-break-inside:avoid">'
        r'<span style="color:#1A9E1A;font-weight:800;flex-shrink:0;line-height:1.5">✓</span>'
        r'<span style="flex:1">\1</span></div>',
        text, flags=_re.MULTILINE,
    )
    # generic bullets ("- x" / "• x") → cyan-arrow row
    text = _re.sub(
        r"^[•\-]\s+(.+)$",
        r'<div style="display:flex;align-items:flex-start;gap:9px;margin:5px 0 5px 3px;'
        r'break-inside:avoid;-webkit-column-break-inside:avoid">'
        r'<span style="color:#1A6FFF;font-weight:800;flex-shrink:0;line-height:1.5">▸</span>'
        r'<span style="flex:1">\1</span></div>',
        text, flags=_re.MULTILINE,
    )
    # horizontal rule ---
    text = _re.sub(
        r"^---$",
        r'<hr style="border:none;border-top:1px solid rgba(255,255,255,.08);margin:10px 0">',
        text, flags=_re.MULTILINE,
    )
    # bold **text**
    text = _re.sub(r"\*\*(.+?)\*\*", r'<b style="color:#EDEDED">\1</b>', text)
    # inline code `text`
    text = _re.sub(
        r"`(.+?)`",
        r'<span style="font-family:monospace;font-size:.78rem;color:#1A6FFF;'
        r'background:rgba(26,111,255,.12);padding:1px 6px;border-radius:4px;'
        r'border:1px solid rgba(26,111,255,.22)">\1</span>',
        text,
    )
    # italic citation _(text)_
    text = _re.sub(
        r"_\((.+?)\)_",
        r'<span style="opacity:.55;font-size:.73rem">(\1)</span>',
        text,
    )
    # paragraph breaks → HTML br
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    # collapse <br>s adjacent to our block-level rows (headers/bullets) to avoid huge gaps
    text = _re.sub(r"(</div>)(<br>)+", r"\1", text)
    text = _re.sub(r"(<br>)+(<div style=\"display:flex)", r"\2", text)
    return text


# --------------------------------------------------------------------------- #
# Theme constants
# --------------------------------------------------------------------------- #
_FAVICON = ROOT / "frontend" / "public" / "logo.svg"
st.set_page_config(page_title="Adaptive Offers · Observability",
                   page_icon=str(_FAVICON) if _FAVICON.exists() else "🛰️", layout="wide",
                   initial_sidebar_state="expanded")

# Paleta: dark navy base + electric blue + vivid green + gold + orange-red
# Complementa o visual dark com a paleta BR (azul marinho, verde, ouro, laranja).
BG     = "#000000"      # fundo puro
PANEL  = "#030D24"      # navy panel (novo — dá profundidade)
PANEL2 = "#010A1A"      # navy mais escuro
GRID   = "#0D1F42"      # navy grid lines
TEXT   = "#EDEDED"
MUTED  = "#8899BB"      # azul-acinzentado (novo)

# Cores primárias — paleta nova integrada
VIOLET    = "#0033CC"   # azul royal profundo
CYAN      = "#1A6FFF"   # azul elétrico (entre royal e neon)
GREEN     = "#1A9E1A"   # verde vívido (novo)
LIME      = "#A0C830"   # verde-limão (novo — exploração)
GOLD      = "#FFC200"   # amarelo-ouro (novo — destaque de valor)
AMBER     = "#FF9A00"   # laranja quente (novo)
RED       = "#E84000"   # laranja-vermelho (novo)
ACCENT_LT = "#1A6FFF"   # azul claro para pills

# Políticas: cada uma com uma cor distinta da nova paleta
POLICY_COLORS = {
    "linucb":        VIOLET,    # azul royal — LinUCB (UCB determinístico)
    "lin_thompson":  CYAN,      # azul elétrico — LinThompson (Bayesiano linear)
    "thompson":      GREEN,     # verde — Thompson Beta-Bernoulli
    "nilos_ucb":     GOLD,      # ouro — UCB variance-aware
    "baseline":      "#3A4A6B", # azul-cinza neutro
}
POLICY_LABEL = {
    "linucb":       "LinUCB",
    "lin_thompson": "LinThompson",
    "thompson":     "Thompson",
    "nilos_ucb":    "Nilos-UCB",
    "baseline":     "Baseline",
}


def logo_svg(size: int = 34, gid: str = "lg") -> str:
    """Inline SVG do logo do projeto (vetorial, escala perfeita em qualquer tamanho).

    Conceito: 3 barras ascendentes = braços do bandit · linha = aprendizado ·
    ponto dourado = braço escolhido. ``gid`` deve ser único por uso na página
    para os IDs de gradiente/filtro não colidirem entre instâncias."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;flex-shrink:0">'
        f'<defs>'
        f'<linearGradient id="{gid}G" x1="4" y1="4" x2="44" y2="44" gradientUnits="userSpaceOnUse">'
        f'<stop stop-color="{VIOLET}"/><stop offset="1" stop-color="{CYAN}"/></linearGradient>'
        f'<filter id="{gid}F" x="-40%" y="-40%" width="180%" height="180%">'
        f'<feGaussianBlur stdDeviation="1.4" result="b"/>'
        f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        f'</defs>'
        f'<rect x="2" y="2" width="44" height="44" rx="13" fill="#030D24" '
        f'stroke="url(#{gid}G)" stroke-width="2"/>'
        f'<rect x="11.5" y="28" width="5.2" height="9" rx="2" fill="{CYAN}" opacity="0.50"/>'
        f'<rect x="20.4" y="22" width="5.2" height="15" rx="2" fill="{CYAN}" opacity="0.78"/>'
        f'<rect x="29.3" y="15" width="5.2" height="22" rx="2" fill="url(#{gid}G)"/>'
        f'<path d="M11 30.5 L23 24 L32 12.5" stroke="#FFFFFF" stroke-width="1.7" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="0.9" fill="none"/>'
        f'<circle cx="32" cy="12.5" r="3.7" fill="{GOLD}" stroke="#FFFFFF" '
        f'stroke-width="1.3" filter="url(#{gid}F)"/>'
        f'</svg>'
    )


def _hero_bg_layer() -> str:
    """Fundo (data-URI) = a FOTO do hero COM as fórmulas (na ordem do treino)
    ASSADAS dentro do tubo glow (caracol), usando a luminância da própria foto como
    máscara — assim ficam só sobre o tubo, sem vazar. O PIL renderiza e compõe
    (controle exato de pixel). Em caso de falha, cai para a foto pura.
    """
    import base64
    import io

    public = ROOT / "frontend" / "public"
    for name, mime in (("hero-bg.png", "image/png"), ("hero-bg.jpg", "image/jpeg"),
                       ("hero-bg.jpeg", "image/jpeg")):
        f = public / name
        if not f.exists():
            continue
        try:
            import numpy as _np
            from PIL import Image, ImageDraw, ImageFilter, ImageFont

            photo = Image.open(f).convert("RGB")
            photo.thumbnail((1600, 1600))
            IW, IH = photo.size
            fs = max(15, int(IH * 0.033))
            font = None
            for fp_ in ("C:/Windows/Fonts/seguisym.ttf", "C:/Windows/Fonts/segoeui.ttf"):
                try:
                    font = ImageFont.truetype(fp_, fs)
                    break
                except Exception:
                    pass
            if font is None:
                font = ImageFont.load_default()
            # Sequência NA ORDEM do loop de treino do bandit contextual.
            line = (
                "(1) contexto x → (2) θ=inv(A)·b → (3) UCB=x^T·θ+α·√(x^T·inv(A)·x) → "
                "(4) a*=argmax UCB → (5) reward=margem·P(conv|x) → (6) A+=x·x^T ; b+=r·x → "
                "(7) regret=Σ(μ*−μ) → "
            ) * 2
            txt = Image.new("L", (IW, IH), 0)
            dd = ImageDraw.Draw(txt)
            y, k = 0, 0
            while y < IH:
                dd.text((-180 - (k % 4) * 65, y), line, fill=255, font=font)
                y += int(fs * 1.32)
                k += 1
            # máscara do tubo: blur leve (preenche miolos) + curva que ZERA o halo
            # fraco (sem vazar) e ENCHE o corpo do tubo (sem gaps).
            g = photo.convert("L").filter(ImageFilter.GaussianBlur(2))
            M = _np.interp(_np.asarray(g, dtype=_np.float32) / 255.0,
                           [0, 0.11, 0.19, 0.36, 1.0],
                           [0, 0, 0.85, 1.0, 1.0]).astype(_np.float32)
            P = _np.asarray(photo, dtype=_np.float32)
            T = _np.asarray(txt, dtype=_np.float32) / 255.0
            a = (T * M).clip(0, 1)[..., None]
            top = _np.array([235., 245., 255.], dtype=_np.float32)
            out = (255 - (255 - P) * (255 - top * a) / 255).clip(0, 255).astype("uint8")  # screen
            buf = io.BytesIO()
            Image.fromarray(out).save(buf, format="JPEG", quality=84, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f'url("data:image/jpeg;base64,{b64}") center/cover fixed no-repeat, '
        except Exception:
            data = f.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return f'url("data:{mime};base64,{b64}") center/cover fixed no-repeat, '
    return ""


HERO_BG = _hero_bg_layer()
# Só a parte url("data:…") — usada na camada animada de fundo (sem `fixed`, que
# entra em conflito com transform). Vazio se não houver imagem de hero.
HERO_URL = HERO_BG.split(" center/cover")[0] if HERO_BG else ""

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      #MainMenu, footer, [data-testid="stToolbar"] {{visibility: hidden;}}
      /* ── Top header bar — make it dark instead of the default white ─ */
      [data-testid="stHeader"], header[data-testid="stHeader"] {{
        background:rgba(0,0,0,0) !important; box-shadow:none !important;}}
      [data-testid="stHeader"] * {{color:{TEXT} !important;}}
      [data-testid="stDecoration"] {{display:none !important;}}
      [data-testid="collapsedControl"],
      button[data-testid="collapsedControl"],
      [data-testid="stSidebarCollapsedControl"],
      button[data-testid="stSidebarCollapsedControl"] {{
        visibility: visible !important; opacity: 1 !important;
        display: flex !important; pointer-events: all !important; z-index: 999999 !important;}}
      [data-testid="stSidebarCollapseButton"] {{display: none !important;}}
      html, body, [class*="css"], .stApp {{font-family:'Inter',system-ui,sans-serif;}}
      /* base preta na raiz; .stApp transparente p/ a imagem do ::before aparecer */
      html, body {{background:{BG};}}
      .stApp {{background:transparent;}}
      /* ── Imagem de fundo: CRESCE do centro (máscara radial expande) até
            preencher; depois deriva de leve. Sem leque, sem sujeira. ────── */
      .stApp::before {{
        content:""; position:fixed; inset:-6%; z-index:0; pointer-events:none;
        background:{HERO_URL} center/cover no-repeat;
        opacity:0; will-change:transform,opacity,-webkit-mask-size;
        -webkit-mask:radial-gradient(circle at 50% 47%,
            #000 42%, rgba(0,0,0,.5) 66%, transparent 84%) 50% 47% / 0% 0% no-repeat;
        mask:radial-gradient(circle at 50% 47%,
            #000 42%, rgba(0,0,0,.5) 66%, transparent 84%) 50% 47% / 0% 0% no-repeat;
        animation:heroGrow 2.8s cubic-bezier(.2,.7,.2,1) both;}}
      /* conteúdo sempre ACIMA do fundo (fórmulas já estão assadas na foto) */
      [data-testid="stMain"], [data-testid="stSidebar"],
      [data-testid="stHeader"] {{position:relative; z-index:1;}}
      @keyframes heroGrow {{
        0%   {{opacity:0;   transform:scale(1.02);
               -webkit-mask-size:0% 0%; mask-size:0% 0%;}}
        60%  {{opacity:.34;}}
        100% {{opacity:.34; transform:scale(1.05);
               -webkit-mask-size:340% 340%; mask-size:340% 340%;}}}}
      @media (prefers-reduced-motion: reduce) {{
        .stApp::before {{animation:none; opacity:.34;
          filter:none; -webkit-mask:none; mask:none;}}}}
      /* ── Layout base ─────────────────────────────────────────── */
      .block-container {{padding-top:1rem;padding-bottom:0.5rem;max-width:1400px;}}
      [data-testid="stSidebar"] {{background:{PANEL2};border-right:1px solid rgba(255,255,255,.05);}}
      div[data-testid="column"] {{padding:0 8px;}}
      button[data-testid="StyledFullScreenButton"] {{display:none !important;}}
      /* ── Sidebar — legibilidade máxima ───────────────────────── */
      [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{padding-top:1.4rem;}}
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] div {{
        color:#DCE4F5 !important; font-weight:600 !important; font-size:.84rem !important;
        letter-spacing:.01em;}}
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color:#A7B6D6 !important; font-size:.76rem !important; font-weight:500;}}
      [data-testid="stSidebar"] hr {{margin:0.9rem 0 !important;
        border-color:rgba(255,255,255,.07) !important;}}
      /* slider ticks/value mais legíveis */
      [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
      [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {{
        color:{MUTED} !important; font-size:.68rem !important;}}
      [data-testid="stSidebar"] [data-testid="stThumbValue"] {{
        color:{RED} !important; font-weight:800 !important;}}
      /* help "?" tooltip icon */
      [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {{opacity:.65;}}
      /* cabeçalho de seção da sidebar — consistente, com acento */
      .side-sect {{display:flex;align-items:center;gap:9px;margin:6px 0 12px;
        padding:7px 11px;border-radius:9px;
        font-size:.72rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;
        color:#E2EAFB;
        background:linear-gradient(90deg,{hex_rgba(CYAN,.16)},rgba(0,0,0,0));
        border:1px solid {hex_rgba(CYAN,.14)};
        border-left:3px solid {CYAN};}}
      .side-sect::before {{content:"";width:4px;height:14px;border-radius:2px;flex-shrink:0;
        background:linear-gradient(180deg,{VIOLET},{CYAN});
        box-shadow:0 0 8px {hex_rgba(CYAN,.55)};}}
      .side-card {{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);
        border-radius:11px;padding:12px 13px;}}
      .side-legend-row {{display:flex;align-items:baseline;gap:8px;padding:4px 0;
        font-size:.80rem;color:{MUTED};line-height:1.35;}}
      .side-legend-row b {{color:{TEXT};font-weight:700;}}
      .side-svc {{display:flex;align-items:center;justify-content:space-between;
        padding:8px 11px;margin-bottom:7px;border-radius:10px;
        background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);}}
      .side-svc .svc-name {{font-size:.84rem;font-weight:700;color:{TEXT};}}
      .side-svc .svc-cmd {{font-size:.66rem;color:{MUTED};font-family:monospace;
        display:block;margin-top:2px;}}
      /* ── Expander (rastreio de execução) — card escuro coeso ──── */
      [data-testid="stExpander"] {{
        background:rgba(0,0,0,0.55) !important;
        border:1px solid rgba(255,255,255,.07) !important;
        border-radius:12px !important; overflow:hidden;
        box-shadow:0 1px 3px rgba(0,0,0,.50), 0 4px 14px rgba(0,0,0,.35);}}
      [data-testid="stExpander"] details,
      [data-testid="stExpander"] details > summary,
      [data-testid="stExpanderHeader"],
      [data-testid="stExpanderDetails"] {{
        background:transparent !important; border:none !important;}}
      [data-testid="stExpander"] summary {{
        font-weight:700 !important; color:{TEXT} !important; font-size:.86rem;
        padding:12px 14px !important; background:transparent !important;}}
      [data-testid="stExpander"] summary p,
      [data-testid="stExpander"] summary span {{color:{TEXT} !important;}}
      [data-testid="stExpander"] summary:hover,
      [data-testid="stExpander"] summary:hover p {{color:{CYAN} !important;}}
      [data-testid="stExpander"] summary svg {{fill:{CYAN} !important;}}
      [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{padding:4px 14px 14px;}}
      /* ── st.metric — tiles escuros (usado no rastreio) ────────── */
      [data-testid="stMetric"] {{
        background:rgba(0,0,0,0.42); border:1px solid rgba(255,255,255,.07);
        border-radius:10px; padding:10px 14px;}}
      [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
        color:{MUTED} !important; font-weight:600 !important;}}
      [data-testid="stMetricValue"] {{color:{TEXT} !important; font-weight:800 !important;}}
      /* ── Card elevado — preto com leve transparência ─────────── */
      .card-elevated,
      div[data-testid="stPlotlyChart"],
      div[data-testid="stDataFrame"] {{
        background:rgba(0,0,0,0.72);
        border-radius:12px;
        box-shadow:0 1px 3px rgba(0,0,0,.60), 0 4px 16px rgba(0,0,0,.45);
        padding:10px 12px 12px;
        border:1px solid rgba(255,255,255,.05);
        overflow:hidden;
        transition:box-shadow .20s ease, transform .15s ease;
        backdrop-filter:blur(12px);
        margin-bottom:0;}}
      .card-elevated:hover,
      div[data-testid="stPlotlyChart"]:hover,
      div[data-testid="stDataFrame"]:hover {{
        box-shadow:0 2px 6px rgba(0,0,0,.65), 0 8px 26px rgba(26,111,255,.20);
        transform:translateY(-1px);
        border-color:rgba(26,111,255,.25);}}
      div[data-testid="stPlotlyChart"] > div,
      .js-plotly-plot, .plot-container {{overflow:visible !important;}}
      /* ── Hero topbar ──────────────────────────────────────────── */
      .topbar {{
        position:relative; overflow:hidden;
        display:flex;align-items:center;justify-content:space-between;gap:18px;
        background:
          radial-gradient(120% 180% at 88% -40%, {hex_rgba(CYAN,.20)} 0%, rgba(0,0,0,0) 55%),
          linear-gradient(135deg, {hex_rgba(VIOLET,.24)} 0%, rgba(0,0,0,.84) 52%, {hex_rgba(CYAN,.12)} 100%);
        border-radius:18px;padding:26px 30px;margin-bottom:14px;
        border:1px solid {hex_rgba(CYAN,.20)};
        box-shadow:0 2px 8px rgba(0,0,0,.60), 0 14px 44px {hex_rgba(VIOLET,.20)};}}
      .topbar::after {{
        content:"";position:absolute;left:0;top:0;bottom:0;width:5px;
        background:linear-gradient(180deg,{VIOLET},{CYAN});}}
      .topbar .hero-row {{display:flex;align-items:center;gap:16px;}}
      .topbar .hero-logo {{display:flex;align-items:center;
        filter:drop-shadow(0 4px 14px {hex_rgba(CYAN,.45)});}}
      .topbar .eyebrow {{color:{CYAN};font-size:.66rem;font-weight:800;
        letter-spacing:.24em;text-transform:uppercase;margin-bottom:6px;
        display:flex;align-items:center;gap:8px;}}
      .topbar .eyebrow::before {{content:"";width:16px;height:2px;border-radius:2px;
        background:{CYAN};display:inline-block;}}
      .topbar h1 {{margin:0;font-size:2.15rem;font-weight:800;letter-spacing:-.035em;
        line-height:1.0;color:{TEXT};display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;}}
      .topbar .hero-grad {{
        background:linear-gradient(92deg,#FFFFFF 0%,#A9C7FF 55%,{CYAN} 100%);
        -webkit-background-clip:text;background-clip:text;
        -webkit-text-fill-color:transparent;color:transparent;}}
      .topbar .hero-board {{font-weight:600;color:{MUTED};font-size:1.0rem;
        letter-spacing:.01em;text-transform:none;
        border-left:1px solid rgba(255,255,255,.15);padding-left:12px;}}
      .topbar .sub {{color:#A7B6D6;font-size:.92rem;margin-top:12px;font-weight:500;
        line-height:1.5;}}
      .topbar .hero-badges {{display:flex;flex-direction:column;gap:7px;align-items:flex-end;
        flex-shrink:0;}}
      /* ── Badges de status ─────────────────────────────────────── */
      .stat {{display:inline-block;padding:4px 10px;border-radius:999px;font-size:.72rem;
        font-weight:700;margin-left:5px;background:rgba(255,255,255,.06);
        border:1px solid rgba(255,255,255,.10);letter-spacing:.02em;}}
      .on  {{color:{GREEN};background:rgba(26,158,26,.12);border-color:rgba(26,158,26,.30);}}
      .off {{color:#64748B;background:rgba(100,116,139,.08);border-color:rgba(100,116,139,.15);}}
      /* ── Seções ───────────────────────────────────────────────── */
      .sect {{color:{MUTED};font-size:.73rem;font-weight:700;text-transform:uppercase;
        letter-spacing:.10em;margin:10px 0 2px;border-left:3px solid {VIOLET};padding-left:10px;}}
      .sect-desc {{color:{MUTED};font-size:.71rem;margin:0 0 5px 13px;line-height:1.40;
        padding-left:10px;border-left:1px solid rgba(255,255,255,.06);}}
      /* ── Pills de reason codes ────────────────────────────────── */
      .pill {{display:inline-block;padding:3px 10px;border-radius:999px;
        background:{hex_rgba(CYAN,.14)};color:{CYAN};font-size:.71rem;font-weight:700;
        margin:2px 5px 2px 0;border:1px solid {hex_rgba(CYAN,.28)};}}
      /* ── Card de resultado do explorador ─────────────────────── */
      .result {{
        background:rgba(0,0,0,0.80);border-radius:12px;padding:18px 22px;
        border:1px solid rgba(255,255,255,.05);
        box-shadow:0 1px 3px rgba(0,0,0,.60), 0 6px 22px rgba(0,0,0,.45);}}
      .result .arm {{font-size:1.40rem;font-weight:800;color:{ACCENT_LT};letter-spacing:-.01em;}}
      /* ── Botão primary ────────────────────────────────────────── */
      .stButton > button {{
        border-radius:10px;font-weight:700;
        background:linear-gradient(135deg,{VIOLET} 0%,{CYAN} 100%) !important;
        color:white !important;border:none !important;
        box-shadow:0 2px 6px rgba(0,0,0,.40), 0 4px 14px {hex_rgba(VIOLET,.35)};}}
      .stButton > button:hover {{filter:brightness(1.10);transform:translateY(-1px);
        box-shadow:0 4px 12px rgba(0,0,0,.45), 0 6px 20px {hex_rgba(VIOLET,.45)};}}
      /* ── Inputs (number, text) — dark instead of default white ──── */
      [data-testid="stNumberInput"] input,
      [data-testid="stTextInput"] input,
      [data-baseweb="input"], [data-baseweb="base-input"] {{
        background:{PANEL} !important; color:{TEXT} !important;
        border-color:rgba(255,255,255,.10) !important;}}
      [data-testid="stNumberInput"] > div,
      [data-testid="stTextInput"] > div {{
        background:{PANEL} !important;
        border:1px solid rgba(255,255,255,.10) !important; border-radius:8px;}}
      [data-testid="stNumberInput"] button,
      [data-testid="stNumberInputStepUp"],
      [data-testid="stNumberInputStepDown"] {{
        background:{PANEL2} !important; color:{TEXT} !important;
        border-color:rgba(255,255,255,.10) !important;}}
      [data-testid="stNumberInput"] button:hover {{
        background:{hex_rgba(CYAN,.18)} !important; color:{CYAN} !important;}}
      /* ── Selectboxes / dropdowns — dark instead of default white ── */
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background:{PANEL} !important; color:{TEXT} !important;
        border:1px solid rgba(255,255,255,.10) !important; border-radius:8px;}}
      [data-testid="stSelectbox"] div[data-baseweb="select"] span {{
        color:{TEXT} !important;}}
      [data-testid="stSelectbox"] svg {{fill:{MUTED} !important;}}
      /* dropdown popover menu (rendered in a portal at body root) */
      div[data-baseweb="popover"] ul,
      div[data-baseweb="popover"] [role="listbox"] {{
        background:{PANEL} !important;
        border:1px solid rgba(255,255,255,.10) !important;}}
      div[data-baseweb="popover"] li,
      div[data-baseweb="popover"] [role="option"] {{
        background:{PANEL} !important; color:{TEXT} !important;}}
      div[data-baseweb="popover"] li:hover,
      div[data-baseweb="popover"] [role="option"]:hover {{
        background:{hex_rgba(CYAN,.18)} !important; color:{CYAN} !important;}}
      .svc {{font-size:.82rem;padding:4px 0;color:{TEXT};}}
      .sim-pending {{
        background:{hex_rgba(GOLD,.10)};border:1px solid {hex_rgba(GOLD,.28)};
        color:{GOLD};border-radius:8px;padding:5px 10px;font-size:.76rem;
        font-weight:600;text-align:center;margin-top:5px;}}
      /* ── Timeline feed ───────────────────────────────────────── */
      .tl-wrap {{
        background:rgba(0,0,0,0.72);border-radius:12px;padding:14px 16px;
        border:1px solid rgba(255,255,255,.05);
        box-shadow:0 1px 3px rgba(0,0,0,.60), 0 4px 16px rgba(0,0,0,.45);}}
      @keyframes livepulse {{
        0%,100%{{opacity:1;box-shadow:0 0 0 0 rgba(26,158,26,.5);}}
        50%{{opacity:.8;box-shadow:0 0 0 7px rgba(26,158,26,0);}}}}
      .live-dot {{width:7px;height:7px;border-radius:50%;background:{GREEN};
        animation:livepulse 2s infinite;display:inline-block;
        margin-right:6px;vertical-align:middle;}}
      .tl-item {{display:grid;grid-template-columns:64px 10px 50px 1fr 56px;
        align-items:center;gap:8px;padding:8px 0;
        border-bottom:1px solid rgba(255,255,255,.05);}}
      .tl-item:last-child {{border-bottom:none;}}
      .tl-time {{color:{TEXT};font-size:10px;font-family:monospace;text-align:right;
        font-weight:600;line-height:1.25;}}
      .tl-date {{display:block;color:{MUTED};font-size:8px;font-weight:500;}}
      .tl-dot {{width:9px;height:9px;border-radius:50%;}}
      .tl-badge {{font-size:7px;font-weight:800;padding:2px 5px;border-radius:4px;
        text-align:center;letter-spacing:.04em;}}
      .tl-body {{min-width:0;}}
      .tl-name {{font-size:11px;color:{TEXT};font-weight:600;
        overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}}
      .tl-meta {{font-size:8px;color:{MUTED};margin-top:2px;}}
      .tl-val {{font-size:11px;font-weight:800;text-align:right;}}
      /* ── RAG citations ───────────────────────────────────────── */
      .rag-card {{
        background:rgba(0,0,0,0.72);border-radius:10px;
        padding:11px 13px;margin-bottom:8px;
        border:1px solid rgba(255,255,255,.05);
        box-shadow:0 1px 3px rgba(0,0,0,.55), 0 3px 10px rgba(0,0,0,.38);
        transition:box-shadow .18s, transform .15s;}}
      .rag-card:hover {{
        box-shadow:0 2px 6px rgba(0,0,0,.55), 0 6px 18px rgba(26,111,255,.20);
        transform:translateY(-1px);}}
      .rag-hdr {{display:flex;align-items:center;gap:8px;margin-bottom:7px;}}
      .rag-src {{font-size:.68rem;font-weight:700;color:{CYAN};
        background:{hex_rgba(CYAN,.12)};padding:2px 7px;border-radius:4px;
        border:1px solid {hex_rgba(CYAN,.22)};white-space:nowrap;}}
      .rag-rank {{font-size:.68rem;font-weight:700;color:{MUTED};
        background:rgba(255,255,255,.05);padding:2px 5px;border-radius:4px;
        border:1px solid rgba(255,255,255,.08);}}
      .rag-bar-bg {{flex:1;height:5px;background:rgba(255,255,255,.07);
        border-radius:3px;overflow:hidden;}}
      .rag-bar-fg {{height:100%;border-radius:3px;}}
      .rag-score {{font-size:.69rem;font-weight:800;min-width:36px;text-align:right;}}
      .rag-txt {{font-size:.78rem;color:{MUTED};line-height:1.55;}}
      .rag-highlight {{color:{TEXT};font-weight:600;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Experiment (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="🛰️ Preparando o experimento (treino das 5 políticas) — só nesta primeira carga, fica em cache depois…")
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
    # LinThompson importado diretamente para evitar dependência do registry em cache
    results["lin_thompson"] = run_simulation(
        LinThompson(arms, dim=len(CONTEXT_FEATURES), seed=seed),
        processed, bundle, horizon=horizon, seed=seed,
    )
    return processed, bundle, results


def downsample(arr, points: int = 48):
    a = np.asarray(arr, dtype=float)
    if len(a) <= points:
        return a
    return a[np.linspace(0, len(a) - 1, points).astype(int)]


def tile(col, label: str, value: str, series, color: str, desc: str = "") -> None:
    s = downsample(series)
    n = len(s)
    # x axis: round indices scaled to readable labels
    x_idx = np.linspace(1, n, n).astype(int)
    # Tooltip label depends on units
    if "R$" in value:
        hover_fmt = "round %{x}<br><b>R$ %{y:,.1f}</b><extra></extra>"
    elif "%" in value:
        hover_fmt = "round %{x}<br><b>%{y:.2%}</b><extra></extra>"
    else:
        hover_fmt = "round %{x}<br><b>%{y:,.2f}</b><extra></extra>"

    fig = go.Figure(go.Scatter(
        x=x_idx, y=s, mode="lines",
        line={"color": color, "width": 2.4, "shape": "spline", "smoothing": 0.7},
        fill="tozeroy", fillcolor=hex_rgba(color, 0.12),
        hovertemplate=hover_fmt,
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.88)", bordercolor=color,
                        font_size=11, font_family="Inter"),
    ))
    # Marcador no ponto atual (último valor) — destaca "onde estamos agora".
    fig.add_trace(go.Scatter(
        x=[x_idx[-1]], y=[s[-1]], mode="markers",
        marker={"color": color, "size": 8, "line": {"color": "#FFFFFF", "width": 1.6}},
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        showlegend=False,
        height=148,
        margin={"l": 8, "r": 8, "t": 42, "b": 6},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[
            {"text": label.upper(), "x": 0.01, "y": 1.0, "xref": "paper", "yref": "paper",
             "showarrow": False, "xanchor": "left", "yanchor": "bottom",
             "font": {"size": 10, "color": MUTED, "family": "Inter"}},
            {"text": value, "x": 0.01, "y": 0.52, "xref": "paper", "yref": "paper",
             "showarrow": False, "xanchor": "left",
             "font": {"size": 28, "color": color, "family": "Inter", "weight": 800}},
        ],
    )
    col.plotly_chart(fig, config={**NO_BAR, "scrollZoom": False}, **fill())
    if desc:
        col.markdown(
            f'<div style="margin:-8px 2px 5px;padding:4px 10px 5px;'
            f'background:rgba(0,0,0,0.50);border-radius:0 0 10px 10px;'
            f'border:1px solid rgba(255,255,255,.08);border-top:none">'
            f'<span style="font-size:.68rem;color:{MUTED};line-height:1.45">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def gauge(value: float, title: str, vmax: float, color: str, suffix: str = "",
          sub: str = "", ref: float | None = None, lower_is_better: bool = False) -> go.Figure:
    # Semantic zones: green = good, gold = warning, red = bad
    if lower_is_better:
        steps = [
            {"range": [0, vmax * 0.25],           "color": hex_rgba(GREEN, 0.42)},
            {"range": [vmax * 0.25, vmax * 0.55],  "color": hex_rgba(GOLD,  0.28)},
            {"range": [vmax * 0.55, vmax],         "color": hex_rgba(RED,   0.24)},
        ]
    else:
        steps = [
            {"range": [0, vmax * 0.33],            "color": hex_rgba(RED,   0.20)},
            {"range": [vmax * 0.33, vmax * 0.65],  "color": hex_rgba(GOLD,  0.26)},
            {"range": [vmax * 0.65, vmax],         "color": hex_rgba(GREEN, 0.40)},
        ]
    ref_threshold = (
        {"line": {"color": GOLD, "width": 3}, "thickness": 0.85, "value": ref}
        if ref is not None else None
    )
    # Delta vs baseline direto no medidor (insight): cor verde/vermelho conforme
    # a direção for boa — invertida quando "menor é melhor" (ex.: regret).
    indicator_mode = "gauge+number+delta" if ref is not None else "gauge+number"
    delta_cfg = (
        {
            "reference": ref,
            "suffix": suffix,
            "valueformat": ".1f",
            "increasing": {"color": RED if lower_is_better else GREEN},
            "decreasing": {"color": GREEN if lower_is_better else RED},
            "font": {"size": 13, "family": "Inter"},
            "position": "bottom",
        }
        if ref is not None else None
    )
    fig = go.Figure(go.Indicator(
        mode=indicator_mode,
        value=value,
        delta=delta_cfg,
        number={
            "suffix": suffix,
            "font": {"size": 36, "color": color, "family": "Inter"},
            "valueformat": ".1f",
        },
        title={
            "text": f"<b style='font-size:14px;color:{TEXT};letter-spacing:.04em'>{title}</b>",
        },
        gauge={
            "axis": {
                "range": [0, vmax],
                "tickwidth": 0,
                "tickcolor": "rgba(0,0,0,0)",
                "showticklabels": False,
                "visible": False,
            },
            "bar": {
                "color": color,
                "thickness": 0.52,
                "line": {"color": hex_rgba(color, 0.55), "width": 2},
            },
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps,
            "threshold": ref_threshold,
        },
    ))
    annotations = []
    if sub:
        annotations.append(dict(
            text=sub, x=0.5, y=-0.06, xref="paper", yref="paper",
            showarrow=False, font=dict(size=9, color=MUTED, family="Inter"),
            align="center",
        ))
    fig.update_layout(
        height=244,
        margin={"l": 20, "r": 20, "t": 54, "b": 42, "autoexpand": True},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Inter"},
        hoverlabel={"bgcolor": "rgba(0,0,0,0.90)", "bordercolor": CYAN,
                    "font_size": 12, "font_family": "Inter", "font_color": TEXT},
        annotations=annotations,
    )
    return fig


def style_panel(fig: go.Figure, title: str, height: int = 260,
                legend_x: float = 0.98, legend_y: float = 0.98,
                legend_xanchor: str = "right", legend_yanchor: str = "top",
                legend_orientation: str = "v", hovermode: str | bool = "closest") -> go.Figure:
    """Apply dark theme, title and margin. Legend always INSIDE the plot area
    (legend_y ≤ 1.0) so it can never overlap the title that lives in the margin.

    ``hovermode`` defaults to ``"closest"``; pass ``"x unified"`` on multi-series
    time charts to get a single shared tooltip (richer cross-series comparison)."""
    fig.update_layout(
        template="plotly_dark", height=height,
        title={"text": f"<b>{title}</b>",
               "font": {"size": 14, "color": TEXT, "family": "Inter"},
               "x": 0.01, "xanchor": "left", "pad": {"b": 4}},
        # t=46 is enough for just the title line; legend is now inside the chart.
        margin={"l": 14, "r": 14, "t": 46, "b": 28, "autoexpand": True},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": TEXT},
        hovermode=hovermode,
        legend={
            "orientation": legend_orientation,
            "x": legend_x, "y": legend_y,
            "xanchor": legend_xanchor, "yanchor": legend_yanchor,
            "font": {"size": 9, "family": "Inter", "color": TEXT},
            "bgcolor": "rgba(0,0,0,0.62)",
            "bordercolor": "rgba(255,255,255,.08)", "borderwidth": 1,
            "itemclick": "toggleothers", "itemdoubleclick": "toggle",
            "itemsizing": "constant",
        },
        hoverlabel={
            "bgcolor": "rgba(3,13,36,0.94)",
            "bordercolor": CYAN,
            "font_size": 12,
            "font_family": "Inter",
            "font_color": TEXT,
            "namelength": -1,
            "align": "left",
        },
        transition={"duration": 350, "easing": "cubic-in-out"},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False,
                     tickfont=dict(size=11, color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False, showline=False,
                     tickfont=dict(size=11, color=MUTED), automargin=True)
    return fig


def p_policy_scoreboard(title: str) -> go.Figure:
    """Scoreboard normalizado: todos os braços × 4 métricas em chart único."""
    metric_defs = [
        ("lift_vs_baseline_pct", "Lift vs Baseline",  lambda v: f"+{v:.0f}%"),
        ("conversion_rate",      "Conversão",          lambda v: f"{v:.1%}"),
        ("regret_ratio",         "Regret (inv.)",      lambda v: f"{v:.1%}"),
        ("exploration_rate",     "Exploração",         lambda v: f"{v:.1%}"),
    ]
    fig = go.Figure()
    for p in ["linucb", "lin_thompson", "thompson", "nilos_ucb", "baseline"]:
        row = sdf[sdf["policy"] == p]
        if row.empty:
            continue
        r = row.iloc[0]
        c    = POLICY_COLORS.get(p, MUTED)
        lbl  = POLICY_LABEL.get(p, p)
        xs, ys, txts = [], [], []
        for colname, mlbl, fmt in metric_defs:
            v_raw  = float(r[colname])
            all_v  = sdf[colname].astype(float).values
            lo, hi = all_v.min(), all_v.max()
            denom  = (hi - lo) or 1e-9
            # Regret: lower is better → invert score
            v_norm = ((1 - (v_raw - lo) / denom) * 100
                      if "regret" in colname
                      else (v_raw - lo) / denom * 100)
            xs.append(max(v_norm, 2))   # floor at 2 so bar is always visible
            ys.append(mlbl)
            txts.append(fmt(v_raw))
        fig.add_trace(go.Bar(
            x=xs, y=ys, orientation="h", name=lbl, legendgroup=p,
            marker=dict(color=c, opacity=0.88, line=dict(width=0)),
            text=txts, textposition="inside", insidetextanchor="end",
            textfont=dict(size=11, color="white", family="Inter"),
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                "%{y}: <b>%{text}</b><br>"
                "<i>Score normalizado: %{x:.0f}/100</i><extra></extra>"
            ),
            hoverlabel=dict(bgcolor="rgba(0,0,0,0.90)", bordercolor=c,
                            font_size=12, font_family="Inter"),
        ))
    fig.update_layout(barmode="group")
    fig.update_xaxes(range=[0, 110], showticklabels=False, showgrid=False,
                     title_text="← pior   |   melhor →",
                     title_font=dict(size=9, color=MUTED))
    fig.update_yaxes(automargin=True, tickfont=dict(size=11, color=TEXT))
    return style_panel(fig, title, height=300)


def recent_decisions(n: int = 15) -> pd.DataFrame:
    p = ROOT / "artifacts" / "decisions" / "audit.jsonl"
    if not p.exists():
        return pd.DataFrame()
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()][-n:]
    rows = [json.loads(ln) for ln in lines]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).iloc[::-1]
    _s = lambda col, default: df[col] if col in df.columns else pd.Series([default] * len(df))
    # Timestamp local (UTC-3, horário de Brasília) → data e hora separadas
    _local = (
        pd.to_datetime(_s("ts", ""), utc=True, errors="coerce") - pd.Timedelta(hours=3)
        if "ts" in df.columns else pd.Series([pd.NaT] * len(df))
    )
    return pd.DataFrame({
        "Data": _local.dt.strftime("%d/%m/%Y").fillna(""),
        "Hora": _local.dt.strftime("%H:%M:%S").fillna(""),
        "Oferta":   _s("arm_name", _s("arm_id", "—")),
        "Explorado": _s("explored", False).fillna(False).astype(bool),
        "Valor":    _s("expected_reward", 0.0).astype(float),
        "Politica": _s("policy_name", ""),
    })


# --------------------------------------------------------------------------- #
# Session state — separates slider position from actual simulation trigger
# --------------------------------------------------------------------------- #
if "sim_horizon" not in st.session_state:
    st.session_state["sim_horizon"] = 2000   # default rápido (≈3s)
if "sim_seed" not in st.session_state:
    st.session_state["sim_seed"] = 123

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:11px;padding:2px 0 4px">'
        f'<span style="filter:drop-shadow(0 3px 10px {hex_rgba(CYAN,.45)})">'
        f'{logo_svg(40, gid="side")}</span>'
        f'<div style="min-width:0">'
        f'<div style="font-size:1.12rem;font-weight:800;letter-spacing:-.02em;line-height:1.05;'
        f'background:linear-gradient(92deg,#FFFFFF 0%,#A9C7FF 60%,{CYAN} 100%);'
        f'-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;'
        f'color:transparent">Adaptive Offers</div>'
        f'<div style="font-size:.66rem;color:{CYAN};font-weight:700;letter-spacing:.16em;'
        f'text-transform:uppercase;margin-top:3px">FIAP 7MLET · Grupo 64</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Simulação ────────────────────────────────────────────────────────────
    st.markdown('<div class="side-sect">⚙️ Simulação</div>', unsafe_allow_html=True)
    new_horizon = st.select_slider(
        "Horizonte de simulação",
        options=[500, 1000, 2000, 4000, 6000, 8000, 12000, 20000],
        value=st.session_state["sim_horizon"],
        help="Número de rounds simulados. Quanto maior, mais preciso mas mais lento.",
    )
    new_seed = st.number_input("Seed aleatória", value=st.session_state["sim_seed"], step=1,
                               help="Altera a seed para obter resultados diferentes.")

    pending = (int(new_horizon) != st.session_state["sim_horizon"]
               or int(new_seed) != st.session_state["sim_seed"])
    if pending:
        st.markdown(
            f'<div class="sim-pending">⚠️ Configuração alterada — clique em Simular</div>',
            unsafe_allow_html=True,
        )

    sim_btn = st.button(
        f"▶ Simular ({new_horizon:,} rounds)",
        type="primary", use_container_width=True,
        help="Executa as 5 políticas com o horizonte selecionado (cacheia o resultado).",
    )
    if sim_btn:
        st.session_state["sim_horizon"] = int(new_horizon)
        st.session_state["sim_seed"]    = int(new_seed)
        st.rerun()

    horizon = st.session_state["sim_horizon"]
    seed    = st.session_state["sim_seed"]

    st.markdown(
        f'<div style="display:flex;gap:8px;margin:4px 0 2px">'
        f'<div style="flex:1;background:rgba(255,255,255,.03);'
        f'border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:9px 8px;'
        f'text-align:center">'
        f'<div style="font-size:1.08rem;font-weight:800;color:{CYAN};line-height:1">'
        f'{horizon:,}</div>'
        f'<div style="font-size:.60rem;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:.10em;margin-top:3px">rounds</div></div>'
        f'<div style="flex:1;background:rgba(255,255,255,.03);'
        f'border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:9px 8px;'
        f'text-align:center">'
        f'<div style="font-size:1.08rem;font-weight:800;color:{GOLD};line-height:1">'
        f'{seed}</div>'
        f'<div style="font-size:.60rem;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:.10em;margin-top:3px">seed</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Serviços ─────────────────────────────────────────────────────────────
    st.markdown('<div class="side-sect">🔌 Serviços</div>', unsafe_allow_html=True)
    api_up, mlf_up = port_open(8000), port_open(5000) or port_open(5001) or port_open(5050)

    def badge(up):  # noqa: ANN001
        return '<span class="stat on">● online</span>' if up else '<span class="stat off">● offline</span>'

    st.markdown(
        f'<div class="side-svc"><div><span class="svc-name">API REST</span>'
        f'<span class="svc-cmd">adaptive-offers serve</span></div>{badge(api_up)}</div>'
        f'<div class="side-svc"><div><span class="svc-name">MLflow</span>'
        f'<span class="svc-cmd">mlflow ui --port 5000</span></div>{badge(mlf_up)}</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Legenda ──────────────────────────────────────────────────────────────
    st.markdown('<div class="side-sect">📌 Como ler</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-card">'
        f'<div class="side-legend-row"><span style="color:{GREEN}">↑</span>'
        f'<span><b>Reward</b> — margem × P(conversão)</span></div>'
        f'<div class="side-legend-row"><span style="color:{RED}">↓</span>'
        f'<span><b>Regret</b> — distância do ótimo</span></div>'
        f'<div class="side-legend-row"><span style="color:{GOLD}">◆</span>'
        f'<span><b>Lift</b> — ganho vs baseline greedy</span></div>'
        f'<div class="side-legend-row"><span style="color:{CYAN}">🔍</span>'
        f'<span><b>Exploração</b> · <span style="color:{GREEN}">🎯</span> '
        f'<b>Explotação</b></span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Professional legend added after experiment load (see below)

processed, bundle, results = load_experiment(int(horizon), int(seed))
summary = compare_results(list(results.values()))
sdf = pd.DataFrame(summary)
best = summary[0]
best_res = results[best["policy"]]
base_res = results["baseline"]
qrep = quality_report(processed)

# NOTA: overlay de fórmulas seguindo a curva da imagem foi REMOVIDO — o Streamlit
# sanitiza <text>/<textPath> no st.markdown, então o SVG não renderizava (virava
# texto solto no topo). As fórmulas e código vivem na seção "📐 Fórmulas & código".

# ── Legenda profissional de políticas (sidebar, com métricas) ──────────────
with st.sidebar:
    st.divider()
    st.markdown('<div class="side-sect">🏆 Benchmark de políticas</div>',
                unsafe_allow_html=True)
    _pol_order = ["linucb", "lin_thompson", "thompson", "nilos_ucb", "baseline"]
    _pol_rank  = {s["policy"]: i for i, s in enumerate(summary)}
    for _p in _pol_order:
        _s   = next((x for x in summary if x["policy"] == _p), {})
        _col = POLICY_COLORS.get(_p, MUTED)
        _lbl = POLICY_LABEL.get(_p, _p)
        _is_best = _p == best["policy"]
        _regret  = _s.get("regret_ratio", 0) * 100
        _lift    = _s.get("lift_vs_baseline_pct") or 0
        _conv    = _s.get("conversion_rate", 0) * 100
        _rank    = _pol_rank.get(_p, 3)
        _star_list = ["★★★★★", "★★★★", "★★★☆", "★★☆☆", "★☆☆☆"]
        _stars     = _star_list[min(_rank, len(_star_list) - 1)]
        _badge   = (
            f'<span style="font-size:7px;background:{hex_rgba(_col,.30)};color:{_col};'
            f'border-radius:3px;padding:1px 5px;border:1px solid {hex_rgba(_col,.65)};'
            f'font-weight:800;margin-left:5px">★ ATIVO</span>'
        ) if _is_best else ""
        st.markdown(
            f'<div style="border-left:3px solid {_col};border-radius:0 12px 12px 0;'
            f'background:linear-gradient(90deg,{hex_rgba(_col,.08)} 0%,rgba(0,0,0,0) 100%);'
            f'padding:8px 10px;margin:5px 0;border:1px solid {hex_rgba(_col,.18)};'
            f'border-left:3px solid {_col}">'
            f'<div style="display:flex;align-items:center;margin-bottom:6px">'
            f'  <div style="width:7px;height:7px;border-radius:50%;background:{_col};'
            f'  flex-shrink:0;margin-right:6px;box-shadow:0 0 4px {_col}"></div>'
            f'  <span style="font-size:.82rem;font-weight:700;color:{TEXT}">{_lbl}</span>'
            f'  {_badge}'
            f'  <span style="margin-left:auto;font-size:9px;color:{_col};'
            f'  font-weight:700">{_stars}</span>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px">'
            f'  <div style="background:rgba(255,255,255,.04);border-radius:6px;'
            f'  padding:4px;text-align:center">'
            f'    <div style="font-size:10px;font-weight:800;color:{RED}">'
            f'    {_regret:.1f}%</div>'
            f'    <div style="font-size:7px;color:{MUTED}">regret</div></div>'
            f'  <div style="background:rgba(255,255,255,.04);border-radius:6px;'
            f'  padding:4px;text-align:center">'
            f'    <div style="font-size:10px;font-weight:800;color:{GREEN}">'
            f'    +{_lift:.0f}%</div>'
            f'    <div style="font-size:7px;color:{MUTED}">lift</div></div>'
            f'  <div style="background:rgba(255,255,255,.04);border-radius:6px;'
            f'  padding:4px;text-align:center">'
            f'    <div style="font-size:10px;font-weight:800;color:{CYAN}">'
            f'    {_conv:.1f}%</div>'
            f'    <div style="font-size:7px;color:{MUTED}">conv</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# Topbar
# --------------------------------------------------------------------------- #
api_up, mlf_up = port_open(8000), port_open(5000)
_active_pol = POLICY_LABEL.get(best["policy"], best["policy"])
st.markdown(
    '<div class="topbar">'
    '<div class="hero-left">'
    f'<div class="hero-row">'
    f'<span class="hero-logo">{logo_svg(56, gid="hero")}</span>'
    f'<div>'
    f'<div class="eyebrow">FIAP Pós-Tech · 7MLET · Grupo 64</div>'
    f'<h1><span class="hero-grad">Adaptive Offers</span>'
    f'<span class="hero-board">Observability Board</span></h1>'
    f'</div></div>'
    f'<div class="sub">Plataforma de <b style="color:#EDEDED">multi-armed bandit</b> '
    f'para decisão de ofertas financeiras em tempo real · '
    f'política ativa <b style="color:{CYAN}">{_active_pol}</b></div>'
    '</div>'
    '<div class="hero-badges">'
    f'<span class="stat {"on" if api_up else "off"}">API REST {"●" if api_up else "○"}</span>'
    f'<span class="stat {"on" if mlf_up else "off"}">MLflow {"●" if mlf_up else "○"}</span>'
    '<span class="stat on">BI Dashboard ●</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Row 1 — big KPI tiles with sparklines
# --------------------------------------------------------------------------- #
st.markdown('<div class="sect">⚡ Métricas em tempo real</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sect-desc">KPIs computados sobre <b>{horizon:,}</b> rodadas da política ativa '
    f'<b style="color:{POLICY_COLORS.get(best["policy"], VIOLET)}">'
    f'{POLICY_LABEL.get(best["policy"], best["policy"])}</b>. '
    f'Sparklines mostram a trajetória de aprendizado ao longo do tempo — '
    f'o bandido melhora progressivamente à medida que acumula evidências sobre cada braço.</div>',
    unsafe_allow_html=True,
)
k1, k2, k3, k4 = st.columns(4)
_conv_raw = np.asarray(best_res.converted, dtype=float)
_smooth_w = max(1, len(_conv_raw) // 30)  # rolling window ≈ 3% of horizon
conv_curve = np.convolve(_conv_raw, np.ones(_smooth_w) / _smooth_w, mode="same")
lift_curve = best_res.cumulative_reward - base_res.cumulative_reward
tile(k1, "Reward / 1k impressões", f"R$ {best['reward_per_1k']:,.0f}", best_res.cumulative_reward, VIOLET,
     desc="Receita total ÷ 1.000 rodadas · proxy de RPM financeiro · "
          "<b>↑ maior = política mais lucrativa</b>")
tile(k2, "Regret ratio", f"{best['regret_ratio']:.1%}", best_res.cumulative_regret, RED,
     desc="% receita perdida vs oráculo ótimo (hindsight) · mede sub-optimalidade residual · "
          "<b>↓ menor = aprendizado mais eficiente</b>")
tile(k3, "Conversão", f"{best['conversion_rate']:.1%}", conv_curve, CYAN,
     desc="Fração acumulada de rodadas com conversão real · calibrado em 20k clientes UCI · "
          "<b>↑ maior = melhor seleção de oferta</b>")
tile(k4, "Lift vs baseline", f"+{best.get('lift_vs_baseline_pct', 0):.0f}%", lift_curve, GREEN,
     desc="Ganho incremental de receita vs greedy puro · prova o valor do aprendizado contextual · "
          "<b>↑ positivo = bandit supera baseline</b>")

# --------------------------------------------------------------------------- #
# Dense panel grid (New Relic style) — compact builders
# --------------------------------------------------------------------------- #
GRID_H = 250
PLABELS = [POLICY_LABEL.get(p, p) for p in sdf["policy"]]
PCOLORS = [POLICY_COLORS.get(p, VIOLET) for p in sdf["policy"]]


def p_lollipop(value_col: str, title: str, money: bool = False,
               ref_zone: tuple[float, float] | None = None,
               ref_label: str = "") -> go.Figure:
    """Horizontal lollipop — cleaner than bars for ranked policy comparisons."""
    o = sdf.sort_values(value_col)
    labels = [POLICY_LABEL.get(p, p) for p in o["policy"]]
    values = o[value_col].tolist()
    colors = [POLICY_COLORS.get(p, VIOLET) for p in o["policy"]]
    fmt = lambda v: f"R$ {v:,.0f}" if money else (f"{v:.1%}" if v < 1 else f"{v:,.1f}")
    txt = [fmt(v) for v in values]
    max_v = max(values) if values else 1
    x_max = max_v * 1.65
    fig = go.Figure()
    # optional reference zone (e.g. ideal exploration 10-20%)
    if ref_zone:
        lo, hi = ref_zone
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=-0.5, y1=len(labels) - 0.5,
                      fillcolor=hex_rgba(GREEN, 0.08), line=dict(width=0), layer="below")
        fig.add_vline(x=lo, line=dict(color=hex_rgba(GREEN, 0.4), width=1, dash="dot"))
        fig.add_vline(x=hi, line=dict(color=hex_rgba(GREEN, 0.4), width=1, dash="dot"))
        if ref_label:
            fig.add_annotation(text=ref_label, x=(lo + hi) / 2, y=len(labels) - 0.1,
                               showarrow=False, font=dict(size=8, color=GREEN, family="Inter"),
                               xanchor="center", yanchor="bottom")
    for i, (v, c) in enumerate(zip(values, colors)):
        fig.add_shape(type="line", x0=0, x1=v, y0=i, y1=i,
                      line=dict(color=hex_rgba(c, 0.40), width=2.5, dash="dot"))
    fig.add_trace(go.Scatter(
        x=values, y=labels, mode="markers+text",
        marker=dict(size=15, color=colors, line=dict(color="white", width=1.8)),
        text=txt, textposition="middle right",
        textfont=dict(size=10, color=TEXT, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Valor: <b>%{text}</b><extra></extra>",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.90)", bordercolor=CYAN,
                        font_size=12, font_family="Inter"),
    ))
    fig.update_xaxes(range=[0, x_max], showticklabels=False, automargin=True)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11, color=TEXT))
    return style_panel(fig, title, height=GRID_H + 20)


def p_lollipop_v(x, y, title: str, color: str) -> go.Figure:
    """Vertical lollipop — for categorical dataset signals."""
    x_l, y_l = list(x), list(y)
    max_y = max(y_l) if y_l else 1
    fig = go.Figure()
    for xi, yi in zip(x_l, y_l):
        fig.add_shape(type="line", x0=xi, x1=xi, y0=0, y1=yi,
                      line=dict(color=hex_rgba(color, 0.35), width=2, dash="dot"))
    fig.add_trace(go.Scatter(
        x=x_l, y=y_l, mode="markers+text",
        marker=dict(size=13, color=color, line=dict(color="white", width=1.8)),
        text=[f"{v:.1%}" for v in y_l], textposition="top center",
        textfont=dict(size=10, color=TEXT, family="Inter"),
        hovertemplate="<b>%{x}</b><br>Taxa: <b>%{text}</b><extra></extra>",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.90)", bordercolor=color,
                        font_size=12, font_family="Inter"),
    ))
    fig.update_xaxes(automargin=True, tickfont=dict(size=9, color=TEXT),
                     tickangle=-20)
    fig.update_yaxes(tickformat=".0%", range=[0, max_y * 1.45], showgrid=False, zeroline=False,
                     automargin=True)
    return style_panel(fig, title, height=GRID_H + 30)


def p_arms_bar(labels, values, title: str, colors) -> go.Figure:
    """Ranked horizontal bars for arm pull distribution — cleaner than treemap."""
    _l, _v, _c = zip(*sorted(zip(list(labels), list(values), list(colors)), key=lambda x: x[1])) if list(values) else ([], [], [])
    total = sum(_v) or 1
    pct = [v / total * 100 for v in _v]
    fig = go.Figure(go.Bar(
        x=list(_v), y=list(_l), orientation="h",
        marker=dict(color=list(_c), line=dict(width=0),
                    opacity=[0.72 + 0.28 * (i / max(len(_v) - 1, 1)) for i in range(len(_v))]),
        text=[f"{pc:.1f}%" for pc in pct],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(size=11, color="white", family="Inter"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Pulls totais: <b>%{x:,}</b><br>"
            "Participação: <b>%{text}</b><extra></extra>"
        ),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.90)", bordercolor=CYAN,
                        font_size=12, font_family="Inter"),
    ))
    fig.update_xaxes(showticklabels=False, automargin=True)
    fig.update_yaxes(automargin=True, tickfont=dict(size=10, color=TEXT))
    return style_panel(fig, title, height=GRID_H + 20)


def p_bar_cat(x, y, title: str, color: str) -> go.Figure:
    """Horizontal bar chart for small-N categorical comparisons (2-6 categories).

    Preferred over lollipop when n < 4: bars give a clearer magnitude comparison
    and fill the visual space naturally even with just 2 categories.
    Data is sorted by value (ascending = best appears at top in horizontal bars).
    """
    pairs = sorted(zip(list(y), list(x)), key=lambda t: t[0])
    vals  = [p[0] for p in pairs]
    cats  = [p[1] for p in pairs]
    n     = len(vals)
    alphas = [0.55 + 0.35 * (i / max(n - 1, 1)) for i in range(n)]
    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h",
        marker=dict(
            color=[hex_rgba(color, a) for a in alphas],
            line=dict(width=0),
        ),
        text=[f"{v:.1%}" for v in vals],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=12, color=TEXT, family="Inter", weight=700),
        hovertemplate="<b>%{y}</b><br>Taxa de conversão: <b>%{text}</b><extra></extra>",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.90)", bordercolor=color,
                        font_size=12, font_family="Inter"),
    ))
    max_v = max(vals) if vals else 1
    fig.update_xaxes(range=[0, max_v * 1.45], tickformat=".0%",
                     showgrid=False, showticklabels=False, automargin=True)
    fig.update_yaxes(automargin=True, tickfont=dict(size=12, color=TEXT, family="Inter"))
    return style_panel(fig, title, height=GRID_H + 30)


def p_heatmap_corr(df: pd.DataFrame, title: str) -> go.Figure:
    """Feature × target correlation heatmap — standard ML exploratory analysis."""
    num_cols = [c for c in ["age", "euribor3m", "campaign", "pdays", "previous", "subscribed"]
                if c in df.columns]
    rename = {"age": "Idade", "euribor3m": "Euribor 3m", "campaign": "Contatos camp.",
              "pdays": "Dias ult. contato", "previous": "Contatos ant.", "subscribed": "Subscreveu"}
    corr = df[num_cols].corr().round(2)
    xl = [rename.get(c, c) for c in corr.columns]
    yl = [rename.get(c, c) for c in corr.index]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=xl, y=yl,
        colorscale=[[0.0, RED], [0.5, PANEL2], [1.0, CYAN]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=9, family="Inter"),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.85,
                      tickfont=dict(size=9, color=MUTED), tickformat=".1f",
                      bgcolor="rgba(0,0,0,0)", borderwidth=0),
    ))
    fig.update_xaxes(tickangle=-35, automargin=True, tickfont=dict(size=9, color=MUTED))
    fig.update_yaxes(automargin=True, tickfont=dict(size=9, color=MUTED))
    return style_panel(fig, title, height=GRID_H)


def p_lift_curve(title: str) -> go.Figure:
    """Cumulative lift vs baseline with 95% confidence bands (paired z-CI)."""
    base_arr  = np.array(base_res.cumulative_reward, dtype=float)
    base_step = np.array(base_res.realized_reward,   dtype=float)
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    for name, res in results.items():
        if name == "baseline":
            continue
        arr  = np.array(res.cumulative_reward, dtype=float)
        step = np.array(res.realized_reward,   dtype=float)
        lift = arr - base_arr

        # Per-step paired difference → expanding std → 95% CI for cumulative sum
        d       = step - base_step
        t_arr   = np.arange(1, len(d) + 1, dtype=float)
        run_std = pd.Series(d).expanding(min_periods=2).std().fillna(0).values
        ci_95   = 1.96 * run_std * np.sqrt(t_arr)

        pts = min(80, len(lift))
        idx = np.linspace(0, len(lift) - 1, pts).astype(int)
        col     = POLICY_COLORS.get(name, VIOLET)
        is_best = name == best["policy"]

        # CI band rendered first (sits below main line in z-order)
        fig.add_trace(go.Scatter(
            x=np.concatenate([idx, idx[::-1]]),
            y=np.concatenate([lift[idx] + ci_95[idx], (lift[idx] - ci_95[idx])[::-1]]),
            fill="toself",
            fillcolor=hex_rgba(col, 0.12 if is_best else 0.04),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Main line
        fig.add_trace(go.Scatter(
            x=idx, y=lift[idx], mode="lines",
            name=POLICY_LABEL.get(name, name),
            line=dict(color=col, width=3.0 if is_best else 1.4,
                      dash="solid" if is_best else "dot",
                      shape="spline", smoothing=0.5),
            opacity=1.0 if is_best else 0.6,
            hovertemplate=f"<b>{POLICY_LABEL.get(name, name)}</b><br>step %{{x}}: +R$ %{{y:,.0f}} (IC 95%)<extra></extra>",
        ))
    return style_panel(fig, title, height=GRID_H + 20)


def _msprt_evalue(d: np.ndarray) -> np.ndarray:
    """Mixture Sequential Probability Ratio Test (Johari et al. 2017).

    Always-valid e-value for H₀: mean(d) = 0 with normal-mixture prior.
    Setting τ = σ̂ (unit effect size) gives the closed form:

        M_t = sqrt(1 / (1 + t)) · exp(S_t² / (2 σ̂² (1 + t)))

    M_t is a non-negative super-martingale under H₀; the type-I error is
    controlled at α for ANY stopping time: P(∃t: M_t ≥ 1/α | H₀) ≤ α.
    Reject H₀ (stop experiment) when M_t ≥ 20  (α = 0.05).
    """
    sigma_hat = max(float(np.std(d)), 1e-9)
    S   = np.cumsum(d)
    t   = np.arange(1, len(d) + 1, dtype=float)
    exp_arg = np.clip(S ** 2 / (2.0 * sigma_hat ** 2 * (1.0 + t)), 0.0, 500.0)
    return np.sqrt(1.0 / (1.0 + t)) * np.exp(exp_arg)


def p_stopping_criterion(title: str) -> go.Figure:
    """mSPRT e-value: always-valid sequential test for experiment stopping.

    H₀: mean(reward_bandit − reward_baseline) = 0.
    E-value M_t = sqrt(1/(1+t)) · exp(S_t²/(2σ̂²(1+t))).
    Reject (stop) when M_t ≥ 20 (α = 0.05) — valid at any stopping time,
    no peeking inflation unlike the classical z-test.
    Reference: Johari, Koomen, Pekelis & Walsh (2017) SIGMETRICS.
    """
    THRESHOLD = 20.0           # 1/α, α = 0.05
    LOG_THRESH = np.log10(THRESHOLD)   # ≈ 1.301

    base_step = np.array(base_res.realized_reward, dtype=float)
    n_rounds  = len(base_step)
    sustained = max(3, n_rounds // 200)

    fig = go.Figure()
    fig.add_hline(
        y=LOG_THRESH,
        line=dict(color=GOLD, width=1.5, dash="dash"),
        annotation_text=f"M_t = {THRESHOLD:.0f}  (α = 0.05, always-valid)",
        annotation_position="top left",
        annotation_font=dict(size=8, color=GOLD, family="Inter"),
    )

    best_conv_round: int | None = None
    for name, res in results.items():
        if name == "baseline":
            continue
        d      = np.array(res.realized_reward, dtype=float) - base_step
        e_val  = _msprt_evalue(d)
        log_ev = np.log10(np.maximum(e_val, 1e-12))

        pts = min(100, len(log_ev))
        idx = np.linspace(0, len(log_ev) - 1, pts).astype(int)
        col     = POLICY_COLORS.get(name, VIOLET)
        is_best = name == best["policy"]

        # First sustained crossing of M_t ≥ 20 (log10 ≥ LOG_THRESH)
        conv_round: int | None = None
        for t in range(len(log_ev) - sustained):
            if np.all(log_ev[t: t + sustained] >= LOG_THRESH):
                conv_round = t
                break
        if is_best and conv_round is not None:
            best_conv_round = conv_round

        fig.add_trace(go.Scatter(
            x=idx, y=log_ev[idx], mode="lines",
            name=POLICY_LABEL.get(name, name),
            line=dict(color=col, width=3.0 if is_best else 1.4,
                      dash="solid" if is_best else "dot",
                      shape="spline", smoothing=0.4),
            opacity=1.0 if is_best else 0.55,
            hovertemplate=(
                f"<b>{POLICY_LABEL.get(name, name)}</b><br>"
                "round %{x} → log₁₀(M_t) = %{y:.3f}<extra></extra>"
            ),
        ))

    if best_conv_round is not None:
        best_col  = POLICY_COLORS.get(best["policy"], VIOLET)
        best_lbl  = POLICY_LABEL.get(best["policy"], best["policy"])
        saving_pct = (1 - best_conv_round / n_rounds) * 100
        fig.add_shape(type="line",
                      x0=best_conv_round, x1=best_conv_round, y0=0, y1=1,
                      yref="paper",
                      line=dict(color=best_col, width=1.5, dash="dot"))
        fig.add_annotation(
            x=best_conv_round, y=0.92, yref="paper",
            text=(f"⏹ <b>{best_lbl}</b> converge<br>"
                  f"round {best_conv_round} de {n_rounds}<br>"
                  f"economiza <b>{saving_pct:.0f}%</b> do horizonte"),
            showarrow=True, arrowhead=2, arrowcolor=best_col, arrowwidth=1.4,
            ax=48, ay=0,
            font=dict(size=8, color=TEXT, family="Inter"),
            bgcolor="rgba(0,0,0,0.80)", bordercolor=best_col, borderwidth=1,
            borderpad=5,
        )

    fig.update_yaxes(title_text="log₁₀(e-value)", title_font=dict(size=9, color=MUTED),
                     tickfont=dict(size=9, color=MUTED))
    return style_panel(fig, title, height=GRID_H + 30)


def p_window_regret(title: str) -> go.Figure:
    """Regret per time window — reveals where each policy struggled or excelled."""
    n_win = 8
    win_size = max(1, int(horizon) // n_win)
    fig = go.Figure()
    # plot baseline last so best policy stays on top
    order = [n for n in results if n != "baseline"] + ["baseline"]
    for name in order:
        res = results[name]
        reg = np.array(res.cumulative_regret, dtype=float)
        incr = np.diff(np.concatenate([[0.0], reg]))
        wins = [incr[i * win_size:(i + 1) * win_size].mean() for i in range(n_win)]
        x_lbl = [f"t{int((i + 1) * win_size)}" for i in range(n_win)]
        col = POLICY_COLORS.get(name, VIOLET)
        is_best = name == best["policy"]
        fig.add_trace(go.Scatter(
            x=x_lbl, y=wins, mode="lines+markers",
            name=POLICY_LABEL.get(name, name),
            line=dict(color=col, width=3.2 if is_best else 1.4,
                      dash="solid" if is_best else "dot",
                      shape="spline", smoothing=0.6),
            marker=dict(size=8 if is_best else 4, color=col,
                        line=dict(color="white", width=1.5 if is_best else 0)),
            opacity=1.0 if is_best else 0.55,
            hovertemplate=f"<b>{POLICY_LABEL.get(name, name)}</b><br>%{{x}}: %{{y:.4f}} regret/step<extra></extra>",
        ))
    fig.update_xaxes(tickangle=-35, automargin=True,
                     tickfont=dict(size=8, color=MUTED))
    fig.update_yaxes(tickformat=".3f", automargin=True,
                     tickfont=dict(size=9, color=MUTED))
    return style_panel(fig, title, height=GRID_H + 20)


def p_policy_heatmap(title: str) -> go.Figure:
    """Policy × metric matrix — standard ML model-comparison table as heatmap.

    Works in any column width; shows actual values inside each cell.
    """
    met = ["conversion_rate", "reward_per_1k", "exploration_rate", "lift_vs_baseline_pct"]
    met_labels = ["Conv. %", "R$/1k", "Exploração", "Lift %"]
    met_explain = ["Taxa de conversão", "Receita por 1k rodadas", "Taxa de exploração", "Lift vs baseline"]
    pol_labels = [POLICY_LABEL.get(p, p) for p in sdf["policy"]]

    # Normalise each column 0-100 for colour intensity
    norm_mat, text_mat = [], []
    for m in met:
        col_vals = sdf[m].values.astype(float)
        lo, hi = col_vals.min(), col_vals.max()
        norm_mat.append(((col_vals - lo) / (hi - lo) * 100).tolist() if hi > lo
                        else [50.0] * len(col_vals))

    for _, row in sdf.iterrows():
        row_txt = []
        for m in met:
            v = row[m]
            if m == "reward_per_1k":
                row_txt.append(f"R${v:,.0f}")
            elif m == "lift_vs_baseline_pct":
                row_txt.append(f"{v:+.0f}%")
            else:
                row_txt.append(f"{v:.1%}")
        text_mat.append(row_txt)

    # Transpose norm_mat: shape [n_policies][n_metrics]
    z = list(map(list, zip(*norm_mat)))

    # Build rich tooltip matrix with metric explanation
    hover_mat = []
    for ri, (_, row) in enumerate(sdf.iterrows()):
        hover_row = []
        for mi, (m, mlbl, mexp) in enumerate(zip(met, met_labels, met_explain)):
            v = row[m]
            if m == "reward_per_1k":
                fv = f"R$ {v:,.0f}"
            elif m == "lift_vs_baseline_pct":
                fv = f"{v:+.0f}%"
            else:
                fv = f"{v:.1%}"
            hover_row.append(f"<b>{pol_labels[ri]}</b><br>{mexp}<br>Valor: <b>{fv}</b>")
        hover_mat.append(hover_row)

    fig = go.Figure(go.Heatmap(
        z=z, x=met_labels, y=pol_labels,
        colorscale=[[0.0, PANEL2], [0.45, VIOLET], [1.0, CYAN]],
        zmin=0, zmax=100, showscale=False,
        text=text_mat, texttemplate="%{text}",
        textfont=dict(size=11, color="white", family="Inter"),
        customdata=hover_mat,
        hovertemplate="%{customdata}<extra></extra>",
        xgap=4, ygap=4,
    ))
    fig.update_xaxes(tickfont=dict(size=11, color=TEXT, family="Inter"), tickangle=-30,
                     side="top", automargin=True)
    fig.update_yaxes(tickfont=dict(size=11, color=TEXT), automargin=True)
    fig.update_layout(
        height=GRID_H + 80,
        margin=dict(l=12, r=12, t=80, b=12, autoexpand=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXT),
        title=dict(text=title, font=dict(size=13, color=TEXT), x=0.01, xanchor="left",
                   pad=dict(b=6)),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.90)", bordercolor=CYAN,
                        font_size=12, font_family="Inter", font_color=TEXT),
    )
    return fig


def p_stat(col, title: str, rows: list[tuple[str, str]], height: int | None = GRID_H) -> None:
    h_style = f"height:{height}px;" if height is not None else ""
    items = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 0;border-bottom:1px solid {GRID};gap:8px" '
        f'title="{lab}">'
        f'<span style="color:{MUTED};font-size:.81rem;flex-shrink:0">{lab}</span>'
        f'<span style="font-weight:700;color:{TEXT};font-size:.85rem;text-align:right;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{val}</span>'
        f'</div>' for lab, val in rows)
    col.markdown(
        f'<div style="background:rgba(0,0,0,0.72);backdrop-filter:blur(12px);'
        f'border-radius:12px;border:none;'
        f'padding:14px 16px;{h_style}'
        f'box-shadow:0 1px 3px rgba(0,0,0,.55),0 4px 14px rgba(0,0,0,.38);">'
        f'<div style="color:{TEXT};font-weight:700;font-size:12px;margin-bottom:7px;'
        f'padding-bottom:6px;border-bottom:1px solid rgba(255,255,255,.07)">{title}</div>'
        f'{items}</div>', unsafe_allow_html=True)


# pre-computed series
seg = processed.copy()
seg["age_band"] = pd.cut(seg["age"], [17, 30, 45, 60, 100], labels=["≤30", "31-45", "46-60", "60+"])
by_age = seg.groupby("age_band", observed=True)["subscribed"].mean()
by_pout = target_rate_by(processed, "poutcome")["subscription_rate"].sort_values(ascending=False)
by_contact = target_rate_by(processed, "contact")["subscription_rate"]
pulls = pd.Series(best_res.arm_pulls)
pulls = pulls[pulls > 0].sort_values(ascending=False)
_arm_name_map = {a.offer_id: a.name for a in bundle.catalog}
pulls_named = pulls.copy()
pulls_named.index = [_arm_name_map.get(i, i) for i in pulls.index]
regret_fig = go.Figure()
_best_idx, _best_cum = regret_curve(best_res, points=70)
_base_idx, _base_cum = regret_curve(base_res, points=70)
regret_fig.add_trace(go.Scatter(
    x=np.concatenate([_best_idx, _base_idx[::-1]]),
    y=np.concatenate([_best_cum, _base_cum[::-1]]),
    fill="toself", fillcolor=hex_rgba(GREEN, 0.07),
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))
for name, res in results.items():
    idx, cum = regret_curve(res, points=70)
    _lbl = POLICY_LABEL.get(name, name)
    _col = POLICY_COLORS.get(name, VIOLET)
    regret_fig.add_trace(go.Scatter(
        x=idx, y=cum, mode="lines", name=_lbl,
        line={"width": 2.8 if name == best["policy"] else 1.6,
              "color": _col, "shape": "spline", "smoothing": 0.5},
        opacity=1.0 if name == best["policy"] else 0.65,
        hovertemplate=(
            f"<b>{_lbl}</b><br>"
            "Step: <b>%{x:,}</b><br>"
            "Regret acumulado: <b>%{y:,.0f}</b><extra></extra>"
        ),
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.90)", bordercolor=_col,
                        font_size=12, font_family="Inter"),
    ))

# --- Grid row A: gauges ---------------------------------------------------- #
st.markdown('<div class="sect">🎛️ Indicadores de performance da política</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sect-desc">Gauges da política ativa <b style="color:{TEXT}">'
    f'{POLICY_LABEL.get(best["policy"], best["policy"])}</b>. '
    f'<span style="color:{GREEN}">Verde</span> = zona ótima · '
    f'<span style="color:{GOLD}">Amarelo</span> = atenção · '
    f'<span style="color:{RED}">Vermelho</span> = crítico. '
    f'Linha tracejada dourada = referência da política baseline greedy.</div>',
    unsafe_allow_html=True,
)
a1, a2, a3, a4 = st.columns(4)
_base = next((s for s in summary if s["policy"] == "baseline"), {})
_b_regret  = _base.get("regret_ratio", 1.0) * 100
_b_conv    = _base.get("conversion_rate", 0.055) * 100
_b_explor  = _base.get("exploration_rate", 0.0) * 100
a1.plotly_chart(gauge(
    best["regret_ratio"] * 100, "Regret ratio", 50, RED, "%",
    sub=f"baseline: {_b_regret:.0f}%  ·  ↓ menor = melhor",
    lower_is_better=True,
), config=NO_BAR, **fill())
a2.plotly_chart(gauge(
    best["exploration_rate"] * 100, "Exploração", 30, VIOLET, "%",
    sub=f"baseline: {_b_explor:.0f}%  ·  alvo 10-20%",
    ref=15.0,
), config=NO_BAR, **fill())
a3.plotly_chart(gauge(
    best["conversion_rate"] * 100, "Conversão", 20, CYAN, "%",
    sub=f"baseline: {_b_conv:.1f}%  ·  ↑ maior = melhor",
    ref=_b_conv,
), config=NO_BAR, **fill())
a4.plotly_chart(gauge(
    best.get("lift_vs_baseline_pct", 0), "Lift vs Baseline", 100, GREEN, "%",
    sub="ganho de reward vs greedy  ·  ↑ maior = melhor",
    ref=0.0,
), config=NO_BAR, **fill())

# --- Grid row B: experiment results ---------------------------------------- #
def p_timeseries(series_fn, title, *, smooth=False, pct=False, money=False,
                 height=None, legend_left=True) -> go.Figure:
    """Linha temporal por round — uma série por política, destacando a ativa.

    ``series_fn(res)`` extrai o array por round; ``smooth`` aplica média móvel
    (≈4% do horizonte) para revelar tendência sob o ruído."""
    fig = go.Figure()
    yfmt = ":.1%" if pct else (":,.0f" if money else ":,.2f")
    pre = "R$ " if money else ""
    for name, res in results.items():
        y = np.asarray(series_fn(res), dtype=float)
        if smooth and len(y) > 4:
            w = max(1, len(y) // 25)
            y = pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()
        n = len(y)
        if n == 0:
            continue
        pts = min(110, n)
        idx = np.linspace(0, n - 1, pts).astype(int)
        col = POLICY_COLORS.get(name, VIOLET)
        is_best = name == best["policy"]
        fig.add_trace(go.Scatter(
            x=(idx + 1), y=y[idx], mode="lines",
            name=POLICY_LABEL.get(name, name), legendgroup=name,
            line=dict(color=col, width=3 if is_best else 1.6,
                      shape="spline", smoothing=0.6,
                      dash="solid" if is_best else "dot"),
            opacity=1.0 if is_best else 0.62,
            hovertemplate=f"<b>{pre}%{{y{yfmt}}}</b><extra>{POLICY_LABEL.get(name, name)}</extra>",
        ))
    fig.update_xaxes(title_text="rounds →", title_font=dict(size=9, color=MUTED))
    if pct:
        fig.update_yaxes(tickformat=".0%")
    lx, lxa = (0.02, "left") if legend_left else (0.98, "right")
    return style_panel(fig, title, height=height or (GRID_H + 36),
                       legend_x=lx, legend_xanchor=lxa, hovermode="x unified")


st.markdown('<div class="sect">📊 Resultados do experimento</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sect-desc">Comparação das 5 políticas no mesmo cenário/seed. '
    '<b>Valor acumulado</b>: reward total por política. '
    '<b>Regret acumulado</b>: distância do oráculo — área sombreada = receita recuperada vs baseline. '
    '<b>Pulls por oferta</b>: braços preferidos pela política ativa. '
    '<b>Conversão por política</b>: % rodadas que resultaram em conversão real — métrica de qualidade de aprendizado.</div>',
    unsafe_allow_html=True,
)
b1, b2, b3, b4 = st.columns(4)
b1.plotly_chart(p_lollipop("cumulative_reward", "💰 Valor acumulado por política", money=True), config=NO_BAR, **fill())
_rf = style_panel(regret_fig, "📉 Regret acumulado", height=GRID_H + 20)
_rf.update_layout(legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"))
b2.plotly_chart(_rf, config=NO_BAR, **fill())
b3.plotly_chart(p_arms_bar(pulls_named.index, pulls_named.values, "🎯 Pulls por oferta",
                            [VIOLET, CYAN, GREEN, GOLD, AMBER, RED]),
                config=NO_BAR, **fill())
b4.plotly_chart(p_lollipop("conversion_rate", "📊 Conversão por política"), config=NO_BAR, **fill())

# Segunda linha — trajetórias temporais (mais explicativas que os rankings acima)
b5, b6 = st.columns(2)
b5.plotly_chart(
    p_timeseries(lambda r: r.cumulative_reward,
                 "💹 Reward acumulado ao longo dos rounds", money=True),
    config=NO_BAR, **fill())
b6.plotly_chart(
    p_timeseries(lambda r: r.converted, "🎯 Taxa de conversão (janela móvel)",
                 smooth=True, pct=True),
    config=NO_BAR, **fill())
st.markdown(
    '<div class="sect-desc" style="margin-top:6px">'
    '<b>Reward acumulado</b>: a política ativa (linha cheia) se descola das demais à medida '
    'que aprende — a inclinação é a taxa de ganho. '
    '<b>Taxa de conversão</b> (janela móvel, %): fração de rodadas que converteram em cada '
    'janela — sobe e converge conforme o bandit refina a seleção (forma distinta do cumulativo).</div>',
    unsafe_allow_html=True,
)

# --- Grid row C: ML learning dynamics -------------------------------------- #
st.markdown('<div class="sect">🧠 Dinâmica de aprendizado do modelo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sect-desc">'
    '<b>Lift cumulativo</b>: Σ(reward_bandit − reward_baseline) ao longo do tempo — área acima de zero prova ganho real vs greedy puro. '
    '<b>Regret por janela</b>: regret incremental médio em 8 janelas de tempo — curva descendente = modelo convergindo para braço ótimo. '
    '<b>Taxa de exploração</b>: % de rodadas onde o bandit testou braços alternativos — UCB/Thompson regulam automaticamente; 0% = baseline greedy puro. '
    '<b>Heatmap multi-métrica</b>: intensidade de cor = posição relativa normalizada 0–100 entre políticas; valor exato dentro de cada célula.</div>',
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
c1.plotly_chart(p_lift_curve("📈 Lift cumulativo vs baseline"), config=NO_BAR, **fill())
c2.plotly_chart(p_window_regret("⚡ Regret por janela (convergência)"), config=NO_BAR, **fill())
c3.plotly_chart(p_lollipop("exploration_rate", "🔍 Taxa de exploração",
                            ref_zone=(0.10, 0.20), ref_label="zona ideal"), config=NO_BAR, **fill())
c4.plotly_chart(p_policy_heatmap("🔢 Comparação multi-métrica"), config=NO_BAR, **fill())

# Segunda linha — dinâmica temporal do aprendizado (exploração e convergência)
c5, c6 = st.columns(2)
c5.plotly_chart(
    p_timeseries(lambda r: r.explored,
                 "🔭 Exploração → explotação ao longo do tempo", smooth=True, pct=True),
    config=NO_BAR, **fill())
c6.plotly_chart(
    p_timeseries(lambda r: r.instant_regret,
                 "📉 Regret instantâneo (suavizado) — convergência", smooth=True, money=True),
    config=NO_BAR, **fill())
st.markdown(
    '<div class="sect-desc" style="margin-top:6px">'
    '<b>Exploração ao longo do tempo</b>: fração de rodadas testando braços alternativos '
    '(média móvel). Cai conforme o modelo ganha confiança — a transição '
    'exploração→explotação que define o aprendizado por bandit (baseline fica em 0%). '
    '<b>Regret instantâneo</b>: perda por decisão vs oráculo, suavizada — '
    'a queda em direção a zero é a <i>prova visual da convergência</i>.</div>',
    unsafe_allow_html=True,
)

# --- Monte Carlo & tendência ------------------------------------------------ #
def p_montecarlo(title: str, n_paths: int = 240) -> go.Figure:
    """Monte Carlo: reamostra (bootstrap) os rewards por round da política ativa
    em N trajetórias → leque de incerteza (P5–P95, P25–P75) do reward acumulado."""
    rng = np.random.default_rng(7)
    base = np.asarray(best_res.realized_reward, dtype=float)
    n = len(base)
    paths = np.cumsum(base[rng.integers(0, n, size=(n_paths, n))], axis=1)
    q5, q25, q50, q75, q95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    x = np.arange(1, n + 1)
    sel = np.linspace(0, n - 1, min(120, n)).astype(int)
    fig = go.Figure()
    band = [(q95, q5, hex_rgba(CYAN, .10), "P5–P95"),
            (q75, q25, hex_rgba(CYAN, .20), "P25–P75")]
    for hi, lo, fc, nm in band:
        fig.add_trace(go.Scatter(x=x[sel], y=hi[sel], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x[sel], y=lo[sel], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=fc, name=nm, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x[sel], y=q50[sel], mode="lines", name="mediana",
        line=dict(color=CYAN, width=2.6, shape="spline", smoothing=0.5),
        hovertemplate="round %{x}<br>mediana R$ %{y:,.0f}<extra></extra>"))
    fig.update_xaxes(title_text="rounds →", title_font=dict(size=9, color=MUTED))
    return style_panel(fig, title, height=GRID_H + 40, legend_x=0.02, legend_xanchor="left")


def p_trend(title: str) -> go.Figure:
    """Tendência: reward médio por janela (barras) + reta de regressão (OLS)."""
    base = np.asarray(best_res.realized_reward, dtype=float)
    n = len(base)
    w = max(1, n // 22)
    nb = n // w
    wm = base[:nb * w].reshape(nb, w).mean(axis=1)
    xw = np.arange(nb) * w + w / 2
    coef = np.polyfit(xw, wm, 1)
    trend = np.polyval(coef, xw)
    up = coef[0] >= 0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xw, y=wm, name="reward/round (janela)",
                         marker=dict(color=hex_rgba(GOLD, .55), line=dict(width=0)),
                         hovertemplate="round ~%{x:.0f}<br>R$ %{y:,.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=xw, y=trend, mode="lines", name=f"tendência {'↗' if up else '↘'}",
        line=dict(color=GREEN if up else RED, width=3, dash="dash"),
        hovertemplate="tendência R$ %{y:,.1f}<extra></extra>"))
    fig.update_xaxes(title_text="rounds →", title_font=dict(size=9, color=MUTED))
    return style_panel(fig, title, height=GRID_H + 40, legend_x=0.02, legend_xanchor="left")


st.markdown('<div class="sect">🎲 Monte Carlo & tendência</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sect-desc">'
    '<b>Monte Carlo</b>: 240 trajetórias por bootstrap dos rewards da política ativa — '
    'o leque mostra a incerteza do reward acumulado (P5–P95 e P25–P75) em torno da mediana. '
    '<b>Tendência</b>: reward médio por janela com reta de regressão (OLS) — '
    'inclinação positiva indica que a política continua melhorando ao longo dos rounds.</div>',
    unsafe_allow_html=True,
)
mc1, mc2 = st.columns(2)
mc1.plotly_chart(p_montecarlo("🎲 Monte Carlo — leque de reward acumulado"),
                 config=NO_BAR, **fill())
mc2.plotly_chart(p_trend("📈 Tendência do reward por janela (OLS)"),
                 config=NO_BAR, **fill())

# --- Grid row C2: Stopping criterion --------------------------------------- #
st.markdown('<div class="sect">🛑 Critério de parada do experimento</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sect-desc">'
    '<b>mSPRT e-value</b> (Johari, Koomen, Pekelis &amp; Walsh, SIGMETRICS 2017): teste sempre-válido '
    'para H₀: <i>lift médio = 0</i>. '
    'O e-value M_t = √(1/(1+t)) · exp(S_t²/2σ̂²(1+t)) é uma super-martingala não-negativa sob H₀ — '
    'o erro tipo I é controlado em α = 5% para <b>qualquer</b> momento de parada, eliminando o '
    '<i>peeking problem</i> do z-test clássico. '
    'Rejeitar H₀ quando M_t ≥ 20 (= 1/α). '
    'Gráfico em log₁₀: linha dourada tracejada = limiar de significância (log₁₀20 ≈ 1.30). '
    'As bandas de IC 95% no gráfico de lift são complementares (aproximação assintótica). '
    'Marcador = round de convergência da melhor política; percentual = economia vs horizonte completo.</div>',
    unsafe_allow_html=True,
)

# Compute sample-efficiency stats inline — uses mSPRT e-values (always-valid)
_base_step   = np.array(base_res.realized_reward, dtype=float)
_n_rounds    = len(_base_step)
_sustained   = max(3, _n_rounds // 200)
_LOG_THRESH  = np.log10(20.0)   # M_t >= 20 ↔ α = 0.05
_conv_rounds: dict[str, int | None] = {}
for _name, _res in results.items():
    if _name == "baseline":
        continue
    _d     = np.array(_res.realized_reward, dtype=float) - _base_step
    _ev    = _msprt_evalue(_d)
    _lev   = np.log10(np.maximum(_ev, 1e-12))
    _cr: int | None = None
    for _t in range(len(_lev) - _sustained):
        if np.all(_lev[_t: _t + _sustained] >= _LOG_THRESH):
            _cr = _t
            break
    _conv_rounds[_name] = _cr

_best_cr   = _conv_rounds.get(best["policy"])
_save_pct  = (1 - _best_cr / _n_rounds) * 100 if _best_cr is not None else None
_best_lbl  = POLICY_LABEL.get(best["policy"], best["policy"])

stop_chart, stop_stat = st.columns([3, 1])
stop_chart.plotly_chart(
    p_stopping_criterion("📐 Z-score sequencial · critério de parada estatístico"),
    config=NO_BAR, **fill(),
)
with stop_stat:
    if _best_cr is not None:
        _card_color = GREEN if _save_pct and _save_pct >= 40 else GOLD
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.72);border-radius:12px;'
            f'border:1px solid rgba(255,255,255,.05);padding:18px 16px 16px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,.60),0 4px 16px rgba(0,0,0,.45);">'
            f'<div style="font-size:.68rem;color:{MUTED};font-weight:700;'
            f'letter-spacing:.08em;margin-bottom:6px">EARLY STOPPING</div>'
            f'<div style="font-size:2.2rem;font-weight:900;color:{_card_color};'
            f'line-height:1.1;margin-bottom:4px">{_save_pct:.0f}%</div>'
            f'<div style="font-size:.75rem;color:{TEXT};margin-bottom:12px">'
            f'de rodadas economizadas</div>'
            f'<hr style="border-color:rgba(255,255,255,.06);margin:8px 0">'
            f'<div style="font-size:.70rem;color:{MUTED};line-height:1.55">'
            f'<b style="color:{TEXT}">{_best_lbl}</b> atinge significância estatística '
            f'(z ≥ 1.96) na rodada <b style="color:{_card_color}">{_best_cr}</b> '
            f'de <b>{_n_rounds}</b>.<br><br>'
            f'Em produção, o experimento poderia parar aqui e '
            f'explorar apenas a política vencedora.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Mini convergence table for all policies
        _rows_html = ""
        for _n, _cr2 in _conv_rounds.items():
            _lbl2 = POLICY_LABEL.get(_n, _n)
            _col2 = POLICY_COLORS.get(_n, MUTED)
            _cr_txt = str(_cr2) if _cr2 is not None else "—"
            _rows_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04)">'
                f'<span style="color:{_col2};font-size:.68rem;font-weight:700">{_lbl2}</span>'
                f'<span style="color:{TEXT};font-size:.68rem">t={_cr_txt}</span></div>'
            )
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.72);border-radius:12px;'
            f'border:1px solid rgba(255,255,255,.05);padding:12px 16px;margin-top:8px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,.60),0 4px 16px rgba(0,0,0,.45);">'
            f'<div style="font-size:.68rem;color:{MUTED};font-weight:700;'
            f'letter-spacing:.08em;margin-bottom:8px">CONVERGÊNCIA POR POLÍTICA</div>'
            f'{_rows_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:{MUTED};font-size:.75rem;padding:12px">'
            'Nenhuma política atingiu convergência sustentada com horizonte atual.<br>'
            'Aumente o horizonte de simulação.</div>',
            unsafe_allow_html=True,
        )

# --- Fórmulas & código dos algoritmos (apêndice técnico, igual aos slides) -- #
st.markdown('<div class="sect">📐 Fórmulas & código dos algoritmos</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sect-desc">As fórmulas e o código real usados no treino de cada política '
    'bandit — o mesmo conteúdo do apêndice técnico do pitch.</div>',
    unsafe_allow_html=True,
)
with st.expander("Ver fórmulas e código por algoritmo", expanded=False):
    _ta, _tb, _tc, _tdd = st.tabs(
        ["LinUCB ★", "Thompson (Beta)", "LinThompson", "Nilos-UCB · Baseline"])
    with _ta:
        st.latex(r"\hat\theta_a = A_a^{-1} b_a \qquad "
                 r"p_a = x^\top \hat\theta_a + \alpha\sqrt{x^\top A_a^{-1} x}")
        st.caption("Regressão ridge por braço; o termo α·√(·) adiciona exploração "
                   "otimista (intervalo de confiança superior).")
        st.code(
            "A_a += x @ x.T              # covariância do braço\n"
            "b_a += reward * x           # recompensa acumulada\n"
            "theta = inv(A_a) @ b_a      # estimativa ridge\n"
            "ucb   = x @ theta + alpha * sqrt(x @ inv(A_a) @ x)\n"
            "arm   = argmax_a(ucb)       # braço escolhido",
            language="python")
    with _tb:
        st.latex(r"\theta_a \sim \mathrm{Beta}(\alpha_a,\beta_a) \qquad "
                 r"a^{*} = \arg\max_a \theta_a")
        st.caption("Bayesiano Beta-Bernoulli: amostra uma taxa por braço; "
                   "sucesso incrementa α, falha incrementa β.")
        st.code(
            "theta = rng.beta(alpha, beta)   # amostra por braço\n"
            "arm   = argmax(theta)\n"
            "alpha[arm] += reward            # update conjugado\n"
            "beta[arm]  += (1 - reward)",
            language="python")
    with _tc:
        st.latex(r"\theta \sim \mathcal{N}(\hat\theta,\,A^{-1}) \qquad "
                 r"a^{*} = \arg\max_a x^\top \theta_a")
        st.caption("Thompson linear: amostra um vetor de pesos da posterior "
                   "gaussiana e escolhe o braço de maior valor esperado.")
        st.code(
            "theta = rng.multivariate_normal(inv(A) @ b, inv(A))\n"
            "arm   = argmax_a(x @ theta_a)",
            language="python")
    with _tdd:
        st.latex(r"\text{Nilos-UCB:}\; p_a = \hat\mu_a + "
                 r"\sqrt{\tfrac{2\ln t}{n_a}}\,\hat\sigma_a \qquad "
                 r"\text{Baseline:}\; a^{*} = \arg\max_a \hat\mu_a")
        st.caption("Nilos-UCB pondera a incerteza pela variância observada; "
                   "o baseline é greedy puro (sem exploração) — controle do experimento.")
        st.code(
            "# Nilos-UCB (variance-aware)\n"
            "bonus = sqrt(2 * log(t) / n_a) * sigma_a\n"
            "arm   = argmax_a(mu_a + bonus)\n"
            "# Baseline greedy:  arm = argmax_a(mu_a)",
            language="python")

# --- Grid row D: dataset signals ------------------------------------------- #
st.markdown('<div class="sect">🧬 Análise exploratória da base</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sect-desc">Sinais do dataset Bank Marketing que alimentam as features do bandit contextual. '
    '<b>Poutcome</b> e <b>faixa etária</b>: lollipop ordenado por taxa de conversão — o bandit aprende a priorizar segmentos de maior probabilidade. '
    '<b>Canal de contato</b>: barra horizontal comparando cellular vs telephone (n=2 → barra mais adequada que lollipop). '
    '<b>Correlação</b>: heatmap Pearson entre features numéricas e variável alvo.</div>',
    unsafe_allow_html=True,
)
d1, d2, d3, d4 = st.columns(4)
d1.plotly_chart(p_lollipop_v(by_pout.index, by_pout.values, "Conversão · poutcome", CYAN), config=NO_BAR, **fill())
d2.plotly_chart(p_bar_cat(by_contact.index, by_contact.values, "Conversão · canal de contato", GREEN), config=NO_BAR, **fill())
d3.plotly_chart(p_lollipop_v(by_age.index.astype(str), by_age.values, "Conversão · faixa etária", AMBER), config=NO_BAR, **fill())
d4.plotly_chart(p_heatmap_corr(processed, "🔬 Correlação de features"), config=NO_BAR, **fill())

# --- Grid row E: comparison + ops ------------------------------------------ #
st.markdown('<div class="sect">🧭 Comparação de políticas & operação</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sect-desc">Scoreboard normalizado 0-100 por métrica entre as 5 políticas. '
    'Regret é invertido (menor bruto = maior score). '
    'Cada grupo de 4 barras é uma métrica — compare a altura relativa entre políticas.</div>',
    unsafe_allow_html=True,
)
e_score, e3 = st.columns([6, 3])
e_score.plotly_chart(
    p_policy_scoreboard("🏆 Scoreboard — comparação multi-métrica (score normalizado 0-100)"),
    config=NO_BAR, **fill()
)
p_stat(e3, "📦 Base factual", [
    ("Registros", f"{qrep['n_rows']:,}"),
    ("Conversão", f"{qrep['target']['positive_rate']:.1%}"),
    ("Desbalanceamento", f"{qrep['target']['imbalance_ratio']:.0f}:1"),
    ("Duplicatas", str(qrep["n_duplicates"])),
    ("Ofertas (braços)", str(len(bundle.catalog))),
])

# --- Grid row F: Live feed timeline ----------------------------------------- #
st.markdown('<div class="sect">📡 Feed de decisões ao vivo</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sect-desc">'
    f'Streaming das últimas decisões do bandit em tempo real. '
    f'<span style="color:{GREEN};font-weight:600">Verde = Explotação</span> '
    f'(melhor braço estimado pelo modelo) · '
    f'<span style="color:{CYAN};font-weight:600">Ciano = Exploração</span> '
    f'(testando braços alternativos para reduzir incerteza). '
    f'Atualiza automaticamente a cada 30s.</div>',
    unsafe_allow_html=True,
)

@st.fragment(run_every=30)
def _feed_panel():
    feed = recent_decisions(15)
    audit_path = ROOT / "artifacts" / "decisions" / "audit.jsonl"
    total_count = 0
    if audit_path.exists():
        total_count = sum(
            1 for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        )

    f_left, f_right = st.columns([7, 3])

    with f_left:
        if feed.empty:
            st.markdown(
                f'<div class="tl-wrap" style="text-align:center;padding:50px 20px">'
                f'<div style="font-size:2rem;margin-bottom:12px">📭</div>'
                f'<div style="color:{TEXT};font-size:.9rem;font-weight:600">Sem decisões ainda</div>'
                f'<div style="color:{MUTED};font-size:.78rem;margin-top:5px;line-height:1.5">'
                f'Use o explorador abaixo para gerar decisões ao vivo.<br>'
                f'Cada decisão será registrada aqui em tempo real.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            max_val = feed["Valor"].max() or 1
            items_html = ""
            for i, (_, row) in enumerate(feed.iterrows()):
                exp   = bool(row["Explorado"])
                mc    = CYAN if exp else GREEN
                mlbl  = "EXPLOR." if exp else "EXPLOIT"
                mbg   = hex_rgba(CYAN, 0.18) if exp else hex_rgba(GREEN, 0.15)
                pct   = row["Valor"] / max_val * 100
                pol   = POLICY_LABEL.get(str(row.get("Politica", "")), str(row.get("Politica", "")))
                new_b = (
                    f'<span style="font-size:7px;background:{hex_rgba(GOLD,.28)};color:{GOLD};'
                    f'border-radius:3px;padding:1px 4px;margin-left:5px;'
                    f'border:1px solid {hex_rgba(GOLD,.5)};vertical-align:middle">'
                    f'NOVO</span>'
                ) if i == 0 else ""
                # mini value bar within the body
                bar_html = (
                    f'<div style="height:3px;background:rgba(255,255,255,.06);'
                    f'border-radius:2px;overflow:hidden;margin-top:4px;width:100%">'
                    f'<div style="width:{pct:.0f}%;height:100%;background:{hex_rgba(mc,.45)};'
                    f'border-radius:2px"></div></div>'
                )
                items_html += (
                    f'<div class="tl-item">'
                    f'<span class="tl-time">'
                    f'<span class="tl-date">{row["Data"]}</span>{row["Hora"]}</span>'
                    f'<div class="tl-dot" style="background:{mc};box-shadow:0 0 7px {mc}"></div>'
                    f'<span class="tl-badge" style="background:{mbg};color:{mc}">{mlbl}</span>'
                    f'<div class="tl-body">'
                    f'  <div class="tl-name">{row["Oferta"]}{new_b}</div>'
                    f'  <div class="tl-meta">{pol if pol else "bandit"}</div>'
                    f'  {bar_html}'
                    f'</div>'
                    f'<span class="tl-val" style="color:{mc}">R${row["Valor"]:.1f}</span>'
                    f'</div>'
                )
            header_html = (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid {GRID}">'
                f'<div style="color:{TEXT};font-weight:800;font-size:14px">'
                f'<span class="live-dot"></span>Últimas {len(feed)} decisões</div>'
                f'<div style="display:flex;align-items:center;gap:10px">'
                f'<span style="color:{GREEN};font-size:.72rem;font-weight:600">'
                f'AO VIVO · 30s</span>'
                f'<span style="color:{MUTED};font-size:.72rem">'
                f'{total_count:,} no log total</span>'
                f'</div></div>'
            )
            st.markdown(
                f'<div class="tl-wrap">{header_html}'
                f'<div style="overflow-y:auto;max-height:300px;padding-right:4px">'
                f'{items_html}</div></div>',
                unsafe_allow_html=True,
            )

    with f_right:
        if feed.empty:
            st.markdown(
                f'<div style="background:rgba(0,0,0,0.72);backdrop-filter:blur(12px);'
                f'border-radius:12px;padding:20px;text-align:center;'
                f'box-shadow:0 1px 3px rgba(0,0,0,.55),0 4px 14px rgba(0,0,0,.38);">'
                f'<div style="color:{MUTED};font-size:.83rem">Aguardando decisões…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            n_e   = int(feed["Explorado"].sum())
            n_ex  = len(feed) - n_e
            ep    = n_e / len(feed) * 100
            top   = feed["Oferta"].value_counts().index[0] if len(feed) > 0 else "—"
            avg_v = feed["Valor"].mean()
            top_count = int(feed["Oferta"].value_counts().iloc[0])
            st.markdown(
                f'<div style="background:rgba(0,0,0,0.72);backdrop-filter:blur(12px);'
                f'border-radius:12px;padding:16px 15px;height:100%;'
                f'box-shadow:0 1px 3px rgba(0,0,0,.55),0 4px 14px rgba(0,0,0,.38);">'
                # header
                f'<div style="color:{TEXT};font-weight:700;font-size:13px;'
                f'margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid {GRID}">'
                f'📊 Resumo do feed</div>'
                # total
                f'<div style="margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid {GRID}">'
                f'  <div style="color:{MUTED};font-size:.68rem;font-weight:700;'
                f'  letter-spacing:.06em;margin-bottom:4px">TOTAL NO LOG</div>'
                f'  <div style="color:{TEXT};font-size:1.8rem;font-weight:900;line-height:1">'
                f'  {total_count:,}</div>'
                f'  <div style="color:{MUTED};font-size:.68rem;margin-top:2px">'
                f'  decisões auditáveis registradas</div>'
                f'</div>'
                # explore ratio
                f'<div style="margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid {GRID}">'
                f'  <div style="color:{MUTED};font-size:.68rem;font-weight:700;'
                f'  letter-spacing:.06em;margin-bottom:6px">EXPLORAÇÃO vs EXPLOTAÇÃO</div>'
                f'  <div style="display:flex;gap:10px;margin-bottom:7px">'
                f'    <div style="text-align:center;flex:1">'
                f'      <div style="color:{CYAN};font-size:1.1rem;font-weight:800">{n_e}</div>'
                f'      <div style="color:{CYAN};font-size:.68rem">🔍 exploração</div>'
                f'    </div>'
                f'    <div style="text-align:center;flex:1">'
                f'      <div style="color:{GREEN};font-size:1.1rem;font-weight:800">{n_ex}</div>'
                f'      <div style="color:{GREEN};font-size:.68rem">🎯 explotação</div>'
                f'    </div>'
                f'  </div>'
                f'  <div style="height:8px;background:rgba(255,255,255,.07);'
                f'  border-radius:4px;overflow:hidden">'
                f'    <div style="width:{ep:.0f}%;height:100%;background:{CYAN};'
                f'    border-radius:4px"></div>'
                f'  </div>'
                f'  <div style="display:flex;justify-content:space-between;margin-top:4px">'
                f'    <span style="color:{CYAN};font-size:.68rem">{ep:.0f}%</span>'
                f'    <span style="color:{GREEN};font-size:.68rem">{100-ep:.0f}%</span>'
                f'  </div>'
                f'</div>'
                # avg value
                f'<div style="margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid {GRID}">'
                f'  <div style="color:{MUTED};font-size:.68rem;font-weight:700;'
                f'  letter-spacing:.06em;margin-bottom:4px">VALOR MÉDIO / DECISÃO</div>'
                f'  <div style="color:{GREEN};font-size:1.3rem;font-weight:800">'
                f'  R$ {avg_v:.1f}</div>'
                f'  <div style="color:{MUTED};font-size:.68rem">P(conv) × margem</div>'
                f'</div>'
                # top arm
                f'<div>'
                f'  <div style="color:{MUTED};font-size:.68rem;font-weight:700;'
                f'  letter-spacing:.06em;margin-bottom:5px">OFERTA MAIS SELECIONADA</div>'
                f'  <div style="color:{GOLD};font-size:.85rem;font-weight:700;line-height:1.3">'
                f'  {str(top)[:30]}</div>'
                f'  <div style="color:{MUTED};font-size:.68rem;margin-top:3px">'
                f'  {top_count}× nesta amostra</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

_feed_panel()

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
    # Faz o fundo CRESCER de novo como feedback da decisão: @keyframes de nome
    # único por decisão → reinicia a animação (CSS não re-dispara sozinho em rerun).
    # String flush-left (sem indentação) p/ não virar bloco de código no Streamlit.
    _dn = st.session_state.get("_decide_n", 0) + 1
    st.session_state["_decide_n"] = _dn
    st.markdown(
        "<style>"
        f"@keyframes heroRegrow{_dn}{{"
        "0%{opacity:0;-webkit-mask-size:0% 0%;mask-size:0% 0%;transform:scale(1.02);}"
        "60%{opacity:.26;}"
        "100%{opacity:.26;-webkit-mask-size:340% 340%;mask-size:340% 340%;transform:scale(1.05);}}"
        ".stApp::before{"
        f"animation:heroRegrow{_dn} 2.4s cubic-bezier(.2,.7,.2,1) both,"
        "heroDrift 44s ease-in-out 2.4s infinite alternate !important;}"
        "</style>",
        unsafe_allow_html=True,
    )
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

    # ── HEADLINE CARD ────────────────────────────────────────────────────────
    chosen_arm  = bdf_e[bdf_e["Escolhida"]].iloc[0] if any(bdf_e["Escolhida"]) else bdf_e.iloc[0]
    p_conv      = chosen_arm["p"]
    p_margin    = chosen_arm["Margem"]
    mode_color  = AMBER if rec.explored else GREEN
    mode_label  = "🔍 Exploração" if rec.explored else "🎯 Explotação"
    mode_desc   = "testando alternativa promissora" if rec.explored else "melhor estimativa atual"
    pills = " ".join(f'<span class="pill">{c}</span>' for c in rec.reason_codes)
    st.markdown(
        f'<div class="result">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px">'
        f'<div style="flex:1">'
        f'  <div class="arm">🎁 {rec.arm_name}</div>'
        f'  <div style="display:flex;align-items:center;gap:10px;margin:6px 0 10px">'
        f'    <span style="color:{mode_color};font-weight:700;font-size:.9rem">{mode_label}</span>'
        f'    <span style="color:{MUTED};font-size:.8rem">({mode_desc})</span>'
        f'    <span style="color:{MUTED};font-size:.8rem">·</span>'
        f'    <span style="color:{MUTED};font-size:.8rem">política '
        f'      <b style="color:{TEXT}">{rec.policy_name}@{rec.policy_version}</b>'
        f'    </span>'
        f'    <span style="color:{MUTED};font-size:.8rem">· {len(elig_ids)} de {len(cat)} elegíveis</span>'
        f'  </div>'
        f'  <div>{pills}</div>'
        f'</div>'
        f'<div style="text-align:right;flex-shrink:0">'
        f'  <div style="font-size:2.2rem;font-weight:900;color:{GREEN};line-height:1">R$ {rec.expected_reward:.1f}</div>'
        f'  <div style="color:{MUTED};font-size:.75rem;margin-top:2px">valor esperado</div>'
        f'  <div style="color:{MUTED};font-size:.72rem;margin-top:4px">'
        f'    {p_conv:.1%} × R${p_margin:.0f} = R${p_conv*p_margin:.1f}'
        f'  </div>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── CHARTS ───────────────────────────────────────────────────────────────
    def _div(label: str) -> None:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:11px;margin:26px 0 16px">'
            f'<span style="width:5px;height:18px;border-radius:3px;flex-shrink:0;'
            f'background:linear-gradient(180deg,{VIOLET},{CYAN});'
            f'box-shadow:0 0 9px {hex_rgba(CYAN,.55)}"></span>'
            f'<span style="color:{TEXT};font-size:.82rem;font-weight:800;letter-spacing:.06em">'
            f'{label}</span>'
            f'<div style="height:1px;flex:1;'
            f'background:linear-gradient(90deg,{hex_rgba(CYAN,.30)},rgba(255,255,255,0))"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    _div("📊 DISTRIBUIÇÃO DE VALOR POR OFERTA ELEGÍVEL")
    bar_h = max(400, len(bdf_e) * 64 + 100)
    g1, g2 = st.columns(2)

    # Dois gráficos com PALETAS DISTINTAS para não se confundirem:
    #   • Valor esperado (R$)  → tema DOURADO/ÂMBAR (dinheiro)
    #   • Probabilidade P(conv) → tema CIANO (chance)
    # Verde permanece como a oferta SELECIONADA em ambos (semântica consistente).
    def _bar_emphasis(chosen_flags) -> tuple[list, list]:
        """Borda branca + maior largura só na barra escolhida (faz ela 'saltar')."""
        line_w = [2.2 if c else 0 for c in chosen_flags]
        line_c = ["#FFFFFF" if c else "rgba(0,0,0,0)" for c in chosen_flags]
        return line_w, line_c

    # ── Gráfico 1: Valor esperado · tema ÂMBAR/OURO ─────────────────────────
    bdf_sorted_val = bdf_e.sort_values("Valor")
    _chosen_v = list(bdf_sorted_val["Escolhida"])
    bar_colors = [GREEN if c else AMBER for c in _chosen_v]
    lw_v, lc_v = _bar_emphasis(_chosen_v)
    max_val = bdf_sorted_val["Valor"].max() or 1
    _p_v   = bdf_sorted_val["p"].to_numpy(dtype=float)
    _marg_v = np.where(_p_v > 0, bdf_sorted_val["Valor"].to_numpy(dtype=float) / _p_v, 0.0)
    val_fig = go.Figure(go.Bar(
        x=bdf_sorted_val["Valor"], y=bdf_sorted_val["Oferta"], orientation="h",
        marker=dict(color=bar_colors, opacity=0.92, line=dict(width=lw_v, color=lc_v)),
        customdata=np.column_stack([_p_v, _marg_v]),
        text=[f"R$ {v:.1f}{'  ✅' if c else ''}"
              for v, c in zip(bdf_sorted_val["Valor"], _chosen_v, strict=False)],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=11, color="white", family="Inter"),
        hovertemplate=("<b>%{y}</b><br>Valor esperado: <b>R$ %{x:.1f}</b><br>"
                       "P(conv) %{customdata[0]:.1%} × Margem R$ %{customdata[1]:.0f}"
                       "<extra></extra>"),
    ))
    val_fig.add_vline(x=max_val, line=dict(color=GOLD, width=1.5, dash="dash"),
                      annotation_text="melhor", annotation_font_color=GOLD,
                      annotation_font_size=9)
    val_fig.update_xaxes(range=[0, max_val * 1.12], showticklabels=False, showgrid=False)
    val_fig.update_yaxes(tickfont=dict(size=10, color=TEXT))
    _vfig = style_panel(val_fig, "💰 Valor esperado · R$ (P(conv) × Margem)", height=bar_h)
    _vfig.update_layout(margin=dict(l=175, r=12, t=44, b=10))
    g1.plotly_chart(_vfig, config=NO_BAR, **fill())

    # ── Gráfico 2: P(conversão) · tema CIANO ────────────────────────────────
    p_sorted = bdf_e.sort_values("p")
    _chosen_p = list(p_sorted["Escolhida"])
    p_max = p_sorted["p"].max() or 0.01
    p_colors = [GREEN if c else CYAN for c in _chosen_p]
    lw_p, lc_p = _bar_emphasis(_chosen_p)
    prob_fig = go.Figure(go.Bar(
        x=p_sorted["p"], y=p_sorted["Oferta"], orientation="h",
        marker=dict(color=p_colors, opacity=0.92, line=dict(width=lw_p, color=lc_p)),
        customdata=p_sorted["Valor"].to_numpy(dtype=float),
        text=[f"{v:.1%}{'  ✅' if c else ''}"
              for v, c in zip(p_sorted["p"], _chosen_p, strict=False)],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=11, color="white", family="Inter"),
        hovertemplate=("<b>%{y}</b><br>P(conversão) = <b>%{x:.1%}</b><br>"
                       "→ Valor esperado R$ %{customdata:.1f}<extra></extra>"),
    ))
    prob_fig.add_vline(x=p_max, line=dict(color=CYAN, width=1.5, dash="dash"),
                       annotation_text="máx", annotation_font_color=CYAN,
                       annotation_font_size=9)
    prob_fig.update_xaxes(range=[0, p_max * 1.12], tickformat=".0%",
                          showticklabels=False, showgrid=False)
    prob_fig.update_yaxes(tickfont=dict(size=10, color=TEXT))
    _pfig = style_panel(prob_fig, "🎯 Probabilidade de conversão P(conv)", height=bar_h)
    _pfig.update_layout(margin=dict(l=175, r=12, t=44, b=10))
    g2.plotly_chart(_pfig, config=NO_BAR, **fill())

    # ── ANÁLISE ──────────────────────────────────────────────────────────────
    _div("📋 DETALHAMENTO — OFERTAS ELEGÍVEIS · FÓRMULA · PERFIL DO CLIENTE")
    h1, h2, h3 = st.columns([1.15, 1.05, 0.90])

    # ── Col 1: Tabela estilizada ─────────────────────────────────────────────
    with h1:
        tbl_rows = bdf_e.sort_values("Valor", ascending=False).to_dict("records")
        row_html = ""
        for row in tbl_rows:
            chosen = row["Escolhida"]
            bg     = f"background:linear-gradient(90deg,{hex_rgba(GREEN,.09)} 0%,rgba(0,0,0,0) 100%);" if chosen else ""
            border = f"border-left:3px solid {GREEN};" if chosen else "border-left:3px solid transparent;"
            tag    = f'<span style="color:{GREEN};font-size:.78rem;font-weight:800">✅</span>' if chosen else ""
            row_html += (
                f'<tr style="{bg}{border}border-bottom:1px solid rgba(255,255,255,.05)">'
                f'<td style="padding:7px 10px;font-size:.78rem;color:{TEXT};font-weight:{"700" if chosen else "400"}'
                f';white-space:nowrap">{tag} {row["Oferta"]}</td>'
                f'<td style="padding:7px 8px;font-size:.78rem;color:{CYAN};text-align:right;font-weight:700">'
                f'{row["p"]:.1%}</td>'
                f'<td style="padding:7px 8px;font-size:.78rem;color:{GOLD};text-align:right">'
                f'R$ {int(row["Margem"])}</td>'
                f'<td style="padding:7px 8px;font-size:.82rem;color:{"#1A9E1A" if chosen else TEXT};'
                f'text-align:right;font-weight:{"800" if chosen else "600"}">'
                f'R$ {row["Valor"]:.1f}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.72);backdrop-filter:blur(12px);'
            f'border-radius:12px;padding:14px 14px 8px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,.55),0 4px 14px rgba(0,0,0,.38);">'
            f'<div style="color:{TEXT};font-weight:700;font-size:13px;margin-bottom:4px">'
            f'🧾 Todas as ofertas elegíveis</div>'
            f'<div style="color:{MUTED};font-size:.70rem;margin-bottom:10px">'
            f'P(conv) = conversão estimada pelo bandit contextual · '
            f'<span style="color:{GREEN};font-weight:700">verde = selecionada</span></div>'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr style="border-bottom:1px solid {GRID}">'
            f'<th style="padding:4px 10px 8px;font-size:.68rem;color:{MUTED};text-align:left;'
            f'font-weight:700;letter-spacing:.06em">OFERTA</th>'
            f'<th style="padding:4px 8px 8px;font-size:.68rem;color:{MUTED};text-align:right;'
            f'font-weight:700;letter-spacing:.06em">P(conv)</th>'
            f'<th style="padding:4px 8px 8px;font-size:.68rem;color:{MUTED};text-align:right;'
            f'font-weight:700;letter-spacing:.06em">MARGEM</th>'
            f'<th style="padding:4px 8px 8px;font-size:.68rem;color:{MUTED};text-align:right;'
            f'font-weight:700;letter-spacing:.06em">VALOR ESP.</th>'
            f'</tr></thead>'
            f'<tbody>{row_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

    # ── Col 2: Decomposição + Oferta ────────────────────────────────────────
    with h2:
        st.markdown(
            f'<div style="background:rgba(0,0,0,0.72);backdrop-filter:blur(12px);'
            f'border-radius:12px;padding:16px;margin-bottom:0;'
            f'box-shadow:0 1px 3px rgba(0,0,0,.55),0 4px 14px rgba(0,0,0,.38);">'
            f'<div style="color:{TEXT};font-weight:700;font-size:13px;margin-bottom:10px;'
            f'padding-bottom:7px;border-bottom:1px solid rgba(255,255,255,.07)">'
            f'⚖️ Fórmula de valor esperado</div>'
            f'<div style="display:flex;align-items:center;justify-content:space-around;'
            f'text-align:center;gap:4px">'
            f'<div>'
            f'  <div style="font-size:1.7rem;font-weight:900;color:{CYAN};line-height:1">'
            f'  {p_conv:.1%}</div>'
            f'  <div style="color:{MUTED};font-size:.70rem;margin-top:4px">P(conv)</div>'
            f'  <div style="color:{MUTED};font-size:.64rem">prob. conversão</div>'
            f'</div>'
            f'<div style="font-size:1.6rem;color:{MUTED};font-weight:200;line-height:1">×</div>'
            f'<div>'
            f'  <div style="font-size:1.7rem;font-weight:900;color:{GOLD};line-height:1">'
            f'  R${p_margin:.0f}</div>'
            f'  <div style="color:{MUTED};font-size:.70rem;margin-top:4px">Margem</div>'
            f'  <div style="color:{MUTED};font-size:.64rem">valor da oferta</div>'
            f'</div>'
            f'<div style="font-size:1.6rem;color:{MUTED};font-weight:200;line-height:1">=</div>'
            f'<div>'
            f'  <div style="font-size:1.7rem;font-weight:900;color:{GREEN};line-height:1">'
            f'  R${p_conv*p_margin:.1f}</div>'
            f'  <div style="color:{MUTED};font-size:.70rem;margin-top:4px">Valor esp.</div>'
            f'  <div style="color:{MUTED};font-size:.64rem">reward esperado</div>'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        arm = by_id[rec.arm_id]
        rules = []
        if getattr(arm, "requires_no_default", False):
            rules.append("sem default")
        if getattr(arm, "requires_no_loan", False):
            rules.append("sem empréstimo")
        if getattr(arm, "min_age", None):
            rules.append(f"idade ≥ {arm.min_age}")
        p_stat(h2, f"🎯 Oferta selecionada", [
            ("Categoria", arm.category),
            ("Margem", f"R$ {arm.margin:.0f}"),
            ("Suitability", arm.suitability_tier),
            ("Regras", ", ".join(rules) or "nenhuma"),
            ("Elegível", "✅" if is_eligible(arm, ctx) else "❌"),
        ], height=None)

    # ── Col 3: Perfil + Modo ─────────────────────────────────────────────
    with h3:
        p_stat(h3, "👤 Perfil do cliente", [
            ("Idade", str(age)), ("Canal", contact),
            ("Resultado anterior", poutcome), ("Default", default),
            ("Empréstimo", loan), ("Euribor 3m", f"{euribor:.1f}%"),
        ], height=None)
        st.markdown(
            f'<div style="background:{hex_rgba(mode_color, 0.10)};backdrop-filter:blur(6px);'
            f'border:2px solid {hex_rgba(mode_color, 0.50)};'
            f'border-radius:14px;padding:14px 16px;margin-top:10px;text-align:center;'
            f'box-shadow:0 6px 20px {hex_rgba(mode_color,0.22)}">'
            f'<div style="font-size:1.15rem;font-weight:900;color:{mode_color};'
            f'letter-spacing:-.01em;line-height:1">{mode_label}</div>'
            f'<div style="color:{MUTED};font-size:.74rem;margin-top:5px;font-weight:500">'
            f'{mode_desc}</div>'
            f'<div style="font-size:.68rem;color:{hex_rgba(mode_color,.65)};margin-top:6px;'
            f'font-weight:600;letter-spacing:.04em">'
            f'{"exploração reduz incerteza do modelo" if rec.explored else "explotação maximiza reward esperado"}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── RAZÕES + ASSISTENTE ──────────────────────────────────────────────────
    _div("🧠 RACIOCÍNIO DA DECISÃO & ASSISTENTE LLM + RAG")

    # Reason codes — faixa de mini-cards em GRID responsivo (largura total).
    reason_items = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:11px;'
        f'padding:11px 13px;border-radius:0 10px 10px 0;'
        f'background:linear-gradient(90deg,{hex_rgba(CYAN,.08)},rgba(0,0,0,0));'
        f'border:1px solid rgba(255,255,255,.06);border-left:3px solid {CYAN}">'
        f'<span style="flex-shrink:0;width:22px;height:22px;border-radius:7px;'
        f'background:{hex_rgba(CYAN,.16)};color:{CYAN};font-size:.72rem;font-weight:800;'
        f'display:flex;align-items:center;justify-content:center;'
        f'border:1px solid {hex_rgba(CYAN,.30)}">{i}</span>'
        f'<div style="min-width:0">'
        f'<div style="color:{CYAN};font-size:.75rem;font-weight:800;'
        f'letter-spacing:.03em;font-family:monospace;margin-bottom:3px">{r["code"]}</div>'
        f'<div style="color:{MUTED};font-size:.76rem;line-height:1.45">'
        f'{r["description"]}</div>'
        f'</div>'
        f'</div>'
        for i, r in enumerate(rec.reasons, 1)
    )
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.72);backdrop-filter:blur(12px);'
        f'border-radius:12px;padding:16px;margin-bottom:12px;'
        f'box-shadow:0 1px 3px rgba(0,0,0,.55),0 4px 14px rgba(0,0,0,.38);">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'flex-wrap:wrap;gap:8px;margin-bottom:12px;'
        f'padding-bottom:9px;border-bottom:1px solid rgba(255,255,255,.07)">'
        f'<span style="color:{TEXT};font-weight:800;font-size:14px">🧠 Por que esta decisão</span>'
        f'<span style="color:{MUTED};font-size:.71rem">'
        f'{len(rec.reasons)} reason codes · política '
        f'<b style="color:{CYAN}">{rec.policy_name}</b></span>'
        f'</div>'
        f'<div style="display:grid;gap:8px;'
        f'grid-template-columns:repeat(auto-fit,minmax(250px,1fr))">{reason_items}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Assistente LLM + RAG — largura TOTAL (texto longo lê melhor sem coluna estreita).
    _prov = exp.get("provider", "offline")
    if _prov in ("anthropic", "azure_openai"):
        _badge_txt, _badge_color = "● Claude online", GREEN
        _badge_bg, _badge_border = hex_rgba(GREEN, .13), hex_rgba(GREEN, .35)
    elif _prov not in ("offline", ""):
        _badge_txt, _badge_color = "⚡ análise ML", VIOLET
        _badge_bg, _badge_border = hex_rgba(VIOLET, .15), hex_rgba(VIOLET, .40)
    else:
        _badge_txt, _badge_color = "● offline", "#64748B"
        _badge_bg, _badge_border = "rgba(100,116,139,.10)", "rgba(100,116,139,.20)"
    provider_badge = (
        f'<span style="background:{_badge_bg};color:{_badge_color};'
        f'border:1px solid {_badge_border};border-radius:5px;'
        f'padding:3px 10px;font-size:.70rem;font-weight:700;white-space:nowrap">{_badge_txt}</span>'
    )
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.72);backdrop-filter:blur(12px);'
        f'border-radius:12px;padding:0 20px 18px;overflow:hidden;'
        f'box-shadow:0 1px 3px rgba(0,0,0,.55),0 4px 14px rgba(0,0,0,.38);">'
        f'<div style="height:3px;margin:0 -20px 0;'
        f'background:linear-gradient(90deg,{VIOLET},{CYAN},{VIOLET})"></div>'
        f'<div style="display:flex;align-items:center;gap:11px;margin:15px 0 13px;'
        f'padding-bottom:11px;border-bottom:1px solid rgba(255,255,255,.07)">'
        f'<span style="width:32px;height:32px;border-radius:9px;flex-shrink:0;'
        f'display:flex;align-items:center;justify-content:center;font-size:17px;'
        f'background:linear-gradient(135deg,{hex_rgba(VIOLET,.30)},{hex_rgba(CYAN,.20)});'
        f'border:1px solid {hex_rgba(CYAN,.30)}">🤖</span>'
        f'<div style="flex:1;min-width:0">'
        f'<div style="color:{TEXT};font-weight:800;font-size:14px;line-height:1.1">'
        f'Assistente LLM + RAG</div>'
        f'<div style="color:{MUTED};font-size:.68rem;margin-top:2px">'
        f'explicação em linguagem natural · grounded nas políticas comerciais</div>'
        f'</div>'
        f'{provider_badge}'
        f'</div>'
        f'<div style="color:{TEXT};font-size:.86rem;line-height:1.8;'
        f'columns:2;column-gap:34px;column-rule:1px solid rgba(255,255,255,.06)">'
        f'{_md2html(exp["answer"])}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if exp.get("citations"):
        st.markdown(
            f'<div style="color:{TEXT};font-weight:700;font-size:13px;margin:18px 0 4px">'
            f'📄 Citações de política comercial</div>'
            f'<div style="color:{MUTED};font-size:.73rem;margin-bottom:12px">'
            f'Chunks da política comercial recuperados por similaridade semântica e injetados no '
            f'prompt do LLM. Score indica relevância: '
            f'<span style="color:{GREEN}">alto (&gt;0.3)</span> · '
            f'<span style="color:{GOLD}">médio (0.1-0.3)</span> · '
            f'<span style="color:{MUTED}">baixo (&lt;0.1)</span>.</div>',
            unsafe_allow_html=True,
        )
        rag_html = '<div class="rag-wrap">'
        for i, c in enumerate(exp["citations"]):
            score    = float(c.get("score", 0))
            if score >= 0.3:
                sc_color, bar_color, rel_lbl = GREEN, GREEN, "alta"
            elif score >= 0.1:
                sc_color, bar_color, rel_lbl = GOLD, GOLD, "média"
            else:
                sc_color, bar_color, rel_lbl = MUTED, MUTED, "baixa"
            bar_pct  = min(100, score * 250)
            src      = str(c.get("source", "policy.md"))
            text     = str(c.get("text", "")).strip()
            rank_lbl = ["1º", "2º", "3º", "4º", "5º"][i] if i < 5 else f"{i+1}º"
            rag_html += (
                f'<div class="rag-card">'
                f'<div class="rag-hdr">'
                f'  <span class="rag-rank">{rank_lbl}</span>'
                f'  <span class="rag-src">📄 {src}</span>'
                f'  <div class="rag-bar-bg">'
                f'    <div class="rag-bar-fg" style="width:{bar_pct:.0f}%;'
                f'    background:{bar_color}"></div>'
                f'  </div>'
                f'  <span style="font-size:.62rem;font-weight:700;color:{sc_color};'
                f'  background:{hex_rgba(sc_color,.14)};border:1px solid {hex_rgba(sc_color,.30)};'
                f'  border-radius:4px;padding:1px 6px;text-transform:uppercase;'
                f'  letter-spacing:.04em;white-space:nowrap">{rel_lbl}</span>'
                f'  <span class="rag-score" style="color:{sc_color}">{score:.2f}</span>'
                f'</div>'
                f'<div class="rag-txt">{text}</div>'
                f'</div>'
            )
        rag_html += '</div>'
        st.markdown(rag_html, unsafe_allow_html=True)

    # --- LLM trace expander -----------------------------------------------
    _trace = exp.get("_trace", {})
    if _trace:
        with st.expander("🔬 Rastreio de execução — LLM · RAG · Tempos", expanded=False):
            t1, t2, t3 = st.columns(3)
            t1.metric("RAG retrieval", f"{_trace.get('rag_ms', 0):.0f} ms",
                      help="Tempo de busca TF-IDF nos chunks das políticas comerciais")
            t2.metric("Geração LLM", f"{_trace.get('llm_ms', 0):.0f} ms",
                      help="Tempo total de geração do texto de explicação")
            t3.metric("Total", f"{_trace.get('total_ms', 0):.0f} ms",
                      help="Tempo end-to-end da chamada ao Assistant")
            st.markdown(
                f'<div style="display:flex;gap:10px;margin:10px 0;flex-wrap:wrap">'
                f'<span style="font-size:.74rem;background:rgba(255,255,255,.05);'
                f'border-radius:6px;padding:4px 10px;color:{MUTED}">'
                f'<b style="color:{TEXT}">Provider:</b> {_trace.get("provider", exp.get("provider","?"))}</span>'
                f'<span style="font-size:.74rem;background:rgba(255,255,255,.05);'
                f'border-radius:6px;padding:4px 10px;color:{MUTED}">'
                f'<b style="color:{TEXT}">Modelo:</b> {_trace.get("model","—")}</span>'
                f'<span style="font-size:.74rem;background:rgba(255,255,255,.05);'
                f'border-radius:6px;padding:4px 10px;color:{MUTED}">'
                f'<b style="color:{TEXT}">Chunks RAG:</b> {_trace.get("chunks_retrieved",0)}</span>'
                f'<span style="font-size:.74rem;background:rgba(255,255,255,.05);'
                f'border-radius:6px;padding:4px 10px;color:{MUTED}">'
                f'<b style="color:{TEXT}">Query RAG:</b> {_trace.get("rag_query","—")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.divider()
st.markdown(
    f'<div style="text-align:center;color:{MUTED};font-size:.82rem;padding:8px 0 4px">'
    f'<b style="color:{TEXT}">Adaptive Offers Platform</b> · © 2026 '
    f'<b style="color:{CYAN}">Dione Braga</b> — Grupo 64 · FIAP Pós-Tech 7MLET'
    '<br/><span style="font-size:.76rem">Licença MIT · '
    '<a href="https://github.com/dionebraga/datathon-7mlet-grupo-64" '
    f'style="color:{MUTED};text-decoration:none">github.com/dionebraga/datathon-7mlet-grupo-64</a>'
    '</span></div>', unsafe_allow_html=True)
