# kto_cards.py
# -------------------------------------------------------------------
# Generate all dashboard visuals in uniform "card" style.
# Only required arg: --label "<your label>"
# -------------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import textwrap
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
from matplotlib.patches import FancyBboxPatch, Wedge, Circle
from Themes import thema_structuur

# -------------------- THEME (uniform card styling) --------------------
BG     = "#062B36"   # page background
CARD   = "#2E3136"   # card background
TEXT   = "#E5E7EB"   # main text
SUBTLE = "#A7B0B7"   # secondary text (ticks)
GRID   = "#3A3D42"   # subtle grid

# Accents used in charts
TEAL   = "#10B981"   # green accent
PURPLE = "#7C3AED"   # purple accent
WHITE  = "#FFFFFF"


def apply_theme():
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",        # we draw the card manually
        "axes.edgecolor": "none",
        "axes.labelcolor": TEXT,
        "xtick.color": SUBTLE,
        "ytick.color": SUBTLE,
        "text.color": TEXT,
        "axes.titleweight": "bold",
        "font.size": 11,
    })

def start_card(figsize=(11, 6), dpi=160, title=None,
               margins=(0.02, 0.06, 0.02, 0.06),  # (left, bottom, right, top) in fig coords
               rounding=0.06, pad=0.018):
    """Create a figure with a rounded 'card' background and optional title."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax_card = fig.add_axes([0, 0, 1, 1]); ax_card.axis("off")
    ax_card.set_xlim(0, 1); ax_card.set_ylim(0, 1)

    l, b, r, t = margins
    rect = (l, b, 1 - l - r, 1 - b - t)  # x, y, w, h in figure coords

    ax_card.add_patch(FancyBboxPatch(
        (rect[0], rect[1]), rect[2], rect[3],
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        facecolor=CARD, edgecolor=CARD, zorder=1
    ))

    if title:
        # Check if the title is "Prioriteitenmatrix" to make it larger
        if title == "Prioriteitenmatrix":
            fontsize = 32
        else:
            fontsize = 26
            
        fig.text(rect[0] + 0.04, rect[1] + rect[3] - 0.06,
                 title, fontsize=fontsize, weight="bold", color=TEXT,
                 ha="left", va="center")

    fig.patch.set_facecolor("none")
    return fig, ax_card, rect  # Use rect to place inner axes

def add_inner_axes(fig, rect, padding=(0.04, 0.12, 0.04, 0.04)):
    """Add an inner plotting axes inside the card with consistent padding."""
    l = rect[0] + padding[0]
    b = rect[1] + padding[1]
    w = rect[2] - padding[0] - padding[2]
    h = rect[3] - padding[1] - padding[3]
    ax = fig.add_axes([l, b, w, h])

    ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=SUBTLE)
    return ax


# -------------------- UTILS --------------------
def load_df(excel_path: Union[str, Path], sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(excel_path, sheet_name=sheet_name)

def filter_by_label(df: pd.DataFrame, label: Optional[str]) -> pd.DataFrame:
    """Filter on Label column if present; otherwise returns df unchanged."""
    if label is None:
        return df
    for c in ["Label", "label", "LABEL"]:
        if c in df.columns:
            return df[df[c].astype(str) == str(label)].copy()
    return df  # if no label column, skip filtering

def ensure_outdir(outdir: Union[str, Path]) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_fig(fig, path: Path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)


# -------------------- PLOTTERS --------------------

def plot_respondent_count(df: pd.DataFrame, outdir: Path,
                          title: Optional[str] = None,
                          figsize: Optional[Tuple[float, float]] = None) -> Optional[Path]:
    """Donut chart showing respondent status distribution with KPI boxes at top."""
    if "sys_respondentStatus" not in df.columns:
        return None
    
    # Calculate counts
    no_response = int((df["sys_respondentStatus"] == 1).sum())
    volledig = int((df["sys_respondentStatus"] == 3).sum())
    gedeeltelijk = int((df["sys_respondentStatus"] == 2).sum())
    uitgenodigd_tot = no_response + volledig + gedeeltelijk
    
    if uitgenodigd_tot == 0:
        return None
    
    respons = (volledig / uitgenodigd_tot) * 100
    
    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (11, 6), title=title or "Aantal respondenten")
    
    # KPI boxes function
    def stat_box(x, y, value, label):
        fig.text(x, y, f"{value:,}".replace(",", "."),
                fontsize=20, weight="bold", ha="center", va="center", color=TEXT)
        fig.text(x, y-0.06, label,
                fontsize=12, ha="center", va="center", color=SUBTLE)
    
    # Position KPI boxes at top of card
    stat_y = rect[1] + rect[3] - 0.20
    stat_box(rect[0] + 0.14, stat_y, uitgenodigd_tot, "Totaal uitgenodigd")
    stat_box(rect[0] + 0.38, stat_y, volledig, "Volledig ingevuld")
    stat_box(rect[0] + 0.62, stat_y, gedeeltelijk, "Gedeeltelijk ingevuld")
    stat_box(rect[0] + 0.86, stat_y, no_response, "Niet ingevuld")
    
    # Donut chart
    sizes = [volledig, gedeeltelijk, no_response]
    colors = [TEAL, PURPLE, "#4B4F55"]
    
    # Position donut lower in the card - make it bigger
    pie_w, pie_h = 0.55, 0.55
    pie_left = rect[0] + (rect[2] - pie_w) / 2  # centered
    pie_bottom = rect[1] + 0.05
    ax_pie = fig.add_axes([pie_left, pie_bottom, pie_w, pie_h])
    ax_pie.set_facecolor("none")
    ax_pie.set_aspect("equal")
    
    wedges, _ = ax_pie.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.30, edgecolor=CARD)
    )
    
    # Response percentage in center
    ax_pie.text(0, 0.00, f"{respons:.1f}%", ha="center", va="center",
                fontsize=30, weight="bold", color=TEXT)
    ax_pie.text(0, -0.23, "Respons", ha="center", va="center",
                fontsize=11, color=SUBTLE)
    
    # Legend to the right of donut
    lx = pie_left + pie_w + 0.02
    legend_items = [("Volledig", TEAL), ("Gedeeltelijk", PURPLE), ("Niet ingevuld", "#4B4F55")]
    for i, (lab, col) in enumerate(legend_items):
        y = pie_bottom + pie_h*0.62 - i*0.09
        ax_card.add_patch(FancyBboxPatch((lx, y), 0.022, 0.045, 
                                        boxstyle="square,pad=0.002",
                                        facecolor=col, edgecolor="none"))
        fig.text(lx + 0.03, y + 0.022, lab, va="center", color=TEXT, fontsize=12)
    
    p = outdir / "blok_respondent_count.png"
    save_fig(fig, p)
    return p


def plot_aanbeveling_distribution(df: pd.DataFrame, outdir: Path,
                                  title: Optional[str] = None,
                                  figsize: Optional[Tuple[float, float]] = None,
                                  column: Optional[str] = None) -> Optional[Path]:
    """Bar chart 0..10 for Blok1_NPS_KPI_page6_Text."""
    col = column or "Blok1_NPS_KPI_page6_Text"
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    idx = list(range(0, 11))
    counts = s.value_counts().reindex(idx, fill_value=0)

    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (11.5, 6.0), title=title or "Aanbeveling bij collega's en relaties")
    ax = add_inner_axes(fig, rect, padding=(0.06, 0.16, 0.06, 0.08))

    bars = ax.bar(idx, counts.values, width=0.75, color=PURPLE)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.9, alpha=0.5)
    ax.set_xlabel("Score")
    ax.set_ylabel("Aantal")
    ax.set_xticks(idx)
    ymax = max(int(counts.max()), 1)
    ax.set_ylim(0, ymax * 1.25)

    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+ymax*0.04,
                f"{int(b.get_height())}", ha="center", va="bottom", color=SUBTLE, fontsize=10)

    p = outdir / "blok_aanbeveling_distributie.png"
    save_fig(fig, p); return p


def create_nps_gauge(df: pd.DataFrame, outdir: Path,
                     title: Optional[str] = None,
                     figsize: Optional[Tuple[float, float]] = None,
                     column: Optional[str] = None) -> Optional[Path]:
    """
    Create an NPS gauge visualization (US style).
    - Promoters: 9–10
    - Passives: 7–8
    - Detractors: 0–6
    - NPS = %Promoters - %Detractors, displayed as an integer (−100..100)
    """
    from matplotlib.patches import Polygon
    
    column_name = column or "Blok1_NPS_KPI_page6_Text"
    if column_name not in df.columns:
        return None
    
    # --- Calculate NPS (US style) ---
    s = pd.to_numeric(pd.Series(df[column_name]), errors="coerce").dropna()
    if len(s) == 0:
        nps_score = 0
    else:
        promoters = (s >= 9).sum()
        detractors = (s <= 6).sum()
        total = len(s)

        pct_promoters = (promoters / total) * 100.0
        pct_detractors = (detractors / total) * 100.0
        nps_score = int(round(pct_promoters - pct_detractors))
        nps_score = int(np.clip(nps_score, -100, 100))  # safety clamp

    # --------- SETTINGS ----------
    TITLE       = title or "NPS"
    SCORE       = nps_score
    VMIN, VMAX  = -100.0, 100.0

    FIGSIZE    = figsize or (7.2, 4.6)
    RADIUS     = 0.98
    RING_THICK = 0.34
    # -----------------------------

    # Needle angle
    frac = float(np.clip((SCORE - VMIN) / max(1e-9, (VMAX - VMIN)), 0, 1))
    theta = np.deg2rad(180 * (1 - frac))   # 180° (left) -> 0° (right)

    # Figure + rounded card
    apply_theme()
    fig, ax_card, rect = start_card(figsize=FIGSIZE, title=TITLE)

    # Gauge axes
    ax = fig.add_axes([rect[0]+0.06, rect[1]+0.22, rect[2]-0.12, 0.50])
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.25, 1.15)

    # White half-donut
    ax.add_patch(Wedge((0, 0), RADIUS, 0, 180, width=RING_THICK,
                       facecolor=WHITE, edgecolor="none", zorder=1))

    # ----- Polygon needle -----
    tip_len    = RADIUS * 0.86
    base_back  = RADIUS * 0.08
    base_width = RADIUS * 0.18

    tx, ty = tip_len * np.cos(theta), tip_len * np.sin(theta)
    bx, by = -base_back * np.cos(theta), -base_back * np.sin(theta)
    px = np.cos(theta + np.pi/2) * (base_width/2)
    py = np.sin(theta + np.pi/2) * (base_width/2)

    needle = Polygon([[bx - px, by - py],
                      [bx + px, by + py],
                      [tx,      ty     ]],
                     closed=True, facecolor=TEAL, edgecolor="none", zorder=3)
    ax.add_patch(needle)

    # Scale labels underneath gauge - moved further to the left and right
    ax.text(-RADIUS*0.85, -0.15, "-100", ha="center", va="center",
            fontsize=14, color=WHITE, weight="bold")
    ax.text(RADIUS*0.85, -0.15, "+100", ha="center", va="center",
            fontsize=14, color=WHITE, weight="bold")

    # Big number (US formatting: whole number, no comma replacement)
    fig.text(rect[0]+rect[2]/2, rect[1]+0.12, f"{SCORE:d}", ha="center", va="center",
             fontsize=36, weight="bold", color=TEAL)

    p = outdir / "blok_nps_gauge.png"
    save_fig(fig, p)
    return p


def plot_nps_white_gauge_from_scores(df: pd.DataFrame, outdir: Path,
                                     title: Optional[str] = None,
                                     figsize: Optional[Tuple[float, float]] = None,
                                     column: Optional[str] = None) -> Optional[Path]:
    """NPS gauge using proper NPS calculation (Promoters - Detractors)"""
    return create_nps_gauge(df, outdir, title=title, figsize=figsize, column=column)


def plot_duur_samenwerking_gauges(df: pd.DataFrame, outdir: Path,
                                  title: Optional[str] = None,
                                  figsize: Optional[Tuple[float, float]] = None,
                                  column: Optional[str] = None) -> Optional[Path]:
    """
    Card-style HALF-donut gauges (full width, compact height):
    - one mini-axes per gauge → no touching
    - true circles (equal aspect), top-half ring
    - % centered inside each gauge, label below
    """
    column_name = column or "Blok7_3 Duur samenwerking_page74_Text"
    title = title or "Duur samenwerking"
    
    # Get data from the column
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in dataframe")
        return None
    
    # Count values and calculate percentages
    counts = df[column_name].value_counts().sort_index()
    total = len(df[column_name].dropna())
    
    if total == 0:
        print("No valid data found in the column")
        return None
    
    # Calculate percentages
    percentages = [(count / total) * 100 for count in counts.values]
    values = counts.index.tolist()
    
    # Create proper labels based on values
    labels = []
    for value in values:
        if value == 1:
            labels.append("0-1 jaar")
        elif value == 2:
            labels.append("1-2 jaar")
        elif value == 3:
            labels.append("2-5 jaar")
        elif value == 4:
            labels.append("5 jaar")
        else:
            labels.append(str(value))  # fallback for unexpected values
    
    n = len(percentages)

    # ==== Figure + rounded card ====
    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (11, 4.8), title=title)

    # ==== Layout for mini-axes (full width, compact height) ====
    inner_left, inner_right = rect[0] + 0.06, rect[0] + rect[2] - 0.06        # within the card
    inner_bottom, inner_height = rect[1] + 0.34, 0.30     # compact row height (move up/down here)
    gutter = 0.02                               # gap between gauges

    cell_w = (inner_right - inner_left - gutter*(n-1)) / n
    gauge_h = inner_height                      # each mini-axes height
    centers = [inner_left + i*(cell_w + gutter) for i in range(n)]

    # geometry inside each mini-axes (data coords) - BIGGER GAUGES
    RADIUS = 1.5                                 # increased from 1.0 to make gauge bigger
    WIDTH  = 0.5                                # ring thickness relative to radius
    pct_fs, lab_fs = 22, 18                     # increased font sizes from 16,14 to 22,18
    RING_BG = "#4B4F55"

    for i, (pct, lab) in enumerate(zip(percentages, labels)):
        # one mini-axes per gauge
        axg = fig.add_axes([centers[i], inner_bottom, cell_w, gauge_h])
        axg.set_aspect("equal")
        axg.set_xlim(-1.8, 1.8)                 # adjusted limits for bigger gauge
        axg.set_ylim(-1.0, 1.6)                 # adjusted limits for bigger gauge
        axg.axis("off")

        pct = float(max(0, min(100, pct)))

        # background half ring (top half 0..180°)
        axg.add_patch(Wedge((0, 0), RADIUS, 0, 180,
                            width=WIDTH, facecolor=RING_BG, edgecolor="none"))
        # value arc (fill left → right along the top)
        if pct > 0:
            start = 180 - (pct/100.0)*180.0
            axg.add_patch(Wedge((0, 0), RADIUS, start, 180,
                                width=WIDTH, facecolor=TEAL, edgecolor="none"))

        # % centered inside the half donut
        axg.text(0, 0.2, f"{int(round(pct))}%", ha="center", va="center",
                 color=TEXT, fontsize=pct_fs)

        # label below
        axg.text(0, -0.7, str(lab), ha="center", va="center",
                 color=TEXT, fontsize=lab_fs, weight="bold")

    p = outdir / "duur_samenwerking.png"
    save_fig(fig, p)
    return p


def plot_afgenomen_diensten_from_flags(df: pd.DataFrame, outdir: Path, label: Optional[str] = None,
                                       title: Optional[str] = None,
                                       figsize: Optional[Tuple[float, float]] = None) -> Optional[Path]:
    """Horizontal bar chart of services taken, from 0/1 flag columns."""
    
    # Determine which FLAG_MAP to use based on label
    if label == "PDG Health Services":
        FLAG_MAP = {
            "RI&E": "WelkeDiensten_PDG_page95_Text_1",
            "Verdiepende onderzoeken": "WelkeDiensten_PDG_page95_Text_2",
            "PMO /PAGO": "WelkeDiensten_PDG_page95_Text_3",
            "FIT-check, Fysio Vitaal en overig": "WelkeDiensten_PDG_page95_Text_4",
        }
    elif label == "Resolu":
        FLAG_MAP = {
            "Detachering casemanager": "Introductie_Resolu_Diensten_page123_Text_1",
            "Bezwaar en beroep":       "Introductie_Resolu_Diensten_page123_Text_2",
            "Schadelastbeheersing":    "Introductie_Resolu_Diensten_page123_Text_3",
            "Ziektewet-begeleiding":   "Introductie_Resolu_Diensten_page123_Text_4",
            "Inzet artsencapaciteit":  "Introductie_Resolu_Diensten_page123_Text_5",
            "WGA-begeleiding":         "Introductie_Resolu_Diensten_page123_Text_6",
            "WW-ondersteuning":        "Introductie_Resolu_Diensten_page123_Text_7",
            "Training/workshop":       "Introductie_Resolu_Diensten_page123_Text_8",
        }
    else:
        # Default to Resolu mapping for backward compatibility
        FLAG_MAP = {
            "Detachering casemanager": "Introductie_Resolu_Diensten_page123_Text_1",
            "Bezwaar en beroep":       "Introductie_Resolu_Diensten_page123_Text_2",
            "Schadelastbeheersing":    "Introductie_Resolu_Diensten_page123_Text_3",
            "Ziektewet-begeleiding":   "Introductie_Resolu_Diensten_page123_Text_4",
            "Inzet artsencapaciteit":  "Introductie_Resolu_Diensten_page123_Text_5",
            "WGA-begeleiding":         "Introductie_Resolu_Diensten_page123_Text_6",
            "WW-ondersteuning":        "Introductie_Resolu_Diensten_page123_Text_7",
            "Training/workshop":       "Introductie_Resolu_Diensten_page123_Text_8",
        }

    counts = {}
    for service_label, col in FLAG_MAP.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        counts[service_label] = int((s == 1).sum())
    counts = {k:v for k,v in counts.items() if v>0}
    if not counts:
        return None

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k,_ in items][::-1]
    values = [v for _,v in items][::-1]

    COLORS = ["#69D2E7", "#38BDF8", "#2D7DA6", "#3B82F6", "#7C3AED", "#C241A4", "#EC4899"]
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]

    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (12.0, 6.8), title=title or "Afgenomen diensten")
    ax = add_inner_axes(fig, rect, padding=(0.24, 0.16, 0.08, 0.10))

    y = np.arange(len(labels))
    bars = ax.barh(y, values, height=0.7, color=bar_colors, edgecolor="none")
    ax.set_yticks(y, labels)
    # Make Y-axis labels bigger, bold, and white
    for tick in ax.get_yticklabels():
        tick.set_color(WHITE)
        tick.set_fontsize(14)
        tick.set_weight("bold")
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=1.0, alpha=0.6)

    xmax = max(values)
    for b, v in zip(bars, values):
        ax.text(b.get_width() + xmax*0.02, b.get_y()+b.get_height()/2,
                f"{int(v)}", va="center", ha="left", color=TEXT, fontsize=10, weight="bold")
    ax.set_xlim(0, xmax*1.15)

    p = outdir / "blok_afgenomen_diensten.png"
    save_fig(fig, p); return p


def create_omgang_informatie_card(df: pd.DataFrame, outdir: Path,
                                  title: Optional[str] = None,
                                  figsize: Optional[Tuple[float, float]] = None,
                                  column: Optional[str] = None) -> Optional[Path]:
    """Pie chart based on 'Blok8_TevredenheidPrivacyInformatie_page114_Text' (1..5 mapped) - PDG only."""
    col = column or "Blok8_TevredenheidPrivacyInformatie_page114_Text"
    if col not in df.columns:
        return None

    code2label = {5:"Zeer tevreden", 4:"Tevreden", 3:"Neutraal", 2:"Ontevreden", 1:"Zeer ontevreden"}
    pie_colors = ["#4E6591", "#2D7DA6", "#69D2E7", "#8AD0DF", "#6B57A8"]

    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    # Filter to only valid 1-5 values (exclude 6 if present)
    s = s[s.isin([1,2,3,4,5])]
    
    if len(s) == 0:
        return None
    
    # Calculate weighted average (same logic as aspecten functions)
    counts = s.value_counts().reindex([1,2,3,4,5], fill_value=0).astype(float).values
    fracs_for_calc = counts / counts.sum()
    weighted_avg = (np.arange(1,6) * fracs_for_calc).sum()
    
    order = [5,4,3,2,1]
    labels = [code2label[k] for k in order]
    vals = s.map(code2label).value_counts().reindex(labels, fill_value=0).values.astype(float)
    fracs = vals / vals.sum()

    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (12.8, 4.4), title=title or "Omgang informatie")
    # left text block
    ax_text = fig.add_axes([rect[0]+0.04, rect[1]+0.16, 0.38, rect[3]-0.24]); ax_text.axis("off")
    q = "Tevredenheid omgang\nprivacygevoelige informatie?"
    ax_text.text(0.0, 0.80, q, fontsize=17, weight="bold", color=TEXT, ha="left", va="top", linespacing=1.35)

    # pie - make it smaller to leave more room for legend
    ax_pie = fig.add_axes([rect[0]+0.44, rect[1]+0.08, 0.32, rect[3]-0.16])
    ax_pie.set_aspect("equal"); ax_pie.axis("off")
    ax_pie.pie(fracs, colors=pie_colors[:len(fracs)], startangle=120, counterclock=False,
               wedgeprops=dict(width=1.0, edgecolor=CARD))

    # legend - move left and make wider to fit text properly
    ax_leg = fig.add_axes([rect[0]+0.78, rect[1]+0.16, 0.20, rect[3]-0.24]); ax_leg.axis("off")
    for i, (lab, frac, color) in enumerate(zip(labels, fracs, pie_colors[:len(fracs)])):
        y = 0.92 - i*0.18
        ax_leg.add_patch(FancyBboxPatch((0.0, y-0.03), 0.04, 0.06,
                                        boxstyle="square,pad=0.01",
                                        facecolor=color, edgecolor="none"))
        ax_leg.text(0.06, y, f"{lab}\n{frac*100:.1f}%", fontsize=11, weight="bold",
                    color=TEXT, ha="left", va="center")

    # Add weighted average value display (similar to aspecten charts)
    ax_leg.text(0.06, -0.05, f"μ {weighted_avg:.2f}".replace(".", ","), color=TEAL, ha="left", va="center",
                fontsize=16, weight="bold")

    # Add weighted average explanation text at bottom right (same as aspecten functions)
    fig.text(0.94, 0.06, "* μ is het gewogen gemiddelde (1-5 schaal)", color=TEXT, fontsize=11, 
             ha="right", va="bottom", weight="bold")

    p = outdir / "blok_omgang_informatie.png"
    save_fig(fig, p); return p


def create_informatievoorziening_card(df: pd.DataFrame, outdir: Path,
                                      title: Optional[str] = None,
                                      figsize: Optional[Tuple[float, float]] = None,
                                      column: Optional[str] = None) -> Optional[Path]:
    """Pie chart based on 'Blok7_VoorafGeinformeerd_Resolu_page128_Text' (1..5 mapped)."""
    col = column or "Blok7_VoorafGeinformeerd_Resolu_page128_Text"
    if col not in df.columns:
        return None

    code2label = {5:"Zeer tevreden", 4:"Tevreden", 3:"Neutraal", 2:"Ontevreden", 1:"Zeer ontevreden"}
    pie_colors = ["#4E6591", "#2D7DA6", "#69D2E7", "#8AD0DF", "#6B57A8"]

    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    # Filter to only valid 1-5 values (exclude 6 if present)
    s = s[s.isin([1,2,3,4,5])]
    
    if len(s) == 0:
        return None
    
    # Calculate weighted average (same logic as aspecten functions)
    counts = s.value_counts().reindex([1,2,3,4,5], fill_value=0).astype(float).values
    fracs_for_calc = counts / counts.sum()
    weighted_avg = (np.arange(1,6) * fracs_for_calc).sum()
    
    order = [5,4,3,2,1]
    labels = [code2label[k] for k in order]
    vals = s.map(code2label).value_counts().reindex(labels, fill_value=0).values.astype(float)
    fracs = vals / vals.sum()

    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (12.8, 4.4), title=title or "Informatievoorziening")
    # left text block
    ax_text = fig.add_axes([rect[0]+0.04, rect[1]+0.16, 0.38, rect[3]-0.24]); ax_text.axis("off")
    q = "Bent u goed geïnformeerd\nvoor het afnemen van de dienst?"
    ax_text.text(0.0, 0.80, q, fontsize=17, weight="bold", color=TEXT, ha="left", va="top", linespacing=1.35)

    # pie - make it smaller to leave more room for legend
    ax_pie = fig.add_axes([rect[0]+0.44, rect[1]+0.08, 0.32, rect[3]-0.16])
    ax_pie.set_aspect("equal"); ax_pie.axis("off")
    ax_pie.pie(fracs, colors=pie_colors[:len(fracs)], startangle=120, counterclock=False,
               wedgeprops=dict(width=1.0, edgecolor=CARD))

    # legend - move left and make wider to fit text properly
    ax_leg = fig.add_axes([rect[0]+0.78, rect[1]+0.16, 0.20, rect[3]-0.24]); ax_leg.axis("off")
    for i, (lab, frac, color) in enumerate(zip(labels, fracs, pie_colors[:len(fracs)])):
        y = 0.92 - i*0.18
        ax_leg.add_patch(FancyBboxPatch((0.0, y-0.03), 0.04, 0.06,
                                        boxstyle="square,pad=0.01",
                                        facecolor=color, edgecolor="none"))
        ax_leg.text(0.06, y, f"{lab}\n{frac*100:.1f}%", fontsize=11, weight="bold",
                    color=TEXT, ha="left", va="center")

    # Add weighted average value display (similar to aspecten charts)
    ax_leg.text(0.06, -0.05, f"μ {weighted_avg:.2f}".replace(".", ","), color=TEAL, ha="left", va="center",
                fontsize=16, weight="bold")

    # Add weighted average explanation text at bottom right (same as aspecten functions)
    fig.text(0.94, 0.06, "* μ is het gewogen gemiddelde (1-5 schaal)", color=TEXT, fontsize=11, 
             ha="right", va="bottom", weight="bold")

    p = outdir / "blok_informatievoorziening.png"
    save_fig(fig, p); return p


def draw_smiley_impact_card(df: pd.DataFrame, outdir: Path,
                            title: Optional[str] = None,
                            figsize: Optional[Tuple[float, float]] = None,
                            column: Optional[str] = None) -> Optional[Path]:
    """Smileys card based on 'Blok1_Alg1_page9_Text' (1..5), ignore value 6."""
    col = column or "Blok1_Alg1_page9_Text"
    if col not in df.columns:
        return None

    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    s = s[s.between(1,5)]  # drop '6=weet ik niet'
    if s.empty: return None
    mean_value = float(s.mean())

    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (12.0, 5.8),
                                    title=title or "Impact op verzuim, inzetbaarheid\nen preventie")
    ax = fig.add_axes([rect[0]+0.06, rect[1]+0.16, rect[2]-0.12, rect[3]-0.28])
    ax.set_xlim(0,8); ax.set_ylim(0,4); ax.set_aspect("equal"); ax.axis("off")

    green = "#00cc66"; white = "#ffffff"

    full_green = int(mean_value)
    partial_green = mean_value - full_green
    positions = [(1,2), (2.5,2), (4,2), (5.5,2), (7,2)]

    for i,(x,y) in enumerate(positions):
        if i < full_green:
            face = Circle((x,y), 0.4, color=green, zorder=2); ax.add_patch(face)
        elif i == full_green and partial_green >= 0.5:
            ax.add_patch(Circle((x,y), 0.4, color=white, zorder=2))
            ax.add_patch(Wedge((x,y), 0.4, 90, 270, facecolor=green, zorder=2.5))
        else:
            ax.add_patch(Circle((x,y), 0.4, color=white, zorder=2))
        # eyes + smile
        ax.add_patch(Circle((x-0.15, y+0.1), 0.05, color='black', zorder=3))
        ax.add_patch(Circle((x+0.15, y+0.1), 0.05, color='black', zorder=3))
        ax.add_patch(Wedge((x,y-0.03), 0.22, 200, 340, width=0.03, color='black', zorder=3))

    ax.text(1, 0.8, 'zeer weinig', fontsize=14, color=WHITE, ha='center', weight='bold')
    ax.text(7, 0.8, 'Zeer veel',   fontsize=14, color=WHITE, ha='center', weight='bold')
    fig.text(rect[0]+rect[2]/2, rect[1]+0.12, f"{mean_value:.1f}".replace(".", ","),
             fontsize=36, color=green, ha="center", weight="bold")

    p = outdir / "blok_impact_smileys.png"
    save_fig(fig, p); return p


def create_functiegroep_card(df: pd.DataFrame, outdir: Path,
                             title: Optional[str] = None,
                             figsize: Optional[Tuple[float, float]] = None,
                             column: Optional[str] = None) -> Optional[Path]:
    """Horizontal bar chart showing function group distribution."""
    FUNCTIE_MAP = {
        "Directie/werkgever": "1",
        "HR": "2", 
        "Operations": "3",
        "Finance": "4",
        "Anders": "5"
    }
    
    COLUMN_NAME = column or "Blok4_1_page22_Text"
    
    if COLUMN_NAME not in df.columns:
        return None
    
    # Count occurrences of each function group
    counts = {}
    column_data = df[COLUMN_NAME].dropna()  # Remove NaN values first
    
    for label, value in FUNCTIE_MAP.items():
        # Handle both string and float comparisons
        value_float = float(value)
        count = int((column_data == value_float).sum())
        if count > 0:
            counts[label] = count
    
    if not counts:
        return None
    
    # Sort by count (descending) but display ascending for horizontal bars
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    
    # Purple-forward palette to match other cards but with more purple accents
    COLORS = ["#7C3AED", "#8B5CF6", "#A78BFA", "#C084FC", "#C241A4", "#EC4899", "#9333EA"]
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    
    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (12.0, 6.8), title=title or "Functiegroep")
    
    # Match layout of 'Afgenomen diensten' card
    ax = add_inner_axes(fig, rect, padding=(0.24, 0.16, 0.08, 0.10))
    
    y = np.arange(len(labels))
    
    # Bars with same style as services card
    bars = ax.barh(y, values, height=0.7, color=bar_colors, edgecolor="none")
    
    ax.set_yticks(y, labels)
    # Make Y-axis labels bigger, bold, and white
    for tick in ax.get_yticklabels():
        tick.set_color(WHITE)
        tick.set_fontsize(14)
        tick.set_weight("bold")
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=1.0, alpha=0.6)
    
    # Add value labels (same sizing as services card)
    xmax = max(values)
    for b, v in zip(bars, values):
        ax.text(b.get_width() + xmax * 0.02, b.get_y() + b.get_height() / 2,
                f"{int(v)}", va="center", ha="left", color=TEXT, fontsize=11, weight="bold")
    
    ax.set_xlim(0, xmax * 1.15)
    
    p = outdir / "blok_functiegroep.png"
    save_fig(fig, p)
    return p


def create_yes_no_card(df: pd.DataFrame, outdir: Path,
                       title: Optional[str] = None,
                       column: Optional[str] = None,
                       figsize: Optional[Tuple[float, float]] = None) -> Path:
    """Yes/No big numbers. If column not provided/absent, uses dummy 62/38."""
    if column and column in df.columns:
        s = df[column].astype(str).str.strip().str.lower()
        yes = int((s == "ja").sum())
        no  = int((s == "nee").sum())
    else:
        # dummy
        yes, no = 62, 38

    total = max(yes+no, 1)
    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (11.5, 5.4), title=title or "Betrokken bij keuze samenwerking")

    ax = fig.add_axes([rect[0]+0.06, rect[1]+0.16, rect[2]-0.12, rect[3]-0.28]); ax.axis("off")
    # JA
    fig.text(rect[0]+0.25, rect[1]+0.52, "Ja", fontsize=60, weight="bold", color=TEAL, ha="center")
    fig.text(rect[0]+0.25, rect[1]+0.35, f"{yes} ({yes/total*100:.0f}%)", fontsize=26, weight="bold", ha="center")
    # NEE
    fig.text(rect[0]+0.75, rect[1]+0.52, "Nee", fontsize=60, weight="bold", color="#EF4444", ha="center")
    fig.text(rect[0]+0.75, rect[1]+0.35, f"{no} ({no/total*100:.0f}%)", fontsize=26, weight="bold", ha="center")

    p = outdir / "blok_ja_nee.png"
    save_fig(fig, p); return p


# ---- Improved Likert chart with percentage display (from notebook) ----
def create_aspecten_likert_card(df: pd.DataFrame,
                                aspect_map: Dict[str,str],
                                outdir: Path,
                                title: Optional[str] = None,
                                sort_by: Optional[str] = "mean",
                                figsize: Optional[Tuple[float, float]] = None) -> Optional[Path]:
    """
    Create an aspecten likert chart with percentage display and dynamic label fitting.
    Based on the improved notebook implementation.
    """
    # colors
    C_NEG2 = "#EF4444"; C_NEG1 = "#F59E0B"; C_NEUT = "#9CA3AF"; C_POS1 = "#34D399"; C_POS2 = "#10B981"
    COLORS = [C_NEG2, C_NEG1, C_NEUT, C_POS1, C_POS2]

    # Also supports string labels if they ever appear
    txt2code = {
        "zeer ontevreden": 1, "very dissatisfied": 1,
        "ontevreden": 2,     "dissatisfied": 2,
        "neutraal": 3,       "neutral": 3,
        "tevreden": 4,       "satisfied": 4,
        "zeer tevreden": 5,  "very satisfied": 5,
        "geen mening": np.nan, "nvt": np.nan
    }
    def normalize_ser(s: pd.Series) -> pd.Series:
        # numeric path: keep only 1..5; treat 6 ("geen mening") as NaN
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().any():
            return sn.where(sn.isin([1,2,3,4,5]), np.nan)
        # textual path
        st = s.astype(str).str.strip().str.lower().map(txt2code)
        return pd.to_numeric(st, errors="coerce")

    rows = []
    for label, col in aspect_map.items():
        if col not in df.columns:
            rows.append((label, np.array([0,0,0,0,0], dtype=float), 0, np.nan))
            continue
        s = normalize_ser(df[col]).dropna().astype(int)
        N = len(s)
        if N == 0:
            rows.append((label, np.array([0,0,0,0,0], dtype=float), 0, np.nan))
            continue
        counts = s.value_counts().reindex([1,2,3,4,5], fill_value=0).astype(float).values
        fracs  = counts / counts.sum()
        mean   = (np.arange(1,6) * fracs).sum()
        rows.append((label, fracs, int(counts.sum()), float(mean)))

    # Sort with highest μ first - ensure NaN values are placed at the end
    if sort_by == "mean":
        rows.sort(key=lambda r: np.nan_to_num(r[3], nan=-np.inf), reverse=False)
    elif sort_by == "positive":
        rows.sort(key=lambda r: (r[1][3] + r[1][4]), reverse=True)
    elif sort_by in ("very_pos", "label5"):
        rows.sort(key=lambda r: r[1][4], reverse=True)

    labels = [r[0] for r in rows]
    frmat  = np.vstack([r[1] for r in rows]) if rows else np.zeros((0,5))
    Ns     = [r[2] for r in rows]
    means  = [r[3] for r in rows]

    k = len(labels)
    if k == 0:
        print("Geen aspecten om te tonen.")
        return None

    apply_theme()
    fig = plt.figure(figsize=figsize or (16.0, 1.0*k + 3.8), dpi=160)
    ax_card = fig.add_axes([0,0,1,1]); ax_card.axis("off")
    ax_card.set_xlim(0,1); ax_card.set_ylim(0,1)
    ax_card.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.88,
                                     boxstyle="round,pad=0.018,rounding_size=0.06",
                                     facecolor=CARD, edgecolor=CARD))
    fig.text(0.06, 0.84, title or "Vergelijking van aspecten professionals", fontsize=28, weight="bold", color=TEXT, ha="left", va="center")

    # Dynamic left margin so long labels sit inside the card
    max_label_len = max((len(label) for label in labels), default=0)
    label_space = min(0.35, 0.08 + max_label_len * 0.008)
    left = max(0.08, label_space)
    right = 0.80
    width = right - left
    bottom, height = 0.20, 0.58

    ax = fig.add_axes([left, bottom, width, height])
    ax.set_facecolor("none")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=1.0, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, k-0.5)

    # Verbeterde label weergave
    label_fontsize = min(16, max(12, 250 / max(max_label_len, 1)))
    ax.set_yticks(range(k), labels, color=TEXT, fontsize=label_fontsize, weight="bold")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%","25%","50%","75%","100%"], color=SUBTLE, fontsize=12)
    ax.tick_params(axis="y", colors=TEXT, pad=12)

    h = 0.72
    thresh = 0.08
    for i in range(k):
        f1, f2, f3, f4, f5 = frmat[i]
        leftpos = 0.0
        for w, c in zip([f1,f2,f3,f4,f5], [C_NEG2,C_NEG1,C_NEUT,C_POS1,C_POS2]):
            if w <= 0: 
                continue
            ax.barh(i, w, left=leftpos, color=c, edgecolor="none", height=h)
            if w >= thresh:
                ax.text(leftpos + w/2, i, f"{w*100:.0f}%", color="white",
                        ha="center", va="center", fontsize=12, weight="bold")
            leftpos += w

        m = means[i]
        text_x = 1.04
        if not np.isnan(m):
            ax.text(text_x, i, f"μ {m:.2f}".replace(".", ","), color=TEXT, ha="left", va="center",
                    fontsize=12, weight="bold", transform=ax.transData, clip_on=False)
        ax.text(text_x, i-0.25, f"N={Ns[i]}", color=SUBTLE, ha="left", va="center",
                fontsize=11, transform=ax.transData, clip_on=False)

    # Verbeterde legenda
    legend_labels = ["Zeer ontevreden (1)", "Ontevreden (2)", "Neutraal (3)", "Tevreden (4)", "Zeer tevreden (5)"]
    legend_y = 0.105
    for j, (lab, col) in enumerate(zip(legend_labels, COLORS)):
        x0 = 0.06 + j*0.176
        ax_card.add_patch(FancyBboxPatch((x0, legend_y), 0.028, 0.04,
                                         boxstyle="round,pad=0.004,rounding_size=0.006",
                                         facecolor=col, edgecolor="none", transform=ax_card.transAxes))
        fig.text(x0 + 0.032, legend_y + 0.02, lab, color=SUBTLE, fontsize=12, va="center", weight="bold")

    # Uitleg tekst rechtsonder
    fig.text(0.94, 0.06, "* μ is het gewogen gemiddelde (1-5 schaal)", color=TEXT, fontsize=11, 
             ha="right", va="bottom", weight="bold")

    p = outdir / "blok_aspecten_likert.png"
    save_fig(fig, p)
    return p


def create_aspecten_systeem_likert_card(df: pd.DataFrame,
                                        aspect_map: Dict[str,str],
                                        outdir: Path,
                                        title: Optional[str] = None,
                                        sort_by: Optional[str] = "mean",
                                        figsize: Optional[Tuple[float, float]] = None) -> Optional[Path]:
    """
    Create an aspecten likert chart for systemen with percentage display and dynamic label fitting.
    Based on the improved notebook implementation for systems.
    """
    if not aspect_map:
        return None
    
    # colors
    C_NEG2 = "#EF4444"; C_NEG1 = "#F59E0B"; C_NEUT = "#9CA3AF"; C_POS1 = "#34D399"; C_POS2 = "#10B981"
    COLORS = [C_NEG2, C_NEG1, C_NEUT, C_POS1, C_POS2]

    # Also supports string labels if they ever appear
    txt2code = {
        "zeer ontevreden": 1, "very dissatisfied": 1,
        "ontevreden": 2,     "dissatisfied": 2,
        "neutraal": 3,       "neutral": 3,
        "tevreden": 4,       "satisfied": 4,
        "zeer tevreden": 5,  "very satisfied": 5,
        "geen mening": np.nan, "nvt": np.nan
    }
    def normalize_ser(s: pd.Series) -> pd.Series:
        # numeric path: keep only 1..5; treat 6 ("geen mening") as NaN
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().any():
            return sn.where(sn.isin([1,2,3,4,5]), np.nan)
        # textual path
        st = s.astype(str).str.strip().str.lower().map(txt2code)
        return pd.to_numeric(st, errors="coerce")

    rows = []
    for label, col in aspect_map.items():
        if col not in df.columns:
            rows.append((label, np.array([0,0,0,0,0], dtype=float), 0, np.nan))
            continue
        s = normalize_ser(df[col]).dropna().astype(int)
        N = len(s)
        if N == 0:
            rows.append((label, np.array([0,0,0,0,0], dtype=float), 0, np.nan))
            continue
        counts = s.value_counts().reindex([1,2,3,4,5], fill_value=0).astype(float).values
        fracs  = counts / counts.sum()
        mean   = (np.arange(1,6) * fracs).sum()
        rows.append((label, fracs, int(counts.sum()), float(mean)))

    # Sort with highest μ first - ensure NaN values are placed at the end
    if sort_by == "mean":
        rows.sort(key=lambda r: np.nan_to_num(r[3], nan=-np.inf), reverse=False)
    elif sort_by == "positive":
        rows.sort(key=lambda r: (r[1][3] + r[1][4]), reverse=True)
    elif sort_by in ("very_pos", "label5"):
        rows.sort(key=lambda r: r[1][4], reverse=True)

    labels = [r[0] for r in rows]
    frmat  = np.vstack([r[1] for r in rows]) if rows else np.zeros((0,5))
    Ns     = [r[2] for r in rows]
    means  = [r[3] for r in rows]

    k = len(labels)
    if k == 0:
        print("Geen aspecten om te tonen.")
        return None

    apply_theme()
    fig = plt.figure(figsize=figsize or (16.0, 1.0*k + 3.8), dpi=160)
    ax_card = fig.add_axes([0,0,1,1]); ax_card.axis("off")
    ax_card.set_xlim(0,1); ax_card.set_ylim(0,1)
    ax_card.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.88,
                                     boxstyle="round,pad=0.018,rounding_size=0.06",
                                     facecolor=CARD, edgecolor=CARD))
    fig.text(0.06, 0.84, title or "Vergelijking van aspecten systeem", fontsize=28, weight="bold", color=TEXT, ha="left", va="center")

    # Dynamic left margin so long labels sit inside the card
    max_label_len = max((len(label) for label in labels), default=0)
    label_space = min(0.35, 0.08 + max_label_len * 0.008)
    left = max(0.08, label_space)
    right = 0.80
    width = right - left
    bottom, height = 0.20, 0.58

    ax = fig.add_axes([left, bottom, width, height])
    ax.set_facecolor("none")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=1.0, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, k-0.5)

    # Verbeterde label weergave
    label_fontsize = min(16, max(12, 250 / max(max_label_len, 1)))
    ax.set_yticks(range(k), labels, color=TEXT, fontsize=label_fontsize, weight="bold")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%","25%","50%","75%","100%"], color=SUBTLE, fontsize=12)
    ax.tick_params(axis="y", colors=TEXT, pad=12)

    h = 0.72
    thresh = 0.08
    for i in range(k):
        f1, f2, f3, f4, f5 = frmat[i]
        leftpos = 0.0
        for w, c in zip([f1,f2,f3,f4,f5], [C_NEG2,C_NEG1,C_NEUT,C_POS1,C_POS2]):
            if w <= 0: 
                continue
            ax.barh(i, w, left=leftpos, color=c, edgecolor="none", height=h)
            if w >= thresh:
                ax.text(leftpos + w/2, i, f"{w*100:.0f}%", color="white",
                        ha="center", va="center", fontsize=12, weight="bold")
            leftpos += w

        m = means[i]
        text_x = 1.04
        if not np.isnan(m):
            ax.text(text_x, i, f"μ {m:.2f}".replace(".", ","), color=TEXT, ha="left", va="center",
                    fontsize=12, weight="bold", transform=ax.transData, clip_on=False)
        ax.text(text_x, i-0.25, f"N={Ns[i]}", color=SUBTLE, ha="left", va="center",
                fontsize=11, transform=ax.transData, clip_on=False)

    # Verbeterde legenda
    legend_labels = ["Zeer ontevreden (1)", "Ontevreden (2)", "Neutraal (3)", "Tevreden (4)", "Zeer tevreden (5)"]
    legend_y = 0.105
    for j, (lab, col) in enumerate(zip(legend_labels, COLORS)):
        x0 = 0.06 + j*0.176
        ax_card.add_patch(FancyBboxPatch((x0, legend_y), 0.028, 0.04,
                                         boxstyle="round,pad=0.004,rounding_size=0.006",
                                         facecolor=col, edgecolor="none", transform=ax_card.transAxes))
        fig.text(x0 + 0.032, legend_y + 0.02, lab, color=SUBTLE, fontsize=12, va="center", weight="bold")

    # Uitleg tekst rechtsonder
    fig.text(0.94, 0.06, "* μ is het gewogen gemiddelde (1-5 schaal)", color=TEXT, fontsize=11, 
             ha="right", va="bottom", weight="bold")

    p = outdir / "blok_aspecten_systeem_likert.png"
    save_fig(fig, p)
    return p


def create_priority_matrix_card(df: pd.DataFrame, outdir: Path, 
                                n_themes: int = 6,
                                themes_column: str = "THEMES",
                                nps_column: str = "NPS",
                                mainthemes: bool = True,
                                vertical_grid_position: Optional[float] = None,
                                horizontal_grid_position: Optional[float] = None,
                                title: Optional[str] = None,
                                figsize: Optional[Tuple[float, float]] = None) -> Optional[Path]:
    """
    Create a priority matrix card showing theme importance vs satisfaction.
    
    Args:
        df: DataFrame with themes and NPS data
        outdir: Output directory
        n_themes: Number of top themes to show
        themes_column: Column name containing themes (JSON string or dict)
        nps_column: Column name containing NPS scores
        mainthemes: If True, reduce themes to main themes only
        vertical_grid_position: Custom vertical grid line position
        horizontal_grid_position: Custom horizontal grid line position
    """
    if themes_column not in df.columns or nps_column not in df.columns:
        return None
    
    # Build set of valid theme strings
    valid_theme_strings = set()
    # Add main themes directly
    for main in thema_structuur.keys():
        valid_theme_strings.add(main)
    # Add "MainTheme > SubTheme" format
    for main, subs in thema_structuur.items():
        for sub in subs:
            valid_theme_strings.add(f"{main} > {sub}")

    def extract_theme_keys(row):
        if isinstance(row, str):
            try:
                d = json.loads(row)
                return [k for k in d.keys() if k.strip() and (k in valid_theme_strings or k in thema_structuur)]
            except Exception:
                return []
        elif isinstance(row, dict):
            return [k for k in row.keys() if k.strip() and (k in valid_theme_strings or k in thema_structuur)]
        return []

    df_work = df.copy()
    df_work['THEMES_PARSED'] = df_work[themes_column].apply(extract_theme_keys)

    # If mainthemes, reduce each theme to its main theme
    if mainthemes:
        def get_main_themes(theme_list):
            result = []
            for t in theme_list:
                if '>' in t and t in valid_theme_strings:
                    result.append(t.split('>')[0].strip())
                elif t in thema_structuur:  # Direct main theme
                    result.append(t)
                elif t in valid_theme_strings:  # Other valid format
                    result.append(t)
            return result
        df_work['THEMES_PARSED'] = df_work['THEMES_PARSED'].apply(get_main_themes)

    # Count theme occurrences
    theme_counts = Counter(
        item for sublist in df_work['THEMES_PARSED'] for item in sublist if item.strip()
    )
    all_themes = [theme for theme in theme_counts if theme.strip()]

    # Create binary columns for all themes
    for theme in all_themes:
        col_name = theme.replace(" ", "_").replace("&", "and")
        df_work[col_name] = df_work['THEMES_PARSED'].apply(lambda x: 1 if theme in x else 0)

    # Clean the NPS column
    df_work[nps_column] = pd.to_numeric(df_work[nps_column], errors='coerce')
    df_work = df_work.dropna(subset=[nps_column])

    # Calculate mean NPS and correlation for each theme
    theme_means = {}
    theme_correlations = {}
    for theme in all_themes:
        col_name = theme.replace(" ", "_").replace("&", "and")
        theme_means[theme] = df_work.loc[df_work[col_name] == 1, nps_column].mean()
        theme_correlations[theme] = df_work[col_name].corr(df_work[nps_column])

    # Calculate percentages for each theme
    theme_percentages = {}
    total_rows = len(df_work)
    for theme in all_themes:
        col_name = theme.replace(" ", "_").replace("&", "and")
        theme_percentages[theme] = (df_work[col_name].sum() / total_rows) * 100

    # Calculate global means
    all_percentages = [theme_percentages[theme] for theme in all_themes]
    all_mean_scores = [theme_means[theme] for theme in all_themes]
    mean_percentage = np.mean(all_percentages) if all_percentages else 0
    mean_mean_score = np.mean(all_mean_scores) if all_mean_scores else 0

    # Select top N themes for plotting
    top_themes = [theme for theme, count in theme_counts.most_common(n_themes)]
    # Filter out themes with NaN mean, percentage, or correlation
    filtered_themes = [theme for theme in top_themes if not (
        pd.isna(theme_percentages.get(theme)) or pd.isna(theme_means.get(theme)) or pd.isna(theme_correlations.get(theme))
    )]
    
    if not filtered_themes:
        return None
        
    themes = list(filtered_themes)
    percentages = [theme_percentages[theme] for theme in themes]
    mean_scores = [theme_means[theme] for theme in themes]
    
    # Determine grid positions - use custom values if provided, otherwise use global means
    v_grid = vertical_grid_position if vertical_grid_position is not None else mean_percentage
    h_grid = horizontal_grid_position if horizontal_grid_position is not None else mean_mean_score
    
    # Assign quadrants for top N themes
    quadrant_labels = []
    for percentage, mean_score in zip(percentages, mean_scores):
        if percentage >= v_grid and mean_score >= h_grid:
            quadrant_labels.append('Belangrijk met hoge Score')
        elif percentage >= v_grid and mean_score < h_grid:
            quadrant_labels.append('Belangrijk met lage Score')
        elif percentage < v_grid and mean_score >= h_grid:
            quadrant_labels.append('Minder belangrijk met hoge Score')
        else:
            quadrant_labels.append('Minder belangrijk met lage Score')

    # Apply card theme
    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (16, 12), title=title or "Prioriteitenmatrix")
    
    # Create inner axes for the scatter plot
    ax = add_inner_axes(fig, rect, padding=(0.08, 0.12, 0.08, 0.08))
    
    # Color mapping for quadrants - using card theme colors
    category_colors = {
        'Belangrijk met hoge Score': TEAL,      # green
        'Belangrijk met lage Score': '#EF4444', # red
        'Minder belangrijk met hoge Score': SUBTLE,  # grey
        'Minder belangrijk met lage Score': SUBTLE   # grey
    }
    
    # Create the scatter plot
    for i, (x, y, theme, quad) in enumerate(zip(percentages, mean_scores, themes, quadrant_labels)):
        color = category_colors[quad]
        # Make dots much bigger
        size = 500 if quad.startswith("Belangrijk") else 300
        
        # Plot the point
        ax.scatter(x, y, c=color, s=size, alpha=0.8, edgecolors='white', linewidth=2)
        
        # Add theme label - make important themes bold
        fontweight = 'bold' if quad.startswith("Belangrijk") else 'normal'
        fontsize = 12 if quad.startswith("Belangrijk") else 11
        
        # Position text with better offset to avoid overlap
        ax.annotate(theme, (x, y), xytext=(10, 10), textcoords='offset points',
                   fontsize=fontsize, color=TEXT, ha='left', va='bottom',
                   weight=fontweight, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD, edgecolor='none', alpha=0.8))
    
    # Calculate better axis limits with more space
    x_margin = max(percentages) * 0.3 if percentages else 10  # 30% margin
    y_range = max(mean_scores) - min(mean_scores) if mean_scores and len(mean_scores) > 1 else 20
    y_margin = max(y_range * 0.3, 10)  # At least 30% margin or 10 units
    
    x_min = -5  # Start below 0 for better visibility
    x_max = max(percentages) + x_margin if percentages else 35
    y_min = min(mean_scores) - y_margin if mean_scores else -40
    y_max = max(mean_scores) + y_margin if mean_scores else 50
    
    # Draw grid lines
    ax.axvline(x=v_grid, color=TEXT, linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=h_grid, color=TEXT, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Set axis labels and limits
    ax.set_xlabel(f"Aantal keer genoemd (%) van {total_rows} totaal", color=TEXT, fontsize=14)
    ax.set_ylabel("Gemiddelde NPS", color=TEXT, fontsize=14)
    
    # Set limits with better padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Style the axes
    ax.grid(True, alpha=0.3, color=GRID)
    ax.tick_params(colors=SUBTLE)
    
    # Add legend with bigger markers
    legend_elements = []
    for quad, color in category_colors.items():
        if quad in quadrant_labels:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=12, label=quad))
    
    if legend_elements:
        legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
                          facecolor=CARD, edgecolor='none', labelcolor=TEXT, fontsize=12)
        legend.get_frame().set_alpha(0.9)
        # Add border to legend
        legend.get_frame().set_linewidth(1)
        legend.get_frame().set_edgecolor(SUBTLE)
    
    p = outdir / "blok_priority_matrix.png"
    save_fig(fig, p)
    return p


# -------------------- OPEN ANSWER ANALYSE CARD --------------------
def create_open_answer_analysis_card(df: pd.DataFrame,
                                     outdir: Path,
                                     themes_column: str = "THEMES",
                                     title: Optional[str] = None,
                                     top_k: int = 3,
                                     subtitle_label: Optional[str] = None,
                                     figsize: Optional[Tuple[float, float]] = None) -> Optional[Path]:
    """Create a text card summarizing top-K themes with % mentioned and a representative quote.

    Rules:
    - Only consider rows where THEMES is not NaN
    - Each theme counts at most once per row (unique per row)
    - % is computed over all non-NaN THEMES rows
    - For each theme, choose the longest quote among rows that contain that theme
    - THEMES values are expected to be JSON objects mapping theme -> quote (string)
    """
    if themes_column not in df.columns:
        return None

    df_non_null = df[df[themes_column].notna()].copy()
    total_non_null = len(df_non_null)
    if total_non_null == 0:
        return None

    theme_to_count: Dict[str, int] = {}
    theme_to_best_quote: Dict[str, str] = {}

    def parse_themes_cell(cell) -> Dict[str, str]:
        if isinstance(cell, dict):
            return {str(k): ("" if v is None else str(v)) for k, v in cell.items()}
        if isinstance(cell, str):
            try:
                obj = json.loads(cell)
                if isinstance(obj, dict):
                    return {str(k): ("" if v is None else str(v)) for k, v in obj.items()}
            except Exception:
                return {}
        return {}

    for _, row in df_non_null.iterrows():
        mapping = parse_themes_cell(row[themes_column])
        if not mapping:
            continue
        # unique per row
        unique_themes = set(k for k in mapping.keys() if str(k).strip())
        for theme in unique_themes:
            theme_to_count[theme] = theme_to_count.get(theme, 0) + 1
            quote_candidate = mapping.get(theme, "").strip()
            prev_best = theme_to_best_quote.get(theme, "")
            if len(quote_candidate) > len(prev_best):
                theme_to_best_quote[theme] = quote_candidate

    if not theme_to_count:
        return None

    # Sort by count desc, then theme label asc for stability
    sorted_items = sorted(theme_to_count.items(), key=lambda kv: (-kv[1], kv[0]))
    top_items = sorted_items[:max(1, min(top_k, len(sorted_items)))]

    # Prepare display tuples: (theme, pct, quote)
    display_rows: List[Tuple[str, float, str]] = []
    for theme, cnt in top_items:
        pct = (cnt / total_non_null) * 100.0
        quote = theme_to_best_quote.get(theme, "")
        display_rows.append((theme, pct, quote))

    # Figure layout (bigger card)
    apply_theme()
    fig, ax_card, rect = start_card(figsize=figsize or (16.0, 10.0), title=title or "Open antwoord tekstanalyse")

    # Determine subtitle label if not provided
    if subtitle_label is None and "Label" in df.columns:
        try:
            uniq = df["Label"].dropna().astype(str).unique()
            if len(uniq) >= 1:
                subtitle_label = str(uniq[0])
        except Exception:
            subtitle_label = None

    # Subtitle text
    subtitle_text = (
        f"Waarom zou je de diensten van {subtitle_label} wel of niet aanbevelen?"
        if subtitle_label else
        "Waarom zou je de diensten wel of niet aanbevelen?"
    )
    fig.text(rect[0] + 0.06, rect[1] + rect[3] - 0.12, subtitle_text,
             fontsize=16, color=SUBTLE, ha="left", va="center")

    # Two-column layout
    margin = 0.06
    content_left = rect[0] + margin
    content_right = rect[0] + rect[2] - margin
    content_width = content_right - content_left
    
    # Column setup
    col_gap = 0.04
    left_col_width = (content_width - col_gap) / 2
    right_col_width = left_col_width
    
    left_col_left = content_left
    left_col_right = left_col_left + left_col_width
    right_col_left = left_col_right + col_gap
    
    # Column headers
    header_y = rect[1] + rect[3] - 0.20
    fig.text(left_col_left + left_col_width/2, header_y, "3 meest genoemde subthema's", 
             fontsize=18, weight="bold", ha="center", va="center", color=TEXT)
    fig.text(right_col_left + right_col_width/2, header_y, "Quotes", 
             fontsize=18, weight="bold", ha="center", va="center", color=TEXT)
    
    # Theme colors - consistent with card theme (no red/green)
    theme_colors = [PURPLE, "#4E6591", "#2D7DA6"]  # purple, blue, darker blue
    
    # Calculate available space for content
    content_start_y = header_y - 0.06
    content_bottom = rect[1] + 0.08  # Bottom margin
    available_height = content_start_y - content_bottom
    
    # Calculate card dimensions to fit equally in available space with gaps
    num_cards = len(display_rows)
    card_gap = 0.05  # Increased gap between cards to prevent touching
    total_gap_space = card_gap * (num_cards - 1)
    card_height = (available_height - total_gap_space) / num_cards
    
    # Ensure minimum card height
    card_height = max(card_height, 0.12)
    
    # Render each theme and quote
    for idx, (theme, pct, quote) in enumerate(display_rows, start=1):
        theme_color = theme_colors[(idx-1) % len(theme_colors)]
        
        # Calculate position for this card (from top down)
        card_top = content_start_y - (idx-1) * (card_height + card_gap)
        card_bottom = card_top - card_height
        
        # Theme card background
        ax_card.add_patch(FancyBboxPatch(
            (left_col_left, card_bottom), left_col_width, card_height,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=theme_color, edgecolor="none", alpha=0.9, zorder=1
        ))
        
        # Parse theme into main and sub
        theme_parts = theme.split(" > ")
        main_theme = theme_parts[0] if theme_parts else theme
        sub_theme = theme_parts[1] if len(theme_parts) > 1 else ""
        
        # Theme title - adjust positioning to avoid collision
        theme_text = f"{idx}. {main_theme}"
        title_y = card_bottom + card_height * 0.75
        fig.text(left_col_left + 0.02, title_y, theme_text, 
                fontsize=13, weight="bold", ha="left", va="center", color=TEXT)
        
        # Sub-theme with label
        if sub_theme:
            # "Subtheme:" label
            subtheme_label_y = card_bottom + card_height * 0.50
            fig.text(left_col_left + 0.02, subtheme_label_y, "Subthema:", 
                    fontsize=12, ha="left", va="center", color=SUBTLE)
            
            # Actual subtheme text - bigger and bold in grey
            subtheme_y = card_bottom + card_height * 0.32
            fig.text(left_col_left + 0.02, subtheme_y, sub_theme, 
                    fontsize=13, weight="bold", ha="left", va="center", color=SUBTLE)
        
        # Percentage - positioned to avoid collision, smaller font
        pct_y = card_bottom + card_height * 0.25
        fig.text(left_col_left + left_col_width - 0.01, pct_y, f"{pct:.0f}%", 
                fontsize=16, weight="bold", ha="right", va="center", color=TEAL)
        fig.text(left_col_left + left_col_width - 0.01, pct_y - 0.03, "genoemd", 
                fontsize=10, weight="bold", ha="right", va="center", color=TEAL)
        
        # Right column: Quote box - same height as theme card
        if quote:
            wrapped = textwrap.fill(quote, width=45)
            
            # Quote box background
            ax_card.add_patch(FancyBboxPatch(
                (right_col_left, card_bottom), right_col_width, card_height,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                facecolor="#3A3D42", edgecolor=SUBTLE, linewidth=1.5, zorder=1
            ))
            # Quote text with inline citations - bigger font
            text_y = card_bottom + card_height/2
            quote_with_marks = f'"{wrapped}"'
            fig.text(right_col_left + right_col_width/2, text_y, quote_with_marks, 
                    fontsize=15, ha="center", va="center", color=TEXT, style="italic")

    p = outdir / "blok_open_antwoord_analyse.png"
    save_fig(fig, p)
    return p

# -------------------- RUNNER --------------------
def get_custom_params(plot_key: str, customizations: Optional[Dict] = None) -> Dict:
    """Extract custom parameters for a specific plot."""
    if not customizations or plot_key not in customizations:
        return {}
    params = customizations[plot_key]
    result: Dict[str, Union[str, Tuple[float, float]]] = {}
    if isinstance(params, dict):
        if 'title' in params and params['title']:
            result['title'] = params['title']
        if 'width' in params and 'height' in params and params['width'] and params['height']:
            result['figsize'] = (float(params['width']), float(params['height']))
        if 'column' in params and params['column']:
            result['column'] = params['column']
        # Priority matrix custom quadrant lines
        if plot_key == "priority_matrix":
            v = params.get('vertical_grid_position')
            h = params.get('horizontal_grid_position')
            try:
                if v is not None and v != "":
                    result['vertical_grid_position'] = float(v)
            except Exception:
                pass
            try:
                if h is not None and h != "":
                    result['horizontal_grid_position'] = float(h)
            except Exception:
                pass
    return result


def run_all(excel: Union[str, Path],
            sheet: str,
            label: Optional[str],
            outdir: Union[str, Path],
            customizations: Optional[Dict] = None) -> Dict[str, Union[str, List[str]]]:
    apply_theme()
    outdir = ensure_outdir(outdir)
    df = load_df(excel, sheet)

    # make sure that rows with missing NPS scores have sys_respondentStatus = 1
    df = df.copy()
    
    # Handle different NPS column names
    nps_col = None
    if "Blok1_NPS_KPI_page6_Text" in df.columns:
        nps_col = "Blok1_NPS_KPI_page6_Text"
    elif "NPS" in df.columns:
        nps_col = "NPS"
    
    # Only update sys_respondentStatus if we have the columns and NPS column exists
    if nps_col and "sys_respondentStatus" in df.columns:
        mask = df[nps_col].isna()
        df["sys_respondentStatus"] = df["sys_respondentStatus"].where(~mask, 1)

    df_lab = filter_by_label(df, label)


    # For respondent count, use full dataframe (all statuses)
    # For all other plots, filter to only status 2 and 3 (gedeeltelijk + volledig)
    df_filtered = df_lab[(df_lab["sys_respondentStatus"] == 2) | (df_lab["sys_respondentStatus"] == 3)] if "sys_respondentStatus" in df_lab.columns else df_lab

    outputs = {}

    # Use full dataframe for respondent count
    custom_params = get_custom_params("respondent_count", customizations)
    p = plot_respondent_count(df_lab, outdir, **custom_params);                outputs["respondent_count"] = str(p) if p else None
    
    # Use filtered dataframe for all other plots
    custom_params = get_custom_params("aanbeveling", customizations)
    p = plot_aanbeveling_distribution(df_filtered, outdir, **custom_params);        outputs["aanbeveling"] = str(p) if p else None
    custom_params = get_custom_params("nps_gauge", customizations)
    p = plot_nps_white_gauge_from_scores(df_filtered, outdir, **custom_params);     outputs["nps_gauge"]   = str(p) if p else None
    custom_params = get_custom_params("duur", customizations)
    p = plot_duur_samenwerking_gauges(df_filtered, outdir, **custom_params);        outputs["duur"]        = str(p) if p else None
    custom_params = get_custom_params("diensten", customizations)
    p = plot_afgenomen_diensten_from_flags(df_filtered, outdir, label, **custom_params);   outputs["diensten"]    = str(p) if p else None
    custom_params = get_custom_params("info_pie", customizations)
    p = create_informatievoorziening_card(df_filtered, outdir, **custom_params);    outputs["info_pie"]    = str(p) if p else None
    
    # PDG-specific chart for omgang informatie
    if label == "PDG Health Services":
        custom_params = get_custom_params("omgang_info", customizations)
        p = create_omgang_informatie_card(df_filtered, outdir, **custom_params);    outputs["omgang_info"] = str(p) if p else None
    
    custom_params = get_custom_params("smileys", customizations)
    p = draw_smiley_impact_card(df_filtered, outdir, **custom_params);              outputs["smileys"]     = str(p) if p else None
    custom_params = get_custom_params("functiegroep", customizations)
    p = create_functiegroep_card(df_filtered, outdir, **custom_params);             outputs["functiegroep"] = str(p) if p else None
    # p = create_yes_no_card(df_filtered, outdir, title="Betrokken bij keuze samenwerking"); outputs["ja_nee"] = str(p)

    # Likert aspects — label-specific mapping based on notebook implementation
    aspect_map = {}
    if label == "Resolu":
        aspect_map = {
            "Communicatie": "Blok_Resolu_Professional_page124_Question1",
            "Bereikbaarheid en zichtbaarheid": "Blok_Resolu_Professional_page124_Question2",
            "Meedenken": "Blok_Resolu_Professional_page124_Question3",
            "Bereikbaarheid en zichtbaarheid": "Blok_Resolu_Professional_page124_Question4",
            "Kennis en expertise": "Blok_Resolu_Professional_page124_Question5",
        }
    elif label == "PDG Health Services":
        aspect_map = {
            "Contact met professionals": "Blok8_CategorieProfessional_page117_Question1",
            "Manier van terugkoppeling": "Blok8_CategorieProfessional_page117_Question2",
            "Communicatie": "Blok8_CategorieProfessional_page117_Question3",
            "Bereikbaarheid": "Blok8_CategorieProfessional_page117_Question4",
            "Snelheid van leveren": "Blok8_CategorieProfessional_page117_Question5",
            "Toepasbaarheid & kwaliteit adviezen": "Blok8_CategorieProfessional_page117_Question6",
            "Het aanbod": "Blok8_CategorieProfessional_page117_Question7",
        }
    
    # Filter to only existing columns
    aspect_map = {k:v for k,v in aspect_map.items() if v in df_filtered.columns}
    if aspect_map:
        custom_params = get_custom_params("aspecten_likert", customizations)
        p = create_aspecten_likert_card(df_filtered, aspect_map, outdir, sort_by="mean", **custom_params)
        outputs["aspecten_likert"] = str(p) if p else None

    # Systemen aspects — only for PDG Health Services
    systeem_aspect_map = {}
    if label == "PDG Health Services":
        systeem_aspect_map = {
            "Gebruiksvriendelijkheid": "Blok8_Systeem_page119_Question1",
            "Functionaliteit": "Blok8_Systeem_page119_Question2",
            "Beschikbaarheid": "Blok8_Systeem_page119_Question3",
            "Support": "Blok8_Systeem_page119_Question4"
        }
    
    # Filter to only existing columns
    systeem_aspect_map = {k:v for k,v in systeem_aspect_map.items() if v in df_filtered.columns}
    if systeem_aspect_map:
        custom_params = get_custom_params("aspecten_systeem_likert", customizations)
        p = create_aspecten_systeem_likert_card(df_filtered, systeem_aspect_map, outdir, sort_by="mean", **custom_params)
        outputs["aspecten_systeem_likert"] = str(p) if p else None

    # Priority matrix and open answer analysis - requires THEMES and NPS columns (if they exist)
    if "THEMES" in df_filtered.columns:
        non_null_themes_count = int(df_filtered["THEMES"].notna().sum())
        
        # Priority matrix - only when count of non-NaN THEMES > 70
        if non_null_themes_count > 70:
            nps_col_for_matrix = None
            if "NPS" in df_filtered.columns:
                nps_col_for_matrix = "NPS"
            elif "Blok1_NPS_KPI_page6_Text" in df_filtered.columns:
                nps_col_for_matrix = "Blok1_NPS_KPI_page6_Text"
            
            if nps_col_for_matrix:
                custom_params = get_custom_params("priority_matrix", customizations)
                p = create_priority_matrix_card(df_filtered, outdir, n_themes=6, themes_column="THEMES", nps_column=nps_col_for_matrix, **custom_params)
                outputs["priority_matrix"] = str(p) if p else None

        # Open answer analysis card - always try to render when THEMES exists
        custom_params = get_custom_params("open_antwoord_analyse", customizations)
        p = create_open_answer_analysis_card(
            df_filtered,
            outdir,
            themes_column="THEMES",
            top_k=3,
            subtitle_label=label,
            **custom_params
        )
        outputs["open_antwoord_analyse"] = str(p) if p else None

    # clean up None
    outputs = {k:v for k,v in outputs.items() if v not in [None, "None", [], [""]]}
    return outputs

def main():
    parser = argparse.ArgumentParser(description="Generate KTO dashboard visuals (card style)")
    parser.add_argument("--label", required=True, help="Label to filter on (matches the 'Label' column if present)")
    parser.add_argument("--excel", default="/Users/jessevdsluis/Library/CloudStorage/OneDrive-Gedeeldebibliotheken-SpringX/SpringXSharepoint - Documenten/SpringX Analytics/Klanten/Paradigma/CX/KTO/Rapportages/KTO_Themes.xlsx", help="Path to Excel file")
    parser.add_argument("--outdir", default="./plots", help="Output folder for PNGs")
    args = parser.parse_args()

    paths = run_all(args.excel, "Sheet1", args.label, args.outdir)
    for k, v in paths.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
