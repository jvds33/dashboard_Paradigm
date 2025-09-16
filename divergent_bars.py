"""
    divergent_bars.py
    -----------------
    Build Plotly "divergent comparative bar" charts that compare NPS effect when a theme
    is mentioned vs not mentioned, and output figures for the top 3 best and top 3 worst themes.

    Assumptions (aligned with Prio_matrix.py logic):
      - Input data has a themes column (default "THEMES") containing either:
          * a JSON string mapping theme path(s) -> count/weight (e.g. {"Service > Speed": 1, ...}), or
          * a Python dict with the same structure, or
          * a list of theme strings.
      - Input data has an NPS column (default "NPS") with numeric values.
      - "Main themes" are derived by taking the text before the first " > " if mainthemes=True.

    Output:
      - Two Plotly figures: one for the top 3 best themes (largest positive mentioned-vs-not delta),
        and one for the top 3 worst themes (most negative delta).
      - If run as a script, saves HTML files and also shows the figures in a browser window.

    Usage:
      python divergent_bars.py --csv path/to/data.csv --themes-column THEMES --nps-column NPS --mainthemes 1

    Notes:
      - This module re-implements a custom divergent chart similar to
        `divergent_comparative_bar_chart(...)` referenced in the notebook.
      - Designed to be robust to various theme encodings.
    """

from __future__ import annotations

import json
import math
import re
from typing import Iterable, Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- Theme parsing helpers ----------

def _ensure_iterable_themes(x: Any) -> List[str]:
    """Return a flat list of theme strings for one row.

    Accepts JSON strings, dicts, lists, or a single string.
    Values in dicts are ignored (only keys are used).
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        # Try JSON parse first
        try:
            parsed = json.loads(x)
        except Exception:
            # Treat as a single theme string or a semicolon/comma separated list
            if ';' in x or ',' in x:
                parts = [p.strip() for p in re.split(r'[;,]', x) if p.strip()]
                return parts
            return [x]
        else:
            # Parsed JSON can be dict or list
            if isinstance(parsed, dict):
                return list(parsed.keys())
            if isinstance(parsed, list):
                out = []
                for item in parsed:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        out.extend(list(item.keys()))
                return out
            return []
    if isinstance(x, dict):
        return list(x.keys())
    if isinstance(x, (list, tuple, set)):
        out = []
        for item in x:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.extend(list(item.keys()))
        return out
    # Unknown type
    return []


def _to_main_theme(theme: str) -> str:
    """Reduce 'Main > Sub' -> 'Main'."""
    if not isinstance(theme, str):
        return ''
    return theme.split('>')[0].strip()


# ---------- Core computation ----------

def compute_theme_effects(
    data: pd.DataFrame,
    themes_column: str = "THEMES",
    nps_column: str = "NPS",
    mainthemes: bool = True,
    min_count: int = 20,
) -> pd.DataFrame:
    """Compute NPS for rows where each theme IS mentioned vs NOT mentioned.

    Returns a DataFrame with:
      - theme
      - mentioned_score, not_mentioned_score (means)
      - mentioned_n, not_mentioned_n (counts)
      - effect = mentioned_score - not_mentioned_score
    Only keeps rows where both cohorts have at least 1 sample and total >= min_count.
    """
    if themes_column not in data.columns:
        raise KeyError(f"Missing themes column: {themes_column}")
    if nps_column not in data.columns:
        raise KeyError(f"Missing NPS column: {nps_column}")

    # Precompute each row's theme list (optionally reduced to main themes)
    theme_lists: List[List[str]] = []
    all_themes: List[str] = []
    for val in data[themes_column].tolist():
        themes = _ensure_iterable_themes(val)
        if mainthemes:
            themes = [_to_main_theme(t) for t in themes if isinstance(t, str)]
        # keep non-empty
        themes = [t for t in themes if t]
        theme_lists.append(themes)
        all_themes.extend(themes)

    if not all_themes:
        # No themes found; return empty df
        return pd.DataFrame(columns=[
            "theme","mentioned_score","not_mentioned_score","mentioned_n","not_mentioned_n","effect"
        ])

    unique_themes = sorted(set(all_themes))
    nps = data[nps_column].astype(float).to_numpy()

    results = []
    for theme in unique_themes:
        mentioned_mask = np.array([theme in lst for lst in theme_lists], dtype=bool)
        not_mentioned_mask = ~mentioned_mask

        # Compute cohort stats; ignore NaNs in NPS
        m_scores = nps[mentioned_mask]
        nm_scores = nps[not_mentioned_mask]

        m_scores = m_scores[~np.isnan(m_scores)]
        nm_scores = nm_scores[~np.isnan(nm_scores)]

        m_n = int(m_scores.size)
        nm_n = int(nm_scores.size)
        total = m_n + nm_n

        if m_n == 0 or nm_n == 0 or total < min_count:
            continue

        m_mean = float(np.mean(m_scores))
        nm_mean = float(np.mean(nm_scores))
        effect = m_mean - nm_mean

        results.append({
            "theme": theme,
            "mentioned_score": m_mean,
            "not_mentioned_score": nm_mean,
            "mentioned_n": m_n,
            "not_mentioned_n": nm_n,
            "effect": effect,
            "total_n": total,
        })

    df = pd.DataFrame(results).sort_values("effect", ascending=False).reset_index(drop=True)
    return df


# ---------- Plotting ----------

def divergent_comparative_bar_chart(
    theme_name: str,
    not_mentioned_score: float, not_mentioned_n: int,
    mentioned_score: float, mentioned_n: int,
    x_name: str = "gast-NPS",
    show_theme_label: bool = True,
) -> go.Figure:
    """Make a 'divergent comparative bar' for a single theme.

    This chart shows two opposing horizontal bars:
      - Left: NPS when theme is NOT mentioned
      - Right: NPS when theme IS mentioned

    Bars diverge from a central baseline (0). We also annotate counts.
    """
    # Bars extend from 0; left bars are negative values for display
    left_val = -float(not_mentioned_score)
    right_val = float(mentioned_score)

    title_theme = theme_name.split('>')[-1].strip() if theme_name else theme_name

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[left_val], y=[title_theme],
        orientation="h",
        name=f"Niet genoemd (n={not_mentioned_n})",
        text=[f"{not_mentioned_score:.1f}"],
        textposition="outside",
        hovertemplate=f"Niet genoemd<extra></extra><br>{x_name}: {{text}}<br>n: {not_mentioned_n}",
    ))
    fig.add_trace(go.Bar(
        x=[right_val], y=[title_theme],
        orientation="h",
        name=f"Wel genoemd (n={mentioned_n})",
        text=[f"{mentioned_score:.1f}"],
        textposition="outside",
        hovertemplate=f"Wel genoemd<extra></extra><br>{x_name}: {{text}}<br>n: {mentioned_n}",
    ))

    # Layout with a central vertical line at 0
    max_abs = max(abs(left_val), abs(right_val), 10)
    fig.update_layout(
        barmode="relative",
        bargap=0.2,
        template="plotly_white",
        title=dict(text=f"Divergente vergelijking – {title_theme}", x=0.5),
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            range=[-max_abs * 1.25, max_abs * 1.25],
            title=x_name
        ),
        yaxis=dict(showticklabels=show_theme_label),
        showlegend=True,
        height=220,
        margin=dict(l=80, r=40, t=60, b=50),
    )

    return fig


# ---------- Top 3 Best / Worst wrappers ----------

def build_top3_divergent_figures(
    data: pd.DataFrame,
    themes_column: str = "THEMES",
    nps_column: str = "NPS",
    mainthemes: bool = True,
    min_count: int = 20,
) -> Tuple[List[go.Figure], List[go.Figure], pd.DataFrame]:
    """Compute theme effects and return two lists of figures (best, worst) plus the effects table."""

    effects = compute_theme_effects(
        data,
        themes_column=themes_column,
        nps_column=nps_column,
        mainthemes=mainthemes,
        min_count=min_count,
    )

    if effects.empty:
        return [], [], effects

    # Top 3 best (largest positive effect) and worst (most negative effect)
    best3 = effects.sort_values("effect", ascending=False).head(3)
    worst3 = effects.sort_values("effect", ascending=True).head(3)

    best_figs: List[go.Figure] = []
    for _, row in best3.iterrows():
        best_figs.append(divergent_comparative_bar_chart(
            theme_name=row["theme"],
            not_mentioned_score=row["not_mentioned_score"],
            not_mentioned_n=int(row["not_mentioned_n"]),
            mentioned_score=row["mentioned_score"],
            mentioned_n=int(row["mentioned_n"]),
            x_name=nps_column,
        ))

    worst_figs: List[go.Figure] = []
    for _, row in worst3.iterrows():
        worst_figs.append(divergent_comparative_bar_chart(
            theme_name=row["theme"],
            not_mentioned_score=row["not_mentioned_score"],
            not_mentioned_n=int(row["not_mentioned_n"]),
            mentioned_score=row["mentioned_score"],
            mentioned_n=int(row["mentioned_n"]),
            x_name=nps_column,
        ))

    return best_figs, worst_figs, effects


# ---------- Script entrypoint ----------

def _save_figs_as_html(figs: List[go.Figure], out_path: str, title: str):
    """Save multiple small figures into one HTML by vertical stacking."""
    if not figs:
        return None
    # stack by creating a tall subplot canvas
    rows = len(figs)
    canvas = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.12)
    for i, fig in enumerate(figs, start=1):
        for tr in fig.data:
            canvas.add_trace(tr, row=i, col=1)
        # carry over layout-adjusted height
        canvas.update_yaxes(showticklabels=True, row=i, col=1)
    canvas.update_layout(height=rows * 260, title=title, template="plotly_white", showlegend=True)
    canvas.write_html(out_path)
    return out_path


def _cli():
    import argparse, sys, os
    parser = argparse.ArgumentParser(description="Build divergent comparative bars for top 3 best/worst themes.")
    parser.add_argument("--csv", type=str, help="Path to CSV with data.")
    parser.add_argument("--themes-column", type=str, default="THEMES", help="Name of themes column.")
    parser.add_argument("--nps-column", type=str, default="NPS", help="Name of NPS column.")
    parser.add_argument("--mainthemes", type=int, default=1, help="Reduce to main themes (1) or keep full path (0).")
    parser.add_argument("--min-count", type=int, default=20, help="Minimum total sample size for a theme to be kept.")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save the HTML outputs.")
    args = parser.parse_args()

    if not args.csv or not os.path.exists(args.csv):
        print("No CSV found. Exiting.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    best_figs, worst_figs, effects = build_top3_divergent_figures(
        df,
        themes_column=args.themes_column,
        nps_column=args.nps_column,
        mainthemes=bool(args.mainthemes),
        min_count=int(args.min_count),
    )

    os.makedirs(args.outdir, exist_ok=True)
    best_path = os.path.join(args.outdir, "top3_best_divergent.html")
    worst_path = os.path.join(args.outdir, "top3_worst_divergent.html")
    effects_path = os.path.join(args.outdir, "theme_effects.csv")

    if best_figs:
        _save_figs_as_html(best_figs, best_path, title="Top 3 Beste Thema's – Divergente vergelijking")
        print(f"Saved best charts -> {best_path}")
    else:
        print("No best charts produced (no qualifying themes).")

    if worst_figs:
        _save_figs_as_html(worst_figs, worst_path, title="Top 3 Slechtste Thema's – Divergente vergelijking")
        print(f"Saved worst charts -> {worst_path}")
    else:
        print("No worst charts produced (no qualifying themes).")

    effects.to_csv(effects_path, index=False)
    print(f"Saved effects table -> {effects_path}")

if __name__ == "__main__":
    _cli()
