import json
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    # If used as a package module
    from .Themes import thema_structuur  # type: ignore
except Exception:  # pragma: no cover - fallback when run as a script
    # If run directly from the project root
    from Themes import thema_structuur  # type: ignore


DataLike = Union[str, pd.DataFrame]


def _load_dataframe(data: DataLike) -> pd.DataFrame:
    """Load a dataframe from a path or return a copy if already a DataFrame."""
    if isinstance(data, str):
        if data.lower().endswith(".csv"):
            return pd.read_csv(data)
        return pd.read_excel(data)
    return data.copy()


def _valid_theme_strings() -> set:
    """Build a set of valid theme keys based on `thema_structuur`.

    Includes both main themes and "Main > Sub" combinations.
    """
    valid = set()
    for main, subs in thema_structuur.items():
        valid.add(main)
        for sub in subs:
            valid.add(f"{main} > {sub}")
    return valid


def _extract_theme_keys(row: Union[str, dict], valid: set) -> List[str]:
    """Extract subtheme pairs from THEMES JSON using the notebook logic.

    Falls back to simple key filtering if parsing fails.
    """
    def _pairs_from(any_json) -> List[str]:
        try:
            if isinstance(any_json, str):
                return extract_theme_subtheme_pairs(any_json, thema_structuur)
            if isinstance(any_json, dict):
                return extract_theme_subtheme_pairs(json.dumps(any_json), thema_structuur)
        except Exception:
            pass
        # Fallback: simple key filter
        if isinstance(any_json, str):
            try:
                d = json.loads(any_json)
                return [k for k in d.keys() if isinstance(k, str) and k.strip() and k in valid]
            except Exception:
                return []
        if isinstance(any_json, dict):
            return [k for k in any_json.keys() if isinstance(k, str) and k.strip() and k in valid]
        return []

    return _pairs_from(row)


def extract_theme_subtheme_pairs(
    themes_json: str,
    GX_themes: dict,
) -> List[str]:
    """
    Extract valid "Main > Sub" pairs from a JSON string, following the notebook logic.
    - If key contains a subtheme, validate against GX_themes
    - If key is a main theme, scan its value text for any known subtheme mentions
    """
    if themes_json is None:
        return []
    try:
        themes_dict = json.loads(themes_json)
        pairs: List[str] = []
        for theme_key, theme_value in themes_dict.items():
            main_theme = str(theme_key).split(" > ")[0].strip()
            if " > " in str(theme_key):
                subtheme = str(theme_key).split(" > ")[1].strip()
                if main_theme in GX_themes and isinstance(GX_themes[main_theme], list):
                    if subtheme in GX_themes[main_theme]:
                        pairs.append(f"{main_theme} > {subtheme}")
            elif main_theme in GX_themes and isinstance(GX_themes[main_theme], list):
                feedback_text = str(theme_value).lower()
                for subtheme in GX_themes[main_theme]:
                    if str(subtheme).lower() in feedback_text:
                        pairs.append(f"{main_theme} > {subtheme}")
        return pairs
    except Exception:
        return []


def _reduce_to_main_themes(theme_list: List[str]) -> List[str]:
    """Reduce each theme to its main theme label when in the form 'Main > Sub'."""
    result: List[str] = []
    for t in theme_list:
        if ">" in t:
            result.append(t.split(">")[0].strip())
        elif t in thema_structuur:
            result.append(t)
    return result


def _safe_col_name(theme: str) -> str:
    """Convert a theme label to a safe column name (match `Prio_matrix.py`)."""
    return theme.replace(" ", "_").replace("&", "and")


def prepare_theme_matrix(
    data: DataLike,
    themes_column: str = "THEMES",
    nps_column: str = "NPS",
    mainthemes: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare a binary theme matrix and a clean NPS column from the input data.

    Mirrors the parsing and transformation style used in `Prio_matrix.py`.

    Returns the transformed dataframe and the list of canonical theme labels
    (matching the human-readable names, not the safe column names).
    """
    df = _load_dataframe(data)

    valid = _valid_theme_strings()
    df["THEMES_PARSED"] = df[themes_column].apply(lambda r: _extract_theme_keys(r, valid))

    if mainthemes:
        df["THEMES_PARSED"] = df["THEMES_PARSED"].apply(_reduce_to_main_themes)

    # Build the universe of themes present
    theme_universe: List[str] = sorted({t for lst in df["THEMES_PARSED"] for t in lst})

    # Create 0/1 columns per theme
    for theme in theme_universe:
        col = _safe_col_name(theme)
        df[col] = df["THEMES_PARSED"].apply(lambda x, theme=theme: 1 if theme in x else 0)

    # Clean NPS
    df[nps_column] = pd.to_numeric(df[nps_column], errors="coerce")
    df = df.dropna(subset=[nps_column])

    return df, theme_universe


def compute_theme_effects(
    df: pd.DataFrame,
    theme_labels: List[str],
    nps_column: str = "NPS",
    min_mentioned: int = 3,
) -> pd.DataFrame:
    """
    For each theme, compute the mean NPS when the theme is mentioned vs not mentioned.

    Returns a dataframe with: theme, mentioned_mean, not_mentioned_mean, effect, n_mentioned, n_not_mentioned.
    Themes with too few mentions (below `min_mentioned`) are filtered out.
    """
    rows: List[dict] = []
    total_n = len(df)
    if total_n == 0:
        return pd.DataFrame(columns=[
            "theme", "mentioned_mean", "not_mentioned_mean", "effect", "n_mentioned", "n_not_mentioned", "pct_mentioned"
        ])

    for theme in theme_labels:
        col = _safe_col_name(theme)
        if col not in df.columns:
            continue
        mask_1 = df[col] == 1
        n1 = int(mask_1.sum())
        n0 = int(total_n - n1)
        if n1 < min_mentioned or n0 <= 0:
            continue
        m1 = float(df.loc[mask_1, nps_column].mean())
        m0 = float(df.loc[~mask_1, nps_column].mean())
        rows.append({
            "theme": theme,
            "mentioned_mean": m1,
            "not_mentioned_mean": m0,
            "effect": m1 - m0,
            "n_mentioned": n1,
            "n_not_mentioned": n0,
            "pct_mentioned": (n1 / total_n) * 100.0,
        })

    return pd.DataFrame(rows).sort_values("effect", ascending=False).reset_index(drop=True)


def _convert_nps_to_american_categories(y: pd.Series) -> pd.Series:
    """Convert raw NPS scores 0-10 into american categories {-100, 0, 100}."""
    y = pd.to_numeric(y, errors="coerce")
    return (
        np.where(y <= 6, -100, np.where(y <= 8, 0, np.where(y <= 10, 100, np.nan)))
    )


def compute_theme_effects_ebm(
    df: pd.DataFrame,
    theme_labels: List[str],
    nps_column: str = "NPS",
    interactions: int = 1,
    min_mentioned: int = 3,
) -> pd.DataFrame:
    """
    Mirror the notebook's EBM approach:
    - Target is NPS converted to american categories (-100/0/100)
    - Use subtheme one-hots as X
    - Filter rows with at least one theme mentioned
    - Fit EBM Regressor and pull per-feature term scores for 0 vs 1
    """
    try:
        from interpret.glassbox import ExplainableBoostingRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "interpret is required for method='ebm'. Install with: pip install interpret"
        ) from exc

    # Build X over available theme columns
    present_cols = [c for c in [
        _safe_col_name(t) for t in theme_labels
    ] if c in df.columns]
    if not present_cols:
        return pd.DataFrame(columns=[
            "theme", "not_mentioned_score", "mentioned_score", "effect", "n_mentioned", "n_not_mentioned"
        ])

    X = df[present_cols].astype(int)
    # Keep only rows with at least one theme mentioned
    X_sum = X.sum(axis=1)
    keep_mask = X_sum > 0
    X = X.loc[keep_mask]
    y_raw = df.loc[keep_mask, nps_column]

    # Convert NPS to american categories
    y = pd.Series(_convert_nps_to_american_categories(y_raw), index=y_raw.index).astype(float)
    # Drop rows where y is missing after conversion
    not_na_mask = ~np.isnan(y.values)
    X = X.loc[not_na_mask]
    y = y.loc[not_na_mask]

    # Fit EBM
    ebm = ExplainableBoostingRegressor(interactions=interactions)
    ebm.fit(X, y)

    # Extract per-feature scores and counts
    rows: List[dict] = []
    for i, col in enumerate(X.columns):
        scores = ebm.term_scores_[i]
        # Robust indexing: prefer [1] (0 bin) and [2] (1 bin) if available, else fallback to [0], [1]
        if isinstance(scores, (list, np.ndarray)) and len(scores) >= 3:
            score_0 = float(scores[1])
            score_1 = float(scores[2])
        elif isinstance(scores, (list, np.ndarray)) and len(scores) >= 2:
            score_0 = float(scores[0])
            score_1 = float(scores[1])
        else:
            # If EBM provides unexpected shape, skip
            continue

        n1 = int((X[col] == 1).sum())
        n0 = int((X[col] == 0).sum())
        if n1 < min_mentioned or n0 <= 0:
            continue

        # Map back to human-readable theme name
        # Reverse lookup via theme_labels -> safe name
        # If multiple themes map to same safe name (unlikely), pick first
        matching = [t for t in theme_labels if _safe_col_name(t) == col]
        theme_name = matching[0] if matching else col

        rows.append({
            "theme": theme_name,
            "not_mentioned_score": score_0,
            "mentioned_score": score_1,
            "effect": score_1 - score_0,
            "n_mentioned": n1,
            "n_not_mentioned": n0,
        })

    return pd.DataFrame(rows).sort_values("effect", ascending=False).reset_index(drop=True)


def divergent_comparative_bar_chart(
    theme_name: str,
    not_mentioned_score: float,
    not_mentioned_n: int,
    mentioned_score: float,
    mentioned_n: int,
    custom_thema_title: Union[str, None] = None,
    x_name: str = "NPS",
    niet_label: bool = True,
    thema_label: bool = True,
    dataset_label: Optional[str] = None,
) -> go.Figure:
    """Create the custom Plotly divergent comparative bar chart.

    Mirrors the styling used in the notebook implementation.
    """
    theme = theme_name.split("> ")[-1].strip()
    title_theme = custom_thema_title or theme

    total = mentioned_n + not_mentioned_n
    pct_pos = (mentioned_n / total * 100.0) if total > 0 else 0.0
    pct_neg = (not_mentioned_n / total * 100.0) if total > 0 else 0.0
    total_effect = mentioned_score - not_mentioned_score

    pos_color = "#005444"
    neg_color = "#DE5912"
    mentioned_color = pos_color if mentioned_score >= 0 else neg_color
    mentioned_color2 = pos_color if mentioned_score < 0 else neg_color

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[not_mentioned_score],
        y=[theme],
        orientation="h",
        base=0,
        marker_color=mentioned_color2,
        name="Niet benoemd",
        text=[f"{int(round(pct_neg, 1))}%{('<br>Niet benoemd' if niet_label else '')}"],
        textposition="inside",
        insidetextanchor="middle",
        marker_line=dict(color="gray", width=1),
    ))

    fig.add_trace(go.Bar(
        x=[mentioned_score],
        y=[theme],
        orientation="h",
        base=0,
        marker_color=mentioned_color,
        name="Thema benoemd",
        text=[f"{int(round(pct_pos, 1))}%{('<br>Thema benoemd' if thema_label else '')}"],
        textposition="inside",
        insidetextanchor="middle",
        marker_line=dict(color="gray", width=1),
    ))

    subtitle = None
    if dataset_label:
        subtitle = dict(
            text=f"<sup>{dataset_label}</sup>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.1,
            showarrow=False,
        )

    fig.update_layout(
        barmode="stack",
        title={
            "text": f"Thema: <b>{title_theme}</b>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.8,
        },
        xaxis=dict(
            title=f"Verwacht effect op {x_name}",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
        ),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        margin=dict(l=50, r=30, t=70, b=80),
        annotations=[ann for ann in [
            dict(
                text=f"<b>[ Totale effect: {total_effect:.1f} punten ]</b>",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-1.15,
                showarrow=False,
            ),
            subtitle,
        ] if ann is not None],
        template="plotly_white",
        height=210,
        width=550,
        font=dict(family="Arial, sans-serif", size=12),
    )

    return fig


def build_top_bottom_divergent_bars(
    data: DataLike,
    themes_column: str = "THEMES",
    nps_column: str = "NPS",
    mainthemes: bool = True,
    min_mentioned: int = 3,
    top_k: int = 3,
    label_column: Optional[str] = None,
    method: str = "mean",  # "mean" or "ebm"
) -> Dict[str, List[go.Figure]]:
    """
    Create divergent comparative bars for the top-k best and worst themes
    based on the mean NPS difference between mentioned vs not mentioned.

    Returns a dict with keys 'best' and 'worst', each containing a list of
    Plotly figures in ranking order.
    """
    df, theme_universe = prepare_theme_matrix(
        data,
        themes_column=themes_column,
        nps_column=nps_column,
        mainthemes=mainthemes,
    )
    if method == "ebm":
        effects = compute_theme_effects_ebm(df, theme_universe, nps_column=nps_column, min_mentioned=min_mentioned)
    else:
        effects = compute_theme_effects(df, theme_universe, nps_column=nps_column, min_mentioned=min_mentioned)
    if effects.empty:
        return {"best": [], "worst": []}

    # Top and bottom
    best = effects.nlargest(top_k, "effect")
    worst = effects.nsmallest(top_k, "effect")

    dataset_label = None
    if label_column and label_column in df.columns:
        uniq = pd.unique(df[label_column].astype(str))
        if len(uniq) == 1:
            dataset_label = f"Label: {uniq[0]}"
        elif len(uniq) > 1:
            dataset_label = f"Label: multiple ({len(uniq)})"

    # Build figures
    figures_best: List[go.Figure] = []
    for _, row in best.iterrows():
        fig = divergent_comparative_bar_chart(
            theme_name=str(row["theme"]),
            not_mentioned_score=float(row.get("not_mentioned_mean", row.get("not_mentioned_score", 0.0))),
            not_mentioned_n=int(row["n_not_mentioned"]),
            mentioned_score=float(row.get("mentioned_mean", row.get("mentioned_score", 0.0))),
            mentioned_n=int(row["n_mentioned"]),
            x_name=nps_column,
            dataset_label=dataset_label,
        )
        figures_best.append(fig)

    figures_worst: List[go.Figure] = []
    for _, row in worst.iterrows():
        fig = divergent_comparative_bar_chart(
            theme_name=str(row["theme"]),
            not_mentioned_score=float(row.get("not_mentioned_mean", row.get("not_mentioned_score", 0.0))),
            not_mentioned_n=int(row["n_not_mentioned"]),
            mentioned_score=float(row.get("mentioned_mean", row.get("mentioned_score", 0.0))),
            mentioned_n=int(row["n_mentioned"]),
            x_name=nps_column,
            dataset_label=dataset_label,
        )
        figures_worst.append(fig)

    return {"best": figures_best, "worst": figures_worst}


def _safe_filename(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(">", "-")
        .replace(":", "-")
        .replace("|", "-")
        .replace("*", "-")
        .replace("?", "-")
        .replace('"', "'")
    )


def export_top_bottom_divergent_bars_png(
    data: DataLike,
    out_dir: str,
    file_prefix: str = "divergent",
    themes_column: str = "THEMES",
    nps_column: str = "NPS",
    mainthemes: bool = True,
    min_mentioned: int = 3,
    top_k: int = 3,
    label_column: Optional[str] = None,
    scale: int = 2,
    method: str = "mean",  # "mean" or "ebm"
) -> Dict[str, List[str]]:
    """Save top/bottom divergent bars as PNG to `out_dir`.

    Returns the paths of saved files for 'best' and 'worst'. Requires `kaleido`.
    """
    import os

    figs = build_top_bottom_divergent_bars(
        data=data,
        themes_column=themes_column,
        nps_column=nps_column,
        mainthemes=mainthemes,
        min_mentioned=min_mentioned,
        top_k=top_k,
        label_column=label_column,
        method=method,
    )

    os.makedirs(out_dir, exist_ok=True)

    # compute label suffix for filenames
    df, _ = prepare_theme_matrix(data, themes_column=themes_column, nps_column=nps_column, mainthemes=mainthemes)
    label_suffix = ""
    if label_column and label_column in df.columns:
        uniq = pd.unique(df[label_column].astype(str))
        if len(uniq) == 1:
            label_suffix = f"__label_{_safe_filename(str(uniq[0]))}"
        elif len(uniq) > 1:
            label_suffix = f"__label_multi_{len(uniq)}"

    saved: Dict[str, List[str]] = {"best": [], "worst": []}

    for idx, fig in enumerate(figs.get("best", []), start=1):
        path = os.path.join(out_dir, f"{file_prefix}_best_{idx}{label_suffix}.png")
        fig.write_image(path, scale=scale)
        saved["best"].append(path)
    for idx, fig in enumerate(figs.get("worst", []), start=1):
        path = os.path.join(out_dir, f"{file_prefix}_worst_{idx}{label_suffix}.png")
        fig.write_image(path, scale=scale)
        saved["worst"].append(path)

    return saved


__all__ = [
    "prepare_theme_matrix",
    "compute_theme_effects",
    "divergent_comparative_bar_chart",
    "build_top_bottom_divergent_bars",
    "export_top_bottom_divergent_bars_png",
]


