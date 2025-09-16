import pandas as pd
import numpy as np
import json
from collections import Counter
import plotly.express as px
from .Themes import thema_structuur  # Import thema_structuur


def plot_prioriteitenmatrix(
    data,
    plot_title="Prioriteitenmatrix",
    n_themes=6,
    themes_column="THEMES",
    nps_column="NPS",
    mainthemes=True,
    vertical_grid_position=None,  # Changed to None by default
    horizontal_grid_position=None  # Changed to None by default
):
    """
    Plots a Prioriteitenmatrix for the top N themes in the dataset.

    Args:
        data (str or pd.DataFrame): Path to the dataset (CSV/Excel) or a DataFrame.
        plot_title (str): Title for the plot.
        n_themes (int): Number of top themes to select.
        themes_column (str): Name of the column containing themes (as JSON string or dict).
        nps_column (str): Name of the column containing NPS scores.
        mainthemes (bool): If True, reduce each theme to its main theme.
        vertical_grid_position (float, optional): Custom position for the vertical grid line. If None, uses global mean.
        horizontal_grid_position (float, optional): Custom position for the horizontal grid line. If None, uses global mean.
    """

    # Load data if a path is given, otherwise assume it's a DataFrame
    if isinstance(data, str):
        if data.endswith(".csv"):
            df = pd.read_csv(data)
        else:
            df = pd.read_excel(data)
    else:
        df = data.copy()

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

    df['THEMES_PARSED'] = df[themes_column].apply(extract_theme_keys)

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
        df['THEMES_PARSED'] = df['THEMES_PARSED'].apply(get_main_themes)

    # Count theme occurrences
    theme_counts = Counter(
        item for sublist in df['THEMES_PARSED'] for item in sublist if item.strip()
    )
    all_themes = [theme for theme in theme_counts if theme.strip()]

    # Create binary columns for all themes
    for theme in all_themes:
        col_name = theme.replace(" ", "_").replace("&", "and")
        df[col_name] = df['THEMES_PARSED'].apply(lambda x: 1 if theme in x else 0)

    # Clean the NPS column
    df[nps_column] = pd.to_numeric(df[nps_column], errors='coerce')
    df = df.dropna(subset=[nps_column])

    # Calculate mean NPS and correlation for each theme
    theme_means = {}
    theme_correlations = {}
    for theme in all_themes:
        col_name = theme.replace(" ", "_").replace("&", "and")
        theme_means[theme] = df.loc[df[col_name] == 1, nps_column].mean()
        theme_correlations[theme] = df[col_name].corr(df[nps_column])

    # Calculate percentages for each theme
    theme_percentages = {}
    total_rows = len(df)
    for theme in all_themes:
        col_name = theme.replace(" ", "_").replace("&", "and")
        theme_percentages[theme] = (df[col_name].sum() / total_rows) * 100

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
    themes = list(filtered_themes)
    percentages = [theme_percentages[theme] for theme in themes]
    mean_scores = [theme_means[theme] for theme in themes]
    # Replace NaN correlations with 0 for size
    correlations = [0 if pd.isna(theme_correlations[theme]) else theme_correlations[theme] for theme in themes]

    # Assign quadrants for top N themes
    quadrant_labels = []
    
    # Determine grid positions - use custom values if provided, otherwise use global means
    v_grid = vertical_grid_position if vertical_grid_position is not None else mean_percentage
    h_grid = horizontal_grid_position if horizontal_grid_position is not None else mean_mean_score
    
    for percentage, mean_score in zip(percentages, mean_scores):
        if percentage >= v_grid and mean_score >= h_grid:
            quadrant_labels.append('Belangrijk met hoge Score')
        elif percentage >= v_grid and mean_score < h_grid:
            quadrant_labels.append('Belangrijk met lage Score')
        elif percentage < v_grid and mean_score >= h_grid:
            quadrant_labels.append('Minder belangrijk met hoge Score')
        else:
            quadrant_labels.append('Minder belangrijk met lage Score')

    # Make "Belangrijk" themes bold
    themes_bold = [
        f"<b>{theme}</b>" if quad.startswith("Belangrijk") else theme
        for theme, quad in zip(themes, quadrant_labels)
    ]

    # Show all theme names (including grey quadrants)
    themes_for_plot = themes_bold

    data = {
    'percentage': percentages,
    'mean_score': mean_scores,
    'themes': themes_for_plot,
    'quadrants': quadrant_labels,
    'size': [
        0.5 if quad in ['Minder belangrijk met hoge Score', 'Minder belangrijk met lage Score'] else 1
        for quad in quadrant_labels
    ]  # Smaller size for all dots, even smaller for grey dots
}

    category_orders = {
        "quadrants": [
            'Belangrijk met hoge Score',
            'Belangrijk met lage Score',
            'Minder belangrijk met hoge Score',
            'Minder belangrijk met lage Score'
        ]
    }
    category_colors = {
        'Belangrijk met hoge Score': '#005444',  # green
        'Belangrijk met lage Score': '#FF0000',  # red
        'Minder belangrijk met hoge Score': '#A9A9A9',  # grey
        'Minder belangrijk met lage Score': '#A9A9A9'   # grey
    }

    fig = px.scatter(
        data,
        x='percentage',
        y='mean_score',
        size='size',
        color='quadrants',
        color_discrete_map=category_colors,
        text='themes',
        size_max=30,  # Reduced from 60 to 30
        labels={
            "percentage": "Percentage van Thema (%)",
            "mean_score": "Gemiddelde NPS"
        },
        title=plot_title,
        category_orders=category_orders,
        template="simple_white"
    )

    fig.update_traces(
        marker=dict(opacity=0.8, line=dict(width=0, color='Black'), symbol='circle'),
        textposition='bottom center',
        textfont=dict(size=14, family="League Spartan", color='black'),
        cliponaxis=False
    )

    # Draw grid lines
    # For vertical line, y0 and y1 should cover the y-axis range
    y_min = min(mean_scores + [h_grid]) - 1
    y_max = max(mean_scores + [h_grid]) + 1
    x_max = max(percentages + [v_grid]) * 1.25
    
    fig.add_shape(
        type='line',
        x0=v_grid, x1=v_grid,
        y0=y_min, y1=y_max,
        xref='x', yref='y',
        line=dict(color="Black", width=2, dash="dash")
    )
    fig.add_shape(
        type='line',
        x0=0, x1=x_max,
        y0=h_grid, y1=h_grid,
        xref='x', yref='y',
        line=dict(color="Black", width=2, dash="dash")
    )

    fig.update_layout(
        title_font=dict(size=32, family='League Spartan', color='black'),
        title_x=0.5,
        xaxis_title=dict(text=f"Aantal keer genoemd (%) van {total_rows} totaal", font=dict(size=20)),
        yaxis_title=dict(text="Gemiddelde NPS", font=dict(size=20)),
        legend_title=dict(text="Thema's:", font=dict(size=16)),
        font=dict(size=16, family="League Spartan"),
        margin=dict(l=50, r=200, t=70, b=100),
        legend=dict(
            yanchor="top", y=-0.15, xanchor="center", x=0.5, orientation="h",
            font=dict(size=14),
            bgcolor='rgba(255, 255, 255, 0)',
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True, gridcolor='rgba(211, 211, 211, 0.5)', gridwidth=0.5, zeroline=False, showline=False,
            range=[0, max(percentages + [10]) * 1.25] if percentages else [0, 10]
        ),
        yaxis=dict(
            showgrid=True, gridcolor='rgba(211, 211, 211, 0.5)', gridwidth=0.5, zeroline=False, showline=False
        ),
        height=1000,
        width=1500
    )

    # Return the figure object
    return fig