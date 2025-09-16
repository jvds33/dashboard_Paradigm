from DivergentBars import export_top_bottom_divergent_bars_png
import pandas as pd

file_path = "/Users/jessevdsluis/Library/CloudStorage/OneDrive-Gedeeldebibliotheken-SpringX/SpringXSharepoint - Documenten/SpringX Analytics/Klanten/Paradigma/CX/KTO/Rapportages/KTO_Themes.xlsx"

# Filter to a single label subset
full_df = pd.read_excel(file_path)
resolu_df = full_df[full_df.get("Label").astype(str) == "Resolu"] if "Label" in full_df.columns else full_df

saved = export_top_bottom_divergent_bars_png(
    data=resolu_df,         # pass filtered DataFrame
    out_dir="plots",
    file_prefix="divergent",
    themes_column="THEMES",
    nps_column="NPS",
    mainthemes=False,       # use subthemes
    min_mentioned=3,
    top_k=3,
    label_column="Label",
    scale=2,
    method="ebm"
)

print("Saved PNGs:")
print(saved)