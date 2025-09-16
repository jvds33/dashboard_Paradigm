## KTO Dashboard Generator

Streamlit app and plotting toolkit for generating KTO (customer survey) visualizations. The app wraps the plotting functions in `kto_cards.py` and lets you upload an Excel, pick a sheet and label, tweak titles/sizes, and download the resulting PNGs.

### Features

- **Excel upload and sheet/label picker**: Select any sheet; labels are discovered from your `Label` column
- **One-click generation**: Calls `kto_cards.run_all(...)` to create a consistent set of “card” style visuals
- **Edit per-plot settings**: Change title, figure size, and (where supported) the source column
- **Download**: Save individual PNGs, or download everything as a ZIP with a per-label folder

Generated plots (key → default file):
- `respondent_count` → `blok_respondent_count.png`
- `aanbeveling` (0–10 distribution) → `blok_aanbeveling_distributie.png`
- `nps_gauge` (US-style NPS) → `blok_nps_gauge.png`
- `duur` (duur samenwerking gauges) → `duur_samenwerking.png`
- `diensten` (afgenomen diensten) → `blok_afgenomen_diensten.png`
- `info_pie` (informatievoorziening) → `blok_informatievoorziening.png`
- `omgang_info` (PDG only) → `blok_omgang_informatie.png`
- `smileys` (impact smileys) → `blok_impact_smileys.png`
- `functiegroep` → `blok_functiegroep.png`
- `aspecten_likert` → `blok_aspecten_likert.png`
- `aspecten_systeem_likert` (PDG only) → `blok_aspecten_systeem_likert.png`
- `priority_matrix` (requires THEMES + NPS, only when enough THEMES data) → `blok_priority_matrix.png`
- `open_antwoord_analyse` → `blok_open_antwoord_analyse.png`

Notes:
- Label-specific behavior:
  - `Resolu` and `PDG Health Services` use tailored mappings for services and aspecten
  - PDG unlocks `omgang_info` and `aspecten_systeem_likert`
- Priority matrix renders only if `THEMES` has > 70 non-empty rows and an NPS column exists

### Quick start

```bash
# Install dependencies
uv sync

# Run Streamlit dashboard
streamlit run dashboard.py

# Or use the helper script
python run_dashboard.py
```

App runs at `http://localhost:8501`.

### Using the dashboard

1) Upload an Excel file (.xlsx/.xls)
2) Pick a sheet and a label (from the `Label` column)
3) Click “Generate Dashboard”
4) Optionally edit plots (title, width/height, and column where available) and re-apply
5) Download individual plots or the full ZIP

Per-plot column override is supported for: `aanbeveling`, `nps_gauge`, `duur`, `info_pie`, `omgang_info`, `smileys`, `functiegroep`.

### Programmatic usage (Python)

```python
from kto_cards import run_all

paths = run_all(
    excel="your.xlsx",
    sheet="Sheet1",
    label="Resolu",
    outdir="./plots",
    customizations={
        "nps_gauge": {"title": "NPS", "width": 7.2, "height": 4.6},
        "aanbeveling": {"column": "Blok1_NPS_KPI_page6_Text"}
    }
)
print(paths)
```

### Command-line usage

```bash
python kto_cards.py --label "Your Label" --excel "your.xlsx" --outdir "./plots"
```

CLI uses sheet `Sheet1` internally. To target a different sheet or customize plots, prefer the Streamlit app or import `run_all` in Python.

### Data requirements

- Excel with a `Label` column (case-insensitive: Label/label/LABEL)
- Plot-specific columns (defaults coded in `kto_cards.py`; you can override many via the app)
- For `priority_matrix` and `open_antwoord_analyse`: a `THEMES` column containing JSON objects mapping theme → quote; NPS scores in `Blok1_NPS_KPI_page6_Text` or `NPS`

### Dependencies

- streamlit
- pandas
- matplotlib
- numpy
- openpyxl

Managed via `pyproject.toml`. Install with `uv sync`.
