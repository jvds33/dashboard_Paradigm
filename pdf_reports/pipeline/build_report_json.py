"""CLI entrypoint: build a single report JSON from an Excel file + label config.

Dependencies: pandas, pyyaml. ``openai`` is optional — narrative.py degrades
gracefully when the package or credentials are missing.

Usage:
    python -m pdf_reports.pipeline.build_report_json \\
        --excel data/filtered_arbodienst.xlsx \\
        --label-config pdf_reports/pipeline/labels/immediator.yaml \\
        --tenant-id demo \\
        --tenant-name "Demo Tenant" \\
        --out pdf_reports/app/public/data/demo.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml


# Word-boundary regex: matches the standalone word "test" (case-insensitive),
# so "getest" / "testimonial" / "testdag" do NOT match. Used to drop obvious
# test/spam rows where any open-answer or respondent-identity field contains
# a literal "test".
_TEST_WORD_RE = re.compile(r"\btest\b", re.IGNORECASE)

# Respondent-identity column names that are auto-scanned for the word "test".
# Catches the common case where a tester is seeded into the panel with a name
# like "Test 7" but fills out plausible-looking answers — the open-answer
# heuristic alone then misses the row. Match is case-insensitive on the
# stripped column name.
_IDENTITY_COLUMN_NAMES = {
    "naam",
    "naam klant",
    "voornaam",
    "achternaam",
    "hoofdcontact",
    "hoofdcontact klant",
    "klantnaam",
    "respondent",
    "respondentnaam",
}


_DUTCH_MONTHS = [
    "januari", "februari", "maart", "april", "mei", "juni",
    "juli", "augustus", "september", "oktober", "november", "december",
]


def _compute_wave_date_from_df(df: pd.DataFrame, column: str = "sys_updated") -> str | None:
    """Return the modal year-month of ``column`` formatted as Dutch "Maand JJJJ".

    Used as a fallback when ``--wave-date`` is not provided: the cover slide
    then shows the month in which the overgrote meerderheid van de respondenten
    de vragenlijst heeft ingevuld, in plaats van de eerste/laatste datum.
    """
    if column not in df.columns:
        return None
    series = pd.to_datetime(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    top_period = series.dt.to_period("M").value_counts().idxmax()
    return f"{_DUTCH_MONTHS[top_period.month - 1].capitalize()} {top_period.year}"


def _drop_empty_partials(
    df: pd.DataFrame,
    status_column: str | None,
    partial_status: int = 2,
) -> tuple[pd.DataFrame, list[int]]:
    """Drop status=partial respondents that have no substantive content.

    A "substantive" field is any non-null open-answer column (text columns
    ending in ``_Text`` or containing ``Toelicht`` / ``Suggest`` /
    ``OpenAntwoord``) OR any aspect-rating column (matching ``*_QuestionN``).

    The typical case this catches: someone clicks the NPS slider, leaves it
    on the default (often 0), then closes the survey. They have no quote,
    no theme, no ratings — nothing to report on, but their lone click
    distorts the NPS distribution.
    """
    if not status_column or status_column not in df.columns:
        return df, []

    status_num = pd.to_numeric(df[status_column], errors="coerce")
    is_partial = status_num == partial_status
    if not is_partial.any():
        return df, []

    # A "content column" = any survey response column starting with "Blok"
    # (the questionnaire blocks), EXCEPT the NPS KPI score itself. The NPS
    # click is exactly what we don't want to count, since the whole point
    # of this filter is to drop respondents whose ONLY input is the NPS
    # slider. Everything else (open answers, Likerts, aspect ratings,
    # suggestions, toelichtingen) counts as substantive content.
    nps_kpi_cols = {c for c in df.columns if "NPS_KPI" in c}
    content_cols = [
        c
        for c in df.columns
        if c.startswith("Blok") and c not in nps_kpi_cols
    ]
    if not content_cols:
        return df, []

    # A row "has content" if at least one content column is non-null AND
    # (for object/string columns) non-empty after stripping whitespace.
    def _row_has_content(row) -> bool:
        for col in content_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            if isinstance(val, str) and not val.strip():
                continue
            return True
        return False

    has_content = df.apply(_row_has_content, axis=1)
    drop_mask = is_partial & ~has_content
    dropped = df.index[drop_mask].tolist()
    return df[~drop_mask].copy(), dropped


def _drop_test_rows(df: pd.DataFrame, columns=None) -> tuple[pd.DataFrame, list[int]]:
    """Drop rows where any of the given text columns contains the word "test".

    If ``columns`` is None, auto-detect by scanning all object-dtype columns
    whose name either
      (a) looks like an open-answer field — ends with ``_Text`` or contains
          ``Toelicht`` / ``Suggest`` / ``OpenAntwoord`` — or
      (b) is a respondent-identity field — ``Naam`` / ``Naam klant`` /
          ``Achternaam`` / ``Hoofdcontact klant`` / etc. (see
          ``_IDENTITY_COLUMN_NAMES``).

    Respondent-identity columns catch the common case where a test panelist
    is seeded with a name like "Test 7" but fills in plausible answers — the
    open-answer heuristic alone would leave those rows in, skewing NPS and
    response counts. Returns (cleaned_df, dropped_row_indices).
    """
    if columns is None:
        columns = [
            c
            for c in df.columns
            if df[c].dtype == object
            and (
                c.endswith("_Text")
                or "oelicht" in c
                or "Suggest" in c
                or "OpenAntwoord" in c
                or str(c).strip().lower() in _IDENTITY_COLUMN_NAMES
            )
        ]
    if not columns:
        return df, []

    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        col_mask = df[col].astype(str).str.contains(_TEST_WORD_RE, na=False)
        mask = mask | col_mask
    dropped = df.index[mask].tolist()
    return df[~mask].copy(), dropped

# Make sibling imports work whether this is run as a module or as a script.
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import extractors  # noqa: E402
import loader  # noqa: E402
import narrative  # noqa: E402


def _slugify(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value.strip().lower()).strip("_")


def _format_template(template: str, **kwargs) -> str:
    if template is None:
        return template
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template


def _load_overrides(slug: str, tenant_id: str) -> dict:
    overrides_dir = _PKG_DIR / "overrides"
    candidate = overrides_dir / f"{slug}__{tenant_id}.yaml"
    if not candidate.exists():
        return {}
    try:
        with candidate.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a report JSON for the React PDF app.")
    parser.add_argument("--excel", required=True, help="Path to the Excel data file.")
    parser.add_argument("--sheet", default=None, help="Optional sheet name (default: first sheet).")
    parser.add_argument("--label-config", required=True, help="Path to the label YAML config.")
    parser.add_argument("--tenant-id", required=True, help="Tenant identifier (used in filenames + overrides).")
    parser.add_argument("--tenant-name", default=None, help="Display name for the tenant (default = tenant-id).")
    parser.add_argument("--wave-date", default=None, help="Optional ISO date for this wave.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--no-ai", action="store_true", help="Skip Azure OpenAI narrative generation.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tenant_name = args.tenant_name or args.tenant_id

    label_config_path = Path(args.label_config)
    with label_config_path.open("r", encoding="utf-8") as fh:
        label_config = yaml.safe_load(fh) or {}

    label_value = label_config.get("label")
    slug = label_config.get("slug") or _slugify(label_value or label_config_path.stem)

    filter_cfg = label_config.get("filter", {}) or {}
    label_column = filter_cfg.get("label_column", "Label")
    status_column = filter_cfg.get("status_column")
    status_in = filter_cfg.get("status_in")
    drop_test_rows = bool(filter_cfg.get("drop_test_rows", False))
    drop_test_columns = filter_cfg.get("drop_test_columns")  # optional explicit list
    drop_empty_partials = bool(filter_cfg.get("drop_empty_partials", False))

    df_full = loader.read_excel(args.excel, sheet=args.sheet)

    # Drop obvious test/spam rows up-front so EVERY extractor sees a clean
    # dataset (NPS, top_themes, quotes, …). Off by default — opt-in per label.
    if drop_test_rows:
        df_full, dropped_idx = _drop_test_rows(df_full, columns=drop_test_columns)
        if dropped_idx:
            print(
                f"Dropped {len(dropped_idx)} test/spam row(s) "
                f"(matched word 'test' in open-answer fields): rows {dropped_idx}"
            )

    # Snapshot of "everyone who got the survey" — used by respondent_tiles
    # so the funnel slide still shows the real invited / partial / completed
    # numbers (test rows excluded but empty partials still counted as
    # someone who opened the survey).
    df_for_tiles = df_full.copy()

    # Drop status=partial respondents that filled in nothing substantive
    # (no open answer, no toelichting, no aspect ratings). Their single
    # NPS-click distorts the score without contributing any feedback.
    if drop_empty_partials:
        df_full, dropped_idx = _drop_empty_partials(df_full, status_column)
        if dropped_idx:
            print(
                f"Dropped {len(dropped_idx)} empty partial respondent(s) "
                f"(no open answer / toelichting / aspect rating): rows {dropped_idx}"
            )

    # Label filter is optional: if no label_value or label_column not in df,
    # we skip the label filter and only apply status filter.
    if label_value and label_column in df_full.columns:
        df = loader.filter_by_label(
            df_full,
            label_column=label_column,
            label_value=label_value,
            status_column=status_column,
            status_in=status_in,
        )
    else:
        df = df_full.copy()
        if status_column and status_in is not None and status_column in df.columns:
            try:
                numeric_status = pd.to_numeric(df[status_column], errors="coerce")
                numeric_set = {float(s) for s in status_in}
                df = df[numeric_status.isin(numeric_set)].copy()
            except Exception:
                df = df[df[status_column].astype(str).isin([str(s) for s in status_in])].copy()

    respondent_count = loader.count_respondents(df)

    wave_date = args.wave_date or _compute_wave_date_from_df(df)

    meta = {
        "label_slug": slug,
        "label_name": label_value,
        "tenant_id": args.tenant_id,
        "tenant_name": tenant_name,
        "wave_date": wave_date,
        "respondent_count": respondent_count,
    }

    fmt_kwargs = {
        "tenant_name": tenant_name,
        "tenant_id": args.tenant_id,
        "label_name": label_value or "",
        "wave_date": wave_date or "",
    }

    pages_out = []
    for page_cfg in label_config.get("pages", []) or []:
        page = {
            "id": page_cfg.get("id"),
            "template": page_cfg.get("template"),
            "title": _format_template(page_cfg.get("title"), **fmt_kwargs),
            "subtitle": _format_template(page_cfg.get("subtitle"), **fmt_kwargs),
            "methodology": page_cfg.get("methodology"),
            "methodology_title": page_cfg.get("methodology_title"),
            "orientation": page_cfg.get("orientation", "landscape"),
            "blocks": [],
        }
        if page_cfg.get("methodology_emphasis"):
            page["methodology_emphasis"] = True
        for block_cfg in page_cfg.get("blocks", []) or []:
            extractor_name = block_cfg.get("extractor")
            try:
                extractor_fn = extractors.get(extractor_name)
                # Inject the unfiltered df for extractors that need it (e.g. respondent_tiles
                # wants to count status 1 = invited, which is filtered out for the rest).
                block_cfg_with_full = dict(block_cfg)
                block_cfg_with_full["_full_df"] = df_for_tiles
                block_data = extractor_fn(df, block_cfg_with_full)
            except Exception as exc:  # defensive — never crash on a single block
                block_data = {
                    "type": extractor_name or "unknown",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            # Carry presentational metadata from YAML through to the block
            # payload (extractors focus on data; they don't need to know
            # about titles / subtitles). Only set when the extractor didn't
            # already populate it.
            for meta_key in ("title", "subtitle", "caption"):
                val = block_cfg.get(meta_key)
                if val and not block_data.get(meta_key):
                    block_data[meta_key] = val
            page["blocks"].append(block_data)

        ai_text = None
        if not args.no_ai:
            ai_text = narrative.generate(page["id"], {"meta": meta, "page": page})
        page["narrative"] = {"ai": ai_text, "override": None}
        pages_out.append(page)

    overrides = _load_overrides(slug, args.tenant_id)
    if overrides:
        for page in pages_out:
            pid = page.get("id")
            if pid in overrides and overrides[pid]:
                page["narrative"]["override"] = overrides[pid]

    report = {"meta": meta, "pages": pages_out}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path}")
    print(f"  respondents: {respondent_count}")
    print(f"  pages:       {len(pages_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
