# ThemeRow redesign — three-column card layout

## Context

`ThemeRow` lives in `pdf_reports/app/src/components/TopThemesEditorial.jsx`
and renders one subtheme row inside the "Open antwoorden — top thema's"
PDF-report page. The current layout shows a large rank number on the left,
a title row with NPS chip + mention badge on the right, a divider, and three
quote cells in a grid below.

The user wants a different visual treatment, matching a reference image
they provided: a three-column card with a prominent "count stat" on the
left, three quotes side-by-side in the middle, and a colored NPS circle
on the right.

## Goals

- Redesign `ThemeRow` to the three-column layout described below.
- Keep using the existing theme data contract (no pipeline / data model
  changes).
- Keep rendering three quotes per row, side-by-side.

## Non-goals

- No changes to the upstream data pipeline (no new "werkgeluk" score).
- No changes to `TopThemesEditorial`'s list-level behavior (still a
  vertical stack of `ThemeRow`s).
- No changes to `ThemeQuotesCard` (separate component used elsewhere).

## Design

### Card shell

- `rounded-2xl border border-brand-line bg-brand-bg overflow-hidden`
- `flex items-stretch` — three direct children columns.
- `minHeight` bumped from `168px` to ~`184px` to give the quotes more
  breathing room alongside the left/right stat columns.
- The existing left accent rail (gradient `accent → accentBlue`) is
  removed — the new left column already anchors the card visually.

### Left column — count stat (~150px wide, `shrink-0`)

- Vertical flex column, centered, padded (`px-4 py-4`).
- Large number: `{count}×`
  - `font-extrabold`, `~48px`, color `PARADIGMA.primary`,
    `tabular-nums`, tight leading.
- Label below the number, 2 lines, muted (`text-brand-muted`, ~`10px`,
  `leading-snug`):
  > "genoemd binnen open antwoorden van jouw locatie"
- Horizontal progress bar:
  - Track: `h-1.5 rounded-full bg-brand-surface`
  - Fill: `h-1.5 rounded-full`, width `{percent}%`, background
    `PARADIGMA.accent`.
- Small caption under the bar: `{percent}% van responses`
  (`text-[10px] text-brand-muted`).

### Middle column — title + quotes (`flex-1 min-w-0`)

- Padded `py-4 pr-4`.
- Title block:
  - Small uppercase eyebrow: `{main_theme}` —
    `text-[10px] uppercase tracking-[0.18em] text-brand-muted font-semibold`.
  - Title: `{subtheme}` —
    `text-[18px] font-bold leading-tight text-brand-primary mt-0.5 truncate`,
    `title={subtheme}` for tooltip.
- Thin accent divider (reuse current gradient):
  `h-[2px] mt-3 mb-3 rounded-full`,
  `background: linear-gradient(90deg, ${PARADIGMA.accent} 0%, transparent 60%)`.
- Quotes grid:
  - `grid grid-cols-3 gap-3 flex-1 min-h-0`.
  - Each cell: reuse the existing `QuoteCell` component as-is (big
    curly quote, italic text, `line-clamp-4`).
  - If fewer than 3 quotes, render an empty placeholder cell
    (`rounded-lg bg-brand-surface/50`), matching current behavior.

### Right column — NPS circle (~110px wide, `shrink-0`)

- Vertical flex column, centered, padded (`px-3 py-4`).
- Circle:
  - `w-20 h-20 rounded-full flex items-center justify-center`
  - `background: npsColor(nps_score)` (reuse existing helper)
  - Inside: NPS score — `text-white font-extrabold text-[26px]
    leading-none tabular-nums`, prefixed with `+` when positive.
  - When `nps_score` is null/undefined: render a neutral muted circle
    with an em-dash, OR omit the right column entirely for that row.
    Decision: **render a muted placeholder circle** (`background:
    PARADIGMA.muted`) with an em-dash, so row widths stay consistent
    in the vertical stack.
- Label under the circle, centered, muted, 2 lines:
  - Line 1: `NPS` (`text-[9px] uppercase tracking-widest font-semibold`)
  - Line 2: `n={n_nps ?? '–'}` (`text-[9px]`)

### Removed elements

- Rank number (`01`, `02`, ...) — gone.
- Left accent rail — gone (see card shell).
- Old right-side title badges (NPS chip + mention badge) — replaced by
  the new left and right columns.

## Data contract (unchanged)

`ThemeRow` still consumes:

```
{ rank, subtheme, main_theme, count, percent, quotes, nps_score, n_nps }
```

- `rank` is still received but no longer rendered.
- No new fields are introduced.

## Testing

- Visual check in the local dev harness for `TopThemesPage` with
  representative theme data:
  - Theme with positive NPS (green circle, `+` prefix).
  - Theme with neutral NPS (blue circle).
  - Theme with negative NPS (red circle, `-` prefix).
  - Theme with `nps_score` missing (muted placeholder circle).
  - Theme with fewer than 3 quotes (empty placeholder cell).
- Verify that the vertical stack of rows still fits on the A4 report
  page without overflow. If it doesn't, revisit `minHeight` or the
  number of themes rendered per page.

## Out of scope / follow-ups

- Tuning how many subthemes fit per A4 page (may need a separate pass
  if the taller row pushes content off the page).
- Any design change to `ThemeQuotesCard` or `TopThemesPage` scaffolding.
