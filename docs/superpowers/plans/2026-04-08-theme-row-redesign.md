# ThemeRow redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the `ThemeRow` component in `TopThemesEditorial.jsx` into a three-column card (count stat / title + three quotes / NPS circle), per the approved design spec.

**Architecture:** A single-file edit to `pdf_reports/app/src/components/TopThemesEditorial.jsx`. The `ThemeRow` subcomponent is rewritten; `TopThemesEditorial` and `QuoteCell` remain unchanged (the existing `QuoteCell` is reused inside the new middle column). No data contract changes.

**Tech Stack:** React 19 + Tailwind 3 + inline styles driven by the `PARADIGMA` palette from `src/theme.js`. No unit tests in this package; visual verification happens via the Vite dev server (`npm run dev`) and/or `npm run export-pdf`.

---

## File structure

- **Modify:** `pdf_reports/app/src/components/TopThemesEditorial.jsx`
  - Rewrite the `ThemeRow` function (lines ~38–158).
  - `npsColor` helper is reused as-is.
  - `QuoteCell` is reused as-is for the middle-column quotes.
  - `TopThemesEditorial` (the list wrapper) is unchanged.
- **Reference only:** `docs/superpowers/specs/2026-04-08-theme-row-redesign-design.md`

No new files. No test files (this package has no unit test harness; a Playwright snapshot script exists only for the NPS page and is out of scope here).

---

## Task 1: Rewrite `ThemeRow` with the three-column layout

**Files:**
- Modify: `pdf_reports/app/src/components/TopThemesEditorial.jsx` (`ThemeRow` function, ~lines 38–158)

- [ ] **Step 1: Replace the `ThemeRow` function body**

Open `pdf_reports/app/src/components/TopThemesEditorial.jsx` and replace the entire existing `ThemeRow` function (from `function ThemeRow({ theme }) {` through its closing `}` on ~line 158) with the implementation below. Leave `TopThemesEditorial`, `npsColor`, and `QuoteCell` alone.

```jsx
function ThemeRow({ theme }) {
  const { subtheme, main_theme, count, percent, quotes = [], nps_score, n_nps } = theme;
  const hasNps = nps_score !== null && nps_score !== undefined;
  const npsClr = hasNps ? npsColor(nps_score) : PARADIGMA.muted;
  const npsLabel = hasNps ? (nps_score > 0 ? `+${nps_score}` : `${nps_score}`) : '–';
  const pct = Math.max(0, Math.min(100, Number(percent) || 0));

  return (
    <div
      className="relative flex items-stretch rounded-2xl border border-brand-line bg-brand-bg overflow-hidden"
      style={{ minHeight: 184 }}
    >
      {/* Left column — count stat */}
      <div
        className="shrink-0 flex flex-col justify-center px-4 py-4 border-r border-brand-line"
        style={{ width: 150 }}
      >
        <div
          className="font-extrabold leading-none tracking-tight"
          style={{
            fontSize: 48,
            color: PARADIGMA.primary,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {count}×
        </div>
        <div
          className="mt-2 text-[10px] leading-snug"
          style={{ color: PARADIGMA.muted }}
        >
          genoemd binnen open antwoorden van jouw locatie
        </div>
        <div
          className="mt-3 h-1.5 w-full rounded-full overflow-hidden"
          style={{ background: PARADIGMA.surface }}
        >
          <div
            className="h-full rounded-full"
            style={{ width: `${pct}%`, background: PARADIGMA.accent }}
          />
        </div>
        <div
          className="mt-1 text-[10px]"
          style={{ color: PARADIGMA.muted }}
        >
          {pct}% van responses
        </div>
      </div>

      {/* Middle column — title + three quotes */}
      <div className="flex-1 min-w-0 flex flex-col py-4 px-5">
        <div className="min-w-0">
          <div
            className="text-[10px] uppercase tracking-[0.18em] font-semibold"
            style={{ color: PARADIGMA.muted }}
          >
            {main_theme}
          </div>
          <h3
            className="text-[18px] font-bold leading-tight mt-0.5 truncate"
            style={{ color: PARADIGMA.primary }}
            title={subtheme}
          >
            {subtheme}
          </h3>
        </div>

        <div
          className="h-[2px] mt-3 mb-3 rounded-full"
          style={{
            background: `linear-gradient(90deg, ${PARADIGMA.accent} 0%, transparent 60%)`,
          }}
        />

        <div className="grid grid-cols-3 gap-3 flex-1 min-h-0">
          {[0, 1, 2].map((i) => {
            const q = quotes[i];
            if (!q) {
              return <div key={i} className="rounded-lg bg-brand-surface/50" />;
            }
            return <QuoteCell key={i} quote={q} />;
          })}
        </div>
      </div>

      {/* Right column — NPS circle */}
      <div
        className="shrink-0 flex flex-col items-center justify-center px-3 py-4 border-l border-brand-line"
        style={{ width: 110 }}
      >
        <div
          className="flex items-center justify-center rounded-full"
          style={{
            width: 80,
            height: 80,
            background: npsClr,
          }}
        >
          <div
            className="font-extrabold leading-none tabular-nums"
            style={{ fontSize: 26, color: '#FFFFFF' }}
          >
            {npsLabel}
          </div>
        </div>
        <div
          className="mt-2 text-[9px] uppercase tracking-widest font-semibold text-center"
          style={{ color: PARADIGMA.muted }}
        >
          NPS
        </div>
        <div
          className="text-[9px] text-center"
          style={{ color: PARADIGMA.muted }}
        >
          n={n_nps ?? '–'}
        </div>
      </div>
    </div>
  );
}
```

Notes while editing:
- Keep the existing `import { PARADIGMA } from '../theme.js';` at the top of the file.
- Do not touch `TopThemesEditorial`, `npsColor`, or `QuoteCell`.
- The `rank` field is still destructured off `theme` in the wrapper (`TopThemesEditorial`) and used only as a React key — that stays.

- [ ] **Step 2: Check there are no stale references to the removed props**

Search the file for any remaining references to the old UI that should now be gone:

Run (from repo root):
```bash
grep -n "rankStr\|rank.*padStart\|Mention badge\|Mean NPS chip\|shrink-0 w-\[6px\]" pdf_reports/app/src/components/TopThemesEditorial.jsx
```
Expected: no matches. If any show up, they're leftovers from the old `ThemeRow` — delete them.

- [ ] **Step 3: Start the dev server and visually verify the layout**

Run:
```bash
cd pdf_reports/app && npm run dev
```
Open the URL that Vite prints (typically `http://localhost:5173`) and navigate to a report that includes the "Open antwoorden — top thema's" page (`TopThemesPage`).

Expected:
- Each theme row is a single `rounded-2xl` card with three visible columns: count stat on the left, title + three quote bubbles in the middle, and an NPS circle on the right.
- The left column shows `{count}×`, the label text, a horizontal accent-colored bar filled to `{percent}%`, and the `X% van responses` caption.
- The middle column shows the uppercase `main_theme` eyebrow, the `subtheme` title, the thin accent divider, and three side-by-side quotes (or empty placeholders if fewer than 3 quotes).
- The right column shows a filled circle colored by NPS (green for ≥30, blue for ≥0, red for <0, grey when missing), with the NPS value in white inside, and `NPS` / `n=…` labels underneath.
- The old rank number (`01`, `02`, …) and the old left accent rail are gone.

Stop the dev server with Ctrl+C when done.

- [ ] **Step 4: Commit**

```bash
git add pdf_reports/app/src/components/TopThemesEditorial.jsx
git commit -m "feat(pdf-reports): redesign ThemeRow into three-column card"
```

---

## Self-review notes

- **Spec coverage:** left column (count + label + bar + caption) ✓, middle column (eyebrow + title + divider + 3 quotes grid reusing `QuoteCell`) ✓, right column (NPS circle with `npsColor`, placeholder when missing) ✓, rank number removed ✓, left accent rail removed ✓, `minHeight` bumped to 184 ✓, data contract unchanged ✓.
- **Placeholders:** none — every step has exact code and exact commands.
- **Type consistency:** uses the same field names as the existing `theme` contract (`subtheme`, `main_theme`, `count`, `percent`, `quotes`, `nps_score`, `n_nps`) and reuses the existing `npsColor` and `QuoteCell` symbols defined in the same file.
