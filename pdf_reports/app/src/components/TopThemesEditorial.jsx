import { PARADIGMA } from '../theme.js';

/**
 * Editorial top-themes list. Renders a vertical stack of subtheme rows.
 * Each row is a three-column card: a count-stat column on the left
 * (big `count×` number + percent-of-responses bar), a middle column with
 * the subtheme title and three quote bubbles, and an NPS circle on the
 * right colored by the mean NPS score for that subtheme.
 *
 * Props:
 *   themes: [{ rank, subtheme, main_theme, count, percent, quotes,
 *              nps_score, n_nps }]
 *   n:      number of respondents whose open answers were classified
 */
export default function TopThemesEditorial({ themes = [], n = 0 }) {
  if (!themes.length) {
    return (
      <div className="text-[12px] text-brand-muted italic">
        Geen open antwoorden beschikbaar voor thema-analyse.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 w-full">
      {themes.map((t) => (
        <ThemeRow key={t.rank} theme={t} />
      ))}
    </div>
  );
}

function npsColor(score) {
  if (score === null || score === undefined) return PARADIGMA.muted;
  if (score >= 30) return PARADIGMA.accent;      // strong promoter territory
  if (score >= 0) return PARADIGMA.accentBlue;   // neutral/passive
  return PARADIGMA.danger;                        // detractor territory
}

function ThemeRow({ theme }) {
  const { subtheme, main_theme, count, percent, quotes = [], nps_score, n_nps } = theme;
  const hasNps = nps_score !== null && nps_score !== undefined;
  const npsClr = hasNps ? npsColor(nps_score) : PARADIGMA.muted;
  const npsLabel = hasNps ? (nps_score > 0 ? `+${nps_score}` : `${nps_score}`) : '—';
  const pct = Math.max(0, Math.min(100, Number(percent) || 0));

  return (
    <div
      className="flex items-stretch rounded-2xl border border-brand-line bg-brand-bg overflow-hidden"
      style={{ minHeight: 176 }}
    >
      {/* Left column — count stat */}
      <div
        className="shrink-0 flex flex-col justify-center px-4 py-3 border-r border-brand-line"
        style={{ width: 140 }}
      >
        <div
          className="font-extrabold leading-none tracking-tight"
          style={{
            fontSize: 46,
            color: PARADIGMA.primary,
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {count}×
        </div>
        <div
          className="mt-1.5 text-[10px] leading-snug"
          style={{ color: PARADIGMA.muted }}
        >
          genoemd binnen open antwoorden
        </div>
        <div
          className="mt-2 h-1.5 w-full rounded-full overflow-hidden"
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
        <div className="min-w-0 flex items-baseline gap-3">
          <h3
            className="text-[18px] font-bold leading-tight truncate"
            style={{ color: PARADIGMA.primary }}
            title={subtheme}
          >
            {subtheme}
          </h3>
          <div
            className="shrink-0 text-[12px] font-medium leading-tight truncate"
            style={{ color: PARADIGMA.muted }}
            title={main_theme}
          >
            {main_theme}
          </div>
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
        style={{ width: 104 }}
      >
        <div
          className="flex items-center justify-center rounded-full"
          style={{
            width: 72,
            height: 72,
            background: npsClr,
          }}
        >
          <div
            className="font-extrabold leading-none tabular-nums"
            style={{ fontSize: 24, color: '#FFFFFF' }}
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
          {hasNps ? `n=${n_nps ?? '–'}` : 'n.v.t.'}
        </div>
      </div>
    </div>
  );
}

function QuoteCell({ quote }) {
  return (
    <div
      className="relative rounded-lg bg-brand-surface border border-brand-line px-3 py-2 flex items-center"
      style={{ minHeight: 56 }}
    >
      <span
        aria-hidden
        className="absolute -top-1 left-2 text-[26px] leading-none font-serif select-none"
        style={{ color: PARADIGMA.accent }}
      >
        “
      </span>
      <p
        className="text-[10px] leading-snug italic text-brand-ink pl-3 line-clamp-4"
        style={{ display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}
      >
        {quote}
      </p>
    </div>
  );
}
