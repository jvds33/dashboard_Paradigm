import PageHeader from '../components/PageHeader.jsx';
import NarrativeBlock from '../components/NarrativeBlock.jsx';
import MethodologyCallout from '../components/MethodologyCallout.jsx';

/**
 * "Open antwoorden — top thema's" slide.
 *
 * Layout:
 *   - PageHeader (title + optional subtitle)
 *   - Methodology callout: explains how the themes were derived (text-AI
 *     classification of open answers).
 *   - Block: TopThemesEditorial (vertical numbered list of N subthemes).
 *   - Optional NarrativeBlock at the bottom (override > ai > none).
 *
 * Expects exactly one block of type "top_themes" in page.blocks.
 */
export default function TopThemesPage({ page, blockComponents }) {
  const block = (page.blocks || []).find((b) => b.type === 'top_themes');
  const Renderer = block ? blockComponents['top_themes'] : null;
  const n = block?.n ?? 0;

  return (
    <div className="flex flex-col h-full">
      <PageHeader title={page.title} subtitle={page.subtitle} />

      <div className="shrink-0">
        <MethodologyCallout title="Hoe deze thema's tot stand kwamen">
          respondenten konden hun antwoord op de NPS-vraag{' '}
          <em>“Hoe waarschijnlijk is het dat u Immediator aanbeveelt bij
          collega's en zakenrelaties?”</em> toelichten. Een tekstmodel deelt
          die toelichtingen in hoofd- en subthema's in; de subthema's hieronder
          staan in volgorde van hoe vaak ze genoemd zijn. Per subthema tonen we
          drie representatieve quotes en de gemiddelde NPS van respondenten die
          het thema noemden, zodat je het sentiment rondom dat onderwerp kunt
          aflezen. Op basis van{' '}
          <span className="font-semibold">{n}</span> toelichtingen.
        </MethodologyCallout>
      </div>

      <div className="flex-1 min-h-0 mt-3">
        {Renderer && block ? (
          <Renderer themes={block.themes || []} n={n} n_total={block.n_total} />
        ) : (
          <div className="text-[12px] text-brand-muted italic">
            Geen thema-data beschikbaar.
          </div>
        )}
      </div>

      <NarrativeBlock narrative={page.narrative} />
    </div>
  );
}
