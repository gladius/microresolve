import type { RouteTrace } from '@/api/client';

/**
 * Renders the five layer cards (L0, L1, L2 multi-round, L3, Relations)
 * from a RouteTrace. Used by both the Route page (expanded view) and the
 * Layers page probe tool.
 */
export default function LayerCards({
  original,
  trace,
}: {
  original: string;
  trace: RouteTrace;
}) {
  return (
    <div className="space-y-2">
      <L0Card original={original} trace={trace} />
      <L1Card trace={trace} />
      <L2Card trace={trace} />
      <L3Card trace={trace} />
    </div>
  );
}

// ── Shell ────────────────────────────────────────────────────────────────────

function LayerCard({ tag, name, subtitle, active, children }: {
  tag: string;
  name: string;
  subtitle: string;
  active: boolean;
  children: React.ReactNode;
}) {
  const accent = active ? 'border-emerald-500/40' : 'border-zinc-800';
  const dot    = active ? 'bg-emerald-400' : 'bg-zinc-700';
  return (
    <div className={`bg-zinc-950 border ${accent} rounded-lg p-3`}>
      <div className="flex items-baseline gap-2 mb-1">
        <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
        <span className="text-[10px] font-mono font-bold text-zinc-600">{tag}</span>
        <span className="text-sm font-semibold text-zinc-100">{name}</span>
        <span className="text-[10px] text-zinc-600 ml-auto">{subtitle}</span>
      </div>
      <div className="pl-3.5 mt-1.5">{children}</div>
    </div>
  );
}

// Single-line row for a layer that didn't do anything this query.
function InactiveLine({ tag, name, note }: { tag: string; name: string; note: string }) {
  return (
    <div className="flex items-center gap-2 text-[11px] font-mono px-3 py-1.5 bg-zinc-950 border border-zinc-800/60 rounded">
      <span className="w-1.5 h-1.5 rounded-full bg-zinc-700" />
      <span className="font-bold text-zinc-600">{tag}</span>
      <span className="text-zinc-500">{name}</span>
      <span className="text-zinc-700 italic ml-auto">{note}</span>
    </div>
  );
}

// ── L0 ───────────────────────────────────────────────────────────────────────

function L0Card({ original, trace }: { original: string; trace: RouteTrace }) {
  const changed = trace.l0_corrected.toLowerCase() !== original.toLowerCase();
  if (!changed) return <InactiveLine tag="L0" name="Typo Correction" note="no corrections" />;
  return (
    <LayerCard tag="L0" name="Typo Correction" subtitle="trigram spelling fix" active>
      <div className="font-mono text-xs flex items-center gap-3 flex-wrap">
        <span className="text-zinc-500 line-through">{original}</span>
        <span className="text-zinc-600">→</span>
        <span className="text-amber-400">{trace.l0_corrected}</span>
      </div>
    </LayerCard>
  );
}

// ── L1 ───────────────────────────────────────────────────────────────────────

function L1Card({ trace }: { trace: RouteTrace }) {
  const normalizedChanged = trace.l1_normalized.toLowerCase() !== trace.l0_corrected.toLowerCase();
  const injected = trace.l1_injected || [];
  const active = normalizedChanged || injected.length > 0;
  if (!active) return <InactiveLine tag="L1" name="Lexical Graph" note="no transformations" />;

  return (
    <LayerCard tag="L1" name="Lexical Graph"
      subtitle="morphology · abbreviation · synonym"
      active>
      <div className="space-y-2 font-mono text-xs">
        {normalizedChanged && (
          <div className="flex items-start gap-2">
            <span className="text-zinc-600 w-24 text-[10px] uppercase tracking-wide shrink-0 mt-0.5">normalized</span>
            <span className="text-amber-400 break-all">{trace.l1_normalized}</span>
          </div>
        )}
        {injected.length > 0 && (
          <div className="flex items-start gap-2">
            <span className="text-zinc-600 w-24 text-[10px] uppercase tracking-wide shrink-0 mt-0.5">injected</span>
            <div className="flex flex-wrap gap-1">
              {injected.map(w => (
                <span key={w} className="px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px]">{w}</span>
              ))}
            </div>
          </div>
        )}
        <div className="flex items-start gap-2">
          <span className="text-zinc-600 w-24 text-[10px] uppercase tracking-wide shrink-0 mt-0.5">final</span>
          <span className="text-zinc-300 break-all">{trace.l1_expanded}</span>
        </div>
      </div>
    </LayerCard>
  );
}

// ── L2 ───────────────────────────────────────────────────────────────────────

function L2Card({ trace }: { trace: RouteTrace }) {
  const multi = trace.multi;
  const rounds = multi?.rounds || [];
  const active = rounds.length > 0;
  const roundTop = rounds[0]?.scored[0]?.[1] ?? 1;

  if (!active) return <InactiveLine tag="L2" name="Intent Index" note="no intent matched above threshold" />;

  return (
    <LayerCard tag="L2" name="Intent Index"
      subtitle={`${rounds.length} round${rounds.length === 1 ? '' : 's'} · token consumption`}
      active>

      {active && rounds.map((round, i) => {
        const confirmed = new Set(round.confirmed);
        const consumed  = new Set(round.consumed);
        const isStopRound = round.confirmed.length === 0;
        return (
          <div key={i} className="space-y-1.5 font-mono text-xs mb-3">
            <div className="text-[10px] text-zinc-500 uppercase tracking-wide">
              Round {i + 1} · {round.tokens_in.length} token{round.tokens_in.length === 1 ? '' : 's'}
            </div>

            <div className="flex flex-wrap gap-1 pl-1">
              {round.tokens_in.map((t, ti) => (
                <span key={ti}
                  className={`px-1.5 py-0.5 rounded text-[10px] border ${
                    consumed.has(t)
                      ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-400'
                      : 'bg-zinc-800 border-zinc-700 text-zinc-400'
                  }`}>
                  {t}
                </span>
              ))}
            </div>

            {!isStopRound ? (
              <div className="space-y-1 pt-1 pl-1">
                {round.scored.map(([id, score]) => {
                  const pct = roundTop > 0 ? (score / roundTop) * 100 : 0;
                  const isConfirmed = confirmed.has(id);
                  return (
                    <div key={id} className="flex items-center gap-2">
                      <span className={`w-40 truncate text-right text-[11px] ${isConfirmed ? 'text-emerald-400 font-semibold' : 'text-zinc-500'}`}>
                        {id.split(':').pop()}
                      </span>
                      <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden max-w-xs">
                        <div className={`h-full ${isConfirmed ? 'bg-emerald-500' : 'bg-zinc-600'}`}
                          style={{ width: `${Math.min(100, pct)}%` }} />
                      </div>
                      <span className={`text-[10px] w-10 text-right tabular-nums ${isConfirmed ? 'text-emerald-400' : 'text-zinc-600'}`}>
                        {score.toFixed(3)}
                      </span>
                      {isConfirmed && <span className="text-emerald-400 text-[10px]">✓</span>}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-[10px] text-zinc-600 italic pl-1">
                all scores below threshold — stopping
              </div>
            )}
          </div>
        );
      })}

      {active && multi && (
        <div className="text-[10px] text-zinc-600 border-t border-zinc-800 pt-2 mt-1">
          stop: {multi.stop_reason}
        </div>
      )}
    </LayerCard>
  );
}

// ── L3 ───────────────────────────────────────────────────────────────────────

function L3Card({ trace }: { trace: RouteTrace }) {
  const suppressions = trace.multi?.suppressions || [];
  if (suppressions.length === 0) return <InactiveLine tag="L3" name="Suppression" note="no suppressions" />;

  return (
    <LayerCard tag="L3" name="Suppression"
      subtitle="anti-Hebbian · learned from corrections"
      active>
      <div className="space-y-1.5 font-mono text-xs">
        {suppressions.map(([a, b, strength], i) => (
          <div key={i} className="flex items-center gap-2 flex-wrap">
            <span className="text-emerald-400">{a.split(':').pop()}</span>
            <span className="text-zinc-600 text-[10px]">suppresses</span>
            <span className="text-red-400 line-through">{b.split(':').pop()}</span>
            <span className="text-zinc-600 text-[10px] tabular-nums">strength {strength.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </LayerCard>
  );
}

// ── Compact summary strip (for Route page collapsed view) ────────────────────

export function LayerSummaryStrip({
  trace,
  original,
  onClick,
  expanded,
}: {
  trace: RouteTrace;
  original: string;
  onClick: () => void;
  expanded: boolean;
}) {
  const l0changed = trace.l0_corrected.toLowerCase() !== original.toLowerCase();
  const l1injected = (trace.l1_injected || []).length;
  const l1normalized = trace.l1_normalized.toLowerCase() !== trace.l0_corrected.toLowerCase();
  const l2rounds = trace.multi?.rounds.length || 0;
  const l3suppressions = (trace.multi?.suppressions || []).length;

  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 text-[10px] text-zinc-500 hover:text-zinc-300 font-mono py-1 transition-colors"
    >
      <Pill label="L0" value={l0changed ? 'fix' : '—'}        on={l0changed} />
      <Pill label="L1" value={l1normalized || l1injected ? (l1injected ? `+${l1injected} syn` : 'norm') : '—'} on={l1normalized || l1injected > 0} />
      <Pill label="L2" value={l2rounds ? `${l2rounds} round${l2rounds === 1 ? '' : 's'}` : '—'} on={l2rounds > 0} />
      <Pill label="L3" value={l3suppressions ? `${l3suppressions} sup` : '—'} on={l3suppressions > 0} />
      <span className="text-zinc-700">{expanded ? '▲' : '▼'}</span>
    </button>
  );
}

function Pill({ label, value, on }: { label: string; value: string; on: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 ${on ? 'text-emerald-400' : 'text-zinc-600'}`}>
      <span className="font-bold">{label}</span>
      <span>{value}</span>
    </span>
  );
}
