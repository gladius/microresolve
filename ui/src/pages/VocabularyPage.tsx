import { useState, useEffect, useRef } from 'react';
import Page from '@/components/Page';
import { appHeaders } from '@/api/client';

const BASE = '/api';

interface Edge { from: string; to: string; kind: 'morphological' | 'abbreviation' | 'synonym'; weight: number }

type EdgeKind = 'morphological' | 'synonym' | 'abbreviation';

const COLUMNS: { kind: EdgeKind; label: string; color: string; emptyHint: string }[] = [
  {
    kind: 'morphological',
    label: 'Morphology',
    color: 'text-blue-400 bg-blue-400/10 border-blue-400/30',
    emptyHint: 'No morphology edges yet. These appear as the namespace learns from your LLM calls.',
  },
  {
    kind: 'synonym',
    label: 'Synonyms',
    color: 'text-violet-400 bg-violet-400/10 border-violet-400/30',
    emptyHint: 'No synonym edges yet. These appear as the namespace learns from your LLM calls.',
  },
  {
    kind: 'abbreviation',
    label: 'Abbreviations',
    color: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
    emptyHint: 'No abbreviation edges yet. These appear as the namespace learns from your LLM calls.',
  },
];

async function fetchEdgesByKind(kind: EdgeKind, filter: string): Promise<Edge[]> {
  const params = new URLSearchParams({ limit: '300', kind });
  if (filter) params.set('q', filter);
  const r = await fetch(`${BASE}/layers/l1/edges?${params}`, { headers: appHeaders() });
  const d = await r.json();
  return d.edges ?? [];
}

// ── Single column component ────────────────────────────────────────────────────

function EdgeColumn({ kind, label, color, emptyHint }: (typeof COLUMNS)[number]) {
  const [edges, setEdges] = useState<Edge[]>([]);
  const [filter, setFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = async (f: string) => {
    const result = await fetchEdgesByKind(kind, f);
    setEdges(result);
    setLoading(false);
  };

  useEffect(() => { load(''); }, []);

  const onFilterChange = (val: string) => {
    setFilter(val);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => load(val), 250);
  };

  return (
    <div className="flex flex-col bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden min-h-0 flex-1">
      {/* Column header */}
      <div className="px-4 py-3 border-b border-zinc-800 shrink-0">
        <div className="flex items-center justify-between gap-2 mb-2">
          <span className={`text-xs font-semibold px-2 py-0.5 rounded border ${color}`}>{label}</span>
          <span className="text-[10px] text-zinc-600 tabular-nums">{edges.length} edges</span>
        </div>
        <input
          value={filter}
          onChange={e => onFilterChange(e.target.value)}
          placeholder={`Search ${label.toLowerCase()}…`}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
        />
      </div>

      {/* Edge list */}
      <div className="flex-1 overflow-y-auto divide-y divide-zinc-800/40">
        {loading ? (
          <div className="text-xs text-zinc-600 py-8 text-center">Loading…</div>
        ) : edges.length === 0 ? (
          <div className="text-xs text-zinc-600 py-8 px-4 text-center leading-relaxed">{emptyHint}</div>
        ) : (
          edges.map((e, i) => (
            <div key={i} className="flex items-center gap-2 px-3 py-2 hover:bg-zinc-800/40">
              <span className="font-mono text-xs text-zinc-300 flex-1 truncate">{e.from}</span>
              <span className="text-zinc-600 shrink-0">→</span>
              <span className="font-mono text-xs text-zinc-300 flex-1 truncate">{e.to}</span>
              <span className="text-[10px] text-zinc-500 tabular-nums w-8 text-right shrink-0">
                {e.weight.toFixed(2)}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function VocabularyPage() {
  return (
    <Page title="Vocabulary" subtitle="Morphology · Synonyms · Abbreviations" fullscreen>
      <div className="h-full flex gap-4 p-4 min-h-0">
        {COLUMNS.map(col => (
          <EdgeColumn key={col.kind} {...col} />
        ))}
      </div>
    </Page>
  );
}
