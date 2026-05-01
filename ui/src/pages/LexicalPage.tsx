import { useState, useEffect, useRef } from 'react';
import Page from '@/components/Page';
import { appHeaders } from '@/api/client';
import LayerToggle, { type LayerField } from '@/components/LayerToggle';
import { useAppStore } from '@/store';

const BASE = '/api';

interface Edge { from: string; to: string; kind: 'morphological' | 'abbreviation' | 'synonym'; weight: number }

type EdgeKind = 'morphological' | 'synonym' | 'abbreviation';

const KIND_TO_FIELD: Record<EdgeKind, LayerField> = {
  morphological: 'l1_morphology',
  synonym:       'l1_synonym',
  abbreviation:  'l1_abbreviation',
};

const COLUMNS: { kind: EdgeKind; label: string; color: string; defaultWeight: number; emptyHint: string }[] = [
  {
    kind: 'morphological',
    label: 'Morphology',
    color: 'text-blue-400 bg-blue-400/10 border-blue-400/30',
    defaultWeight: 0.98,
    emptyHint: 'No morphology edges yet. These are added automatically when the namespace routes new vocabulary.',
  },
  {
    kind: 'synonym',
    label: 'Synonyms',
    color: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
    defaultWeight: 0.88,
    emptyHint: 'No synonym edges yet. Add them manually or via import.',
  },
  {
    kind: 'abbreviation',
    label: 'Abbreviations',
    color: 'text-amber-400 bg-amber-400/10 border-amber-400/30',
    defaultWeight: 0.99,
    emptyHint: 'No abbreviation edges yet. These are added automatically when the namespace routes new vocabulary.',
  },
];

// ── API ───────────────────────────────────────────────────────────────────────

async function fetchEdgesByKind(kind: EdgeKind, filter: string): Promise<Edge[]> {
  const params = new URLSearchParams({ limit: '300', kind });
  if (filter) params.set('q', filter);
  const r = await fetch(`${BASE}/layers/l1/edges?${params}`, { headers: appHeaders() });
  const d = await r.json();
  return d.edges ?? [];
}

async function apiAddEdge(from: string, to: string, kind: EdgeKind, weight: number) {
  await fetch(`${BASE}/layers/l1/edges`, {
    method: 'POST',
    headers: appHeaders(),
    body: JSON.stringify({ from, to, kind, weight }),
  });
}

async function apiDeleteEdge(from: string, to: string) {
  await fetch(`${BASE}/layers/l1/edges`, {
    method: 'DELETE',
    headers: appHeaders(),
    body: JSON.stringify({ from, to }),
  });
}

// ── Single column ─────────────────────────────────────────────────────────────

function EdgeColumn({ kind, label, color, defaultWeight, emptyHint }: (typeof COLUMNS)[number]) {
  const [edges, setEdges]   = useState<Edge[]>([]);
  const [filter, setFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [addFrom, setAddFrom] = useState('');
  const [addTo,   setAddTo]   = useState('');
  const [addWeight, setAddWeight] = useState('');
  const [adding, setAdding] = useState(false);
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

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!addFrom.trim() || !addTo.trim()) return;
    setAdding(true);
    try {
      const w = parseFloat(addWeight) || defaultWeight;
      await apiAddEdge(addFrom.trim().toLowerCase(), addTo.trim().toLowerCase(), kind, w);
      setAddFrom(''); setAddTo(''); setAddWeight('');
      await load(filter);
    } finally { setAdding(false); }
  };

  const handleDelete = async (from: string, to: string) => {
    await apiDeleteEdge(from, to);
    await load(filter);
  };

  return (
    <div className="flex flex-col bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden min-h-0 flex-1">
      {/* Header */}
      <div className="px-4 py-3 border-b border-zinc-800 shrink-0 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className={`text-xs font-semibold px-2 py-0.5 rounded border ${color}`}>{label}</span>
          <span className="text-[10px] text-zinc-600 tabular-nums">{edges.length} edges</span>
        </div>
        <LayerToggle field={KIND_TO_FIELD[kind]} label={label} compact />
        <input
          value={filter}
          onChange={e => onFilterChange(e.target.value)}
          placeholder={`Search ${label.toLowerCase()}…`}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
        />
        {/* Add form */}
        <form onSubmit={handleAdd} className="flex items-center gap-1.5">
          <input
            value={addFrom}
            onChange={e => setAddFrom(e.target.value)}
            placeholder="from"
            className="flex-1 min-w-0 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
          />
          <span className="text-zinc-600 text-xs shrink-0">→</span>
          <input
            value={addTo}
            onChange={e => setAddTo(e.target.value)}
            placeholder="to"
            className="flex-1 min-w-0 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
          />
          <input
            value={addWeight}
            onChange={e => setAddWeight(e.target.value)}
            placeholder={defaultWeight.toString()}
            className="w-12 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
          />
          <button
            type="submit"
            disabled={adding || !addFrom.trim() || !addTo.trim()}
            className="px-2 py-1 text-xs bg-zinc-700 hover:bg-zinc-600 disabled:opacity-40 text-zinc-200 rounded transition-colors shrink-0"
          >
            Add
          </button>
        </form>
      </div>

      {/* Edge list */}
      <div className="flex-1 overflow-y-auto divide-y divide-zinc-800/40">
        {loading ? (
          <div className="text-xs text-zinc-600 py-8 text-center">Loading…</div>
        ) : edges.length === 0 ? (
          <div className="text-xs text-zinc-600 py-8 px-4 text-center leading-relaxed">{emptyHint}</div>
        ) : (
          edges.map((e, i) => (
            <div key={i} className="flex items-center gap-2 px-3 py-2 hover:bg-zinc-800/40 group">
              <span className="font-mono text-xs text-zinc-300 flex-1 truncate">{e.from}</span>
              <span className="text-zinc-600 shrink-0">→</span>
              <span className="font-mono text-xs text-zinc-300 flex-1 truncate">{e.to}</span>
              <span className="text-[10px] text-zinc-500 tabular-nums w-8 text-right shrink-0">
                {e.weight.toFixed(2)}
              </span>
              <button
                onClick={() => handleDelete(e.from, e.to)}
                className="text-zinc-700 hover:text-red-400 text-[10px] opacity-0 group-hover:opacity-100 transition-all shrink-0 ml-1"
              >✕</button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function LexicalPage() {
  const { layerStatus } = useAppStore();
  const allOn  = layerStatus.l1m && layerStatus.l1s && layerStatus.l1a;
  const allOff = !layerStatus.l1m && !layerStatus.l1s && !layerStatus.l1a;
  // Only render a badge when something's actionable. The all-on state is
  // the unremarkable normal case — showing a green `on` pill there is just
  // chrome noise. Matches the sidebar convention (no pill when fully on).
  const status: 'off' | 'partial' | null =
    allOff ? 'off' : allOn ? null : 'partial';

  return (
    <Page
      title="L1 — Lexical"
      subtitle={
        <span className="inline-flex items-center gap-2">
          <span>Morphology · Synonyms · Abbreviations</span>
          {status && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider ${
              status === 'off' ? 'bg-zinc-800 text-zinc-500' : 'bg-amber-500/15 text-amber-400'
            }`}>{status}</span>
          )}
        </span>
      }
      fullscreen
    >
      <div className="h-full flex gap-4 p-4 min-h-0">
        {COLUMNS.map(col => (
          <EdgeColumn key={col.kind} {...col} />
        ))}
      </div>
    </Page>
  );
}
