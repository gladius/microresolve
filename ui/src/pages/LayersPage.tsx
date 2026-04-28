import { useState, useEffect } from 'react';
import Page from '@/components/Page';
import { appHeaders } from '@/api/client';

const BASE = '/api';

// ── Types ─────────────────────────────────────────────────────────────────────

interface LayersInfo {
  l0: { vocab_size: number };
  l1: { terms: number; edges_morphological: number; edges_abbreviation: number; edges_synonym: number; edges_total: number };
  l2: { words: number; intents: number };
}

interface Edge { from: string; to: string; kind: 'morphological' | 'abbreviation' | 'synonym'; weight: number }

const KIND_COLORS: Record<string, string> = {
  morphological: 'text-blue-400 bg-blue-400/10 border-blue-400/30',
  abbreviation:  'text-amber-400 bg-amber-400/10 border-amber-400/30',
  synonym:       'text-violet-400 bg-violet-400/10 border-violet-400/30',
};

const KIND_DEFAULTS: Record<string, number> = {
  morphological: 0.98,
  abbreviation: 0.99,
};

// ── API helpers ───────────────────────────────────────────────────────────────

async function fetchInfo(): Promise<LayersInfo> {
  const r = await fetch(`${BASE}/layers/info`, { headers: appHeaders() });
  return r.json();
}

async function fetchEdges(filter = '', kind = 'all'): Promise<{ edges: Edge[]; total: number }> {
  const params = new URLSearchParams({ limit: '200' });
  if (filter) params.set('q', filter);
  if (kind !== 'all') params.set('kind', kind);
  const r = await fetch(`${BASE}/layers/l1/edges?${params}`, { headers: appHeaders() });
  const d = await r.json();
  return { edges: d.edges ?? [], total: d.total ?? 0 };
}

async function addEdge(from: string, to: string, kind: string, weight: number) {
  await fetch(`${BASE}/layers/l1/edges`, {
    method: 'POST', headers: appHeaders(),
    body: JSON.stringify({ from, to, kind, weight }),
  });
}

async function deleteEdge(from: string, to: string) {
  await fetch(`${BASE}/layers/l1/edges`, {
    method: 'DELETE', headers: appHeaders(),
    body: JSON.stringify({ from, to }),
  });
}

async function distill(): Promise<{ ok: boolean; edges_total?: number }> {
  const r = await fetch(`${BASE}/layers/l1/distill`, { method: 'POST', headers: appHeaders() });
  return r.json();
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function LayersPage() {
  const [info, setInfo] = useState<LayersInfo | null>(null);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [edgeTotal, setEdgeTotal] = useState(0);
  const [filter, setFilter] = useState('');
  const [kindFilter, setKindFilter] = useState<string>('all');
  const filterTimer = useState<ReturnType<typeof setTimeout> | null>(null);
  const [distilling, setDistilling] = useState(false);
  const [distillMsg, setDistillMsg] = useState('');

  const [addFrom, setAddFrom] = useState('');
  const [addTo, setAddTo] = useState('');
  const [addKind, setAddKind] = useState<'morphological' | 'abbreviation'>('morphological');
  const [addWeight, setAddWeight] = useState('');
  const [adding, setAdding] = useState(false);

  const reloadEdges = async (f = filter, k = kindFilter) => {
    const { edges: e, total } = await fetchEdges(f, k);
    setEdges(e);
    setEdgeTotal(total);
  };

  const reload = async () => {
    const i = await fetchInfo();
    setInfo(i);
    await reloadEdges();
  };

  useEffect(() => { reload(); }, []);

  const onFilterChange = (val: string) => {
    setFilter(val);
    if (filterTimer[0]) clearTimeout(filterTimer[0]);
    filterTimer[0] = setTimeout(() => reloadEdges(val, kindFilter), 300);
  };

  const onKindChange = (k: string) => {
    setKindFilter(k);
    reloadEdges(filter, k);
  };

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!addFrom.trim() || !addTo.trim()) return;
    setAdding(true);
    try {
      const w = parseFloat(addWeight) || KIND_DEFAULTS[addKind];
      await addEdge(addFrom.trim().toLowerCase(), addTo.trim().toLowerCase(), addKind, w);
      setAddFrom(''); setAddTo(''); setAddWeight('');
      await reload();
    } finally { setAdding(false); }
  };

  const handleDelete = async (from: string, to: string) => {
    await deleteEdge(from, to);
    await reload();
  };

  const handleDistill = async () => {
    setDistilling(true); setDistillMsg('');
    try {
      const r = await distill();
      setDistillMsg(r.ok ? `Done — ${r.edges_total} total edges` : 'Failed (no LLM key configured?)');
      await reload();
    } finally { setDistilling(false); }
  };

  return (
    <Page title="Lexical Graph" subtitle="Tune L1 morphology and abbreviation edges" size="lg">
      <div className="space-y-5">

        {/* Stats row — compact, single line */}
        {info && (
          <div className="flex items-center gap-4 text-xs font-mono text-zinc-500 bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-2">
            <Stat label="L1 terms"    value={info.l1.terms.toLocaleString()} />
            <Sep />
            <Stat label="morphology"  value={info.l1.edges_morphological.toLocaleString()} />
            <Stat label="abbrev"      value={info.l1.edges_abbreviation.toLocaleString()} />
            <Sep />
            <Stat label="L2 vocab"    value={info.l2.words.toLocaleString()} />
            <Stat label="intents"     value={info.l2.intents.toLocaleString()} />
          </div>
        )}

        {/* Distill CTA — morphology + abbreviation edges */}
        <div className="flex items-center justify-between gap-4 bg-zinc-900 border border-zinc-800 rounded-xl p-4">
          <div>
            <div className="text-sm font-medium text-zinc-100">Generate morphology & abbreviations</div>
            <div className="text-xs text-zinc-500 mt-0.5">
              Uses the LLM to produce domain-aware morphological and abbreviation edges.
              For semantic coverage, add training phrases on the Intents page instead.
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {distillMsg && <span className="text-xs text-emerald-400">{distillMsg}</span>}
            <button onClick={handleDistill} disabled={distilling}
              className="px-3 py-1.5 text-xs bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white rounded-lg transition-colors font-medium whitespace-nowrap">
              {distilling ? 'Generating…' : 'Generate via AI'}
            </button>
          </div>
        </div>

        {/* Add edge form */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-3">
          <div className="text-sm font-medium text-zinc-100">Add edge</div>
          <form onSubmit={handleAdd} className="flex items-end gap-2 flex-wrap">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-zinc-600 uppercase tracking-wide">From word</label>
              <input value={addFrom} onChange={e => setAddFrom(e.target.value)}
                placeholder="e.g. canceling"
                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-violet-500 w-36" />
            </div>
            <span className="text-zinc-600 pb-1.5">→</span>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-zinc-600 uppercase tracking-wide">To word</label>
              <input value={addTo} onChange={e => setAddTo(e.target.value)}
                placeholder="e.g. cancel"
                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-violet-500 w-36" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-zinc-600 uppercase tracking-wide">Type</label>
              <select value={addKind} onChange={e => setAddKind(e.target.value as 'morphological' | 'abbreviation')}
                className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-violet-500">
                <option value="morphological">morphological</option>
                <option value="abbreviation">abbreviation</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-zinc-600 uppercase tracking-wide" title="0.0 = very loose · 1.0 = identical">
                Weight <span className="text-zinc-700 normal-case">(?)</span>
              </label>
              <input value={addWeight} onChange={e => setAddWeight(e.target.value)}
                placeholder={`default ${KIND_DEFAULTS[addKind]}`}
                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-violet-500 w-28" />
            </div>
            <button type="submit" disabled={adding || !addFrom || !addTo}
              className="px-3 py-1.5 text-xs bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white rounded transition-colors font-medium">
              Add edge
            </button>
          </form>
        </div>

        {/* Filter + edge list */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center gap-2 flex-wrap">
            <input value={filter} onChange={e => onFilterChange(e.target.value)}
              placeholder="Search edges…"
              className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-600 flex-1 max-w-xs" />
            {(['all', 'morphological', 'abbreviation'] as const).map(k => (
              <button key={k} onClick={() => onKindChange(k)}
                className={`px-2 py-1 text-[10px] rounded border transition-colors ${kindFilter === k ? 'border-violet-500 text-violet-400 bg-violet-500/10' : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'}`}>
                {k}
              </button>
            ))}
            <span className="text-xs text-zinc-600 ml-auto">{edges.length} shown · {edgeTotal.toLocaleString()} total</span>
          </div>

          <div className="max-h-[60vh] overflow-y-auto divide-y divide-zinc-800/40 border border-zinc-800 rounded-lg overflow-hidden">
            {edges.length === 0 ? (
              <div className="text-xs text-zinc-600 py-8 text-center">
                {filter || kindFilter !== 'all'
                  ? 'No edges match filter'
                  : 'No edges yet — add one above or click "Generate via AI"'}
              </div>
            ) : edges.map((e, i) => (
              <div key={i} className="flex items-center gap-2 px-3 py-2 hover:bg-zinc-800/50 group">
                <span className="font-mono text-xs text-zinc-300 w-32 truncate">{e.from}</span>
                <span className="text-zinc-600">→</span>
                <span className="font-mono text-xs text-zinc-300 w-32 truncate">{e.to}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded border font-semibold ${KIND_COLORS[e.kind]}`}>{e.kind.slice(0, 5)}</span>
                <span className="text-[10px] text-zinc-600 tabular-nums w-8">{e.weight.toFixed(2)}</span>
                <button onClick={() => handleDelete(e.from, e.to)}
                  className="ml-auto text-zinc-700 hover:text-red-400 text-[10px] opacity-0 group-hover:opacity-100 transition-all">✕</button>
              </div>
            ))}
          </div>
        </div>

        <div className="text-[11px] text-zinc-600">
          Tip: use the <span className="text-violet-400">Route</span> page to probe a query and expand the layer trace to see how these edges affect routing.
        </div>

      </div>
    </Page>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-baseline gap-1.5">
      <span className="text-zinc-600 uppercase text-[10px] tracking-wide">{label}</span>
      <span className="text-zinc-200">{value}</span>
    </span>
  );
}

function Sep() {
  return <span className="text-zinc-800">·</span>;
}
