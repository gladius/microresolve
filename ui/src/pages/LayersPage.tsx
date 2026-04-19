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

interface ProbeResult {
  l0_corrected: string;
  l1_normalized: string;
  l1_expanded: string;
  l1_injected: string[];
  tokens: string[];
  scores: { id: string; score: number }[];
}

const KIND_COLORS: Record<string, string> = {
  morphological: 'text-blue-400 bg-blue-400/10 border-blue-400/30',
  abbreviation:  'text-amber-400 bg-amber-400/10 border-amber-400/30',
  synonym:       'text-violet-400 bg-violet-400/10 border-violet-400/30',
};

const KIND_DEFAULTS: Record<string, number> = {
  synonym: 0.88,
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

async function probe(query: string): Promise<ProbeResult> {
  const r = await fetch(`${BASE}/layers/l2/probe`, {
    method: 'POST', headers: appHeaders(),
    body: JSON.stringify({ query }),
  });
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
  const [probeQuery, setProbeQuery] = useState('');
  const [probeResult, setProbeResult] = useState<ProbeResult | null>(null);
  const [probing, setProbing] = useState(false);
  const [distilling, setDistilling] = useState(false);
  const [distillMsg, setDistillMsg] = useState('');

  const [addFrom, setAddFrom] = useState('');
  const [addTo, setAddTo] = useState('');
  const [addKind, setAddKind] = useState<'morphological' | 'abbreviation' | 'synonym'>('synonym');
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

  const handleProbe = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!probeQuery.trim()) return;
    setProbing(true);
    try { setProbeResult(await probe(probeQuery)); }
    finally { setProbing(false); }
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

  const maxScore = probeResult?.scores[0]?.score ?? 1;

  return (
    <Page title="Layers" subtitle="Inspect and tune the routing pipeline" size="lg">
      <div className="space-y-5">

        {/* Advanced notice */}
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg px-4 py-2.5 text-xs text-zinc-500 flex items-center gap-2">
          <span className="text-zinc-600">⧉</span>
          <span>
            <span className="text-zinc-400 font-medium">Advanced.</span>{' '}
            Use the Probe tool to debug how a query flows through the pipeline. Edit the L1 lexical graph to fix synonym and morphology issues.
          </span>
        </div>

        {/* ── Stats row ── */}
        {info && (
          <div className="grid grid-cols-3 gap-3">
            {[
              {
                label: 'L0', name: 'Typo Correction',
                desc: 'Trigram spelling fix before routing',
                stats: [`${info.l0.vocab_size.toLocaleString()} vocab terms`],
                color: 'border-zinc-700',
              },
              {
                label: 'L1', name: 'Lexical Graph',
                desc: 'Synonym · morphology · abbreviation edges',
                stats: [
                  `${info.l1.edges_morphological} morphological`,
                  `${info.l1.edges_abbreviation} abbreviation`,
                  `${info.l1.edges_synonym} synonym`,
                ],
                color: 'border-violet-500/30',
              },
              {
                label: 'L2', name: 'Intent Index',
                desc: 'BM25-style inverted index over phrases',
                stats: [`${info.l2.words.toLocaleString()} terms`, `${info.l2.intents} intents`],
                color: 'border-emerald-500/30',
              },
            ].map(({ label, name, desc, stats, color }) => (
              <div key={label} className={`bg-zinc-900 border ${color} rounded-xl p-4`}>
                <div className="flex items-baseline gap-2 mb-1">
                  <span className="text-[10px] font-bold font-mono text-zinc-600">{label}</span>
                  <span className="text-xs font-semibold text-zinc-300">{name}</span>
                </div>
                <div className="text-[10px] text-zinc-600 mb-2">{desc}</div>
                <div className="flex flex-col gap-0.5">
                  {stats.map(s => <span key={s} className="text-xs text-white font-mono">{s}</span>)}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── Probe tool ── */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-4">
          <div>
            <div className="text-sm font-medium text-white">Probe</div>
            <div className="text-xs text-zinc-500 mt-0.5">Trace any query through L0 → L1 → L2 to see exactly what the router does with it</div>
          </div>

          <form onSubmit={handleProbe} className="flex gap-2">
            <input value={probeQuery} onChange={e => setProbeQuery(e.target.value)}
              placeholder="Type a query to trace…"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 font-mono" />
            <button type="submit" disabled={probing || !probeQuery.trim()}
              className="px-4 py-2 text-xs bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white rounded-lg transition-colors font-medium">
              {probing ? 'Probing…' : 'Probe'}
            </button>
          </form>

          {probeResult && (
            <div className="space-y-2 font-mono text-xs border-t border-zinc-800 pt-4">
              <TraceRow layer="L0" label="typo fix"
                changed={probeResult.l0_corrected !== probeQuery}
                original={probeQuery}
                result={probeResult.l0_corrected} />
              <TraceRow layer="L1" label="morphology"
                changed={probeResult.l1_normalized !== probeResult.l0_corrected}
                original={probeResult.l0_corrected}
                result={probeResult.l1_normalized} />

              {probeResult.l1_injected.length > 0 && (
                <div className="flex items-start gap-3 pl-16">
                  <span className="text-zinc-600 shrink-0">injected</span>
                  <div className="flex flex-wrap gap-1">
                    {probeResult.l1_injected.map(w => (
                      <span key={w} className="px-1.5 py-0.5 rounded bg-violet-500/10 border border-violet-500/30 text-violet-400 text-[10px]">{w}</span>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex items-start gap-3">
                <span className="text-zinc-600 w-14 shrink-0 text-right">L2</span>
                <span className="text-zinc-600 w-20 shrink-0">tokens</span>
                <div className="flex flex-wrap gap-1">
                  {probeResult.tokens.map(t => (
                    <span key={t} className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 text-[10px]">{t}</span>
                  ))}
                </div>
              </div>

              {probeResult.scores.length > 0 && (
                <div className="pt-2 space-y-1.5">
                  <div className="text-[10px] text-zinc-600 uppercase font-semibold tracking-wide pl-16">Intent scores</div>
                  {probeResult.scores.map(({ id, score }) => {
                    const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;
                    const isTop = score === maxScore;
                    return (
                      <div key={id} className="flex items-center gap-2 pl-16">
                        <span className={`w-40 truncate text-right text-[11px] ${isTop ? 'text-emerald-400 font-semibold' : 'text-zinc-500'}`}>
                          {id.split(':').pop()}
                        </span>
                        <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${isTop ? 'bg-emerald-500' : 'bg-zinc-600'}`}
                            style={{ width: `${pct}%` }} />
                        </div>
                        <span className={`text-[10px] w-10 text-right tabular-nums ${isTop ? 'text-emerald-400' : 'text-zinc-600'}`}>
                          {score.toFixed(3)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── L1 Edge editor ── */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="text-sm font-medium text-white">L1 — Lexical Graph</div>
              <div className="text-xs text-zinc-500 mt-0.5">
                Edges teach the router that two words are related.
                Morphological and abbreviation edges are found automatically.
                Add synonym edges for domain-specific terms.
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {distillMsg && <span className="text-xs text-emerald-400">{distillMsg}</span>}
              <button onClick={handleDistill} disabled={distilling}
                className="px-3 py-1.5 text-xs bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white rounded-lg transition-colors font-medium whitespace-nowrap">
                {distilling ? 'Generating…' : '⚡ Generate synonyms via AI'}
              </button>
            </div>
          </div>

          {/* Add edge form */}
          <form onSubmit={handleAdd} className="flex items-end gap-2 flex-wrap bg-zinc-800/40 border border-zinc-800 rounded-lg p-3">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-zinc-600 uppercase tracking-wide">From word</label>
              <input value={addFrom} onChange={e => setAddFrom(e.target.value)}
                placeholder="e.g. canceling"
                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 w-36" />
            </div>
            <span className="text-zinc-600 pb-1.5">→</span>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-zinc-600 uppercase tracking-wide">To word</label>
              <input value={addTo} onChange={e => setAddTo(e.target.value)}
                placeholder="e.g. cancel"
                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 w-36" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-zinc-600 uppercase tracking-wide">Type</label>
              <select value={addKind} onChange={e => setAddKind(e.target.value as any)}
                className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-violet-500">
                <option value="synonym">synonym</option>
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
                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 w-28" />
            </div>
            <button type="submit" disabled={adding || !addFrom || !addTo}
              className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded transition-colors font-medium">
              Add edge
            </button>
          </form>

          {/* Filter row */}
          <div className="flex items-center gap-2">
            <input value={filter} onChange={e => onFilterChange(e.target.value)}
              placeholder="Search edges…"
              className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-zinc-600 flex-1 max-w-xs" />
            {(['all', 'morphological', 'abbreviation', 'synonym'] as const).map(k => (
              <button key={k} onClick={() => onKindChange(k)}
                className={`px-2 py-1 text-[10px] rounded border transition-colors ${kindFilter === k ? 'border-violet-500 text-violet-400 bg-violet-500/10' : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'}`}>
                {k}
              </button>
            ))}
            <span className="text-xs text-zinc-600 ml-auto">{edges.length} shown · {edgeTotal.toLocaleString()} total</span>
          </div>

          {/* Edge list */}
          <div className="max-h-72 overflow-y-auto divide-y divide-zinc-800/40 border border-zinc-800 rounded-lg overflow-hidden">
            {edges.length === 0 ? (
              <div className="text-xs text-zinc-600 py-6 text-center">
                {filter || kindFilter !== 'all'
                  ? 'No edges match filter'
                  : 'No edges yet — add one above or generate synonyms via AI'}
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

      </div>
    </Page>
  );
}

// ── Trace row ─────────────────────────────────────────────────────────────────

function TraceRow({ layer, label, changed, original, result }: {
  layer: string; label: string; changed: boolean; original: string; result: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <span className="text-zinc-600 w-14 shrink-0 text-right">{layer}</span>
      <span className="text-zinc-600 w-20 shrink-0">{label}</span>
      {changed ? (
        <span className="text-amber-400">
          {result}
          <span className="text-zinc-600 ml-2 line-through text-[10px]">{original}</span>
        </span>
      ) : (
        <span className="text-zinc-700">→ unchanged</span>
      )}
    </div>
  );
}
