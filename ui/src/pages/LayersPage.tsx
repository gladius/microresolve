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

// ── API helpers ───────────────────────────────────────────────────────────────

async function fetchInfo(): Promise<LayersInfo> {
  const r = await fetch(`${BASE}/layers/info`, { headers: appHeaders() });
  return r.json();
}

async function fetchEdges(): Promise<Edge[]> {
  const r = await fetch(`${BASE}/layers/l1/edges`, { headers: appHeaders() });
  const d = await r.json();
  return d.edges ?? [];
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
  const [filter, setFilter] = useState('');
  const [kindFilter, setKindFilter] = useState<string>('all');
  const [probeQuery, setProbeQuery] = useState('');
  const [probeResult, setProbeResult] = useState<ProbeResult | null>(null);
  const [probing, setProbing] = useState(false);
  const [distilling, setDistilling] = useState(false);
  const [distillMsg, setDistillMsg] = useState('');

  // Add edge form
  const [addFrom, setAddFrom] = useState('');
  const [addTo, setAddTo] = useState('');
  const [addKind, setAddKind] = useState<'morphological' | 'abbreviation' | 'synonym'>('synonym');
  const [addWeight, setAddWeight] = useState('');
  const [adding, setAdding] = useState(false);

  const reload = async () => {
    const [i, e] = await Promise.all([fetchInfo(), fetchEdges()]);
    setInfo(i);
    setEdges(e);
  };

  useEffect(() => { reload(); }, []);

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
      const w = parseFloat(addWeight) || (addKind === 'synonym' ? 0.88 : addKind === 'morphological' ? 0.98 : 0.99);
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

  const filtered = edges.filter(e => {
    if (kindFilter !== 'all' && e.kind !== kindFilter) return false;
    if (filter && !e.from.includes(filter) && !e.to.includes(filter)) return false;
    return true;
  });

  const maxScore = probeResult?.scores[0]?.score ?? 1;

  return (
    <Page title="Layers" subtitle="Inspect and edit L0 · L1 · L2 routing layers" size="lg">
      <div className="space-y-6">

        {/* ── Stats row ── */}
        {info && (
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'L0  Typo Correction', stats: [`${info.l0.vocab_size.toLocaleString()} vocab terms`], color: 'border-zinc-700' },
              { label: 'L1  Lexical Graph',   stats: [`${info.l1.edges_morphological} morph`, `${info.l1.edges_abbreviation} abbrev`, `${info.l1.edges_synonym} synonym`], color: 'border-violet-500/40' },
              { label: 'L2  Intent Index',    stats: [`${info.l2.words.toLocaleString()} words`, `${info.l2.intents} intents`], color: 'border-emerald-500/40' },
            ].map(({ label, stats, color }) => (
              <div key={label} className={`bg-zinc-900 border ${color} rounded-xl p-4`}>
                <div className="text-xs text-zinc-500 font-mono mb-2">{label}</div>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5">
                  {stats.map(s => <span key={s} className="text-sm font-semibold text-white">{s}</span>)}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── L1 Edge editor ── */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-white">L1 — Lexical Graph</div>
              <div className="text-xs text-zinc-500 mt-0.5">Morphology · Abbreviations · Domain synonyms</div>
            </div>
            <div className="flex items-center gap-2">
              {distillMsg && <span className="text-xs text-emerald-400">{distillMsg}</span>}
              <button onClick={handleDistill} disabled={distilling}
                className="px-3 py-1.5 text-xs bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white rounded-lg transition-colors font-medium">
                {distilling ? 'Distilling...' : '⚡ LLM distill'}
              </button>
            </div>
          </div>

          {/* Add edge form */}
          <form onSubmit={handleAdd} className="flex items-center gap-2 flex-wrap">
            <input value={addFrom} onChange={e => setAddFrom(e.target.value)}
              placeholder="from (e.g. canceling)"
              className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 w-36" />
            <span className="text-zinc-600">→</span>
            <input value={addTo} onChange={e => setAddTo(e.target.value)}
              placeholder="to (e.g. cancel)"
              className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 w-36" />
            <select value={addKind} onChange={e => setAddKind(e.target.value as any)}
              className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-violet-500">
              <option value="morphological">morphological</option>
              <option value="abbreviation">abbreviation</option>
              <option value="synonym">synonym</option>
            </select>
            <input value={addWeight} onChange={e => setAddWeight(e.target.value)}
              placeholder="weight (0.88)"
              className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 w-24" />
            <button type="submit" disabled={adding || !addFrom || !addTo}
              className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded transition-colors">
              + Add edge
            </button>
          </form>

          {/* Filter row */}
          <div className="flex items-center gap-2">
            <input value={filter} onChange={e => setFilter(e.target.value)}
              placeholder="Search edges..."
              className="bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-zinc-600 flex-1 max-w-xs" />
            {(['all', 'morphological', 'abbreviation', 'synonym'] as const).map(k => (
              <button key={k} onClick={() => setKindFilter(k)}
                className={`px-2 py-1 text-[10px] rounded border transition-colors ${kindFilter === k ? 'border-violet-500 text-violet-400 bg-violet-500/10' : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'}`}>
                {k}
              </button>
            ))}
            <span className="text-xs text-zinc-600 ml-auto">{filtered.length} edges</span>
          </div>

          {/* Edge list */}
          <div className="max-h-80 overflow-y-auto space-y-1 pr-1">
            {filtered.length === 0 && (
              <div className="text-xs text-zinc-600 py-4 text-center">
                {edges.length === 0 ? 'No edges yet — add one above or run LLM distill' : 'No edges match filter'}
              </div>
            )}
            {filtered.map((e, i) => (
              <div key={i} className="flex items-center gap-2 px-2 py-1 rounded hover:bg-zinc-800/50 group">
                <span className="font-mono text-xs text-zinc-300 w-32 truncate">{e.from}</span>
                <span className="text-zinc-600">→</span>
                <span className="font-mono text-xs text-zinc-300 w-32 truncate">{e.to}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded border font-semibold ${KIND_COLORS[e.kind]}`}>{e.kind.slice(0, 5)}</span>
                <span className="text-[10px] text-zinc-600 w-8">{e.weight.toFixed(2)}</span>
                <button onClick={() => handleDelete(e.from, e.to)}
                  className="ml-auto text-zinc-700 hover:text-red-400 text-[10px] opacity-0 group-hover:opacity-100 transition-all">✕</button>
              </div>
            ))}
          </div>
        </div>

        {/* ── L2 probe ── */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-4">
          <div>
            <div className="text-sm font-medium text-white">L2 — Intent Index probe</div>
            <div className="text-xs text-zinc-500 mt-0.5">Trace any query through all layers live</div>
          </div>

          <form onSubmit={handleProbe} className="flex gap-2">
            <input value={probeQuery} onChange={e => setProbeQuery(e.target.value)}
              placeholder="Enter query to trace through L0 → L1 → L2..."
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 font-mono" />
            <button type="submit" disabled={probing || !probeQuery.trim()}
              className="px-4 py-2 text-xs bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white rounded-lg transition-colors">
              {probing ? 'Probing...' : 'Probe'}
            </button>
          </form>

          {probeResult && (
            <div className="space-y-3 font-mono text-xs">
              {/* L0 */}
              <div className="flex items-start gap-3">
                <span className="text-zinc-600 w-20 shrink-0 pt-0.5">L0  typo</span>
                {probeResult.l0_corrected !== probeQuery
                  ? <span className="text-amber-400">{probeResult.l0_corrected} <span className="text-zinc-600">(was: {probeQuery})</span></span>
                  : <span className="text-zinc-600">no change</span>
                }
              </div>
              {/* L1 normalize */}
              <div className="flex items-start gap-3">
                <span className="text-zinc-600 w-20 shrink-0 pt-0.5">L1  morph</span>
                {probeResult.l1_normalized !== probeResult.l0_corrected
                  ? <span className="text-amber-400">{probeResult.l1_normalized}</span>
                  : <span className="text-zinc-600">no change</span>
                }
              </div>
              {/* L1 inject */}
              {probeResult.l1_injected.length > 0 && (
                <div className="flex items-start gap-3">
                  <span className="text-zinc-600 w-20 shrink-0 pt-0.5">L1  inject</span>
                  <div className="flex flex-wrap gap-1">
                    {probeResult.l1_injected.map(w => (
                      <span key={w} className="px-1.5 py-0.5 rounded bg-violet-500/10 border border-violet-500/30 text-violet-400 text-[10px]">{w}</span>
                    ))}
                  </div>
                </div>
              )}
              {/* L2 tokens */}
              <div className="flex items-start gap-3">
                <span className="text-zinc-600 w-20 shrink-0 pt-0.5">L2  tokens</span>
                <div className="flex flex-wrap gap-1">
                  {probeResult.tokens.map(t => (
                    <span key={t} className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 text-[10px]">{t}</span>
                  ))}
                </div>
              </div>
              {/* L2 scores */}
              {probeResult.scores.length > 0 && (
                <div className="space-y-1.5 pt-1">
                  <div className="text-[10px] text-zinc-600 uppercase font-semibold tracking-wide">Top intent scores</div>
                  {probeResult.scores.map(({ id, score }) => {
                    const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;
                    const isTop = score === maxScore;
                    return (
                      <div key={id} className="flex items-center gap-2">
                        <span className={`w-40 truncate text-right text-[11px] ${isTop ? 'text-emerald-400 font-semibold' : 'text-zinc-500'}`}>
                          {id.split(':').pop()}
                        </span>
                        <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${isTop ? 'bg-emerald-500' : 'bg-zinc-600'}`}
                            style={{ width: `${pct}%` }} />
                        </div>
                        <span className={`text-[10px] w-10 text-right ${isTop ? 'text-emerald-400' : 'text-zinc-600'}`}>
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
      </div>
    </Page>
  );
}
