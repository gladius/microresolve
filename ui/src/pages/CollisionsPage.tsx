import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '@/api/client';
import Page from '@/components/Page';

type CollisionResult = {
  pairs_analyzed: number;
  phrases_added: number;
  dry_run?: boolean;
  pairs: { intent_a: string; intent_b: string; overlap: number; phrases_added_a: number; phrases_added_b: number }[];
};

export default function CollisionsPage() {
  const [searchParams] = useSearchParams();
  const [domains,    setDomains]    = useState<string[]>([]);
  const [domain,     setDomain]     = useState(searchParams.get('domain') ?? '');
  const [threshold,  setThreshold]  = useState(0.15);
  const [phrasesN,   setPhrasesN]   = useState(5);
  const [running,    setRunning]    = useState(false);
  const [result,     setResult]     = useState<CollisionResult | null>(null);
  const [error,      setError]      = useState<string | null>(null);
  const [applied,    setApplied]    = useState(false);

  // Load domains and auto-preview on mount
  const preview = useCallback(async (dom: string, thresh: number) => {
    setRunning(true); setError(null); setResult(null); setApplied(false);
    try {
      const res = await api.discriminateIntents({ domain: dom || undefined, threshold: thresh, dry_run: true });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed');
    } finally { setRunning(false); }
  }, []);

  useEffect(() => {
    api.listIntents().then(intents => {
      const ds = Array.from(new Set(
        intents.map((i: any) => { const c = i.id.indexOf(':'); return c > 0 ? i.id.slice(0, c) : ''; }).filter(Boolean)
      )).sort() as string[];
      setDomains(ds);
    }).catch(() => {});
    preview(searchParams.get('domain') ?? '', 0.15);
  }, [preview]);

  const apply = async () => {
    setRunning(true); setError(null);
    try {
      const res = await api.discriminateIntents({
        domain: domain || undefined,
        threshold,
        phrases_per_pair: phrasesN,
        dry_run: false,
      });
      setResult(res);
      setApplied(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed');
    } finally { setRunning(false); }
  };

  const repreview = () => preview(domain, threshold);

  return (
    <Page title="Fix Collisions" subtitle="Intents with overlapping phrases that may confuse the router">
      <div className="max-w-2xl mx-auto p-6 space-y-6">

        {/* Config bar */}
        <div className="flex items-end gap-4 bg-zinc-800/50 rounded-lg p-4">
          <div className="flex-1">
            <label className="text-[10px] text-zinc-500 uppercase font-semibold block mb-1">Domain</label>
            <select value={domain} onChange={e => { setDomain(e.target.value); setResult(null); }}
              className="w-full bg-zinc-900 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1.5 focus:border-violet-500 focus:outline-none">
              <option value="">All domains</option>
              {domains.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[10px] text-zinc-500 uppercase font-semibold block mb-1">
              Overlap threshold — {Math.round(threshold * 100)}%
            </label>
            <input type="range" min={0.05} max={0.5} step={0.05} value={threshold}
              onChange={e => { setThreshold(Number(e.target.value)); setResult(null); }}
              className="w-36" />
          </div>
          <div>
            <label className="text-[10px] text-zinc-500 uppercase font-semibold block mb-1">Phrases / pair</label>
            <input type="number" min={2} max={10} value={phrasesN}
              onChange={e => setPhrasesN(Number(e.target.value))}
              className="w-14 bg-zinc-900 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1.5 text-center focus:border-violet-500 focus:outline-none" />
          </div>
          <button onClick={repreview} disabled={running}
            className="text-xs px-3 py-1.5 border border-zinc-600 text-zinc-400 rounded hover:border-violet-500/50 hover:text-violet-400 disabled:opacity-40 transition-colors">
            Refresh
          </button>
        </div>

        {/* Status */}
        {running && (
          <div className="flex items-center gap-2 text-xs text-violet-400">
            <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
            {applied ? 'Applying fixes…' : 'Scanning for collisions…'}
          </div>
        )}
        {error && <div className="text-xs text-red-400">{error}</div>}

        {/* Results */}
        {result && !running && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-xs text-zinc-400">
                {result.pairs.length === 0
                  ? 'No collisions found — your intents are well separated.'
                  : `${result.pairs.length} collision${result.pairs.length > 1 ? 's' : ''} found`}
              </div>
              {applied && result.phrases_added > 0 && (
                <span className="text-xs text-emerald-400">✓ {result.phrases_added} phrases added</span>
              )}
              {!applied && result.pairs.length > 0 && (
                <button onClick={apply} disabled={running}
                  className="text-xs px-4 py-1.5 bg-violet-600 hover:bg-violet-500 text-white rounded disabled:opacity-40 transition-colors">
                  Fix all ({result.pairs.length})
                </button>
              )}
            </div>

            <div className="space-y-2">
              {result.pairs.map((p, i) => (
                <div key={i} className={`rounded-lg p-3 text-xs space-y-1.5 border ${
                  applied ? 'bg-emerald-900/10 border-emerald-800/30' : 'bg-zinc-800/50 border-zinc-700/50'
                }`}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-zinc-300">{p.intent_a}</span>
                    <span className="text-zinc-600">↔</span>
                    <span className="font-mono text-zinc-300">{p.intent_b}</span>
                    <span className={`ml-auto font-mono text-[10px] px-1.5 py-0.5 rounded ${
                      p.overlap >= 30 ? 'text-red-400 bg-red-900/20' :
                      p.overlap >= 20 ? 'text-amber-400 bg-amber-900/20' :
                      'text-zinc-400 bg-zinc-800'
                    }`}>{p.overlap}% overlap</span>
                  </div>
                  {applied && (
                    <div className="text-zinc-500">
                      +{p.phrases_added_a} for {p.intent_a.split(':').pop()}
                      &nbsp;·&nbsp;
                      +{p.phrases_added_b} for {p.intent_b.split(':').pop()}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Page>
  );
}
