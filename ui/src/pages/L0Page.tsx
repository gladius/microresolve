import { useEffect, useState } from 'react';
import { useAppStore } from '@/store';
import { api } from '@/api/client';
import Page from '@/components/Page';
import LayerToggle from '@/components/LayerToggle';

type L0Info = {
  namespace: string;
  vocab_size: number;
  ngram_size: number;
  min_term_len: number;
  vocab_sample: string[];
};

export default function L0Page() {
  const { settings } = useAppStore();
  const ns = settings.selectedNamespaceId;
  const [info, setInfo] = useState<L0Info | null>(null);
  const [err,  setErr]  = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [query,  setQuery]  = useState('');
  const [result, setResult] = useState<{ query: string; corrected: string; changed: boolean } | null>(null);
  const [busy,   setBusy]   = useState(false);

  const load = async () => {
    try {
      const d = await api.getL0Info();
      setInfo(d);
      setErr(null);
    } catch (e) { setErr(String(e)); }
  };

  useEffect(() => { load(); /* eslint-disable-next-line */ }, [ns]);

  const test = async () => {
    if (!query.trim()) return;
    setBusy(true);
    try {
      const r = await api.l0Correct(query);
      setResult(r);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const visibleVocab = info && filter
    ? info.vocab_sample.filter(w => w.toLowerCase().includes(filter.toLowerCase()))
    : info?.vocab_sample ?? [];

  return (
    <Page
      title="L0 — Spelling"
      subtitle={<>character n-gram typo correction for <span className="text-emerald-400 font-mono">{ns}</span></>}
      size="md"
    >
      <div className="space-y-6">
        <div className="flex items-center justify-between bg-zinc-900/60 border border-zinc-800 rounded-xl px-4 py-3">
          <div className="text-xs text-zinc-500">
            Status for <span className="text-emerald-400 font-mono">{ns}</span>
          </div>
          <LayerToggle field="l0_enabled" label="L0 — Spelling" />
        </div>

        <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 text-xs text-zinc-500 leading-relaxed space-y-1">
          <div className="text-zinc-300 font-medium text-sm">What this layer does</div>
          <p>
            Before L1 normalization, each query token is checked against the namespace's known vocabulary.
            Unknown tokens are corrected to the closest term by Jaccard similarity of character trigrams,
            gated by Damerau–Levenshtein edit distance.
          </p>
          <p className="text-zinc-600">
            No training, no model — purely algorithmic. Vocabulary is rebuilt from L1 + L2 whenever phrases change.
            CJK tokens pass through untouched.
          </p>
        </div>

        {err && (
          <div className="bg-red-950/40 border border-red-900 rounded-lg px-3 py-2 text-xs text-red-300">
            {err}
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3">
          <Stat label="Vocab terms" value={info ? info.vocab_size.toLocaleString() : '…'} />
          <Stat label="N-gram size" value={info ? `${info.ngram_size}-gram` : '…'} />
          <Stat label="Min term length" value={info ? `${info.min_term_len} chars` : '…'} />
        </div>

        {/* Live correction tester */}
        <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 space-y-3">
          <div className="text-xs text-zinc-400 font-semibold uppercase tracking-wide">Try it</div>
          {info && info.vocab_size === 0 ? (
            <div className="text-xs text-amber-500/80 bg-amber-950/30 border border-amber-900/50 rounded px-3 py-2">
              This namespace has no vocabulary yet. L0 has nothing to correct against — every token will pass through unchanged.
              Add intents on the <a href="/l2" className="underline hover:text-amber-300">L2 — Intents</a> page first, or switch
              to a populated namespace from the sidebar.
            </div>
          ) : (
            <>
              <div className="flex gap-2">
                <input
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && test()}
                  placeholder="Type a misspelling, e.g. 'cahnge passowrd'"
                  className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500"
                />
                <button
                  onClick={test}
                  disabled={busy || !query.trim()}
                  className="px-4 py-1.5 text-sm bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded transition-colors"
                >
                  Correct
                </button>
              </div>
              {result && (
                <div className="text-xs space-y-1.5 pt-1 border-t border-zinc-800">
                  <Row label="Input"     value={result.query}     mono />
                  <Row label="Corrected" value={result.corrected} mono highlight={result.changed} />
                  <div className="text-[10px] text-zinc-500">
                    {result.changed
                      ? 'L0 rewrote one or more tokens before they reached L1/L2.'
                      : 'No correction applied — every token was already in vocabulary, too short (< 4 chars), or below the similarity threshold.'}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Vocab sample */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-xs text-zinc-400 font-semibold uppercase tracking-wide">Vocabulary</div>
            <span className="text-[10px] text-zinc-600">
              {info ? `showing ${Math.min(info.vocab_sample.length, info.vocab_size)} of ${info.vocab_size.toLocaleString()}` : ''}
            </span>
          </div>
          <input
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Filter terms…"
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500"
          />
          <div className="bg-zinc-900/40 border border-zinc-800 rounded-lg p-3 max-h-72 overflow-y-auto">
            {visibleVocab.length === 0 ? (
              <div className="text-xs text-zinc-600 text-center py-4">
                {info && info.vocab_size === 0
                  ? 'No vocabulary yet — add intents and phrases first.'
                  : 'No matches.'}
              </div>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {visibleVocab.map(w => (
                  <span key={w} className="px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-[11px] text-zinc-300 font-mono">
                    {w}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </Page>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg px-3 py-2.5">
      <div className="text-[10px] text-zinc-500 uppercase tracking-wide">{label}</div>
      <div className="text-lg text-zinc-100 font-mono mt-0.5">{value}</div>
    </div>
  );
}

function Row({ label, value, mono, highlight }: {
  label: string; value: string; mono?: boolean; highlight?: boolean;
}) {
  return (
    <div className="flex gap-2">
      <span className="text-zinc-500 w-20 shrink-0">{label}</span>
      <span className={`${mono ? 'font-mono' : ''} ${highlight ? 'text-emerald-400' : 'text-zinc-200'}`}>{value || <span className="text-zinc-600">(empty)</span>}</span>
    </div>
  );
}
