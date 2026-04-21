import { useState, useEffect, useCallback } from 'react';
import { api, type ReviewItem, type ReviewAnalyzeResult } from '@/api/client';
import { useAppStore } from '@/store';
import Page from '@/components/Page';

type FeedEvent =
  | { type: 'item_queued';  id: number; query: string; app_id: string; flag: string | null }
  | { type: 'llm_started';  id: number; query: string }
  | { type: 'llm_done';     id: number; correct: string[]; wrong: string[]; phrases_added: number; summary: string }
  | { type: 'fix_applied';  id: number; phrases_added: number; phrases_replaced: number; version_before: number; version_after: number }
  | { type: 'escalated';    id: number; reason: string };

export default function ReviewPage() {
  const [items,        setItems]        = useState<ReviewItem[]>([]);
  const [selected,     setSelected]     = useState<ReviewItem | null>(null);
  const [intents,      setIntents]      = useState<string[]>([]);
  const [autoLearn,    setAutoLearn]    = useState<boolean | null>(null);
  const [toggling,     setToggling]     = useState(false);
  const [toggleError,  setToggleError]  = useState<string | null>(null);
  const [stats,        setStats]        = useState<{ pending: number; total: number } | null>(null);
  const [loading,      setLoading]      = useState(true);

  const { settings } = useAppStore();
  const nsId = settings.selectedNamespaceId;

  const refresh = useCallback(async () => {
    try {
      const [q, i, nsList, s] = await Promise.all([
        api.getReviewQueue(undefined, 100),
        api.listIntents().then(list => list.map((x: any) => x.id)),
        api.listNamespaces(),
        api.getReviewStats(),
      ]);
      setItems(q.items);
      setIntents(i);
      setStats(s);
      const ns = nsList.find((n: any) => n.id === nsId);
      setAutoLearn(ns?.auto_learn ?? false);
      setSelected(prev => {
        if (!prev && q.items.length > 0) return q.items[0];
        if (prev) return q.items.find((x: ReviewItem) => x.id === prev.id) ?? (q.items[0] ?? null);
        return null;
      });
    } catch { /* */ } finally { setLoading(false); }
  }, [nsId]);

  useEffect(() => { refresh(); }, [refresh]);

  // SSE — refresh queue on worker events
  useEffect(() => {
    const es = new EventSource('/api/events');
    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data) as FeedEvent;
        if (event.type === 'llm_done' || event.type === 'fix_applied' || event.type === 'escalated') {
          setTimeout(refresh, 600);
        }
      } catch { /* */ }
    };
    return () => es.close();
  }, [refresh]);

  const toggle = async () => {
    if (autoLearn === null || toggling) return;
    setToggling(true);
    setToggleError(null);
    try {
      const next = !autoLearn;
      await api.updateNamespace(nsId, { auto_learn: next });
      setAutoLearn(next);
    } catch {
      setToggleError('Save failed');
      setTimeout(() => setToggleError(null), 3000);
    } finally { setToggling(false); }
  };

  const onFixed = useCallback(() => {
    setSelected(null);
    refresh();
  }, [refresh]);

  const visible = items;

  return (
    <Page title="Review" subtitle="All routed queries — reviewed by LLM judge" fullscreen
      actions={
        <div className="flex items-center gap-3">
          {stats && stats.pending > 0 && (
            <span className="text-xs text-amber-400 font-mono">{stats.pending} pending</span>
          )}
          <button onClick={toggle} disabled={toggling || autoLearn === null}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
              autoLearn
                ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/25'
                : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500'
            } ${toggling ? 'opacity-50' : ''}`}
          >
            <span className={`w-2 h-2 rounded-full ${autoLearn ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
            Auto-learn {autoLearn === null ? '…' : autoLearn ? 'On' : 'Off'}
          </button>
          {toggleError && <span className="text-[10px] text-red-400">{toggleError}</span>}
        </div>
      }
    >
      <div className="flex h-full gap-0">

        {/* Left — queue list */}
        <div className="w-72 shrink-0 border-r border-zinc-800 flex flex-col">
          {/* Filter bar */}
          <div className="px-3 py-2.5 border-b border-zinc-800 flex items-center flex-shrink-0">
            <span className="text-[10px] text-zinc-500">{items.length} queued for review</span>
            <button onClick={refresh} className="ml-auto text-[10px] text-zinc-600 hover:text-zinc-400">↺</button>
          </div>

          {/* Queue items */}
          <div className="flex-1 overflow-y-auto divide-y divide-zinc-800/50">
            {visible.length === 0 ? (
              <div className="text-center py-12 px-4 text-zinc-600 text-xs leading-relaxed">
                {items.length === 0
                  ? 'Queue is empty.\nRouted queries appear here for LLM review.'
                  : 'No items match this filter.'}
              </div>
            ) : visible.map(item => (
              <button key={item.id} onClick={() => setSelected(item)}
                className={`w-full text-left px-3 py-3 hover:bg-zinc-800/50 transition-colors ${selected?.id === item.id ? 'bg-zinc-800/70' : ''}`}>
                <div className="flex items-center gap-2 mb-1">
                  {item.detected.length > 0 ? (
                    <span className="text-[10px] text-zinc-400 truncate">{item.detected.join(', ')}</span>
                  ) : (
                    <span className="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded flex-shrink-0 bg-zinc-800 text-zinc-500">no match</span>
                  )}
                  <span className="ml-auto text-[9px] text-zinc-700 flex-shrink-0">
                    {new Date(item.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="text-xs text-zinc-300 truncate font-mono">"{item.query}"</div>
              </button>
            ))}
          </div>
        </div>

        {/* Right — detail */}
        <div className="flex-1 min-w-0 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="w-4 h-4 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" />
            </div>
          ) : selected ? (
            <ReviewDetail item={selected} intents={intents} onFixed={onFixed} onDismiss={onFixed} />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center px-8 gap-3">
              <div className="text-zinc-600 text-3xl mb-1">✓</div>
              <div className="text-zinc-300 text-sm font-medium">Queue is clear</div>
              <div className="text-zinc-600 text-xs max-w-sm leading-relaxed">
                Every routed query is reviewed by the LLM judge. Enable <span className="text-emerald-400">Auto-learn</span> to apply corrections automatically,
                or review and fix them manually here.
              </div>
            </div>
          )}
        </div>
      </div>
    </Page>
  );
}

// ─── Review Detail ────────────────────────────────────────────────────────────

interface PhraseEntry { phrase: string; lang: string; }
interface IntentBlock { intentId: string; phrases: PhraseEntry[]; }

function CopyAnalysisButton({ item, analysis }: { item: ReviewItem; analysis: ReviewAnalyzeResult }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    const lines = [
      `Query: "${item.query}"`, `Flag: ${item.flag}`,
      `Detected: ${item.detected.length > 0 ? item.detected.join(', ') : 'none'}`,
      `Wrong: ${analysis.wrong_detections.length > 0 ? analysis.wrong_detections.join(', ') : 'none'}`,
      `Should be: ${analysis.correct_intents.length > 0 ? analysis.correct_intents.join(', ') : 'no matching intent'}`,
    ];
    if (analysis.summary) lines.push('', `Summary: ${analysis.summary}`);
    navigator.clipboard.writeText(lines.join('\n'));
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button onClick={copy} className="text-[10px] text-zinc-500 hover:text-violet-400 transition-colors px-2 py-0.5 border border-zinc-800 rounded hover:border-violet-500/40">
      {copied ? '✓ copied' : 'copy report'}
    </button>
  );
}

function ReviewDetail({ item, intents, onFixed, onDismiss }: {
  item: ReviewItem; intents: string[]; onFixed: () => void; onDismiss: () => void;
}) {
  const [analyzing,   setAnalyzing]   = useState(false);
  const [analysis,    setAnalysis]    = useState<ReviewAnalyzeResult | null>(null);
  const [blocks,      setBlocks]      = useState<IntentBlock[]>([]);
  const [applyResult, setApplyResult] = useState<{ added: number; resolved: number } | null>(null);
  const [applyError,  setApplyError]  = useState<string | null>(null);
  const { settings } = useAppStore();
  const enabledLangs = settings.languages.length > 0 ? settings.languages : ['en'];

  useEffect(() => { setAnalysis(null); setBlocks([]); }, [item.id]);

  const runAnalysis = async () => {
    setAnalyzing(true);
    setApplyError(null);
    try {
      const result = await api.reviewAnalyze(item.id);
      setAnalysis(result);
      setBlocks(Object.entries(result.phrases_to_add).map(([intentId, phraseList]) => ({
        intentId,
        phrases: (phraseList as string[]).map(s => ({ phrase: s, lang: result.languages[0] || 'en' })),
      })));
    } catch (e) {
      setApplyError('Analysis failed: ' + (e instanceof Error ? e.message : String(e)));
      setTimeout(() => setApplyError(null), 5000);
    } finally { setAnalyzing(false); }
  };

  const setBlockIntent = (i: number, id: string) =>
    setBlocks(prev => prev.map((b, idx) => idx === i ? { ...b, intentId: id } : b));
  const setBlockPhrase = (bi: number, si: number, val: string) =>
    setBlocks(prev => prev.map((b, i) => {
      if (i !== bi) return b;
      const p = [...b.phrases]; p[si] = { ...p[si], phrase: val }; return { ...b, phrases: p };
    }));
  const addPhraseToBlock      = (bi: number) =>
    setBlocks(prev => prev.map((b, i) => i === bi ? { ...b, phrases: [...b.phrases, { phrase: '', lang: 'en' }] } : b));
  const removePhraseFromBlock = (bi: number, si: number) =>
    setBlocks(prev => prev.map((b, i) => i === bi ? { ...b, phrases: b.phrases.filter((_, idx) => idx !== si) } : b));
  const removeBlock = (i: number) => setBlocks(prev => prev.filter((_, idx) => idx !== i));

  const handleApply = async () => {
    const toApply: Record<string, { phrase: string; lang: string }[]> = {};
    for (const block of blocks) {
      if (!block.intentId) continue;
      const phrases = block.phrases.filter(s => s.phrase.trim());
      if (phrases.length > 0) toApply[block.intentId] = [...(toApply[block.intentId] || []), ...phrases];
    }
    if (Object.keys(toApply).length === 0) return;
    try {
      const result = await api.reviewFix(item.id, toApply);
      setApplyResult({ added: result.added, resolved: result.resolved_count ?? 0 });
      setTimeout(onFixed, 1200);
    } catch (e) {
      setApplyError(e instanceof Error ? e.message : 'Apply failed');
      setTimeout(() => setApplyError(null), 4000);
    }
  };

  const totalPhrases = blocks.flatMap(b => b.phrases).filter(s => s.phrase.trim()).length;
  const usedIntents  = new Set(blocks.map(b => b.intentId).filter(Boolean));

  return (
    <div className="p-5 space-y-4 max-w-2xl">
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-zinc-800 text-zinc-400">
          {item.detected.length === 0 ? 'no match' : 'review'}
        </span>
        {item.detected.length > 0 && <span className="text-xs text-zinc-500">detected: {item.detected.join(', ')}</span>}
      </div>

      <div className="bg-zinc-800 rounded-lg p-3">
        <div className="text-[10px] text-zinc-500 mb-1">Query</div>
        <div className="text-zinc-100 font-mono text-sm">"{item.query}"</div>
      </div>

      {analysis && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wide">AI Analysis</div>
            <CopyAnalysisButton item={item} analysis={analysis} />
          </div>
          <div className="bg-zinc-800/50 rounded-lg p-3 grid grid-cols-2 gap-3">
            <div>
              <div className="text-[10px] text-zinc-500 mb-1.5">Detected</div>
              <div className="flex flex-wrap gap-1">
                {item.detected.map(id => (
                  <span key={id} className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${
                    analysis.wrong_detections.includes(id)
                      ? 'text-red-400 bg-red-900/20 border-red-800 line-through'
                      : 'text-emerald-400 bg-emerald-900/20 border-emerald-800'
                  }`}>{id}</span>
                ))}
                {item.detected.length === 0 && <span className="text-[10px] text-red-400">none</span>}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-zinc-500 mb-1.5">Should be</div>
              <div className="flex flex-wrap gap-1">
                {analysis.correct_intents.length > 0
                  ? analysis.correct_intents.map(id => (
                      <span key={id} className="text-[10px] font-mono text-emerald-400 bg-emerald-900/20 border border-emerald-800 px-1.5 py-0.5 rounded">{id}</span>
                    ))
                  : <span className="text-[10px] text-zinc-600 italic">no matching intent</span>}
              </div>
            </div>
          </div>
          {analysis.summary && (
            <div className="text-xs text-zinc-400 bg-zinc-800/50 rounded px-3 py-2">{analysis.summary}</div>
          )}
        </div>
      )}

      {analyzing && (
        <div className="flex items-center gap-2 text-xs text-violet-400">
          <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
          Analyzing...
        </div>
      )}

      <div className="space-y-2">
        {blocks.length > 0 && <div className="text-[10px] text-zinc-500 uppercase font-semibold">Phrases to add</div>}
        {blocks.map((block, bi) => (
          <div key={bi} className="bg-zinc-800 border border-zinc-700 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <select value={block.intentId} onChange={e => setBlockIntent(bi, e.target.value)}
                className="flex-1 bg-zinc-900 border border-zinc-700 text-zinc-100 text-xs rounded px-2 py-1 font-mono focus:border-violet-500 focus:outline-none">
                <option value="">Select intent...</option>
                {intents.filter(id => !usedIntents.has(id) || id === block.intentId).map(id => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
              <button onClick={() => removeBlock(bi)} className="text-[10px] text-zinc-600 hover:text-red-400">remove</button>
            </div>
            {block.phrases.map((entry, si) => (
              <div key={si} className="flex gap-1.5 mb-1">
                <select value={entry.lang}
                  onChange={e => setBlocks(prev => prev.map((b, i) => {
                    if (i !== bi) return b;
                    const p = [...b.phrases]; p[si] = { ...p[si], lang: e.target.value }; return { ...b, phrases: p };
                  }))}
                  className="bg-zinc-900 border border-zinc-700 text-violet-400 text-[10px] rounded px-1 py-1 w-12 focus:outline-none">
                  {enabledLangs.map(lang => <option key={lang} value={lang}>{lang.toUpperCase()}</option>)}
                </select>
                <input value={entry.phrase} onChange={e => setBlockPhrase(bi, si, e.target.value)}
                  placeholder="example phrase"
                  className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 font-mono focus:border-violet-500 focus:outline-none" />
                {block.phrases.length > 1 && (
                  <button onClick={() => removePhraseFromBlock(bi, si)} className="text-zinc-600 hover:text-red-400 text-xs">×</button>
                )}
              </div>
            ))}
            <button onClick={() => addPhraseToBlock(bi)} className="text-[9px] text-zinc-500 hover:text-violet-400 mt-1">+ phrase</button>
          </div>
        ))}
        <button
          onClick={() => setBlocks(prev => [...prev, { intentId: '', phrases: [{ phrase: '', lang: 'en' }] }])}
          className="w-full py-2 text-xs text-zinc-600 hover:text-violet-400 border border-dashed border-zinc-800 hover:border-violet-500/40 rounded-lg transition-colors">
          + Add intent block
        </button>
      </div>

      <div className="flex items-center gap-2 pt-2 border-t border-zinc-800">
        <button onClick={runAnalysis} disabled={analyzing}
          className="text-xs px-3 py-1.5 border border-violet-500/50 text-violet-400 rounded hover:bg-violet-500/10 disabled:opacity-40 transition-colors">
          {analyzing ? 'Analyzing...' : analysis ? 'Re-analyze' : 'Analyze with AI'}
        </button>
        <div className="flex-1">
          {applyResult && (
            <span className="text-xs text-emerald-400">
              ✓ {applyResult.added} phrases added{applyResult.resolved > 0 ? `, ${applyResult.resolved} resolved` : ''}
            </span>
          )}
          {applyError && <span className="text-xs text-red-400">{applyError}</span>}
        </div>
        <button onClick={async () => {
            if (!confirm('Dismiss this item? It will be removed from the queue without training.')) return;
            await api.reviewReject(item.id); onDismiss();
          }}
          className="text-xs px-3 py-1.5 border border-zinc-700 text-zinc-500 rounded hover:text-zinc-100 transition-colors">
          Dismiss
        </button>
        <button onClick={handleApply} disabled={totalPhrases === 0 || !!applyResult}
          className="text-xs px-4 py-1.5 bg-violet-600 hover:bg-violet-500 text-white rounded disabled:opacity-30 transition-colors">
          Apply ({totalPhrases})
        </button>
      </div>
    </div>
  );
}
