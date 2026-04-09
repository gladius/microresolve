import { useState, useEffect, useCallback } from 'react';
import { api, type ReviewItem, type ReviewAnalyzeResult, type AccuracyResult } from '@/api/client';

interface SeedEntry { seed: string; lang: string; }
interface IntentBlock { intentId: string; seeds: SeedEntry[]; }

export default function ReviewPage() {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [intents, setIntents] = useState<string[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [accuracy, setAccuracy] = useState<AccuracyResult | null>(null);
  const [checkingAccuracy, setCheckingAccuracy] = useState(false);
  const [total, setTotal] = useState(0);

  const refresh = useCallback(async () => {
    try {
      const [q, i] = await Promise.all([
        api.getReviewQueue(undefined, 200),
        api.listIntents().then(list => list.map(i => i.id)),
      ]);
      setItems(q.items);
      setTotal(q.total);
      setIntents(i);
    } catch { /* */ }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const checkAccuracy = async () => {
    setCheckingAccuracy(true);
    try { setAccuracy(await api.checkAccuracy()); } catch { /* */ }
    setCheckingAccuracy(false);
  };

  const selected = items.find(i => i.id === selectedId) || null;

  return (
    <div className="flex gap-0 h-[calc(100vh-6rem)] -mx-4">
      {/* Sidebar */}
      <div className="w-72 min-w-[18rem] border-r border-zinc-800 flex flex-col">
        <div className="px-3 py-3 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
          <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Failures ({total})</span>
          <button onClick={refresh} className="text-[10px] text-zinc-500 hover:text-white">Refresh</button>
        </div>

        <div className="px-3 py-2 border-b border-zinc-800">
          <button onClick={checkAccuracy} disabled={checkingAccuracy}
            className="w-full text-xs px-2 py-1.5 border border-zinc-700 text-zinc-400 rounded hover:text-white hover:border-violet-500 disabled:opacity-50">
            {checkingAccuracy ? 'Checking...' : 'Check Accuracy'}
          </button>
          {accuracy && (
            <div className="mt-2 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-emerald-400">High</span>
                <span className="text-white">{accuracy.high} ({accuracy.high_pct.toFixed(1)}%)</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-amber-400">Medium</span>
                <span className="text-white">{accuracy.medium} ({accuracy.medium_pct.toFixed(1)}%)</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-red-400">Low/Miss</span>
                <span className="text-white">{accuracy.low + accuracy.miss}</span>
              </div>
              <div className="flex justify-between text-xs pt-1 border-t border-zinc-800">
                <span className="text-zinc-400 font-semibold">Pass rate</span>
                <span className={`font-semibold ${accuracy.pass_pct >= 80 ? 'text-emerald-400' : accuracy.pass_pct >= 60 ? 'text-amber-400' : 'text-red-400'}`}>
                  {accuracy.pass_pct.toFixed(1)}%
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto">
          {items.map(item => (
            <div key={item.id} onClick={() => setSelectedId(item.id)}
              className={`px-3 py-2 cursor-pointer border-b border-zinc-800/50 transition-colors ${selectedId === item.id ? 'bg-zinc-800/80' : 'hover:bg-zinc-800/40'}`}>
              <div className="flex items-center gap-2">
                <span className={`text-[9px] font-bold uppercase ${item.flag === 'miss' ? 'text-red-400' : item.flag === 'low_confidence' ? 'text-amber-400' : 'text-blue-400'}`}>
                  {item.flag === 'low_confidence' ? 'LOW' : item.flag}
                </span>
                <span className="text-xs text-zinc-400 truncate flex-1">{item.query.slice(0, 40)}</span>
              </div>
            </div>
          ))}
          {items.length === 0 && <div className="text-zinc-600 text-xs text-center py-8 px-4">No failures to review</div>}
        </div>
      </div>

      {/* Detail */}
      <div className="flex-1 overflow-y-auto">
        {selected ? (
          <FailureDetail item={selected} intents={intents} onAction={() => { setSelectedId(null); refresh(); }} />
        ) : (
          <div className="flex items-center justify-center h-full text-zinc-600 text-sm">
            {items.length > 0 ? 'Select a failure to fix' : 'No failures'}
          </div>
        )}
      </div>
    </div>
  );
}

function FailureDetail({ item, intents, onAction }: { item: ReviewItem; intents: string[]; onAction: () => void }) {
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<ReviewAnalyzeResult | null>(null);
  const [wrongSeeds, setWrongSeeds] = useState<Record<string, string[]>>({});
  const [enabledLangs] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem('asv_languages') || '["en"]'); } catch { return ['en']; }
  });

  // Editable seeds (from analysis)
  const [blocks, setBlocks] = useState<IntentBlock[]>([]);

  useEffect(() => { setAnalysis(null); setBlocks([]); setWrongSeeds({}); }, [item.id]);

  const runAnalysis = async () => {
    setAnalyzing(true);
    try {
      const result = await api.reviewAnalyze(item.id);
      setAnalysis(result);

      // Build editable seed blocks from analysis
      const newBlocks: IntentBlock[] = Object.entries(result.seeds_to_add).map(([intentId, seeds]) => ({
        intentId,
        seeds: seeds.map(s => ({ seed: s, lang: result.languages[0] || 'en' })),
      }));
      setBlocks(newBlocks);

      // Fetch seeds for wrong intents
      if (result.wrong_detections.length > 0) {
        const seeds = await api.reviewIntentSeeds(result.wrong_detections);
        setWrongSeeds(seeds);
      }
    } catch (e) {
      alert('Analysis failed: ' + (e instanceof Error ? e.message : 'unknown'));
    } finally {
      setAnalyzing(false);
    }
  };

  // Block editing
  const addBlock = () => setBlocks(prev => [...prev, { intentId: '', seeds: [{ seed: '', lang: 'en' }] }]);
  const removeBlock = (i: number) => setBlocks(prev => prev.filter((_, idx) => idx !== i));
  const setBlockIntent = (i: number, id: string) => setBlocks(prev => prev.map((b, idx) => idx === i ? { ...b, intentId: id } : b));
  const setBlockSeed = (bi: number, si: number, val: string) => setBlocks(prev => prev.map((b, i) => {
    if (i !== bi) return b;
    const seeds = [...b.seeds]; seeds[si] = { ...seeds[si], seed: val }; return { ...b, seeds };
  }));
  const addSeedToBlock = (bi: number) => setBlocks(prev => prev.map((b, i) => i === bi ? { ...b, seeds: [...b.seeds, { seed: '', lang: 'en' }] } : b));
  const removeSeedFromBlock = (bi: number, si: number) => setBlocks(prev => prev.map((b, i) => i === bi ? { ...b, seeds: b.seeds.filter((_, idx) => idx !== si) } : b));

  const handleApply = async () => {
    const toApply: Record<string, { seed: string; lang: string }[]> = {};
    for (const block of blocks) {
      if (!block.intentId) continue;
      const seeds = block.seeds.filter(s => s.seed.trim());
      if (seeds.length > 0) toApply[block.intentId] = [...(toApply[block.intentId] || []), ...seeds];
    }
    if (Object.keys(toApply).length === 0 && (!analysis || analysis.seeds_to_replace.length === 0)) return;
    const result = await api.reviewFix(item.id, toApply);
    const msgs = [`Applied ${result.added} seeds.`];
    if (result.resolved_count > 0) msgs.push(`${result.resolved_count} failures resolved.`);
    if (result.blocked.length > 0) {
      msgs.push(`Blocked ${result.blocked.length}:`);
      result.blocked.forEach(b => msgs.push(`  "${b.seed}": ${b.reason}`));
    }
    alert(msgs.join('\n'));
    onAction();
  };

  const handleDismiss = async () => { await api.reviewReject(item.id); onAction(); };

  const totalSeeds = blocks.flatMap(b => b.seeds).filter(s => s.seed.trim()).length;
  const usedIntents = new Set(blocks.map(b => b.intentId).filter(Boolean));

  return (
    <div className="p-5 space-y-4 max-w-3xl">
      {/* Header */}
      <div className="flex items-center gap-2">
        <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded ${
          item.flag === 'miss' ? 'bg-red-900/30 text-red-400' :
          item.flag === 'low_confidence' ? 'bg-amber-900/30 text-amber-400' : 'bg-blue-900/30 text-blue-400'
        }`}>{item.flag.replace('_', ' ')}</span>
        {item.detected.length > 0 && <span className="text-xs text-zinc-500">Detected: {item.detected.join(', ')}</span>}
      </div>

      {/* Query */}
      <div className="bg-zinc-800 rounded-lg p-4">
        <div className="text-[10px] text-zinc-500 mb-1">Customer query</div>
        <div className="text-white font-mono text-sm">"{item.query}"</div>
      </div>

      {/* Analysis result */}
      {analysis && (
        <>
          {/* Detected vs Correct */}
          <div className="bg-zinc-800/50 rounded-lg p-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-[10px] text-zinc-500 mb-2">Detected (by ASV)</div>
                <div className="flex flex-wrap gap-1.5">
                  {item.detected.map(id => {
                    const isWrong = analysis.wrong_detections.includes(id);
                    return (
                      <span key={id} className={`text-xs font-mono px-2 py-0.5 rounded border ${
                        isWrong ? 'text-red-400 bg-red-900/20 border-red-800 line-through' : 'text-emerald-400 bg-emerald-900/20 border-emerald-800'
                      }`}>{id} {isWrong ? '✗' : '✓'}</span>
                    );
                  })}
                  {item.detected.length === 0 && <span className="text-xs text-red-400">none</span>}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-zinc-500 mb-2">Correct (by AI)</div>
                <div className="flex flex-wrap gap-1.5">
                  {analysis.correct_intents.map(id => (
                    <span key={id} className="text-xs font-mono text-emerald-400 bg-emerald-900/20 border border-emerald-800 px-2 py-0.5 rounded">{id}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* False positive fixes (replacements) */}
          {analysis.seeds_to_replace.length > 0 && (
            <div className="space-y-2">
              <div className="text-[10px] text-zinc-500 uppercase font-semibold">False Positive Fixes</div>
              {analysis.seeds_to_replace.map((r, i) => (
                <div key={i} className="bg-red-900/10 border border-red-800/30 rounded-lg px-3 py-3 text-xs space-y-1">
                  <div><span className="text-red-400 font-mono font-semibold">{r.intent}</span> <span className="text-zinc-500">— false match</span></div>
                  <div className="text-zinc-500">
                    Replace: <span className="text-red-400 font-mono line-through">"{r.old_seed}"</span>
                    {' → '}<span className="text-emerald-400 font-mono">"{r.new_seed}"</span>
                  </div>
                  <div className="text-zinc-600">{r.reason}</div>
                  {wrongSeeds[r.intent] && (
                    <div className="mt-1">
                      <div className="text-[10px] text-zinc-600 mb-1">Current seeds:</div>
                      <div className="flex flex-wrap gap-1">
                        {wrongSeeds[r.intent].map((s, si) => (
                          <span key={si} className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                            s === r.old_seed ? 'bg-red-900/30 text-red-400 border border-red-800' : 'bg-zinc-800 text-zinc-500'
                          }`}>{s}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Blocked seeds */}
          {analysis.seeds_blocked.length > 0 && (
            <div className="text-xs text-amber-400 bg-amber-900/10 border border-amber-800/30 rounded px-3 py-2">
              Guard blocked: {analysis.seeds_blocked.map(b => `"${b.seed}" (${b.reason})`).join(', ')}
            </div>
          )}

          {/* Summary */}
          {analysis.summary && (
            <div className="text-xs text-zinc-400 bg-zinc-800/50 rounded px-3 py-2">{analysis.summary}</div>
          )}
        </>
      )}

      {/* Analyzing spinner */}
      {analyzing && (
        <div className="text-xs text-violet-400 flex items-center gap-2">
          <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
          Analyzing (3-turn review)...
        </div>
      )}

      {/* Seeds to add */}
      <div className="space-y-2">
        {blocks.length > 0 && <div className="text-[10px] text-zinc-500 uppercase font-semibold">Seeds to add</div>}
        {blocks.map((block, bi) => (
          <div key={bi} className="bg-zinc-800 border border-zinc-700 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <select value={block.intentId} onChange={e => setBlockIntent(bi, e.target.value)}
                className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1 font-mono focus:border-violet-500 focus:outline-none flex-1">
                <option value="">Select intent...</option>
                {intents.filter(id => !usedIntents.has(id) || id === block.intentId).map(id => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
              <button onClick={() => removeBlock(bi)} className="text-xs text-zinc-600 hover:text-red-400">Remove</button>
            </div>
            {block.seeds.map((entry, si) => (
              <div key={si} className="flex gap-1.5 mb-1">
                <select value={entry.lang} onChange={e => setBlocks(prev => prev.map((b, i) => {
                  if (i !== bi) return b;
                  const seeds = [...b.seeds]; seeds[si] = { ...seeds[si], lang: e.target.value }; return { ...b, seeds };
                }))} className="bg-zinc-900 border border-zinc-700 text-violet-400 text-[10px] rounded px-1 py-1 w-12 focus:outline-none">
                  {enabledLangs.map(lang => <option key={lang} value={lang}>{lang.toUpperCase()}</option>)}
                </select>
                <input value={entry.seed} onChange={e => setBlockSeed(bi, si, e.target.value)}
                  placeholder="e.g. received wrong item"
                  className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-white font-mono focus:border-violet-500 focus:outline-none" />
                {block.seeds.length > 1 && <button onClick={() => removeSeedFromBlock(bi, si)} className="text-zinc-600 hover:text-red-400 text-xs">×</button>}
              </div>
            ))}
            <button onClick={() => addSeedToBlock(bi)} className="text-[10px] text-zinc-500 hover:text-violet-400 mt-1">+ add seed</button>
          </div>
        ))}
        <button onClick={addBlock}
          className="w-full py-2 text-xs text-zinc-500 hover:text-violet-400 border border-dashed border-zinc-700 hover:border-violet-500 rounded-lg transition-colors">
          + Add intent block
        </button>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3 pt-3 border-t border-zinc-800">
        <button onClick={runAnalysis} disabled={analyzing}
          className="text-xs px-3 py-1.5 border border-violet-500 text-violet-400 rounded hover:bg-violet-500 hover:text-white disabled:opacity-50">
          {analyzing ? 'Analyzing...' : analysis ? 'Re-analyze' : 'Analyze with AI'}
        </button>
        <div className="flex-1" />
        <button onClick={handleDismiss} className="text-xs px-3 py-1.5 border border-zinc-700 text-zinc-400 rounded hover:text-white">Dismiss</button>
        <button onClick={handleApply} disabled={totalSeeds === 0}
          className="text-xs px-4 py-1.5 bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30">
          Apply Fix ({totalSeeds} seeds)
        </button>
      </div>
    </div>
  );
}
