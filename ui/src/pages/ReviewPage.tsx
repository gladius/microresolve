import { useState, useEffect, useCallback } from 'react';
import { api, type ReviewItem, type ReviewStats } from '@/api/client';

interface SeedEntry {
  seed: string;
  lang: string;
}

interface IntentBlock {
  intentId: string;
  seeds: SeedEntry[];
}

export default function ReviewPage() {
  const [section, setSection] = useState('pending');
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [intents, setIntents] = useState<string[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [q, s, i] = await Promise.all([
        api.getReviewQueue(section === 'all' ? undefined : section),
        api.getReviewStats(),
        api.listIntents().then(list => list.map(i => i.id)),
      ]);
      setItems(q.items);
      setStats(s);
      setIntents(i);
    } catch { /* */ }
  }, [section]);

  useEffect(() => { refresh(); }, [refresh]);

  const selected = items.find(i => i.id === selectedId) || null;

  const flagColor = (flag: string) => ({
    miss: 'text-red-400',
    low_confidence: 'text-amber-400',
    ambiguous: 'text-blue-400',
  }[flag] || 'text-zinc-400');

  return (
    <div className="flex gap-0 h-[calc(100vh-6rem)] -mx-4">
      {/* Sidebar */}
      <div className="w-72 min-w-[18rem] border-r border-zinc-800 flex flex-col">
        <div className="px-3 py-3 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
          <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Review</span>
          <button onClick={refresh} className="text-[10px] text-zinc-500 hover:text-white">Refresh</button>
        </div>

        <div className="flex flex-wrap gap-1 px-3 py-2 border-b border-zinc-800">
          {[
            { id: 'pending', label: 'Pending', count: stats?.pending },
            { id: 'fixed', label: 'Fixed', count: stats?.fixed },
            { id: 'approved', label: 'Approved', count: stats?.approved },
            { id: 'rejected', label: 'Rejected', count: stats?.rejected },
            { id: 'auto_resolved', label: 'Auto-resolved', count: stats?.auto_resolved },
            { id: 'auto_applied', label: 'Auto-learn', count: stats?.auto_applied },
            { id: 'all', label: 'All', count: stats?.total },
          ].map(f => (
            <button
              key={f.id}
              onClick={() => { setSection(f.id); setSelectedId(null); }}
              className={`text-[10px] px-2 py-0.5 rounded transition-colors ${
                section === f.id ? 'bg-violet-500/20 text-violet-400' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              {f.label}{f.count ? ` (${f.count})` : ''}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto">
          {items.map(item => (
            <div
              key={item.id}
              onClick={() => setSelectedId(item.id)}
              className={`px-3 py-2 cursor-pointer border-b border-zinc-800/50 transition-colors ${
                selectedId === item.id ? 'bg-zinc-800/80' : 'hover:bg-zinc-800/40'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className={`text-[9px] font-bold uppercase ${flagColor(item.flag)}`}>
                  {item.flag === 'low_confidence' ? 'LOW' : item.flag}
                </span>
                <span className="text-xs text-zinc-400 truncate flex-1">
                  {item.query.slice(0, 40)}{item.query.length > 40 ? '...' : ''}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-[10px] text-zinc-600">{item.session_id || ''}</span>
                <span className={`text-[10px] ml-auto ${
                  item.status === 'pending' ? 'text-amber-400/60' :
                  item.status === 'fixed' ? 'text-cyan-400/60' : 'text-zinc-600'
                }`}>{item.status}</span>
              </div>
            </div>
          ))}
          {items.length === 0 && (
            <div className="text-zinc-600 text-xs text-center py-8 px-4">No {section} items</div>
          )}
        </div>
      </div>

      {/* Detail */}
      <div className="flex-1 overflow-y-auto">
        {selected ? (
          <ReviewDetail item={selected} intents={intents} onAction={() => { setSelectedId(null); refresh(); }} />
        ) : (
          <div className="flex items-center justify-center h-full text-zinc-600 text-sm">Select a review item</div>
        )}
      </div>
    </div>
  );
}

// --- Detail ---

function ReviewDetail({ item, intents, onAction }: {
  item: ReviewItem; intents: string[]; onAction: () => void;
}) {
  const [enabledLangs, setEnabledLangs] = useState<string[]>(['en']);
  const [blocks, setBlocks] = useState<IntentBlock[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);

  // Turn 1 results
  const [correctIntents, setCorrectIntents] = useState<string[]>([]);
  const [wrongDetections, setWrongDetections] = useState<string[]>([]);
  const [detectedLanguages, setDetectedLanguages] = useState<string[]>(['en']);

  // Turn 3 results
  const [wrongIntentSeeds, setWrongIntentSeeds] = useState<Record<string, string[]>>({});
  const [wrongAnalysis, setWrongAnalysis] = useState<{ intent: string; problem_seed: string; reason: string; fix: string }[]>([]);
  const [addRisks, setAddRisks] = useState<{ intent: string; seed: string; risk: string }[]>([]);
  const [safeToApply, setSafeToApply] = useState<boolean | null>(null);
  const [summary, setSummary] = useState('');

  useEffect(() => {
    try {
      const saved = localStorage.getItem('asv_languages');
      if (saved) setEnabledLangs(JSON.parse(saved));
    } catch { /* */ }
  }, []);

  useEffect(() => {
    setBlocks([]);
    setAnalyzed(false);
    setCorrectIntents([]);
    setWrongDetections([]);
    setWrongIntentSeeds({});
    setWrongAnalysis([]);
    setAddRisks([]);
    setSafeToApply(null);
    setSummary('');
  }, [item.id]);

  // Run all 3 turns
  const runAnalysis = async () => {
    setAnalyzing(true);
    setAnalyzed(false);
    try {
      // Turn 1
      const t1 = await api.reviewTurn1(item.id);
      const correct = t1.correct_intents || [];
      const wrong = item.detected.filter(d => !correct.includes(d));
      const langs = t1.languages || ['en'];
      setCorrectIntents(correct);
      setWrongDetections(wrong);
      setDetectedLanguages(langs);

      // Turn 2
      const t2 = await api.reviewTurn2(item.id, correct, langs);
      if (t2.seeds_by_intent) {
        const defaultLang = langs[0] || 'en';
        const newBlocks = Object.entries(t2.seeds_by_intent).map(([intentId, seeds]) => ({
          intentId,
          seeds: seeds.map(s => {
            // Check if seed has language prefix like "[fr] produit authentique"
            const langMatch = s.match(/^\[([a-z]{2})\]\s*(.+)$/);
            if (langMatch) {
              return { seed: langMatch[2], lang: langMatch[1] };
            }
            return { seed: s, lang: defaultLang };
          }),
        }));
        // Remove risky seeds that Turn 3 might flag (we'll check after)
        setBlocks(newBlocks);
      }

      // Fetch current seeds for wrong intents
      if (wrong.length > 0) {
        const seeds = await api.reviewIntentSeeds(wrong);
        setWrongIntentSeeds(seeds);
      }

      // Turn 3 (only if wrong detections)
      if (wrong.length > 0) {
        const seedsMap: Record<string, string[]> = {};
        if (t2.seeds_by_intent) {
          for (const [k, v] of Object.entries(t2.seeds_by_intent)) {
            seedsMap[k] = v;
          }
        }
        const t3 = await api.reviewTurn3(item.id, correct, wrong, seedsMap);
        setWrongAnalysis(t3.wrong_intent_analysis || []);
        setAddRisks(t3.add_risks || []);
        setSafeToApply(t3.safe_to_apply ?? null);
        setSummary(t3.summary || '');

        // Auto-remove seeds flagged as high risk
        if (t3.add_risks && t3.add_risks.length > 0) {
          const riskySeeds = new Set(t3.add_risks.map(r => r.seed));
          setBlocks(prev => prev.map(b => ({
            ...b,
            seeds: b.seeds.filter(s => !riskySeeds.has(s.seed)),
          })).filter(b => b.seeds.length > 0));
        }
      } else {
        setSafeToApply(true);
      }

      setAnalyzed(true);
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
  const setBlockSeedLang = (bi: number, si: number, lang: string) => setBlocks(prev => prev.map((b, i) => {
    if (i !== bi) return b;
    const seeds = [...b.seeds]; seeds[si] = { ...seeds[si], lang }; return { ...b, seeds };
  }));
  const addSeedToBlock = (bi: number) => setBlocks(prev => prev.map((b, i) => i === bi ? { ...b, seeds: [...b.seeds, { seed: '', lang: 'en' }] } : b));
  const removeSeedFromBlock = (bi: number, si: number) => setBlocks(prev => prev.map((b, i) => i === bi ? { ...b, seeds: b.seeds.filter((_, idx) => idx !== si) } : b));

  const [fixResult, setFixResult] = useState<{ added: number; autoResolved: { id: number; query: string; now_detects: string[] }[] } | null>(null);

  const handleApply = async () => {
    const toApply: Record<string, { seed: string; lang: string }[]> = {};
    for (const block of blocks) {
      if (!block.intentId) continue;
      const seeds = block.seeds.filter(s => s.seed.trim());
      if (seeds.length > 0) toApply[block.intentId] = [...(toApply[block.intentId] || []), ...seeds];
    }
    if (Object.keys(toApply).length === 0) return;
    const result = await api.reviewFix(item.id, toApply);
    if (result.auto_resolved_count > 0) {
      setFixResult({ added: result.added, autoResolved: result.auto_resolved });
    } else {
      onAction();
    }
  };

  const handleReject = async () => { await api.reviewReject(item.id); onAction(); };

  const totalSeeds = blocks.flatMap(b => b.seeds).filter(s => s.seed.trim()).length;
  const usedIntents = new Set(blocks.map(b => b.intentId).filter(Boolean));

  return (
    <div className="p-5 space-y-4 max-w-3xl">
      {/* Header */}
      <div className="flex items-center gap-2">
        <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded ${
          item.flag === 'miss' ? 'bg-red-900/30 text-red-400' :
          item.flag === 'low_confidence' ? 'bg-amber-900/30 text-amber-400' :
          'bg-blue-900/30 text-blue-400'
        }`}>{item.flag.replace('_', ' ')}</span>
        {item.session_id && <span className="text-xs text-zinc-500">Session: {item.session_id}</span>}
        <span className={`text-xs ml-auto ${item.status === 'pending' ? 'text-amber-400' : 'text-zinc-500'}`}>{item.status}</span>
      </div>

      {/* Query */}
      <div className="bg-zinc-800 rounded-lg p-4">
        <div className="text-[10px] text-zinc-500 mb-1">Customer query</div>
        <div className="text-white font-mono text-sm">"{item.query}"</div>
      </div>

      {item.status === 'pending' && (
        <>
          {/* Comparison: detected vs correct */}
          {analyzed && (
            <div className="bg-zinc-800/50 rounded-lg p-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-[10px] text-zinc-500 mb-2">Detected (by ASV)</div>
                  <div className="flex flex-wrap gap-1.5">
                    {item.detected.map(id => {
                      const isWrong = wrongDetections.includes(id);
                      return (
                        <span key={id} className={`text-xs font-mono px-2 py-0.5 rounded border ${
                          isWrong
                            ? 'text-red-400 bg-red-900/20 border-red-800 line-through'
                            : 'text-emerald-400 bg-emerald-900/20 border-emerald-800'
                        }`}>
                          {id} {isWrong ? '✗' : '✓'}
                        </span>
                      );
                    })}
                    {item.detected.length === 0 && <span className="text-xs text-red-400">none</span>}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-zinc-500 mb-2">Correct (by AI)</div>
                  <div className="flex flex-wrap gap-1.5">
                    {correctIntents.map(id => (
                      <span key={id} className="text-xs font-mono text-emerald-400 bg-emerald-900/20 border border-emerald-800 px-2 py-0.5 rounded">{id}</span>
                    ))}
                  </div>
                  {detectedLanguages.length > 0 && detectedLanguages[0] !== 'en' && (
                    <div className="text-[10px] text-violet-400 mt-1.5">
                      Language: {detectedLanguages.map(l => l.toUpperCase()).join(', ')}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Wrong intent analysis (inline, not a separate section) */}
          {wrongAnalysis.length > 0 && (
            <div className="space-y-2">
              {wrongAnalysis.map((wa, i) => (
                <div key={i} className="bg-red-900/10 border border-red-800/30 rounded-lg px-3 py-3 text-xs space-y-2">
                  <div>
                    <span className="text-red-400 font-mono font-semibold">{wa.intent}</span>
                    <span className="text-zinc-500"> — false match</span>
                  </div>
                  <div className="text-zinc-500">
                    Problem seed: <span className="text-white font-mono bg-red-900/30 px-1 rounded">"{wa.problem_seed}"</span>
                    <span className="text-zinc-600 ml-1">— {wa.reason}</span>
                  </div>
                  {wrongIntentSeeds[wa.intent] && (
                    <div>
                      <div className="text-[10px] text-zinc-600 mb-1">Current seeds in index:</div>
                      <div className="flex flex-wrap gap-1">
                        {wrongIntentSeeds[wa.intent].map((s, si) => (
                          <span key={si} className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                            s === wa.problem_seed
                              ? 'bg-red-900/30 text-red-400 border border-red-800'
                              : 'bg-zinc-800 text-zinc-500'
                          }`}>{s}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="text-amber-400">Suggested fix: {wa.fix}</div>
                </div>
              ))}
            </div>
          )}

          {/* Removed risky seeds notice */}
          {addRisks.length > 0 && (
            <div className="text-xs text-amber-400 bg-amber-900/10 border border-amber-800/30 rounded px-3 py-2">
              Removed risky seeds: {addRisks.map(r => `"${r.seed}"`).join(', ')} — {addRisks[0]?.risk}
            </div>
          )}

          {/* Safety verdict */}
          {safeToApply !== null && analyzed && (
            <div className={`text-xs rounded px-3 py-2 ${safeToApply ? 'text-emerald-400 bg-emerald-900/10 border border-emerald-800/30' : 'text-red-400 bg-red-900/10 border border-red-800/30'}`}>
              {safeToApply ? '✓ Safe to apply' : '⚠ Review carefully'}{summary && ` — ${summary}`}
            </div>
          )}

          {/* Progress */}
          {analyzing && (
            <div className="text-xs text-violet-400 flex items-center gap-2">
              <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
              Analyzing with AI...
            </div>
          )}

          {/* Seeds to add */}
          <div className="space-y-2">
            {blocks.length > 0 && (
              <div className="text-[10px] text-zinc-500 uppercase font-semibold">Seeds to add</div>
            )}
            {blocks.map((block, bi) => (
              <div key={bi} className="bg-zinc-800 border border-zinc-700 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <select
                    value={block.intentId}
                    onChange={e => setBlockIntent(bi, e.target.value)}
                    className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1 font-mono focus:border-violet-500 focus:outline-none flex-1"
                  >
                    <option value="">Select intent...</option>
                    {intents.filter(id => !usedIntents.has(id) || id === block.intentId).map(id => (
                      <option key={id} value={id}>{id}</option>
                    ))}
                  </select>
                  <button onClick={() => removeBlock(bi)} className="text-xs text-zinc-600 hover:text-red-400">Remove</button>
                </div>
                {block.seeds.map((entry, si) => (
                  <div key={si} className="flex gap-1.5 mb-1">
                    <select
                      value={entry.lang}
                      onChange={e => setBlockSeedLang(bi, si, e.target.value)}
                      className="bg-zinc-900 border border-zinc-700 text-violet-400 text-[10px] rounded px-1 py-1 w-12 focus:outline-none"
                    >
                      {enabledLangs.map(lang => (
                        <option key={lang} value={lang}>{lang.toUpperCase()}</option>
                      ))}
                    </select>
                    <input
                      value={entry.seed}
                      onChange={e => setBlockSeed(bi, si, e.target.value)}
                      placeholder="e.g. received wrong item"
                      className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-white font-mono focus:border-violet-500 focus:outline-none"
                    />
                    {block.seeds.length > 1 && (
                      <button onClick={() => removeSeedFromBlock(bi, si)} className="text-zinc-600 hover:text-red-400 text-xs">×</button>
                    )}
                  </div>
                ))}
                <button onClick={() => addSeedToBlock(bi)} className="text-[10px] text-zinc-500 hover:text-violet-400 mt-1">+ add seed</button>
              </div>
            ))}
            <button
              onClick={addBlock}
              className="w-full py-2 text-xs text-zinc-500 hover:text-violet-400 border border-dashed border-zinc-700 hover:border-violet-500 rounded-lg transition-colors"
            >
              + Add intent block
            </button>
          </div>

          {/* Auto-resolved feedback */}
          {fixResult && (
            <div className="bg-emerald-900/10 border border-emerald-800 rounded-lg p-4 space-y-2">
              <div className="text-xs text-emerald-400 font-semibold">
                Applied {fixResult.added} seeds. Auto-resolved {fixResult.autoResolved.length} other reviews:
              </div>
              {fixResult.autoResolved.map(ar => (
                <div key={ar.id} className="text-xs text-zinc-400">
                  <span className="text-emerald-400">✓</span> "{ar.query}..." → now detects: {ar.now_detects.join(', ')}
                </div>
              ))}
              <button
                onClick={onAction}
                className="text-xs px-3 py-1.5 bg-emerald-600 text-white rounded hover:bg-emerald-500 mt-2"
              >
                Continue reviewing
              </button>
            </div>
          )}

          {/* Actions */}
          {!fixResult && (
          <div className="flex items-center gap-3 pt-3 border-t border-zinc-800">
            <button
              onClick={runAnalysis}
              disabled={analyzing}
              className="text-xs px-3 py-1.5 border border-violet-500 text-violet-400 rounded hover:bg-violet-500 hover:text-white disabled:opacity-50"
            >
              {analyzing ? 'Analyzing...' : analyzed ? 'Re-analyze' : 'Suggest with AI'}
            </button>
            <div className="flex-1" />
            <button onClick={handleReject} className="text-xs px-3 py-1.5 border border-zinc-700 text-zinc-400 rounded hover:text-white">
              Reject
            </button>
            <button
              onClick={handleApply}
              disabled={totalSeeds === 0}
              className="text-xs px-4 py-1.5 bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30"
            >
              Apply Fix ({totalSeeds} seeds)
            </button>
          </div>
          )}
        </>
      )}

      {item.status !== 'pending' && (
        <div className="text-sm text-zinc-500">
          Status: <span className="text-white font-medium">{item.status}</span>
          {item.suggested_intent && <> — {item.suggested_intent}</>}
        </div>
      )}
    </div>
  );
}
