import { useState } from 'react';
import { api } from '@/api/client';
import Page from '@/components/Page';

type SimPhase = 'idle' | 'generating' | 'baseline' | 'fixing' | 'retesting' | 'done' | 'error';
type LearnResult = { query: string; phrases_added: number; summary: string; missed: string[]; wrong: string[] };
type TurnOutcome = { message: string; ground_truth: string[]; before: string; after: string; confirmed: string[]; missed: string[]; extra: string[] };

const PERSONALITIES = ['casual', 'frustrated', 'formal', 'terse', 'rambling', 'polite'];
const LANGUAGES     = ['English', 'Spanish', 'French', 'Chinese', 'German', 'Japanese', 'Portuguese', 'Arabic'];

export default function SimulatePage() {
  const [turns,          setTurns]          = useState(10);
  const [personality,    setPersonality]    = useState('casual');
  const [sophistication, setSophistication] = useState('medium');
  const [verbosity,      setVerbosity]      = useState('short');
  const [language,       setLanguage]       = useState('English');
  const [scenario,       setScenario]       = useState('');
  const [running,        setRunning]        = useState(false);
  const [phase,          setPhase]          = useState<SimPhase>('idle');
  const [phaseLabel,     setPhaseLabel]     = useState('');
  const [before,         setBefore]         = useState<{ strict: number; detected: number } | null>(null);
  const [after,          setAfter]          = useState<{ strict: number; detected: number } | null>(null);
  const [learnLog,       setLearnLog]       = useState<LearnResult[]>([]);
  const [outcomes,       setOutcomes]       = useState<TurnOutcome[]>([]);
  const stopRef = { current: false };

  const calcMetrics = (results: any[]) => {
    const n = results.length;
    const detected = results.filter((r: any) =>
      (r.missed?.length ?? 0) === 0 && (r.promotable?.length ?? 0) === 0
    ).length;
    const strict = results.filter((r: any) => r.status === 'pass').length;
    return {
      strict:   n === 0 ? 0 : Math.round((strict   / n) * 100),
      detected: n === 0 ? 0 : Math.round((detected / n) * 100),
    };
  };

  const run = async () => {
    stopRef.current = false;
    setRunning(true);
    setBefore(null); setAfter(null); setLearnLog([]); setOutcomes([]);

    try {
      setPhase('generating'); setPhaseLabel('Generating queries via LLM...');
      const generated = await api.trainingGenerate({
        personality, sophistication, verbosity, turns, language,
        scenario: scenario || undefined,
      });
      if (stopRef.current) return;

      const queries = generated.turns.map((t: any) => ({
        message: t.customer_message,
        ground_truth: t.ground_truth as string[],
      }));

      setPhase('baseline'); setPhaseLabel('Running baseline...');
      const baseline = await api.trainingRun(queries);
      setBefore(calcMetrics(baseline.results));
      if (stopRef.current) return;

      const failures = baseline.results.filter((r: any) => r.status === 'fail' || r.status === 'partial');
      if (failures.length > 0) {
        setPhase('fixing');
        for (let idx = 0; idx < failures.length; idx++) {
          if (stopRef.current) break;
          const r = failures[idx];
          setPhaseLabel(`Learning ${idx + 1}/${failures.length}...`);
          const detected = [...(r.confirmed ?? []), ...(r.candidates ?? [])];
          try {
            const res = await api.learnNow(r.message, detected, r.ground_truth);
            setLearnLog(prev => [...prev, {
              query: r.message,
              phrases_added: res.phrases_added,
              summary: res.summary,
              missed: res.missed_intents ?? [],
              wrong: res.wrong_detections ?? [],
            }]);
          } catch { /* continue */ }
        }
      }
      if (stopRef.current) return;

      setPhase('retesting'); setPhaseLabel('Re-testing same queries...');
      const retest = await api.trainingRun(queries);

      const outs: TurnOutcome[] = queries.map((q, i) => ({
        message:      q.message,
        ground_truth: q.ground_truth,
        before:       baseline.results[i]?.status ?? 'fail',
        after:        retest.results[i]?.status ?? 'fail',
        confirmed:    retest.results[i]?.confirmed ?? [],
        missed:       retest.results[i]?.missed ?? [],
        extra:        retest.results[i]?.extra ?? [],
      }));

      const afterMetrics = calcMetrics(retest.results);
      setAfter(afterMetrics);
      setOutcomes(outs);
      setPhase('done'); setPhaseLabel('');

    } catch (err) {
      setPhase('error'); setPhaseLabel((err as Error).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Page title="Simulate" subtitle="LLM generates queries — system learns from failures" size="lg">
      <div className="space-y-6">

        {/* Explainer */}
        <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 flex gap-4">
          <div className="text-2xl shrink-0 mt-0.5">◎</div>
          <div className="space-y-1">
            <div className="text-sm font-medium text-white">How it works</div>
            <div className="text-xs text-zinc-500 leading-relaxed">
              LLM generates realistic queries based on your intent definitions — different personalities, languages, edge cases.
              Each failure is learned from immediately. After retesting the same queries, you see before/after accuracy.
              Run multiple times: improvements compound.
            </div>
            <div className="flex gap-4 pt-1 text-[10px] text-zinc-600">
              <span>① Generate queries</span>
              <span>→</span>
              <span>② Baseline accuracy</span>
              <span>→</span>
              <span>③ Learn from failures</span>
              <span>→</span>
              <span>④ Retest</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-500">Queries</label>
              <select value={turns} onChange={e => setTurns(Number(e.target.value))} disabled={running}
                className="bg-zinc-800 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
                {[5, 10, 15, 20, 30].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-500">Personality</label>
              <select value={personality} onChange={e => setPersonality(e.target.value)} disabled={running}
                className="bg-zinc-800 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
                {PERSONALITIES.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-500">Language</label>
              <select value={language} onChange={e => setLanguage(e.target.value)} disabled={running}
                className="bg-zinc-800 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
                {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-500">Sophistication</label>
              <select value={sophistication} onChange={e => setSophistication(e.target.value)} disabled={running}
                className="bg-zinc-800 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
                {['low', 'medium', 'high'].map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-500">Verbosity</label>
              <select value={verbosity} onChange={e => setVerbosity(e.target.value)} disabled={running}
                className="bg-zinc-800 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
                {['short', 'medium', 'long'].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          </div>

          <input value={scenario} onChange={e => setScenario(e.target.value)} disabled={running}
            placeholder="Scenario (optional): e.g. angry customer trying to return a broken laptop"
            className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-600 focus:border-violet-500 focus:outline-none" />

          <div className="flex items-center gap-3">
            <button onClick={run} disabled={running}
              className="px-5 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg disabled:opacity-30 transition-colors font-medium">
              {running ? 'Running...' : 'Run Simulation'}
            </button>
            {running && (
              <button onClick={() => { stopRef.current = true; }} className="text-xs text-red-400 hover:text-red-300">
                Stop
              </button>
            )}
            {phase === 'done' && (
              <button onClick={run} disabled={running}
                className="text-xs px-3 py-1.5 border border-violet-500/50 text-violet-400 rounded hover:bg-violet-500/10 transition-colors">
                Run again (keep improving)
              </button>
            )}
          </div>
        </div>

        {/* Phase indicator */}
        {phase !== 'idle' && (
          <div className="flex items-center gap-2">
            {running && <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin flex-shrink-0" />}
            <span className={`text-sm ${phase === 'error' ? 'text-red-400' : phase === 'done' ? 'text-emerald-400' : 'text-violet-400'}`}>
              {phase === 'done' ? 'Complete' : phase === 'error' ? `Error: ${phaseLabel}` : phaseLabel || phase}
            </span>
          </div>
        )}

        {/* Before/After scores */}
        {(before !== null || after !== null) && (
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-3">
            <div className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Accuracy</div>
            {([
              { label: 'Detection', key: 'detected' as const, note: 'correct intent found' },
              { label: 'Strict',    key: 'strict'   as const, note: 'perfect, no extras' },
            ] as const).map(({ label, key, note }) => {
              const b = before?.[key] ?? null;
              const a = after?.[key] ?? null;
              return (
                <div key={key} className="flex items-center gap-4">
                  <div className="w-20">
                    <div className="text-xs text-zinc-400 font-medium">{label}</div>
                    <div className="text-[10px] text-zinc-600">{note}</div>
                  </div>
                  {b !== null && <div className="text-2xl font-bold text-zinc-400 w-14 text-right">{b}%</div>}
                  {a !== null && b !== null && (
                    <>
                      <span className={`text-xl font-light ${a > b ? 'text-emerald-400' : a < b ? 'text-red-400' : 'text-zinc-600'}`}>
                        {a > b ? '↑' : a < b ? '↓' : '→'}
                      </span>
                      <div className={`text-2xl font-bold w-14 ${a > b ? 'text-emerald-400' : a < b ? 'text-red-400' : 'text-zinc-400'}`}>{a}%</div>
                      {a !== b && (
                        <span className={`text-sm font-semibold ${a > b ? 'text-emerald-500' : 'text-red-500'}`}>
                          {a > b ? '+' : ''}{a - b}pp
                        </span>
                      )}
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Live learn log */}
        {learnLog.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">What was learned</div>
            {learnLog.map((entry, i) => (
              <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 text-xs space-y-1">
                <div className="flex items-center gap-2">
                  <span className={`font-mono font-bold flex-shrink-0 ${entry.phrases_added > 0 ? 'text-emerald-400' : 'text-zinc-600'}`}>
                    {entry.phrases_added > 0 ? `+${entry.phrases_added} phrases` : 'no change'}
                  </span>
                  {entry.missed.length > 0 && <span className="text-zinc-500 truncate">→ {entry.missed.join(', ')}</span>}
                </div>
                <div className="text-zinc-500 truncate font-mono">"{entry.query.slice(0, 80)}"</div>
                {entry.summary && <div className="text-zinc-600 italic">{entry.summary.slice(0, 100)}</div>}
              </div>
            ))}
          </div>
        )}

        {/* Per-turn breakdown */}
        {outcomes.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Results per query</div>
            {outcomes.map((o, i) => {
              const wasGood  = o.before === 'pass';
              const isGood   = o.after  === 'pass';
              const fixed    = !wasGood && isGood;
              const stuck    = !wasGood && !isGood;
              const degraded = wasGood  && !isGood;
              const icon = fixed ? '✓' : stuck ? '✗' : degraded ? '!' : '✓';
              const cls  = fixed ? 'text-emerald-400' : stuck ? 'text-red-400' : degraded ? 'text-amber-400' : 'text-zinc-500';
              return (
                <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 text-xs space-y-1.5">
                  <div className="flex items-start gap-2">
                    <span className={`font-bold flex-shrink-0 mt-0.5 ${cls}`}>{icon}</span>
                    <div className="min-w-0 flex-1">
                      <div className="text-zinc-300 font-mono leading-snug">"{o.message.slice(0, 80)}{o.message.length > 80 ? '…' : ''}"</div>
                      <div className="flex flex-wrap gap-x-4 gap-y-0.5 mt-1 text-[11px]">
                        <span className="text-zinc-600">Expected: <span className="text-zinc-400">{o.ground_truth.map(id => id.split(':')[1] ?? id).join(', ')}</span></span>
                        {o.confirmed.length > 0 && (
                          <span className="text-zinc-600">Got: <span className={isGood ? 'text-emerald-400' : 'text-amber-400'}>{o.confirmed.map(id => id.split(':')[1] ?? id).join(', ')}</span></span>
                        )}
                        {o.missed.length > 0 && (
                          <span className="text-zinc-600">Missed: <span className="text-red-400">{o.missed.map((id: string) => id.split(':')[1] ?? id).join(', ')}</span></span>
                        )}
                      </div>
                      {fixed    && <div className="text-[10px] text-emerald-600 mt-0.5">Fixed this session</div>}
                      {stuck    && <div className="text-[10px] text-red-600 mt-0.5">Still failing — run again</div>}
                      {degraded && <div className="text-[10px] text-amber-600 mt-0.5">Regressed</div>}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Page>
  );
}
