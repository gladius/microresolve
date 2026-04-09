import { useState, useRef } from 'react';
import { api } from '@/api/client';

type Phase = 'idle' | 'generating' | 'baseline' | 'fixing' | 'retesting' | 'done' | 'error';

interface IntentResult {
  intent: string;
  before: { correct: number; total: number };
  after: { correct: number; total: number };
}

interface RunState {
  phase: Phase;
  phaseTime: string;
  queries: { message: string; ground_truth: string[]; before_result: string; before_hit: boolean; after_result: string; after_hit: boolean }[];
  beforeAccuracy: number;
  afterAccuracy: number;
  generated: number;
  failures: number;
  fixed: number;
  stuck: number;
  perIntent: IntentResult[];
  error?: string;
}

const PERSONALITIES = ['casual', 'frustrated', 'formal', 'terse', 'rambling', 'polite'];

export default function AutoImprovePage() {
  const [turns, setTurns] = useState(10);
  const [personality, setPersonality] = useState('casual');
  const [scenario, setScenario] = useState('');
  const [running, setRunning] = useState(false);
  const [run, setRun] = useState<RunState | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const stopRef = useRef(false);

  const runImprove = async () => {
    stopRef.current = false;
    setRunning(true);
    setShowDetails(false);

    const state: RunState = {
      phase: 'generating', phaseTime: '', queries: [],
      beforeAccuracy: 0, afterAccuracy: 0,
      generated: 0, failures: 0, fixed: 0, stuck: 0,
      perIntent: [],
    };
    setRun({ ...state });

    try {
      // Phase 1: Generate
      const t0 = performance.now();
      const generated = await api.trainingGenerate({
        personality, sophistication: 'medium', verbosity: 'short',
        turns, scenario: scenario || undefined,
      });
      state.generated = generated.turns.length;
      state.phaseTime = `Generated ${state.generated} queries in ${((performance.now() - t0) / 1000).toFixed(1)}s`;
      state.phase = 'baseline';
      setRun({ ...state });

      if (stopRef.current) { setRunning(false); return; }

      // Phase 2: Baseline
      const t1 = performance.now();
      const queries = generated.turns.map((t: any) => ({
        message: t.customer_message,
        ground_truth: t.ground_truth as string[],
        before_result: '', before_hit: false,
        after_result: '', after_hit: false,
      }));

      const baseline = await api.trainingRun(
        queries.map((q: any) => ({ message: q.message, ground_truth: q.ground_truth }))
      );

      let beforeCorrect = 0;
      const intentBefore: Record<string, { correct: number; total: number }> = {};

      for (let i = 0; i < baseline.results.length; i++) {
        const r = baseline.results[i];
        queries[i].before_result = r.confirmed.join(', ') || r.candidates.join(', ') || '(none)';
        queries[i].before_hit = r.status === 'pass';
        if (r.status === 'pass') beforeCorrect++;

        for (const gt of r.ground_truth) {
          if (!intentBefore[gt]) intentBefore[gt] = { correct: 0, total: 0 };
          intentBefore[gt].total++;
          if (r.matched.includes(gt)) intentBefore[gt].correct++;
        }
      }

      state.beforeAccuracy = Math.round((beforeCorrect / queries.length) * 100);
      state.queries = queries;
      state.phaseTime = `Baseline: ${state.beforeAccuracy}% in ${((performance.now() - t1) / 1000).toFixed(1)}s`;
      state.phase = 'fixing';
      setRun({ ...state });

      if (stopRef.current) { setRunning(false); return; }

      // Phase 3: Report failures → auto-learn fixes them
      const t2 = performance.now();
      let failures = 0;

      // Enable auto-learn
      await api.setReviewMode('auto_learn');

      for (let i = 0; i < baseline.results.length; i++) {
        const r = baseline.results[i];
        if (r.status === 'pass') continue;
        failures++;

        const detected = [...r.confirmed, ...r.candidates];
        await api.report(
          r.message,
          detected,
          r.missed.length > 0 ? 'miss' : 'low_confidence',
        );

        // Small delay to let auto-learn process
        if (failures % 3 === 0) {
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      }

      // Wait for last batch to process
      if (failures > 0) {
        await new Promise(resolve => setTimeout(resolve, 3000));
      }

      state.failures = failures;
      state.phaseTime = `Fixed ${failures} failures in ${((performance.now() - t2) / 1000).toFixed(1)}s`;
      state.phase = 'retesting';
      setRun({ ...state });

      // Restore manual mode
      await api.setReviewMode('manual');

      if (stopRef.current) { setRunning(false); return; }

      // Phase 4: Re-test
      const t3 = performance.now();
      const retest = await api.trainingRun(
        queries.map((q: any) => ({ message: q.message, ground_truth: q.ground_truth }))
      );

      let afterCorrect = 0;
      let fixed = 0;
      let stuck = 0;
      const intentAfter: Record<string, { correct: number; total: number }> = {};

      for (let i = 0; i < retest.results.length; i++) {
        const r = retest.results[i];
        queries[i].after_result = r.confirmed.join(', ') || r.candidates.join(', ') || '(none)';
        queries[i].after_hit = r.status === 'pass';
        if (r.status === 'pass') afterCorrect++;

        if (!queries[i].before_hit && queries[i].after_hit) fixed++;
        if (!queries[i].before_hit && !queries[i].after_hit) stuck++;

        for (const gt of r.ground_truth) {
          if (!intentAfter[gt]) intentAfter[gt] = { correct: 0, total: 0 };
          intentAfter[gt].total++;
          if (r.matched.includes(gt)) intentAfter[gt].correct++;
        }
      }

      // Build per-intent comparison
      const allIntents = new Set([...Object.keys(intentBefore), ...Object.keys(intentAfter)]);
      const perIntent: IntentResult[] = Array.from(allIntents).map(intent => ({
        intent,
        before: intentBefore[intent] || { correct: 0, total: 0 },
        after: intentAfter[intent] || { correct: 0, total: 0 },
      })).sort((a, b) => {
        const deltaA = (a.after.total > 0 ? a.after.correct / a.after.total : 0) - (a.before.total > 0 ? a.before.correct / a.before.total : 0);
        const deltaB = (b.after.total > 0 ? b.after.correct / b.after.total : 0) - (b.before.total > 0 ? b.before.correct / b.before.total : 0);
        return deltaB - deltaA;
      });

      state.afterAccuracy = Math.round((afterCorrect / queries.length) * 100);
      state.fixed = fixed;
      state.stuck = stuck;
      state.perIntent = perIntent;
      state.queries = queries;
      state.phaseTime = `Complete in ${((performance.now() - t3) / 1000).toFixed(1)}s`;
      state.phase = 'done';
      setRun({ ...state });

    } catch (err) {
      state.phase = 'error';
      state.error = (err as Error).message;
      setRun({ ...state });
    } finally {
      // Ensure manual mode
      api.setReviewMode('manual').catch(() => {});
      setRunning(false);
    }
  };

  const pct = (n: number, t: number) => t > 0 ? Math.round((n / t) * 100) : 0;
  const delta = run ? run.afterAccuracy - run.beforeAccuracy : 0;

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-white">Auto-Improve</h2>
        <p className="text-xs text-zinc-500 mt-1">
          Generate synthetic queries, find failures, fix them automatically. Watch accuracy improve in real time.
        </p>
      </div>

      {/* Config */}
      <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4 space-y-3">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-xs text-zinc-500">Queries</label>
            <select value={turns} onChange={e => setTurns(Number(e.target.value))} disabled={running}
              className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1 focus:outline-none">
              {[5, 10, 15, 20, 30].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-zinc-500">Personality</label>
            <select value={personality} onChange={e => setPersonality(e.target.value)} disabled={running}
              className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1 focus:outline-none">
              {PERSONALITIES.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
        </div>
        <div>
          <input
            value={scenario}
            onChange={e => setScenario(e.target.value)}
            placeholder="Scenario (optional): e.g. angry customer trying to return a broken laptop"
            disabled={running}
            className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-1.5 text-xs text-white focus:border-violet-500 focus:outline-none"
          />
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={runImprove}
            disabled={running}
            className="px-4 py-2 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30"
          >
            {running ? 'Running...' : 'Run Auto-Improve'}
          </button>
          {running && (
            <button onClick={() => { stopRef.current = true; }} className="text-xs text-red-400">Stop</button>
          )}
        </div>
      </div>

      {/* Phase indicator */}
      {run && run.phase !== 'idle' && (
        <div className="flex items-center gap-3">
          {run.phase !== 'done' && run.phase !== 'error' && (
            <div className="w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
          )}
          <span className={`text-xs ${run.phase === 'error' ? 'text-red-400' : run.phase === 'done' ? 'text-emerald-400' : 'text-violet-400'}`}>
            {run.phase === 'generating' && 'Generating synthetic queries...'}
            {run.phase === 'baseline' && 'Running baseline...'}
            {run.phase === 'fixing' && `Fixing ${run.failures} failures with auto-learn...`}
            {run.phase === 'retesting' && 'Re-testing after fixes...'}
            {run.phase === 'done' && run.phaseTime}
            {run.phase === 'error' && `Error: ${run.error}`}
          </span>
        </div>
      )}

      {/* Results */}
      {run && (run.phase === 'done' || run.beforeAccuracy > 0) && (
        <div className="space-y-4">
          {/* Before / After bars */}
          <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-5 space-y-4">
            <div className="flex items-end justify-between">
              <div>
                <div className="text-xs text-zinc-500 mb-1">Before</div>
                <div className="flex items-center gap-3">
                  <div className="w-48 h-6 bg-zinc-700 rounded-full overflow-hidden">
                    <div className="h-full bg-zinc-500 rounded-full transition-all duration-1000" style={{ width: `${run.beforeAccuracy}%` }} />
                  </div>
                  <span className="text-lg font-mono text-zinc-400">{run.beforeAccuracy}%</span>
                </div>
              </div>
              {run.phase === 'done' && delta !== 0 && (
                <span className={`text-2xl font-bold ${delta > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {delta > 0 ? '+' : ''}{delta}%
                </span>
              )}
            </div>

            {run.afterAccuracy > 0 && (
              <div>
                <div className="text-xs text-zinc-500 mb-1">After</div>
                <div className="flex items-center gap-3">
                  <div className="w-48 h-6 bg-zinc-700 rounded-full overflow-hidden">
                    <div className="h-full bg-emerald-500 rounded-full transition-all duration-1000" style={{ width: `${run.afterAccuracy}%` }} />
                  </div>
                  <span className="text-lg font-mono text-emerald-400">{run.afterAccuracy}%</span>
                </div>
              </div>
            )}

            {/* Stats row */}
            {run.phase === 'done' && (
              <div className="flex gap-6 pt-3 border-t border-zinc-700 text-xs">
                <div><span className="text-white font-semibold">{run.generated}</span> <span className="text-zinc-500">generated</span></div>
                <div><span className="text-amber-400 font-semibold">{run.failures}</span> <span className="text-zinc-500">failures</span></div>
                <div><span className="text-emerald-400 font-semibold">{run.fixed}</span> <span className="text-zinc-500">fixed</span></div>
                {run.stuck > 0 && (
                  <div><span className="text-red-400 font-semibold">{run.stuck}</span> <span className="text-zinc-500">stuck</span></div>
                )}
              </div>
            )}
          </div>

          {/* Per-intent breakdown */}
          {run.phase === 'done' && run.perIntent.length > 0 && (
            <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
              <div className="text-xs text-zinc-500 font-semibold uppercase mb-3">Per Intent</div>
              <div className="space-y-2">
                {run.perIntent.map(({ intent, before, after }) => {
                  const bPct = pct(before.correct, before.total);
                  const aPct = pct(after.correct, after.total);
                  const d = aPct - bPct;
                  return (
                    <div key={intent} className="flex items-center gap-3 text-xs">
                      <span className="text-zinc-400 font-mono w-40 truncate">{intent}</span>
                      <div className="flex-1 flex items-center gap-2">
                        <div className="w-16 h-2 bg-zinc-700 rounded-full overflow-hidden">
                          <div className="h-full bg-zinc-500 rounded-full" style={{ width: `${bPct}%` }} />
                        </div>
                        <span className="text-zinc-500 w-8 text-right">{bPct}%</span>
                        <span className="text-zinc-600">→</span>
                        <div className="w-16 h-2 bg-zinc-700 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${aPct >= bPct ? 'bg-emerald-500' : 'bg-red-500'}`} style={{ width: `${aPct}%` }} />
                        </div>
                        <span className={`w-8 text-right ${aPct >= bPct ? 'text-emerald-400' : 'text-red-400'}`}>{aPct}%</span>
                        {d !== 0 && (
                          <span className={`w-12 text-right ${d > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {d > 0 ? '▲' : '▼'} {Math.abs(d)}%
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Query details toggle */}
          {run.phase === 'done' && (
            <div>
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="text-xs text-zinc-500 hover:text-violet-400"
              >
                {showDetails ? 'Hide details' : 'Show query details'}
              </button>

              {showDetails && (
                <div className="mt-2 border border-zinc-800 rounded-lg divide-y divide-zinc-800/50 max-h-80 overflow-y-auto">
                  {run.queries.map((q, i) => (
                    <div key={i} className={`px-3 py-2 text-xs ${q.after_hit ? '' : 'bg-red-900/5'}`}>
                      <div className="text-zinc-300">"{q.message}"</div>
                      <div className="flex items-center gap-4 mt-1">
                        <span className="text-zinc-500">expected: {q.ground_truth.join(', ')}</span>
                        <span className={q.before_hit ? 'text-emerald-400' : 'text-red-400'}>
                          before: {q.before_result}
                        </span>
                        <span className={q.after_hit ? 'text-emerald-400' : 'text-red-400'}>
                          after: {q.after_result}
                        </span>
                        {!q.before_hit && q.after_hit && <span className="text-emerald-400 font-semibold">FIXED</span>}
                        {!q.before_hit && !q.after_hit && <span className="text-red-400">stuck</span>}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Run again */}
          {run.phase === 'done' && (
            <button
              onClick={runImprove}
              disabled={running}
              className="text-xs px-3 py-1.5 border border-violet-500 text-violet-400 rounded hover:bg-violet-500 hover:text-white"
            >
              Run Again (keep improving)
            </button>
          )}
        </div>
      )}
    </div>
  );
}
