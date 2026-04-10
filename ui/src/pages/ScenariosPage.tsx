import { useState, useRef, useEffect, useCallback } from 'react';
import { api } from '@/api/client';

// --- Types ---

type TrainingConfig = {
  personality: string;
  sophistication: string;
  verbosity: string;
  turns: number;
  scenario: string; // empty = random
};

type GeneratedTurn = {
  customer_message: string;
  ground_truth: string[];
  intent_description: string;
  agent_response: string;
};

type RunResult = {
  message: string;
  ground_truth: string[];
  confirmed: string[];
  candidates: string[];
  matched: string[];
  promotable: string[];
  missed: string[];
  extra: string[];
  status: 'pass' | 'partial' | 'fail' | 'promotable';
  details: { id: string; score: number; confidence: string; source: string }[];
};

type Correction = {
  action: string;
  query?: string;
  intent?: string;
  from?: string;
  phrase?: string;
};

type ReviewResult = {
  turnIndex: number;
  analysis: string;
  corrections: Correction[];
};

type TrainingCycle = {
  results: RunResult[];
  pass_count: number;
  confirmed_count?: number;
  promotable_count?: number;
  total: number;
  accuracy: number;
  confirmed_rate?: number;
};

type Phase = 'generating' | 'routing_1' | 'reviewing' | 'applying' | 'routing_2' | 'done' | 'error';

type TrainingSession = {
  id: number;
  config: TrainingConfig;
  phase: Phase;
  generatedTurns: GeneratedTurn[];
  cycle1: TrainingCycle | null;
  reviews: ReviewResult[];
  corrections: Correction[];
  applied: number;
  applyErrors: string[];
  cycle2: TrainingCycle | null;
  error?: string;
};

const PERSONALITIES = ['polite', 'frustrated', 'terse', 'rambling', 'formal', 'casual'];
const SOPHISTICATION = ['low', 'medium', 'high'];
const VERBOSITY = ['short', 'medium', 'long'];

const PHASE_LABELS: Record<Phase, string> = {
  generating: 'Generating queries...',
  routing_1: 'Testing baseline...',
  reviewing: 'Finding & fixing failures...',
  applying: 'Auto-learn processing...',
  routing_2: 'Re-testing after fixes...',
  done: 'Complete',
  error: 'Error',
};

// --- Main Component ---

export default function ScenariosPage() {
  const [config, setConfig] = useState<TrainingConfig>({
    personality: 'polite',
    sophistication: 'medium',
    verbosity: 'short',
    turns: 3,
    scenario: '',
  });
  const [sessions, setSessions] = useState<TrainingSession[]>([]);
  const [activeSession, setActiveSession] = useState<number | null>(null);
  const [copied, setCopied] = useState<number | null>(null);
  const stopRef = useRef(false);
  const nextIdRef = useRef(1);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [sessions, activeSession]);

  const updateSession = useCallback((id: number, patch: Partial<TrainingSession>) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, ...patch } : s));
  }, []);

  const runTraining = useCallback(async () => {
    stopRef.current = false;
    const id = nextIdRef.current++;
    const session: TrainingSession = {
      id,
      config: { ...config },
      phase: 'generating',
      generatedTurns: [],
      cycle1: null,
      reviews: [],
      corrections: [],
      applied: 0,
      applyErrors: [],
      cycle2: null,
    };
    setSessions(prev => [...prev, session]);
    setActiveSession(id);

    try {
      // Phase 1: Generate conversation
      const generated = await api.trainingGenerate({
        personality: config.personality,
        sophistication: config.sophistication,
        verbosity: config.verbosity,
        turns: config.turns,
        scenario: config.scenario || undefined,
      });
      if (stopRef.current) return;
      updateSession(id, { generatedTurns: generated.turns, phase: 'routing_1' });

      // Phase 2: Route cycle 1
      const turns = generated.turns.map((t: GeneratedTurn) => ({
        message: t.customer_message,
        ground_truth: t.ground_truth,
      }));
      const cycle1 = await api.trainingRun(turns) as TrainingCycle;
      if (stopRef.current) return;
      updateSession(id, { cycle1, phase: 'reviewing' });

      // Phase 3+4: Report failures → auto-learn fixes them via seed_pipeline
      await api.setReviewMode('auto');

      let failureCount = 0;
      const reviews: ReviewResult[] = [];

      for (let i = 0; i < cycle1.results.length; i++) {
        const r = cycle1.results[i] as RunResult;
        if (r.status === 'pass') continue;
        if (stopRef.current) break;

        failureCount++;
        const detected = [...r.confirmed, ...r.candidates];

        // Report to review queue — auto-learn picks it up
        await api.report(
          r.message,
          detected,
          r.missed.length > 0 ? 'miss' : 'low_confidence',
        );

        reviews.push({
          turnIndex: i,
          analysis: `Reported: ${r.missed.join(', ')} missed. Auto-learn fixing...`,
          corrections: r.missed.map(intent => ({
            action: 'auto',
            intent,
            phrase: r.message.slice(0, 40),
          })),
        });
        updateSession(id, { reviews: [...reviews] });

        // Give auto-learn time to process every 2 failures
        if (failureCount % 2 === 0) {
          await new Promise(resolve => setTimeout(resolve, 3000));
        }
      }

      // Wait for last auto-learn batch
      if (failureCount > 0) {
        updateSession(id, { phase: 'applying' });
        await new Promise(resolve => setTimeout(resolve, 5000));
      }

      await api.setReviewMode('manual');

      if (stopRef.current) return;
      updateSession(id, {
        corrections: reviews.flatMap(r => r.corrections),
        applied: failureCount,
        applyErrors: [],
        phase: 'routing_2',
      });

      // Phase 5: Route cycle 2
      const cycle2 = await api.trainingRun(turns) as TrainingCycle;
      updateSession(id, { cycle2, phase: 'done' });

    } catch (err) {
      updateSession(id, { phase: 'error', error: (err as Error).message });
    }
  }, [config, updateSession]);

  const active = sessions.find(s => s.id === activeSession);

  // Aggregate stats
  const completedSessions = sessions.filter(s => s.phase === 'done');
  const totalImprovement = completedSessions.length > 0
    ? completedSessions.reduce((sum, s) => sum + ((s.cycle2?.accuracy ?? 0) - (s.cycle1?.accuracy ?? 0)), 0) / completedSessions.length
    : 0;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h1 className="text-lg font-semibold text-white">Auto-Improve</h1>
        <p className="text-xs text-zinc-500 mt-1">
          Generate synthetic queries, find failures, fix them automatically. Watch accuracy improve in real time.
        </p>
      </div>

      {/* Config panel */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
        <div className="grid grid-cols-5 gap-4">
          <SelectField label="Personality" value={config.personality}
            options={PERSONALITIES} onChange={v => setConfig(c => ({ ...c, personality: v }))} />
          <SelectField label="Sophistication" value={config.sophistication}
            options={SOPHISTICATION} onChange={v => setConfig(c => ({ ...c, sophistication: v }))} />
          <SelectField label="Verbosity" value={config.verbosity}
            options={VERBOSITY} onChange={v => setConfig(c => ({ ...c, verbosity: v }))} />
          <div>
            <label className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wide block mb-1">Turns</label>
            <input
              type="number" min={1} max={10} value={config.turns}
              onChange={e => setConfig(c => ({ ...c, turns: Math.max(1, Math.min(10, Number(e.target.value))) }))}
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-white font-mono focus:outline-none focus:border-violet-500"
            />
          </div>
          <div className="col-span-5">
            <label className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wide block mb-1">
              Scenario <span className="text-zinc-600 normal-case">(optional — leave empty for random)</span>
            </label>
            <input
              type="text"
              value={config.scenario}
              onChange={e => setConfig(c => ({ ...c, scenario: e.target.value }))}
              placeholder="e.g. customer received wrong item, wants refund and to speak to a manager"
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-violet-500 placeholder:text-zinc-600"
            />
          </div>
        </div>
        <div className="flex items-center gap-3 mt-4">
          {sessions.some(s => !['done', 'error'].includes(s.phase)) ? (
            <button
              onClick={() => { stopRef.current = true; }}
              className="px-4 py-2 text-sm bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={runTraining}
              className="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition-colors"
            >
              Run Auto-Improve
            </button>
          )}
        </div>
      </div>

      {/* Aggregate stats */}
      {completedSessions.length > 0 && (
        <div className="flex items-center gap-6 text-xs">
          <Stat label="Sessions" value={completedSessions.length} />
          <Stat label="Avg improvement" value={`${totalImprovement >= 0 ? '+' : ''}${Math.round(totalImprovement * 100)}%`}
            color={totalImprovement > 0 ? 'text-emerald-400' : totalImprovement < 0 ? 'text-red-400' : 'text-zinc-400'} />
          <Stat label="Total corrections" value={completedSessions.reduce((s, sess) => s + sess.applied, 0)} color="text-amber-400" />
        </div>
      )}

      {/* Session list + active session */}
      <div className="flex gap-4 min-h-0" style={{ height: 'calc(100vh - 26rem)' }}>
        {/* Session sidebar */}
        {sessions.length > 0 && (
          <div className="w-52 flex-shrink-0 space-y-1 overflow-y-auto">
            {sessions.map(s => (
              <button
                key={s.id}
                onClick={() => setActiveSession(s.id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors ${
                  activeSession === s.id ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:bg-zinc-800/50'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono">#{s.id}</span>
                  {!['done', 'error'].includes(s.phase) && (
                    <span className="text-amber-400 animate-pulse text-[10px]">{PHASE_LABELS[s.phase]?.split('...')[0]}</span>
                  )}
                  {s.phase === 'done' && s.cycle1 && s.cycle2 && (
                    <span className="font-mono">
                      <span className={s.cycle1.accuracy < s.cycle2.accuracy ? 'text-zinc-500' : 'text-zinc-400'}>
                        {Math.round(s.cycle1.accuracy * 100)}%
                      </span>
                      <span className="text-zinc-600 mx-0.5">&rarr;</span>
                      <span className={s.cycle2.accuracy > s.cycle1.accuracy ? 'text-emerald-400' : s.cycle2.accuracy === s.cycle1.accuracy ? 'text-zinc-400' : 'text-red-400'}>
                        {Math.round(s.cycle2.accuracy * 100)}%
                      </span>
                    </span>
                  )}
                  {s.phase === 'error' && <span className="text-red-400">err</span>}
                </div>
                <div className="text-[10px] text-zinc-600 mt-0.5 capitalize truncate">
                  {s.config.personality} / {s.config.sophistication} / {s.config.verbosity}
                  {s.config.scenario && <span className="text-zinc-700"> — {s.config.scenario.slice(0, 30)}</span>}
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Main training view */}
        {active ? (
          <div className="flex-1 overflow-y-auto space-y-4 pr-1">
            <SessionView session={active} copied={copied} onCopy={id => {
              setCopied(id);
              setTimeout(() => setCopied(null), 2000);
            }} />
            <div ref={bottomRef} />
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-zinc-600 text-sm">
            Configure persona, optionally describe a scenario, and click Train.
          </div>
        )}
      </div>
    </div>
  );
}

// --- Session Detail View ---

function SessionView({ session: s, copied, onCopy }: { session: TrainingSession; copied: number | null; onCopy: (id: number) => void }) {
  return (
    <div className="space-y-4">
      {/* Phase indicator */}
      {s.phase !== 'done' && s.phase !== 'error' && (
        <div className="flex items-center gap-3 bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3">
          <PhaseProgress phase={s.phase} />
        </div>
      )}

      {/* Error */}
      {s.phase === 'error' && (
        <div className="bg-red-400/5 border border-red-500/30 rounded-lg px-4 py-3 text-sm text-red-400">
          {s.error}
        </div>
      )}

      {/* Generated Turns */}
      {s.generatedTurns.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-2">
            <SectionHeader title="Generated Conversation" count={s.generatedTurns.length} />
            {s.config.scenario && (
              <span className="text-[10px] text-zinc-600 italic max-w-xs truncate">{s.config.scenario}</span>
            )}
          </div>
          <div className="space-y-1">
            {s.generatedTurns.map((t, i) => (
              <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2">
                <div className="flex items-start gap-3">
                  <span className="text-[10px] text-zinc-600 font-mono w-4 flex-shrink-0 pt-0.5">{i + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-white">{t.customer_message}</div>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {t.ground_truth.map(gt => (
                        <span key={gt} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">{gt}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Cycle 1 Results */}
      {s.cycle1 && (
        <CycleResults label="Cycle 1 — Before Training" cycle={s.cycle1} />
      )}

      {/* Reviews & Corrections */}
      {s.reviews.length > 0 && (
        <section>
          <SectionHeader title="LLM Review" count={s.reviews.length} subtitle="failed turns reviewed" />
          <div className="space-y-2 mt-2">
            {s.reviews.map((rev, i) => (
              <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] text-zinc-500 font-mono">Turn {rev.turnIndex + 1}</span>
                  <span className="text-xs text-zinc-400">{rev.analysis}</span>
                </div>
                {rev.corrections.length > 0 && (
                  <div className="space-y-0.5 mt-1">
                    {rev.corrections.map((c, j) => (
                      <CorrectionRow key={j} correction={c} />
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Applied count */}
      {s.applied > 0 && (
        <div className="flex items-center gap-3 text-xs bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-2.5">
          <span className="text-emerald-400 font-semibold">{s.applied} corrections applied</span>
          {s.applyErrors.length > 0 && (
            <span className="text-red-400/70">{s.applyErrors.length} errors</span>
          )}
        </div>
      )}

      {/* Cycle 2 Results */}
      {s.cycle2 && (
        <CycleResults label="Cycle 2 — After Training" cycle={s.cycle2} />
      )}

      {/* Improvement Summary */}
      {s.phase === 'done' && s.cycle1 && s.cycle2 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3">
          <div className="flex items-center justify-between">
            <ImprovementSummary cycle1={s.cycle1} cycle2={s.cycle2} corrections={s.applied} />
            <button
              onClick={() => {
                navigator.clipboard.writeText(formatSessionMarkdown(s));
                onCopy(s.id);
              }}
              className="text-[10px] text-violet-400 hover:text-violet-300 transition-colors"
            >
              {copied === s.id ? 'Copied!' : 'Copy'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// --- Cycle Results ---

function CycleResults({ label, cycle }: { label: string; cycle: TrainingCycle }) {
  const [expanded, setExpanded] = useState(true);

  return (
    <section>
      <div className="flex items-center gap-3 mb-2 cursor-pointer" onClick={() => setExpanded(!expanded)}>
        <SectionHeader title={label} />
        <AccuracyBadge accuracy={cycle.accuracy} pass={cycle.pass_count} total={cycle.total} confirmedCount={cycle.confirmed_count} promotableCount={cycle.promotable_count} />
        <span className="text-[10px] text-zinc-600">{expanded ? '[-]' : '[+]'}</span>
      </div>
      {expanded && (
        <div className="space-y-1">
          {cycle.results.map((r, i) => (
            <TurnResult key={i} result={r} index={i} />
          ))}
        </div>
      )}
    </section>
  );
}

// --- Turn Result Row ---

function TurnResult({ result: r, index }: { result: RunResult; index: number }) {
  const [expanded, setExpanded] = useState(false);

  const statusColor = ({
    pass: 'border-emerald-500/30 bg-emerald-400/5',
    partial: 'border-amber-500/30 bg-amber-400/5',
    fail: 'border-red-500/30 bg-red-400/5',
    promotable: 'border-amber-500/30 bg-amber-400/5',
  } as Record<string, string>)[r.status] ?? 'border-zinc-700/30 bg-zinc-800/5';

  return (
    <div
      className={`border rounded-lg px-3 py-2 cursor-pointer hover:brightness-110 transition-all ${statusColor}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-3">
        <span className="text-[10px] text-zinc-600 font-mono w-4 flex-shrink-0">{index + 1}</span>
        <StatusBadge status={r.status} />
        <span className="text-sm text-white truncate flex-1">{r.message}</span>
        <div className="flex gap-1 flex-shrink-0">
          {r.ground_truth.map(gt => {
            const matched = r.matched.includes(gt);
            const missed = r.missed.includes(gt);
            return (
              <span key={gt} className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                matched ? 'bg-emerald-400/10 text-emerald-400' :
                missed ? 'bg-red-400/10 text-red-400' :
                'bg-zinc-800 text-zinc-400'
              }`}>
                {gt}
              </span>
            );
          })}
        </div>
      </div>

      {expanded && (
        <div className="mt-2 ml-7 space-y-1.5">
          {/* Confirmed intents */}
          {r.details.filter(d => d.confidence !== 'low').length > 0 && (
            <div className="space-y-0.5">
              <div className="text-[9px] text-emerald-400/60 uppercase font-semibold">Confirmed</div>
              {r.details.filter(d => d.confidence !== 'low').map((d, i) => {
                const isMatched = r.matched.includes(d.id);
                const isExtra = r.extra.includes(d.id);
                return (
                  <div key={i} className="text-xs font-mono flex items-center gap-2">
                    <span className={isMatched ? 'text-emerald-400' : isExtra ? 'text-red-400' : 'text-zinc-400'}>
                      {isMatched ? '+' : isExtra ? '!' : ' '}{d.id}
                    </span>
                    <span className="text-amber-400/70">{d.score.toFixed(2)}</span>
                    <span className={`text-[9px] ${d.confidence === 'high' ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {d.confidence}
                    </span>
                    <span className="text-zinc-600 text-[9px]">{d.source}</span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Candidate intents (dimmed) */}
          {r.details.filter(d => d.confidence === 'low').length > 0 && (
            <div className="space-y-0.5 opacity-40">
              <div className="text-[9px] text-zinc-500 uppercase font-semibold">Candidates</div>
              {r.details.filter(d => d.confidence === 'low').map((d, i) => (
                <div key={i} className="text-xs font-mono flex items-center gap-2">
                  <span className="text-zinc-500"> {d.id}</span>
                  <span className="text-amber-400/70">{d.score.toFixed(2)}</span>
                  <span className="text-zinc-600 text-[9px]">{d.source}</span>
                </div>
              ))}
            </div>
          )}

          {/* Missed */}
          {r.missed.length > 0 && (
            <div className="text-xs">
              <span className="text-red-400/70">Missed: </span>
              <span className="text-red-400 font-mono">{r.missed.join(', ')}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// --- Correction Row ---

function CorrectionRow({ correction: c }: { correction: Correction }) {
  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <span className="px-1.5 py-0.5 rounded text-[9px] uppercase font-semibold text-cyan-400 bg-cyan-400/10">phrase</span>
      <span className="text-zinc-400">"{c.phrase}"</span>
      <span className="text-zinc-600">&rarr;</span>
      <span className="text-emerald-400">{c.intent}</span>
    </div>
  );
}

// --- Phase Progress ---

function PhaseProgress({ phase }: { phase: Phase }) {
  const steps: Phase[] = ['generating', 'routing_1', 'reviewing', 'applying', 'routing_2'];
  const stepLabels = ['Generate', 'Baseline', 'Fix', 'Learn', 'Re-test'];
  const currentIdx = steps.indexOf(phase);

  return (
    <div className="flex items-center gap-1 w-full">
      {steps.map((step, i) => (
        <div key={step} className="flex items-center gap-1 flex-1">
          <div className={`flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-semibold uppercase tracking-wide flex-1 justify-center ${
            i === currentIdx ? 'bg-violet-500/20 text-violet-400 animate-pulse' :
            i < currentIdx ? 'bg-emerald-400/10 text-emerald-400' :
            'bg-zinc-800 text-zinc-600'
          }`}>
            {i < currentIdx ? '\u2713' : ''} {stepLabels[i]}
          </div>
          {i < steps.length - 1 && <span className="text-zinc-700 text-[10px]">&rarr;</span>}
        </div>
      ))}
    </div>
  );
}

// --- Improvement Summary ---

function ImprovementSummary({ cycle1, cycle2, corrections }: { cycle1: TrainingCycle; cycle2: TrainingCycle; corrections: number }) {
  const delta = cycle2.accuracy - cycle1.accuracy;
  const improved = delta > 0;
  const same = delta === 0;

  return (
    <div className="flex items-center gap-4 text-xs">
      <div className="flex items-center gap-2">
        <span className="text-zinc-500">Cycle 1:</span>
        <span className="font-mono text-zinc-400">{Math.round(cycle1.accuracy * 100)}%</span>
        <span className="text-zinc-500">({cycle1.pass_count}/{cycle1.total})</span>
      </div>
      <span className="text-zinc-600">&rarr;</span>
      <div className="flex items-center gap-2">
        <span className="text-zinc-500">Cycle 2:</span>
        <span className={`font-mono font-semibold ${improved ? 'text-emerald-400' : same ? 'text-zinc-400' : 'text-red-400'}`}>
          {Math.round(cycle2.accuracy * 100)}%
        </span>
        <span className="text-zinc-500">({cycle2.pass_count}/{cycle2.total})</span>
      </div>
      <span className={`font-mono font-semibold ${improved ? 'text-emerald-400' : same ? 'text-zinc-500' : 'text-red-400'}`}>
        {delta >= 0 ? '+' : ''}{Math.round(delta * 100)}%
      </span>
      <span className="text-zinc-600">|</span>
      <span className="text-zinc-500">{corrections} corrections</span>
    </div>
  );
}

// --- UI Helpers ---

function SectionHeader({ title, count, subtitle }: { title: string; count?: number; subtitle?: string }) {
  return (
    <div className="flex items-center gap-2">
      <h2 className="text-[10px] text-zinc-500 font-semibold uppercase tracking-wide">{title}</h2>
      {count !== undefined && (
        <span className="text-[10px] text-zinc-600 font-mono">{count}{subtitle ? ` ${subtitle}` : ''}</span>
      )}
    </div>
  );
}

function AccuracyBadge({ accuracy, pass, total, confirmedCount, promotableCount }: {
  accuracy: number; pass: number; total: number; confirmedCount?: number; promotableCount?: number;
}) {
  const pct = Math.round(accuracy * 100);
  const color = pct >= 80 ? 'text-emerald-400 bg-emerald-400/10' :
    pct >= 50 ? 'text-amber-400 bg-amber-400/10' :
    'text-red-400 bg-red-400/10';

  return (
    <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded ${color}`}>
      {pct}% ({pass}/{total})
      {promotableCount != null && promotableCount > 0 && (
        <span className="text-zinc-500 ml-1">({confirmedCount} confirmed, {promotableCount} promotable)</span>
      )}
    </span>
  );
}

function StatusBadge({ status }: { status: 'pass' | 'partial' | 'fail' | 'promotable' }) {
  const styles = {
    pass: 'bg-emerald-400/10 text-emerald-400',
    promotable: 'bg-blue-400/10 text-blue-400',
    partial: 'bg-amber-400/10 text-amber-400',
    fail: 'bg-red-400/10 text-red-400',
  }[status];

  return (
    <span className={`text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded ${styles}`}>
      {status}
    </span>
  );
}

function SelectField({ label, value, options, onChange }: {
  label: string; value: string; options: string[]; onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wide block mb-1">{label}</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-white capitalize focus:outline-none focus:border-violet-500"
      >
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-zinc-500">{label}:</span>
      <span className={`font-mono font-semibold ${color || 'text-zinc-300'}`}>{value}</span>
    </div>
  );
}

// --- Markdown export ---

function formatSessionMarkdown(s: TrainingSession): string {
  let md = `## Training Session #${s.id}\n`;
  md += `**Persona:** ${s.config.personality} / ${s.config.sophistication} / ${s.config.verbosity}\n`;
  if (s.config.scenario) md += `**Scenario:** ${s.config.scenario}\n`;
  md += `\n`;

  if (s.cycle1) {
    md += `### Cycle 1 — Before Training\n`;
    md += `**Accuracy:** ${Math.round(s.cycle1.accuracy * 100)}% (${s.cycle1.pass_count}/${s.cycle1.total} pass)\n\n`;
    for (let i = 0; i < s.cycle1.results.length; i++) {
      const r = s.cycle1.results[i];
      md += `- Turn ${i + 1} [${r.status.toUpperCase()}]: "${r.message}"\n`;
      md += `  Ground truth: ${r.ground_truth.join(', ')} | Confirmed: ${r.confirmed.join(', ') || 'none'}`;
      if (r.candidates.length) md += ` | Candidates: ${r.candidates.join(', ')}`;
      md += `\n`;
      if (r.missed.length) md += `  Missed: ${r.missed.join(', ')}\n`;
    }
    md += `\n`;
  }

  if (s.reviews.length > 0) {
    md += `### Corrections\n`;
    md += `Applied: ${s.applied}\n\n`;
    for (const rev of s.reviews) {
      md += `- Turn ${rev.turnIndex + 1}: ${rev.analysis}\n`;
      for (const c of rev.corrections) {
        md += `  - phrase: "${c.phrase}" -> ${c.intent}\n`;
      }
    }
    md += `\n`;
  }

  if (s.cycle2) {
    md += `### Cycle 2 — After Training\n`;
    md += `**Accuracy:** ${Math.round(s.cycle2.accuracy * 100)}% (${s.cycle2.pass_count}/${s.cycle2.total} pass)\n\n`;
    for (let i = 0; i < s.cycle2.results.length; i++) {
      const r = s.cycle2.results[i];
      md += `- Turn ${i + 1} [${r.status.toUpperCase()}]: confirmed=[${r.confirmed.join(', ') || 'none'}]\n`;
    }
    md += `\n`;
  }

  if (s.cycle1 && s.cycle2) {
    const delta = Math.round((s.cycle2.accuracy - s.cycle1.accuracy) * 100);
    md += `### Summary\n`;
    md += `${Math.round(s.cycle1.accuracy * 100)}% -> ${Math.round(s.cycle2.accuracy * 100)}% (${delta >= 0 ? '+' : ''}${delta}%) with ${s.applied} corrections\n`;
  }

  return md;
}
