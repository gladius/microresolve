import { useState, useRef, useEffect, useCallback } from 'react';
import { api, type MultiRouteOutput, type ReviewItem, type ReviewAnalyzeResult } from '@/api/client';
import { useAppStore } from '@/store';

// ─── Types ────────────────────────────────────────────────────────────────────

type Message =
  | { type: 'query'; text: string }
  | { type: 'result'; result: MultiRouteOutput; latency: number; query: string }
  | { type: 'learn'; text: string }
  | { type: 'error'; text: string };

// Mirrors the server's StudioEvent enum (serde tag = "type")
type FeedEvent =
  | { type: 'item_queued';  id: number; query: string; app_id: string; flag: string | null }
  | { type: 'llm_started';  id: number; query: string }
  | { type: 'llm_done';     id: number; correct: string[]; wrong: string[]; phrases_added: number; summary: string }
  | { type: 'fix_applied';  id: number; phrases_added: number; phrases_replaced: number; version_before: number; version_after: number }
  | { type: 'escalated';    id: number; reason: string };

type SimPhase = 'idle' | 'generating' | 'baseline' | 'fixing' | 'retesting' | 'done' | 'error';

const PERSONALITIES = ['casual', 'frustrated', 'formal', 'terse', 'rambling', 'polite'];

const INTENT_COLORS = [
  'text-emerald-400', 'text-blue-400', 'text-amber-400', 'text-pink-400',
  'text-cyan-400', 'text-violet-400', 'text-orange-400', 'text-lime-400',
];
const INTENT_BG_COLORS = [
  'bg-emerald-400/20', 'bg-blue-400/20', 'bg-amber-400/20', 'bg-pink-400/20',
  'bg-cyan-400/20', 'bg-violet-400/20', 'bg-orange-400/20', 'bg-lime-400/20',
];

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function StudioPage() {
  const [mode, setMode] = useState<'manual' | 'simulate'>('manual');
  const [reviewItems, setReviewItems] = useState<ReviewItem[]>([]);
  const [selectedItem, setSelectedItem] = useState<ReviewItem | null>(null);
  const [selectedFeedEvent, setSelectedFeedEvent] = useState<FeedEvent | null>(null);
  const [intents, setIntents] = useState<string[]>([]);
  const [accuracy, setAccuracy] = useState<{ before: number; after: number } | null>(null);
  // Live feed from SSE — newest first, capped at 200 entries
  const [feedEvents, setFeedEvents] = useState<FeedEvent[]>([]);

  const refreshQueue = useCallback(async () => {
    try {
      const [q, i] = await Promise.all([
        api.getReviewQueue(undefined, 100),
        api.listIntents().then(list => list.map(i => i.id)),
      ]);
      setReviewItems(q.items);
      setIntents(i);
      // Auto-select latest item if nothing selected
      setSelectedItem(prev => {
        if (!prev && q.items.length > 0) return q.items[0];
        // Keep selection if still in queue
        if (prev) return q.items.find(i => i.id === prev.id) ?? (q.items[0] ?? null);
        return null;
      });
    } catch { /* */ }
  }, []);

  useEffect(() => { refreshQueue(); }, [refreshQueue]);

  // SSE live feed
  useEffect(() => {
    const es = new EventSource('/api/events');
    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data) as FeedEvent;
        setFeedEvents(prev => [event, ...prev].slice(0, 200));
        // Refresh queue when worker finishes an item
        if (event.type === 'llm_done' || event.type === 'fix_applied' || event.type === 'escalated') {
          refreshQueue();
        }
      } catch { /* ignore parse errors */ }
    };
    es.onerror = () => {
      // EventSource auto-reconnects — no action needed
    };
    return () => es.close();
  }, [refreshQueue]);

  const onQueued = useCallback(() => {
    // Small delay so server has time to write the log entry
    setTimeout(refreshQueue, 400);
  }, [refreshQueue]);

  const onFixed = useCallback(() => {
    setSelectedItem(null);
    refreshQueue();
  }, [refreshQueue]);

  return (
    <div className="flex gap-0 h-[calc(100vh-4rem)] -mx-4 -mt-4">
      {/* ── Left panel ── */}
      <div className="w-[52%] min-w-0 border-r border-zinc-800 flex flex-col">
        {/* Mode tabs */}
        <div className="flex border-b border-zinc-800 flex-shrink-0">
          <button
            onClick={() => setMode('manual')}
            className={`px-5 py-3 text-xs font-semibold uppercase tracking-wide transition-colors ${
              mode === 'manual' ? 'text-white border-b-2 border-violet-500' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            Manual
          </button>
          <button
            onClick={() => setMode('simulate')}
            className={`px-5 py-3 text-xs font-semibold uppercase tracking-wide transition-colors ${
              mode === 'simulate' ? 'text-white border-b-2 border-violet-500' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            Simulate
          </button>
          {accuracy && (
            <div className="ml-auto flex items-center gap-3 px-4 text-xs">
              <span className="text-zinc-500">before <span className="text-zinc-300 font-mono">{accuracy.before}%</span></span>
              <span className="text-zinc-700">→</span>
              <span className="text-zinc-500">after <span className={`font-mono font-bold ${accuracy.after >= accuracy.before ? 'text-emerald-400' : 'text-red-400'}`}>{accuracy.after}%</span></span>
              {accuracy.after !== accuracy.before && (
                <span className={`font-bold ${accuracy.after > accuracy.before ? 'text-emerald-400' : 'text-red-400'}`}>
                  {accuracy.after > accuracy.before ? '+' : ''}{accuracy.after - accuracy.before}%
                </span>
              )}
            </div>
          )}
        </div>

        {mode === 'manual' ? (
          <ManualPanel onQueued={onQueued} />
        ) : (
          <SimulatePanel onQueued={onQueued} onAccuracy={setAccuracy} />
        )}
      </div>

      {/* ── Right panel ── */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Queue header */}
        <div className="px-4 py-3 border-b border-zinc-800 flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">
            Review Queue
          </span>
          {reviewItems.length > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 font-semibold">
              {reviewItems.length}
            </span>
          )}
          <div className="flex gap-1.5 flex-1 overflow-x-auto ml-2">
            {reviewItems.slice(0, 8).map(item => (
              <button
                key={item.id}
                onClick={() => { setSelectedItem(item); setSelectedFeedEvent(null); }}
                className={`flex-shrink-0 text-[9px] px-2 py-0.5 rounded border font-semibold uppercase transition-colors ${
                  selectedItem?.id === item.id && !selectedFeedEvent
                    ? 'bg-violet-500/20 border-violet-500/50 text-violet-300'
                    : item.flag === 'miss'
                    ? 'border-red-500/30 text-red-400 hover:bg-red-500/10'
                    : 'border-amber-500/30 text-amber-400 hover:bg-amber-500/10'
                }`}
              >
                {item.flag === 'low_confidence' ? 'LOW' : item.flag}
              </button>
            ))}
            {reviewItems.length > 8 && (
              <span className="text-[9px] text-zinc-600 self-center">+{reviewItems.length - 8} more</span>
            )}
          </div>
          <button onClick={refreshQueue} className="text-[10px] text-zinc-600 hover:text-zinc-400 flex-shrink-0">↺</button>
        </div>

        {/* Live feed */}
        {feedEvents.length > 0 && (
          <div className="border-b border-zinc-800 flex-shrink-0 max-h-36 overflow-y-auto">
            <div className="px-3 py-1.5 flex items-center gap-2">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Live</span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            </div>
            {feedEvents.slice(0, 30).map((ev, i) => (
              <button
                key={i}
                onClick={() => { setSelectedFeedEvent(ev); setSelectedItem(null); }}
                className={`w-full text-left px-3 py-1 text-[11px] font-mono hover:bg-zinc-800/60 transition-colors flex items-center gap-2 ${
                  selectedFeedEvent === ev ? 'bg-zinc-800/80' : ''
                }`}
              >
                <FeedEventDot event={ev} />
                <FeedEventLine event={ev} />
              </button>
            ))}
          </div>
        )}

        {/* Detail area */}
        <div className="flex-1 overflow-y-auto">
          {selectedFeedEvent ? (
            <FeedEventDetail event={selectedFeedEvent} onClose={() => setSelectedFeedEvent(null)} />
          ) : selectedItem ? (
            <ReviewDetail
              item={selectedItem}
              intents={intents}
              onFixed={onFixed}
              onDismiss={onFixed}
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center px-8">
              {reviewItems.length === 0 && feedEvents.length === 0 ? (
                <>
                  <div className="text-emerald-400 text-2xl mb-2">✓</div>
                  <div className="text-zinc-400 text-sm">Queue empty</div>
                  <div className="text-zinc-600 text-xs mt-1">Type a query on the left or run a simulation</div>
                </>
              ) : (
                <div className="text-zinc-600 text-sm">Select an item to review</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Feed helpers ─────────────────────────────────────────────────────────────

function FeedEventDot({ event }: { event: FeedEvent }) {
  const cls = (() => {
    switch (event.type) {
      case 'item_queued': return 'bg-amber-400';
      case 'llm_started': return 'bg-blue-400 animate-pulse';
      case 'llm_done': return event.phrases_added > 0 ? 'bg-emerald-400' : 'bg-zinc-500';
      case 'fix_applied': return 'bg-emerald-400';
      case 'escalated': return 'bg-red-400';
    }
  })();
  return <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cls}`} />;
}

function FeedEventLine({ event }: { event: FeedEvent }) {
  switch (event.type) {
    case 'item_queued':
      return <span className="text-zinc-400 truncate">queued #{event.id} <span className="text-zinc-600">{event.flag ?? 'ok'}</span> "{event.query.slice(0, 50)}"</span>;
    case 'llm_started':
      return <span className="text-blue-400 truncate">llm #{event.id} "{event.query.slice(0, 50)}"</span>;
    case 'llm_done':
      return <span className="text-zinc-300 truncate">#{event.id} +{event.phrases_added} phrases <span className="text-zinc-500">{event.summary.slice(0, 50)}</span></span>;
    case 'fix_applied':
      return <span className="text-emerald-400 truncate">fixed #{event.id} v{event.version_before}→v{event.version_after}</span>;
    case 'escalated':
      return <span className="text-red-400 truncate">escalated #{event.id} {event.reason.slice(0, 60)}</span>;
  }
}

function FeedEventDetail({ event, onClose }: { event: FeedEvent; onClose: () => void }) {
  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wide">{event.type.replace(/_/g, ' ')}</h3>
        <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 text-xs">✕ close</button>
      </div>

      {'query' in event && (
        <div>
          <div className="text-[10px] uppercase text-zinc-500 mb-1">Query</div>
          <div className="font-mono text-sm text-white bg-zinc-900 rounded px-3 py-2">{event.query}</div>
        </div>
      )}

      {event.type === 'item_queued' && (
        <div className="text-xs text-zinc-400">
          App: <span className="font-mono text-zinc-200">{event.app_id}</span>
          {event.flag && <> · Flag: <span className="text-amber-400">{event.flag}</span></>}
        </div>
      )}

      {event.type === 'llm_done' && (
        <div className="space-y-3">
          {event.correct.length > 0 && (
            <div>
              <div className="text-[10px] uppercase text-zinc-500 mb-1">Correct intents</div>
              <div className="flex flex-wrap gap-1">
                {event.correct.map(id => (
                  <span key={id} className="px-2 py-0.5 bg-emerald-500/15 text-emerald-400 rounded text-xs font-mono">{id}</span>
                ))}
              </div>
            </div>
          )}
          {event.wrong.length > 0 && (
            <div>
              <div className="text-[10px] uppercase text-zinc-500 mb-1">Wrong detections</div>
              <div className="flex flex-wrap gap-1">
                {event.wrong.map(id => (
                  <span key={id} className="px-2 py-0.5 bg-red-500/15 text-red-400 rounded text-xs font-mono">{id}</span>
                ))}
              </div>
            </div>
          )}
          <div className="text-xs text-zinc-400">
            Phrases added: <span className="text-white font-semibold">{event.phrases_added}</span>
          </div>
          {event.summary && (
            <div className="text-xs text-zinc-300 border-l-2 border-zinc-700 pl-3 italic">{event.summary}</div>
          )}
        </div>
      )}

      {event.type === 'fix_applied' && (
        <div className="space-y-2 text-xs">
          <div className="flex gap-4">
            <div>Phrases added: <span className="text-emerald-400 font-semibold">{event.phrases_added}</span></div>
            <div>Phrases replaced: <span className="text-blue-400 font-semibold">{event.phrases_replaced}</span></div>
          </div>
          <div className="text-zinc-400">
            Router version: <span className="font-mono text-zinc-200">{event.version_before}</span>
            <span className="text-zinc-600 mx-2">→</span>
            <span className="font-mono text-emerald-400">{event.version_after}</span>
          </div>
        </div>
      )}

      {event.type === 'escalated' && (
        <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/20 rounded px-3 py-2">
          {event.reason}
        </div>
      )}

      <div className="text-[10px] text-zinc-600 font-mono">id #{event.id}</div>
    </div>
  );
}

// ─── Manual Panel ─────────────────────────────────────────────────────────────

function ManualPanel({ onQueued }: { onQueued: () => void }) {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const push = (...msgs: Message[]) => setMessages(prev => [...prev, ...msgs]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const raw = input.trim();
    if (!raw) return;
    setInput('');
    inputRef.current?.focus();

    push({ type: 'query', text: raw });
    const t0 = performance.now();
    try {
      const result = await api.routeMulti(raw, 0.3);
      const latency = performance.now() - t0;
      push({ type: 'result', result, latency, query: raw });
      // Low confidence or no result → notify right panel
      const confirmed = result.confirmed ?? [];
      const candidates = result.candidates ?? [];
      if (confirmed.length === 0 || (confirmed[0]?.confidence === 'low')) {
        onQueued();
      } else if (candidates.length > 0 && confirmed.length === 0) {
        onQueued();
      }
      // Always notify so right panel stays fresh
      onQueued();
    } catch (err) {
      push({ type: 'error', text: String(err) });
    }
  };

  return (
    <div className="flex flex-col flex-1 min-h-0 p-4">
      <div className="flex-1 overflow-y-auto space-y-2 pb-3 min-h-0">
        {messages.length === 0 && (
          <div className="text-zinc-600 text-sm text-center py-16 px-4">
            Type a customer query to route it.<br />
            <span className="text-zinc-700 text-xs">Weak or missed routes appear in the review queue →</span>
          </div>
        )}
        {messages.map((msg, i) => (
          <ManualMessage key={i} msg={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 pt-3 border-t border-zinc-800 flex-shrink-0">
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type a customer query..."
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-white text-sm placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
          autoFocus
        />
        <button
          type="submit"
          className="px-4 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-medium transition-colors text-sm"
        >
          Route
        </button>
      </form>
    </div>
  );
}

function ManualMessage({ msg }: { msg: Message }) {
  if (msg.type === 'query') {
    return (
      <div className="font-mono text-sm text-white">
        <span className="text-violet-400">{'> '}</span>{msg.text}
      </div>
    );
  }
  if (msg.type === 'error') return <div className="text-red-400 text-sm pl-4">{msg.text}</div>;
  if (msg.type === 'learn') return <div className="text-emerald-400 text-sm pl-4">{msg.text}</div>;

  const { result, latency, query } = msg;
  const confirmed = result?.confirmed ?? [];
  const candidates = result?.candidates ?? [];
  const allIntents = [...confirmed, ...candidates];

  if (allIntents.length === 0) {
    return <div className="text-red-400/70 text-sm pl-4">No match — added to review queue</div>;
  }

  const bestScore = Math.max(...allIntents.map(i => i.score));

  return (
    <div className="pl-4 space-y-1">
      <HighlightedQuery query={query} intents={allIntents} />
      {confirmed.map((intent, i) => (
        <IntentRow key={intent.id} intent={intent} index={i} bestScore={bestScore} isMulti={allIntents.length > 1} />
      ))}
      {candidates.length > 0 && (
        <>
          <div className="text-[9px] text-zinc-600 uppercase font-semibold pl-1 pt-0.5">Candidates</div>
          {candidates.map((intent, i) => (
            <IntentRow key={intent.id} intent={intent} index={confirmed.length + i} bestScore={bestScore} isMulti={allIntents.length > 1} />
          ))}
        </>
      )}
      <div className="text-zinc-600 text-xs pl-1">
        <span className="text-emerald-800">router {result.routing_us != null ? (result.routing_us < 1000 ? `${result.routing_us}µs` : `${(result.routing_us / 1000).toFixed(1)}ms`) : '—'}</span>
        <span className="text-zinc-800"> · </span>
        <span>http {latency.toFixed(0)}ms</span>
      </div>
    </div>
  );
}

// ─── Simulate Panel ───────────────────────────────────────────────────────────

function SimulatePanel({ onQueued, onAccuracy }: {
  onQueued: () => void;
  onAccuracy: (a: { before: number; after: number }) => void;
}) {
  const [turns, setTurns] = useState(10);
  const [personality, setPersonality] = useState('casual');
  const [sophistication, setSophistication] = useState('medium');
  const [verbosity, setVerbosity] = useState('short');
  const [scenario, setScenario] = useState('');
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<SimPhase>('idle');
  const [phaseLabel, setPhaseLabel] = useState('');
  const [before, setBefore] = useState<number | null>(null);
  const [after, setAfter] = useState<number | null>(null);
  const [stats, setStats] = useState<{ generated: number; failures: number; fixed: number; stuck: number } | null>(null);
  const stopRef = useRef(false);

  const run = async () => {
    stopRef.current = false;
    setRunning(true);
    setBefore(null); setAfter(null); setStats(null);

    try {
      // Generate
      setPhase('generating'); setPhaseLabel('Generating queries with LLM...');
      const generated = await api.trainingGenerate({
        personality, sophistication, verbosity,
        turns, scenario: scenario || undefined,
      });
      if (stopRef.current) { setRunning(false); return; }

      // Baseline
      setPhase('baseline'); setPhaseLabel('Running baseline...');
      const queries = generated.turns.map((t: any) => ({
        message: t.customer_message,
        ground_truth: t.ground_truth as string[],
      }));
      const baseline = await api.trainingRun(queries);
      const beforeCorrect = baseline.results.filter((r: any) => r.status === 'pass').length;
      const beforePct = Math.round((beforeCorrect / queries.length) * 100);
      setBefore(beforePct);
      if (stopRef.current) { setRunning(false); return; }

      // Fix failures
      setPhase('fixing'); setPhaseLabel('Reporting failures to auto-learn...');
      await api.setReviewMode('auto');
      let failures = 0;
      for (let i = 0; i < baseline.results.length; i++) {
        const r = baseline.results[i];
        if (r.status === 'pass') continue;
        failures++;
        const detected = [...r.confirmed, ...r.candidates];
        await api.report(r.message, detected, r.missed?.length > 0 ? 'miss' : 'low_confidence');
        if (failures % 3 === 0) {
          onQueued();
          await new Promise(res => setTimeout(res, 1500));
        }
      }
      if (failures > 0) {
        onQueued();
        await new Promise(res => setTimeout(res, 2500));
      }
      await api.setReviewMode('manual');
      if (stopRef.current) { setRunning(false); return; }

      // Retest
      setPhase('retesting'); setPhaseLabel('Re-testing after fixes...');
      const retest = await api.trainingRun(queries);
      const afterCorrect = retest.results.filter((r: any) => r.status === 'pass').length;
      const afterPct = Math.round((afterCorrect / queries.length) * 100);
      const fixed = retest.results.filter((r: any, i: number) => r.status === 'pass' && baseline.results[i].status !== 'pass').length;
      const stuck = retest.results.filter((r: any, i: number) => r.status !== 'pass' && baseline.results[i].status !== 'pass').length;

      setAfter(afterPct);
      setStats({ generated: queries.length, failures, fixed, stuck });
      onAccuracy({ before: beforePct, after: afterPct });
      onQueued();
      setPhase('done'); setPhaseLabel('');

    } catch (err) {
      setPhase('error'); setPhaseLabel((err as Error).message);
      api.setReviewMode('manual').catch(() => {});
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col flex-1 min-h-0 p-4 space-y-4 overflow-y-auto">
      {/* Config */}
      <div className="space-y-3">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-xs text-zinc-500">Queries</label>
            <select value={turns} onChange={e => setTurns(Number(e.target.value))} disabled={running}
              className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
              {[5, 10, 15, 20, 30].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-zinc-500">Personality</label>
            <select value={personality} onChange={e => setPersonality(e.target.value)} disabled={running}
              className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
              {PERSONALITIES.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-zinc-500">Sophistication</label>
            <select value={sophistication} onChange={e => setSophistication(e.target.value)} disabled={running}
              className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
              {['low', 'medium', 'high'].map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-zinc-500">Verbosity</label>
            <select value={verbosity} onChange={e => setVerbosity(e.target.value)} disabled={running}
              className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
              {['short', 'medium', 'long'].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
        </div>
        <input
          value={scenario}
          onChange={e => setScenario(e.target.value)}
          disabled={running}
          placeholder="Scenario (optional): e.g. angry customer trying to return a broken laptop"
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-xs text-white placeholder-zinc-600 focus:border-violet-500 focus:outline-none"
        />
        <div className="flex items-center gap-3">
          <button onClick={run} disabled={running}
            className="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded disabled:opacity-30 transition-colors">
            {running ? 'Running...' : 'Run Simulation'}
          </button>
          {running && (
            <button onClick={() => { stopRef.current = true; }} className="text-xs text-red-400 hover:text-red-300">Stop</button>
          )}
        </div>
      </div>

      {/* Phase indicator */}
      {phase !== 'idle' && (
        <div className="flex items-center gap-2">
          {running && <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin flex-shrink-0" />}
          <span className={`text-xs ${phase === 'error' ? 'text-red-400' : phase === 'done' ? 'text-emerald-400' : 'text-violet-400'}`}>
            {phase === 'done' ? 'Complete' : phase === 'error' ? `Error: ${phaseLabel}` : phaseLabel || phase}
          </span>
        </div>
      )}

      {/* Results */}
      {(before !== null || after !== null) && (
        <div className="bg-zinc-800/60 border border-zinc-700 rounded-lg p-4 space-y-3">
          {before !== null && (
            <div>
              <div className="text-[10px] text-zinc-500 mb-1.5">Before</div>
              <div className="flex items-center gap-3">
                <div className="flex-1 h-5 bg-zinc-700 rounded-full overflow-hidden">
                  <div className="h-full bg-zinc-500 rounded-full transition-all duration-700" style={{ width: `${before}%` }} />
                </div>
                <span className="text-sm font-mono text-zinc-400 w-10 text-right">{before}%</span>
              </div>
            </div>
          )}
          {after !== null && (
            <div>
              <div className="text-[10px] text-zinc-500 mb-1.5">After</div>
              <div className="flex items-center gap-3">
                <div className="flex-1 h-5 bg-zinc-700 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full transition-all duration-700" style={{ width: `${after}%` }} />
                </div>
                <span className="text-sm font-mono text-emerald-400 w-10 text-right">{after}%</span>
              </div>
            </div>
          )}
          {stats && (
            <div className="flex gap-5 pt-2 border-t border-zinc-700 text-xs">
              <span><span className="text-white font-semibold">{stats.generated}</span> <span className="text-zinc-500">generated</span></span>
              <span><span className="text-amber-400 font-semibold">{stats.failures}</span> <span className="text-zinc-500">failures</span></span>
              <span><span className="text-emerald-400 font-semibold">{stats.fixed}</span> <span className="text-zinc-500">fixed</span></span>
              {stats.stuck > 0 && <span><span className="text-red-400 font-semibold">{stats.stuck}</span> <span className="text-zinc-500">stuck</span></span>}
            </div>
          )}
        </div>
      )}

      {phase === 'done' && (
        <button onClick={run} disabled={running}
          className="text-xs px-3 py-1.5 border border-violet-500/50 text-violet-400 rounded hover:bg-violet-500/10 transition-colors">
          Run again (keep improving)
        </button>
      )}
    </div>
  );
}

// ─── Copy Analysis Button ─────────────────────────────────────────────────────

function CopyAnalysisButton({ item, analysis }: { item: ReviewItem; analysis: ReviewAnalyzeResult }) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    const lines: string[] = [
      `Query: "${item.query}"`,
      `Flag: ${item.flag}`,
      `Detected: ${item.detected.length > 0 ? item.detected.join(', ') : 'none'}`,
      `Wrong: ${analysis.wrong_detections.length > 0 ? analysis.wrong_detections.join(', ') : 'none'}`,
      `Should be: ${analysis.correct_intents.length > 0 ? analysis.correct_intents.join(', ') : 'no matching intent'}`,
    ];

    if (analysis.phrases_to_replace.length > 0) {
      lines.push('');
      lines.push('Suggested replacements:');
      for (const r of analysis.phrases_to_replace) {
        lines.push(`  ${r.intent}: "${r.old_phrase}" → "${r.new_phrase}"`);
        if (r.reason) lines.push(`    reason: ${r.reason}`);
      }
    }

    if (analysis.summary) {
      lines.push('');
      lines.push(`Summary: ${analysis.summary}`);
    }

    navigator.clipboard.writeText(lines.join('\n'));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button onClick={copy} className="text-[10px] text-zinc-500 hover:text-violet-400 transition-colors px-2 py-0.5 border border-zinc-800 rounded hover:border-violet-500/40">
      {copied ? '✓ copied' : 'copy report'}
    </button>
  );
}

// ─── Review Detail (right panel) ─────────────────────────────────────────────

interface PhraseEntry { phrase: string; lang: string; }
interface IntentBlock { intentId: string; phrases: PhraseEntry[]; }

function ReviewDetail({ item, intents, onFixed, onDismiss }: {
  item: ReviewItem;
  intents: string[];
  onFixed: () => void;
  onDismiss: () => void;
}) {
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<ReviewAnalyzeResult | null>(null);
  const [blocks, setBlocks] = useState<IntentBlock[]>([]);
  const [wrongSeeds, setWrongSeeds] = useState<Record<string, string[]>>({});
  const { settings: studioSettings } = useAppStore();
  const enabledLangs = studioSettings.languages;

  useEffect(() => { setAnalysis(null); setBlocks([]); setWrongSeeds({}); }, [item.id]);

  const runAnalysis = async () => {
    setAnalyzing(true);
    try {
      const result = await api.reviewAnalyze(item.id);
      setAnalysis(result);
      setBlocks(Object.entries(result.phrases_to_add).map(([intentId, phraseList]) => ({
        intentId,
        phrases: phraseList.map(s => ({ phrase: s, lang: result.languages[0] || 'en' })),
      })));
      if (result.wrong_detections.length > 0) {
        setWrongSeeds(await api.reviewIntentPhrases(result.wrong_detections));
      }
    } catch (e) {
      alert('Analysis failed: ' + (e instanceof Error ? e.message : String(e)));
    } finally {
      setAnalyzing(false);
    }
  };

  const setBlockIntent = (i: number, id: string) =>
    setBlocks(prev => prev.map((b, idx) => idx === i ? { ...b, intentId: id } : b));
  const setBlockPhrase = (bi: number, si: number, val: string) =>
    setBlocks(prev => prev.map((b, i) => {
      if (i !== bi) return b;
      const phrases = [...b.phrases]; phrases[si] = { ...phrases[si], phrase: val }; return { ...b, phrases };
    }));
  const addPhraseToBlock = (bi: number) =>
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
    const result = await api.reviewFix(item.id, toApply);
    const msgs = [`Applied ${result.added} phrases.`];
    if (result.resolved_count > 0) msgs.push(`${result.resolved_count} failures resolved.`);
    if (result.blocked.length > 0) msgs.push(`Blocked: ${result.blocked.map((b: any) => `"${b.phrase}"`).join(', ')}`);
    alert(msgs.join(' '));
    onFixed();
  };

  const totalPhrases = blocks.flatMap(b => b.phrases).filter(s => s.phrase.trim()).length;
  const usedIntents = new Set(blocks.map(b => b.intentId).filter(Boolean));

  return (
    <div className="p-5 space-y-4">
      {/* Flag + query */}
      <div className="flex items-center gap-2">
        <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${
          item.flag === 'miss' ? 'bg-red-900/30 text-red-400' :
          item.flag === 'low_confidence' ? 'bg-amber-900/30 text-amber-400' : 'bg-blue-900/30 text-blue-400'
        }`}>{item.flag.replace('_', ' ')}</span>
        {item.detected.length > 0 && (
          <span className="text-xs text-zinc-500">detected: {item.detected.join(', ')}</span>
        )}
      </div>

      <div className="bg-zinc-800 rounded-lg p-3">
        <div className="text-[10px] text-zinc-500 mb-1">Query</div>
        <div className="text-white font-mono text-sm">"{item.query}"</div>
      </div>

      {/* Analysis */}
      {analysis && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wide">AI Analysis</div>
            <CopyAnalysisButton item={item} analysis={analysis} />
          </div>
          <div className="bg-zinc-800/50 rounded-lg p-3">
            <div className="grid grid-cols-2 gap-3">
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
                    : <span className="text-[10px] text-zinc-600 italic">no matching intent</span>
                  }
                </div>
              </div>
            </div>
          </div>

          {analysis.phrases_to_replace.map((r, i) => (
            <div key={i} className="bg-red-900/10 border border-red-800/30 rounded px-3 py-2 text-xs space-y-1">
              <span className="text-red-400 font-mono font-semibold">{r.intent}</span>
              <span className="text-zinc-500"> — replace </span>
              <span className="text-red-400 font-mono line-through">"{r.old_phrase}"</span>
              <span className="text-zinc-500"> → </span>
              <span className="text-emerald-400 font-mono">"{r.new_phrase}"</span>
              {wrongSeeds[r.intent] && (() => {
                const all = wrongSeeds[r.intent];
                const idx = all.indexOf(r.old_phrase);
                const nearby = idx === -1
                  ? all.slice(0, 3)
                  : all.slice(Math.max(0, idx - 1), idx + 2);
                return (
                  <div className="flex flex-wrap gap-1 mt-1 items-center">
                    {idx > 1 && <span className="text-[9px] text-zinc-700">···</span>}
                    {nearby.map((s, si) => (
                      <span key={si} className={`text-[9px] font-mono px-1 py-0.5 rounded ${
                        s === r.old_phrase ? 'bg-red-900/30 text-red-400 border border-red-800' : 'bg-zinc-800 text-zinc-500'
                      }`}>{s}</span>
                    ))}
                    {idx !== -1 && idx < all.length - 2 && <span className="text-[9px] text-zinc-700">···</span>}
                    <span className="text-[9px] text-zinc-700 ml-1">{all.length} phrases total</span>
                  </div>
                );
              })()}
            </div>
          ))}

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

      {/* Phrase blocks */}
      <div className="space-y-2">
        {blocks.length > 0 && (
          <div className="text-[10px] text-zinc-500 uppercase font-semibold">Phrases to add</div>
        )}
        {blocks.map((block, bi) => (
          <div key={bi} className="bg-zinc-800 border border-zinc-700 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <select value={block.intentId} onChange={e => setBlockIntent(bi, e.target.value)}
                className="flex-1 bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1 font-mono focus:border-violet-500 focus:outline-none">
                <option value="">Select intent...</option>
                {intents.filter(id => !usedIntents.has(id) || id === block.intentId).map(id => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
              <button onClick={() => removeBlock(bi)} className="text-[10px] text-zinc-600 hover:text-red-400">remove</button>
            </div>
            {block.phrases.map((entry, si) => (
              <div key={si} className="flex gap-1.5 mb-1">
                <select value={entry.lang} onChange={e => setBlocks(prev => prev.map((b, i) => {
                  if (i !== bi) return b;
                  const phrases = [...b.phrases]; phrases[si] = { ...phrases[si], lang: e.target.value }; return { ...b, phrases };
                }))} className="bg-zinc-900 border border-zinc-700 text-violet-400 text-[10px] rounded px-1 py-1 w-12 focus:outline-none">
                  {enabledLangs.map(lang => <option key={lang} value={lang}>{lang.toUpperCase()}</option>)}
                </select>
                <input value={entry.phrase} onChange={e => setBlockPhrase(bi, si, e.target.value)}
                  placeholder="example phrase"
                  className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-white font-mono focus:border-violet-500 focus:outline-none" />
                {block.phrases.length > 1 && (
                  <button onClick={() => removePhraseFromBlock(bi, si)} className="text-zinc-600 hover:text-red-400 text-xs">×</button>
                )}
              </div>
            ))}
            <button onClick={() => addPhraseToBlock(bi)} className="text-[9px] text-zinc-500 hover:text-violet-400 mt-1">+ phrase</button>
          </div>
        ))}
        <button onClick={() => setBlocks(prev => [...prev, { intentId: '', phrases: [{ phrase: '', lang: 'en' }] }])}
          className="w-full py-2 text-xs text-zinc-600 hover:text-violet-400 border border-dashed border-zinc-800 hover:border-violet-500/40 rounded-lg transition-colors">
          + Add intent block
        </button>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 pt-2 border-t border-zinc-800">
        <button onClick={runAnalysis} disabled={analyzing}
          className="text-xs px-3 py-1.5 border border-violet-500/50 text-violet-400 rounded hover:bg-violet-500/10 disabled:opacity-40 transition-colors">
          {analyzing ? 'Analyzing...' : analysis ? 'Re-analyze' : 'Analyze with AI'}
        </button>
        <div className="flex-1" />
        <button onClick={async () => { await api.reviewReject(item.id); onDismiss(); }}
          className="text-xs px-3 py-1.5 border border-zinc-700 text-zinc-500 rounded hover:text-white transition-colors">
          Dismiss
        </button>
        <button onClick={handleApply} disabled={totalPhrases === 0}
          className="text-xs px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded disabled:opacity-30 transition-colors">
          Apply ({totalPhrases})
        </button>
      </div>
    </div>
  );
}

// ─── Shared components ────────────────────────────────────────────────────────

function IntentRow({ intent, index, bestScore, isMulti }: {
  intent: { id: string; score: number; intent_type: string; span: [number, number]; confidence?: string; source?: string };
  index: number; bestScore: number; isMulti: boolean;
}) {
  const relativeScore = intent.score / bestScore;
  const isWeak = isMulti && relativeScore < 0.3;
  const color = INTENT_COLORS[index % INTENT_COLORS.length];
  const bg = INTENT_BG_COLORS[index % INTENT_BG_COLORS.length];
  const conf = intent.confidence || 'low';
  const confStyle = conf === 'high' ? 'text-emerald-400 border-emerald-400/40' : conf === 'medium' ? 'text-amber-400 border-amber-400/40' : 'text-zinc-400 border-zinc-500/40';

  return (
    <div className={`flex items-center gap-2 font-mono text-xs px-2 py-1 rounded ${bg} ${isWeak ? 'opacity-40' : ''}`}>
      <span className={`text-[9px] px-1 py-0.5 rounded border font-bold uppercase ${confStyle}`}>{conf}</span>
      <span className={`font-semibold ${color}`}>{intent.id}</span>
      <span className="text-amber-400">{intent.score.toFixed(2)}</span>
      {isWeak && <span className="text-zinc-600 text-[9px]">weak</span>}
    </div>
  );
}

function HighlightedQuery({ query, intents }: { query: string; intents: { id: string; span: [number, number] }[] }) {
  const charMap = new Array(query.length).fill(-1);
  intents.forEach((intent, idx) => {
    const [start, end] = intent.span;
    for (let i = Math.max(0, start); i < Math.min(query.length, end); i++) charMap[i] = idx;
  });
  const segments: { text: string; intentIdx: number }[] = [];
  let i = 0;
  while (i < query.length) {
    const cur = charMap[i]; let j = i + 1;
    while (j < query.length && charMap[j] === cur) j++;
    segments.push({ text: query.slice(i, j), intentIdx: cur }); i = j;
  }
  return (
    <div className="font-mono text-sm leading-relaxed mb-1">
      {segments.map((seg, i) =>
        seg.intentIdx === -1
          ? <span key={i} className="text-zinc-500">{seg.text}</span>
          : <span key={i} className={`${INTENT_COLORS[seg.intentIdx % INTENT_COLORS.length]} font-semibold`}>{seg.text}</span>
      )}
    </div>
  );
}
