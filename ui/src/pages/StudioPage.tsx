import { useState, useEffect, useCallback } from 'react';
import { api, type MultiRouteOutput, type ReviewItem, type ReviewAnalyzeResult } from '@/api/client';
import { useAppStore } from '@/store';

// ─── Types ────────────────────────────────────────────────────────────────────

type Message =
  | { type: 'query'; text: string }
  | { type: 'result'; result: MultiRouteOutput; latency: number; query: string }
  | { type: 'learning'; query: string }
  | { type: 'learned'; query: string; phrases_added: number; summary: string }
  | { type: 'error'; text: string };

type FeedEvent =
  | { type: 'item_queued';  id: number; query: string; app_id: string; flag: string | null }
  | { type: 'llm_started';  id: number; query: string }
  | { type: 'llm_done';     id: number; correct: string[]; wrong: string[]; phrases_added: number; summary: string }
  | { type: 'fix_applied';  id: number; phrases_added: number; phrases_replaced: number; version_before: number; version_after: number }
  | { type: 'escalated';    id: number; reason: string };

type Tab = 'manual' | 'simulate' | 'review' | 'auto';
type SimPhase = 'idle' | 'generating' | 'baseline' | 'fixing' | 'retesting' | 'done' | 'error';

// Simulate: per-failure learn result shown inline
type LearnResult = { query: string; phrases_added: number; summary: string; missed: string[]; wrong: string[] };
// Simulate: per-turn retest outcome
type TurnOutcome = { message: string; ground_truth: string[]; before: string; after: string; confirmed: string[]; missed: string[]; extra: string[] };

const PERSONALITIES = ['casual', 'frustrated', 'formal', 'terse', 'rambling', 'polite'];
const LANGUAGES     = ['English', 'Spanish', 'French', 'Chinese', 'German', 'Japanese', 'Portuguese', 'Arabic'];

const INTENT_COLORS    = ['text-emerald-400','text-blue-400','text-amber-400','text-pink-400','text-cyan-400','text-violet-400','text-orange-400','text-lime-400'];
const INTENT_BG_COLORS = ['bg-emerald-400/20','bg-blue-400/20','bg-amber-400/20','bg-pink-400/20','bg-cyan-400/20','bg-violet-400/20','bg-orange-400/20','bg-lime-400/20'];

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function StudioPage() {
  const [tab, setTab]                             = useState<Tab>('manual');
  const [reviewItems, setReviewItems]             = useState<ReviewItem[]>([]);
  const [selectedItem, setSelectedItem]           = useState<ReviewItem | null>(null);
  const [selectedFeedEvent, setSelectedFeedEvent] = useState<FeedEvent | null>(null);
  const [intents, setIntents]                     = useState<string[]>([]);
  const [accuracy, setAccuracy]                   = useState<{ before: number; after: number } | null>(null);
  const [feedEvents, setFeedEvents]               = useState<FeedEvent[]>([]);

  const refreshQueue = useCallback(async () => {
    try {
      const [q, i] = await Promise.all([
        api.getReviewQueue(undefined, 100),
        api.listIntents().then(list => list.map(i => i.id)),
      ]);
      setReviewItems(q.items);
      setIntents(i);
      setSelectedItem(prev => {
        if (!prev && q.items.length > 0) return q.items[0];
        if (prev) return q.items.find(i => i.id === prev.id) ?? (q.items[0] ?? null);
        return null;
      });
    } catch { /* */ }
  }, []);

  useEffect(() => { refreshQueue(); }, [refreshQueue]);

  // SSE — shared across all tabs
  useEffect(() => {
    const es = new EventSource('/api/events');
    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data) as FeedEvent;
        setFeedEvents(prev => [event, ...prev].slice(0, 300));
        if (event.type === 'llm_done' || event.type === 'fix_applied' || event.type === 'escalated') {
          refreshQueue();
        }
      } catch { /* */ }
    };
    return () => es.close();
  }, [refreshQueue]);

  const onQueued  = useCallback(() => { setTimeout(refreshQueue, 400); }, [refreshQueue]);
  const onFixed   = useCallback(() => { setSelectedItem(null); refreshQueue(); }, [refreshQueue]);
  const selectItem = useCallback((item: ReviewItem) => {
    setSelectedItem(item); setSelectedFeedEvent(null);
  }, []);

  // Worker events = id !== 0; learn_now events = id === 0
  const workerEvents = feedEvents.filter(e => e.id !== 0);

  return (
    <div className="flex gap-0 h-[calc(100vh-4rem)] -mx-4 -mt-4">

      {/* ── Left panel ── */}
      <div className="w-[52%] min-w-0 border-r border-zinc-800 flex flex-col">

        {/* Tab bar */}
        <div className="flex border-b border-zinc-800 flex-shrink-0">
          {(['manual', 'simulate', 'review', 'auto'] as Tab[]).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-4 py-3 text-xs font-semibold uppercase tracking-wide transition-colors ${
                tab === t ? 'text-white border-b-2 border-violet-500' : 'text-zinc-500 hover:text-zinc-300'
              }`}>
              {t}
              {t === 'review' && reviewItems.length > 0 && (
                <span className="ml-1.5 text-[9px] px-1 py-0.5 rounded-full bg-amber-500/20 text-amber-400 font-bold">
                  {reviewItems.length}
                </span>
              )}
              {t === 'auto' && workerEvents.length > 0 && (
                <span className="ml-1.5 w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
              )}
            </button>
          ))}

          {accuracy && (
            <div className="ml-auto flex items-center gap-2 px-3 text-xs">
              <span className="text-zinc-600">{accuracy.before}%</span>
              <span className="text-zinc-700">→</span>
              <span className={`font-bold ${accuracy.after > accuracy.before ? 'text-emerald-400' : accuracy.after < accuracy.before ? 'text-red-400' : 'text-zinc-400'}`}>
                {accuracy.after}%
                {accuracy.after !== accuracy.before && (
                  <span className="ml-1 text-[10px]">{accuracy.after > accuracy.before ? '+' : ''}{accuracy.after - accuracy.before}</span>
                )}
              </span>
            </div>
          )}
        </div>

        {/* Tab content */}
        {tab === 'manual'   && <ManualPanel onQueued={onQueued} />}
        {tab === 'simulate' && <SimulatePanel onAccuracy={setAccuracy} />}
        {tab === 'review'   && <ReviewPanel items={reviewItems} selectedId={selectedItem?.id ?? null} onSelect={selectItem} onRefresh={refreshQueue} />}
        {tab === 'auto'     && <AutoPanel workerEvents={workerEvents} />}
      </div>

      {/* ── Right panel ── */}
      <div className="flex-1 min-w-0 flex flex-col">

        {/* Review tab: queue item strip — only shown on Review tab */}
        {tab === 'review' && (
          <div className="px-4 py-2.5 border-b border-zinc-800 flex items-center gap-2 flex-shrink-0">
            <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide flex-shrink-0">Queue</span>
            {reviewItems.length > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 font-semibold flex-shrink-0">
                {reviewItems.length}
              </span>
            )}
            <div className="flex gap-1.5 flex-1 overflow-x-auto">
              {reviewItems.slice(0, 8).map(item => (
                <button key={item.id}
                  onClick={() => { setSelectedItem(item); setSelectedFeedEvent(null); }}
                  className={`flex-shrink-0 text-[9px] px-2 py-0.5 rounded border font-semibold uppercase transition-colors ${
                    selectedItem?.id === item.id && !selectedFeedEvent
                      ? 'bg-violet-500/20 border-violet-500/50 text-violet-300'
                      : item.flag === 'miss'
                      ? 'border-red-500/30 text-red-400 hover:bg-red-500/10'
                      : 'border-amber-500/30 text-amber-400 hover:bg-amber-500/10'
                  }`}>
                  {item.flag === 'low_confidence' ? 'LOW' : item.flag}
                </button>
              ))}
              {reviewItems.length > 8 && (
                <span className="text-[9px] text-zinc-600 self-center">+{reviewItems.length - 8}</span>
              )}
            </div>
            <button onClick={refreshQueue} className="text-[10px] text-zinc-600 hover:text-zinc-400 flex-shrink-0">↺</button>
          </div>
        )}

        {/* Live SSE feed — filtered per tab */}
        {(() => {
          // manual/simulate: only learn_now events (id=0), skip fix_applied (redundant with llm_done)
          // auto: worker events only (id≠0)
          // review: all events
          const visibleEvents = feedEvents.filter(ev => {
            if (tab === 'manual' || tab === 'simulate') return ev.id === 0 && ev.type !== 'fix_applied';
            if (tab === 'auto') return ev.id !== 0;
            return true;
          });
          if (visibleEvents.length === 0) return null;
          return (
            <div className="border-b border-zinc-800 flex-shrink-0 max-h-40 overflow-y-auto">
              <div className="px-3 py-1.5 flex items-center gap-2 sticky top-0 bg-zinc-950/80 backdrop-blur">
                <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-600">Live</span>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              </div>
              {visibleEvents.slice(0, 30).map((ev, i) => (
                <button key={i}
                  onClick={() => { setSelectedFeedEvent(ev); setSelectedItem(null); }}
                  className={`w-full text-left px-3 py-1 text-[11px] font-mono hover:bg-zinc-800/60 transition-colors flex items-center gap-2 ${selectedFeedEvent === ev ? 'bg-zinc-800/80' : ''}`}>
                  <FeedEventDot event={ev} />
                  <FeedEventLine event={ev} />
                </button>
              ))}
            </div>
          );
        })()}

        {/* Detail area */}
        <div className="flex-1 overflow-y-auto">
          {selectedFeedEvent ? (
            <FeedEventDetail event={selectedFeedEvent} onClose={() => setSelectedFeedEvent(null)} />
          ) : selectedItem && tab === 'review' ? (
            <ReviewDetail item={selectedItem} intents={intents} onFixed={onFixed} onDismiss={onFixed} />
          ) : (
            <RightPanelEmpty tab={tab} hasEvents={feedEvents.length > 0} />
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Right panel empty state ─────────────────────────────────────────────────

function RightPanelEmpty({ tab, hasEvents }: { tab: Tab; hasEvents: boolean }) {
  const msgs: Record<Tab, { icon: string; title: string; sub: string }> = {
    manual:   { icon: '→', title: 'Route a query', sub: 'Type something in the left panel. Click "Learn Now" on weak results to teach the system — events appear here.' },
    simulate: { icon: '◎', title: 'Run a simulation', sub: 'Configure and run a simulation. Each failure will be learned from in real time — SSE events stream here as it learns.' },
    review:   { icon: '✓', title: 'Queue empty', sub: 'No flagged items to review. Production traffic misses and low-confidence results appear here for manual analysis.' },
    auto:     { icon: '⟳', title: 'Worker idle', sub: 'Turn on Auto-Learn in the left panel to start processing the queue. Worker events stream here live.' },
  };
  if (hasEvents) return <div className="flex items-center justify-center h-full text-zinc-600 text-sm">Select an event to inspect</div>;
  const m = msgs[tab];
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8 gap-2">
      <div className="text-zinc-600 text-2xl mb-1">{m.icon}</div>
      <div className="text-zinc-400 text-sm font-medium">{m.title}</div>
      <div className="text-zinc-600 text-xs max-w-xs leading-relaxed">{m.sub}</div>
    </div>
  );
}

// ─── Feed helpers ─────────────────────────────────────────────────────────────

function FeedEventDot({ event }: { event: FeedEvent }) {
  const cls = (() => {
    switch (event.type) {
      case 'item_queued': return 'bg-amber-400';
      case 'llm_started': return 'bg-blue-400 animate-pulse';
      case 'llm_done':    return event.phrases_added > 0 ? 'bg-emerald-400' : 'bg-zinc-500';
      case 'fix_applied': return 'bg-emerald-400';
      case 'escalated':   return 'bg-red-400';
    }
  })();
  return <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cls}`} />;
}

function FeedEventLine({ event }: { event: FeedEvent }) {
  const src = event.id === 0 ? '' : <span className="text-zinc-700 mr-1">#{event.id}</span>;
  switch (event.type) {
    case 'item_queued': return <span className="text-zinc-400 truncate">{src}queued [{event.flag ?? 'ok'}] "{event.query.slice(0, 50)}"</span>;
    case 'llm_started': return <span className="text-blue-400 truncate">{src}analyzing "{event.query.slice(0, 50)}"</span>;
    case 'llm_done':    return <span className="text-zinc-300 truncate">{src}+{event.phrases_added} phrases <span className="text-zinc-500">{event.summary.slice(0, 50)}</span></span>;
    case 'fix_applied': return <span className="text-emerald-400 truncate">{src}+{event.phrases_added} phrases learned{event.phrases_replaced > 0 ? `, ${event.phrases_replaced} suppressors` : ''}</span>;
    case 'escalated':   return <span className="text-red-400 truncate">{src}escalated {event.reason.slice(0, 60)}</span>;
  }
}

function FeedEventDetail({ event, onClose }: { event: FeedEvent; onClose: () => void }) {
  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wide">{event.type.replace(/_/g, ' ')}</h3>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500 font-mono">
            {event.id === 0 ? 'learn_now' : `worker #${event.id}`}
          </span>
        </div>
        <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 text-xs">✕</button>
      </div>
      {'query' in event && (
        <div>
          <div className="text-[10px] uppercase text-zinc-500 mb-1">Query</div>
          <div className="font-mono text-sm text-white bg-zinc-900 rounded px-3 py-2">{event.query}</div>
        </div>
      )}
      {event.type === 'item_queued' && (
        <div className="text-xs text-zinc-400">
          Namespace: <span className="font-mono text-zinc-200">{event.app_id}</span>
          {event.flag && <> · Flag: <span className="text-amber-400">{event.flag}</span></>}
        </div>
      )}
      {event.type === 'llm_done' && (
        <div className="space-y-3">
          {event.correct.length > 0 && (
            <div>
              <div className="text-[10px] uppercase text-zinc-500 mb-1">Correct intents</div>
              <div className="flex flex-wrap gap-1">
                {event.correct.map(id => <span key={id} className="px-2 py-0.5 bg-emerald-500/15 text-emerald-400 rounded text-xs font-mono">{id}</span>)}
              </div>
            </div>
          )}
          {event.wrong.length > 0 && (
            <div>
              <div className="text-[10px] uppercase text-zinc-500 mb-1">Wrong detections</div>
              <div className="flex flex-wrap gap-1">
                {event.wrong.map(id => <span key={id} className="px-2 py-0.5 bg-red-500/15 text-red-400 rounded text-xs font-mono">{id}</span>)}
              </div>
            </div>
          )}
          <div className="text-xs text-zinc-400">Phrases added: <span className="text-white font-semibold">{event.phrases_added}</span></div>
          {event.summary && <div className="text-xs text-zinc-300 border-l-2 border-zinc-700 pl-3 italic">{event.summary}</div>}
        </div>
      )}
      {event.type === 'fix_applied' && (
        <div className="space-y-2 text-xs">
          <div className="flex gap-4">
            <div>Phrases added: <span className="text-emerald-400 font-semibold">{event.phrases_added}</span></div>
            {event.phrases_replaced > 0 && <div>Suppressors: <span className="text-blue-400 font-semibold">{event.phrases_replaced}</span></div>}
          </div>
        </div>
      )}
      {event.type === 'escalated' && (
        <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/20 rounded px-3 py-2">{event.reason}</div>
      )}
    </div>
  );
}

// ─── Manual Panel ─────────────────────────────────────────────────────────────

function ManualPanel({ onQueued }: { onQueued: () => void }) {
  const [input, setInput]     = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const bottomRef = { current: null as HTMLDivElement | null };
  const inputRef  = { current: null as HTMLInputElement | null };

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const push = (...msgs: Message[]) => setMessages(prev => [...prev, ...msgs]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const raw = input.trim();
    if (!raw) return;
    setInput('');
    push({ type: 'query', text: raw });
    const t0 = performance.now();
    try {
      const result = await api.routeMulti(raw, 0.3, false); // log=false — no queue pollution
      push({ type: 'result', result, latency: performance.now() - t0, query: raw });
      onQueued();
    } catch (err) {
      push({ type: 'error', text: String(err) });
    }
  };

  const handleLearn = async (query: string, detected: string[]) => {
    push({ type: 'learning', query });
    try {
      const r = await api.learnNow(query, detected);
      setMessages(prev => prev.map(m =>
        m.type === 'learning' && m.query === query
          ? { type: 'learned', query, phrases_added: r.phrases_added, summary: r.summary }
          : m
      ));
    } catch (err) {
      push({ type: 'error', text: `Learn failed: ${err}` });
    }
  };

  return (
    <div className="flex flex-col flex-1 min-h-0 p-4">
      <div className="flex-1 overflow-y-auto space-y-2 pb-3 min-h-0">
        {messages.length === 0 && (
          <div className="text-zinc-600 text-sm text-center py-16 px-4">
            Type a customer query to route it.<br />
            <span className="text-zinc-700 text-xs">Misses and weak results can be learned from instantly.</span>
          </div>
        )}
        {messages.map((msg, i) => <ManualMessage key={i} msg={msg} onLearn={handleLearn} />)}
        <div ref={el => { bottomRef.current = el; }} />
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2 pt-3 border-t border-zinc-800 flex-shrink-0">
        <input ref={el => { inputRef.current = el; }} value={input} onChange={e => setInput(e.target.value)}
          placeholder="Type a customer query..."
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-white text-sm placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
          autoFocus />
        <button type="submit"
          className="px-4 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-medium transition-colors text-sm">
          Route
        </button>
      </form>
    </div>
  );
}

function ManualMessage({ msg, onLearn }: { msg: Message; onLearn: (query: string, detected: string[]) => void }) {
  if (msg.type === 'query')    return <div className="font-mono text-sm text-white"><span className="text-violet-400">{'> '}</span>{msg.text}</div>;
  if (msg.type === 'error')    return <div className="text-red-400 text-sm pl-4">{msg.text}</div>;
  if (msg.type === 'learning') return (
    <div className="text-blue-400 text-xs pl-4 flex items-center gap-2">
      <div className="w-2.5 h-2.5 border border-blue-400 border-t-transparent rounded-full animate-spin" />
      Learning from "{msg.query}"...
    </div>
  );
  if (msg.type === 'learned')  return (
    <div className="pl-4 text-xs text-emerald-400">
      ✓ Learned — {msg.phrases_added} edges added
      {msg.summary && <span className="text-zinc-500 ml-2">{msg.summary.slice(0, 80)}</span>}
    </div>
  );

  const { result, latency, query } = msg;
  const confirmed  = result?.confirmed ?? [];
  const candidates = result?.candidates ?? [];
  const allIntents = [...confirmed, ...candidates];
  const detected   = allIntents.map(i => i.id);

  if (allIntents.length === 0) {
    return (
      <div className="pl-4 space-y-1">
        <div className="text-red-400/70 text-sm">No match</div>
        <button onClick={() => onLearn(query, [])}
          className="text-xs px-2 py-1 border border-violet-500/40 text-violet-400 rounded hover:bg-violet-500/10 transition-colors">
          Learn now →
        </button>
      </div>
    );
  }

  const bestScore = Math.max(...allIntents.map(i => i.score));
  const isWeak = confirmed.length > 0 && confirmed[0].confidence !== 'high';

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
      <div className="flex items-center gap-3 pt-0.5">
        <div className="text-zinc-600 text-xs pl-1">
          <span className="text-emerald-800">{result.routing_us != null ? (result.routing_us < 1000 ? `${result.routing_us}µs` : `${(result.routing_us / 1000).toFixed(1)}ms`) : '—'}</span>
          <span className="text-zinc-800"> · </span>
          <span>{latency.toFixed(0)}ms</span>
        </div>
        {isWeak && (
          <button onClick={() => onLearn(query, detected)}
            className="text-[10px] px-2 py-0.5 border border-violet-500/40 text-violet-400 rounded hover:bg-violet-500/10 transition-colors">
            Learn now →
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Simulate Panel ───────────────────────────────────────────────────────────
// Generates queries → baselines → calls learn_now for each failure (no queue)
// → retests → shows before/after. SSE events fire naturally during learn_now awaits.

function SimulatePanel({ onAccuracy }: {
  onAccuracy: (a: { before: number; after: number }) => void;
}) {
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

  const run = async () => {
    stopRef.current = false;
    setRunning(true);
    setBefore(null); setAfter(null); setLearnLog([]); setOutcomes([]);

    try {
      // ── Step 1: Generate ──────────────────────────────────────────────────
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

      // ── Step 2: Baseline ──────────────────────────────────────────────────
      setPhase('baseline'); setPhaseLabel('Running baseline...');
      const baseline = await api.trainingRun(queries);
      // "detected" = all GT intents found in confirmed (extras OK) — shows router recall
      // "strict"   = perfect match, no extras — shows precision
      const calcMetrics = (results: any[]) => {
        const n = results.length;
        const detected = results.filter((r: any) =>
          (r.missed?.length ?? 0) === 0 && (r.promotable?.length ?? 0) === 0
        ).length;
        const strict = results.filter((r: any) => r.status === 'pass').length;
        return {
          strict: n === 0 ? 0 : Math.round((strict / n) * 100),
          detected: n === 0 ? 0 : Math.round((detected / n) * 100),
        };
      };
      setBefore(calcMetrics(baseline.results));
      if (stopRef.current) return;

      // ── Step 3: Learn only queries with MISSED intents (not just extras) ─
      // PARTIAL with no misses = correct intent found + false positives. learn_now
      // can suppress extras but single-step suppressor (Δ=0.05) is too weak vs
      // bootstrap (0.85). Only learn when there are genuinely missed intents.
      const failures = baseline.results.filter((r: any) =>
        r.status === 'fail' ||
        (r.status === 'partial' && (r.missed?.length ?? 0) > 0)
      );
      if (failures.length > 0) {
        setPhase('fixing');
        for (let idx = 0; idx < failures.length; idx++) {
          if (stopRef.current) break;
          const r = failures[idx];
          setPhaseLabel(`Learning ${idx + 1}/${failures.length}...`);
          const detected = [...(r.confirmed ?? []), ...(r.candidates ?? [])];
          try {
            const res = await api.learnNow(r.message, detected);
            setLearnLog(prev => [...prev, {
              query: r.message,
              phrases_added: res.phrases_added,
              summary: res.summary,
              missed: res.missed_intents ?? [],
              wrong: res.wrong_detections ?? [],
            }]);
          } catch { /* continue to next */ }
        }
      }
      if (stopRef.current) return;

      // ── Step 4: Retest ────────────────────────────────────────────────────
      setPhase('retesting'); setPhaseLabel('Re-testing same queries...');
      const retest = await api.trainingRun(queries);

      // Build per-turn outcome list
      const outs: TurnOutcome[] = queries.map((q, i) => ({
        message: q.message,
        ground_truth: q.ground_truth,
        before: baseline.results[i]?.status ?? 'fail',
        after: retest.results[i]?.status ?? 'fail',
        confirmed: retest.results[i]?.confirmed ?? [],
        missed: retest.results[i]?.missed ?? [],
        extra: retest.results[i]?.extra ?? [],
      }));

      const afterMetrics = calcMetrics(retest.results);
      setAfter(afterMetrics);
      setOutcomes(outs);
      onAccuracy({ before: before?.strict ?? 0, after: afterMetrics.strict });
      setPhase('done'); setPhaseLabel('');

    } catch (err) {
      setPhase('error'); setPhaseLabel((err as Error).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-y-auto">
      {/* Controls */}
      <div className="p-4 space-y-3 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-3 flex-wrap">
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
            <label className="text-xs text-zinc-500">Language</label>
            <select value={language} onChange={e => setLanguage(e.target.value)} disabled={running}
              className="bg-zinc-900 border border-zinc-700 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:border-violet-500">
              {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
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
        <input value={scenario} onChange={e => setScenario(e.target.value)} disabled={running}
          placeholder="Scenario (optional): e.g. angry customer trying to return a broken laptop"
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-xs text-white placeholder-zinc-600 focus:border-violet-500 focus:outline-none" />
        <div className="flex items-center gap-3">
          <button onClick={run} disabled={running}
            className="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded disabled:opacity-30 transition-colors">
            {running ? 'Running...' : 'Run Simulation'}
          </button>
          {running && (
            <button onClick={() => { stopRef.current = true; }} className="text-xs text-red-400 hover:text-red-300">
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Progress + results */}
      <div className="flex-1 p-4 space-y-4 min-h-0 overflow-y-auto">

        {/* Phase indicator */}
        {phase !== 'idle' && (
          <div className="flex items-center gap-2">
            {running && <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin flex-shrink-0" />}
            <span className={`text-xs ${phase === 'error' ? 'text-red-400' : phase === 'done' ? 'text-emerald-400' : 'text-violet-400'}`}>
              {phase === 'done' ? 'Complete' : phase === 'error' ? `Error: ${phaseLabel}` : phaseLabel || phase}
            </span>
          </div>
        )}

        {/* Live learn log — as failures are being processed */}
        {learnLog.length > 0 && phase !== 'done' && (
          <div className="space-y-1.5">
            <div className="text-[10px] uppercase text-zinc-500 font-semibold tracking-wide">Learning</div>
            {learnLog.map((entry, i) => (
              <div key={i} className="bg-zinc-900 rounded p-2 text-xs space-y-1">
                <div className="flex items-center gap-2">
                  <span className={`font-mono font-bold flex-shrink-0 ${entry.phrases_added > 0 ? 'text-emerald-400' : 'text-zinc-600'}`}>
                    {entry.phrases_added > 0 ? `+${entry.phrases_added} phrases` : 'no change'}
                  </span>
                  {entry.missed.length > 0 && (
                    <span className="text-zinc-500 truncate">→ {entry.missed.join(', ')}</span>
                  )}
                </div>
                <div className="text-zinc-500 truncate">"{entry.query.slice(0, 60)}"</div>
                {entry.summary && <div className="text-zinc-600 italic text-[11px]">{entry.summary.slice(0, 80)}</div>}
              </div>
            ))}
            {running && phase === 'fixing' && (
              <div className="flex items-center gap-2 text-xs text-zinc-600">
                <div className="w-2 h-2 border border-zinc-600 border-t-transparent rounded-full animate-spin" />
                {phaseLabel}
              </div>
            )}
          </div>
        )}

        {/* Before/After score — two rows: detection recall + strict precision */}
        {(before !== null || after !== null) && (
          <div className="space-y-2">
            {/* Detection row: all GT intents found in confirmed (extras OK) */}
            {[
              { label: 'Detection', key: 'detected' as const, note: 'correct intent found' },
              { label: 'Strict',    key: 'strict'   as const, note: 'perfect, no extras' },
            ].map(({ label, key, note }) => {
              const b = before?.[key] ?? null;
              const a = after?.[key] ?? null;
              return (
                <div key={key} className="flex items-center gap-3">
                  <div className="w-16 text-right">
                    <div className="text-xs text-zinc-500 font-medium">{label}</div>
                    <div className="text-[10px] text-zinc-700">{note}</div>
                  </div>
                  {b !== null && (
                    <div className="text-xl font-bold text-zinc-400 w-12 text-right">{b}%</div>
                  )}
                  {a !== null && b !== null && (
                    <>
                      <span className={`text-lg font-light ${a > b ? 'text-emerald-400' : a < b ? 'text-red-400' : 'text-zinc-600'}`}>
                        {a > b ? '↑' : a < b ? '↓' : '→'}
                      </span>
                      <div className={`text-xl font-bold w-12 ${a > b ? 'text-emerald-400' : a < b ? 'text-red-400' : 'text-zinc-400'}`}>
                        {a}%
                      </div>
                      {a !== b && (
                        <span className={`text-xs font-semibold ${a > b ? 'text-emerald-500' : 'text-red-500'}`}>
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

        {/* Per-turn breakdown — shown after retest completes */}
        {outcomes.length > 0 && (
          <div className="space-y-1">
            <div className="text-[10px] uppercase text-zinc-500 font-semibold tracking-wide">Results per query</div>
            {outcomes.map((o, i) => {
              const wasGood = o.before === 'pass';
              const isGood  = o.after === 'pass';
              const fixed   = !wasGood && isGood;
              const stuck   = !wasGood && !isGood;
              const degraded = wasGood && !isGood;
              const icon = fixed ? '✓' : stuck ? '✗' : degraded ? '!' : '✓';
              const cls  = fixed ? 'text-emerald-400' : stuck ? 'text-red-400' : degraded ? 'text-amber-400' : 'text-zinc-500';
              return (
                <div key={i} className="bg-zinc-900 rounded p-2.5 text-xs space-y-1.5">
                  <div className="flex items-start gap-2">
                    <span className={`font-bold flex-shrink-0 mt-0.5 ${cls}`}>{icon}</span>
                    <div className="min-w-0 flex-1">
                      <div className="text-zinc-300 leading-snug">"{o.message.slice(0, 70)}{o.message.length > 70 ? '…' : ''}"</div>
                      <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1">
                        <span className="text-zinc-600">Expected: <span className="text-zinc-400">{o.ground_truth.map(id => id.split(':')[1] ?? id).join(', ')}</span></span>
                        {o.confirmed.length > 0 && (
                          <span className="text-zinc-600">Got: <span className={isGood ? 'text-emerald-400' : 'text-amber-400'}>{o.confirmed.map(id => id.split(':')[1] ?? id).join(', ')}</span></span>
                        )}
                        {o.missed.length > 0 && (
                          <span className="text-zinc-600">Missed: <span className="text-red-400">{o.missed.map((id: string) => id.split(':')[1] ?? id).join(', ')}</span></span>
                        )}
                      </div>
                      {fixed && <div className="text-[10px] text-emerald-600 mt-0.5">Fixed by this session</div>}
                      {stuck && <div className="text-[10px] text-red-600 mt-0.5">Still failing — run again to try harder</div>}
                      {degraded && <div className="text-[10px] text-amber-600 mt-0.5">Regressed — was passing before</div>}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {phase === 'done' && (
          <button onClick={run} disabled={running}
            className="text-xs px-3 py-1.5 border border-violet-500/50 text-violet-400 rounded hover:bg-violet-500/10 transition-colors">
            Run again (keep improving)
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Review Panel (left — escalation queue list) ──────────────────────────────

function ReviewPanel({ items, selectedId, onSelect, onRefresh }: {
  items: ReviewItem[];
  selectedId: number | null;
  onSelect: (item: ReviewItem) => void;
  onRefresh: () => void;
}) {
  const [filter, setFilter] = useState<'all' | 'miss' | 'low_confidence'>('all');
  const visible = items.filter(i => filter === 'all' || i.flag === filter);

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div className="px-4 py-2.5 border-b border-zinc-800 flex items-center gap-2 flex-shrink-0">
        {(['all', 'miss', 'low_confidence'] as const).map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className={`text-[10px] px-2 py-0.5 rounded uppercase font-semibold transition-colors ${
              filter === f ? 'bg-zinc-700 text-white' : 'text-zinc-500 hover:text-zinc-300'
            }`}>
            {f === 'all' ? `All (${items.length})` : f === 'miss' ? `Miss (${items.filter(i => i.flag === 'miss').length})` : `Low (${items.filter(i => i.flag === 'low_confidence').length})`}
          </button>
        ))}
        <button onClick={onRefresh} className="ml-auto text-[10px] text-zinc-600 hover:text-zinc-400">↺ refresh</button>
      </div>

      <div className="flex-1 overflow-y-auto divide-y divide-zinc-800/50">
        {visible.length === 0 ? (
          <div className="text-center py-16 text-zinc-600 text-sm">No items in queue</div>
        ) : visible.map(item => (
          <button key={item.id} onClick={() => onSelect(item)}
            className={`w-full text-left px-4 py-3 hover:bg-zinc-800/50 transition-colors ${selectedId === item.id ? 'bg-zinc-800/70' : ''}`}>
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded flex-shrink-0 ${
                item.flag === 'miss' ? 'bg-red-900/30 text-red-400' : 'bg-amber-900/30 text-amber-400'
              }`}>{item.flag === 'low_confidence' ? 'LOW' : item.flag}</span>
              {item.detected.length > 0 && (
                <span className="text-[10px] text-zinc-600 truncate">{item.detected.join(', ')}</span>
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
  );
}

// ─── Auto Panel (live worker observer) ───────────────────────────────────────

function AutoPanel({ workerEvents }: { workerEvents: FeedEvent[] }) {
  const { settings } = useAppStore();
  const nsId = settings.selectedNamespaceId;

  const [autoLearn, setAutoLearn] = useState<boolean | null>(null);
  const [stats,     setStats]     = useState<{ pending: number; total: number } | null>(null);
  const [toggling,  setToggling]  = useState(false);

  const load = useCallback(async () => {
    try {
      const [nsList, s] = await Promise.all([
        api.listNamespaces(),
        api.getReviewStats(),
      ]);
      const ns = nsList.find((n: any) => n.id === nsId);
      setAutoLearn(ns?.auto_learn ?? false);
      setStats(s);
    } catch { /* */ }
  }, [nsId]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (workerEvents.length > 0) {
      const t = setTimeout(load, 800);
      return () => clearTimeout(t);
    }
  }, [workerEvents.length, load]);

  const toggle = async () => {
    if (autoLearn === null || toggling) return;
    setToggling(true);
    try {
      const next = !autoLearn;
      await api.updateNamespace(nsId, { auto_learn: next });
      setAutoLearn(next);
    } finally {
      setToggling(false);
    }
  };

  // Split events by type for display
  const recentLearned = workerEvents.filter(e => e.type === 'llm_done').slice(0, 20);
  const recentQueued  = workerEvents.filter(e => e.type === 'item_queued').slice(0, 5);

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-y-auto">

      {/* Status header */}
      <div className="p-4 border-b border-zinc-800 space-y-3 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xs text-zinc-400 font-semibold uppercase tracking-wide">Auto-Learn</span>
            {autoLearn === null ? (
              <span className="text-[10px] text-zinc-600">loading...</span>
            ) : autoLearn ? (
              <span className="flex items-center gap-1.5 text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                ON
              </span>
            ) : (
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-500 border border-zinc-700 font-semibold">OFF</span>
            )}
          </div>
          {autoLearn !== null && (
            <button onClick={toggle} disabled={toggling}
              className={`text-xs px-3 py-1 rounded border transition-colors disabled:opacity-40 ${
                autoLearn
                  ? 'border-zinc-700 text-zinc-400 hover:text-red-400 hover:border-red-500/40'
                  : 'border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/10'
              }`}>
              {toggling ? '...' : autoLearn ? 'Turn off' : 'Turn on'}
            </button>
          )}
        </div>

        {stats && (
          <div className="flex gap-5 text-xs">
            <div>
              <div className="text-zinc-600 text-[10px] mb-0.5">Pending</div>
              <div className={`font-mono font-bold ${stats.pending > 0 ? 'text-amber-400' : 'text-zinc-400'}`}>{stats.pending}</div>
            </div>
            <div>
              <div className="text-zinc-600 text-[10px] mb-0.5">Total processed</div>
              <div className="font-mono text-zinc-300">{stats.total}</div>
            </div>
            <div>
              <div className="text-zinc-600 text-[10px] mb-0.5">Learned this session</div>
              <div className="font-mono text-emerald-400">
                {recentLearned.reduce((sum, e) => sum + (e.type === 'llm_done' ? e.phrases_added : 0), 0)} phrases
              </div>
            </div>
          </div>
        )}

        {autoLearn === false && (
          <div className="text-xs text-zinc-600 bg-zinc-900 rounded px-3 py-2">
            Auto-learn is off. Production traffic misses accumulate but are not processed automatically.
            Turn on to let the background worker analyze and learn from them continuously.
          </div>
        )}
      </div>

      {/* Recently queued (incoming) */}
      {recentQueued.length > 0 && (
        <div className="px-4 pt-3 pb-2">
          <div className="text-[10px] uppercase text-zinc-600 font-semibold tracking-wide mb-2">Incoming</div>
          {recentQueued.map((ev, i) => (
            <div key={i} className="flex items-center gap-2 text-xs font-mono py-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 flex-shrink-0" />
              <span className="text-zinc-500">[{ev.type === 'item_queued' ? ev.flag ?? 'ok' : ''}]</span>
              <span className="text-zinc-400 truncate">"{ev.type === 'item_queued' ? ev.query.slice(0, 50) : ''}"</span>
            </div>
          ))}
        </div>
      )}

      {/* Live learning stream */}
      <div className="px-4 pt-3 flex-1">
        <div className="text-[10px] uppercase text-zinc-600 font-semibold tracking-wide mb-2">
          {recentLearned.length > 0 ? 'What was learned' : workerEvents.length === 0 ? 'No worker activity yet this session' : 'Waiting for worker...'}
        </div>
        {recentLearned.length === 0 && workerEvents.length > 0 && autoLearn === true && (
          <div className="flex items-center gap-2 text-xs text-zinc-600">
            <div className="w-2 h-2 border border-zinc-600 border-t-transparent rounded-full animate-spin" />
            Worker processing...
          </div>
        )}
        {recentLearned.map((ev, i) => {
          if (ev.type !== 'llm_done') return null;
          return (
            <div key={i} className="py-2 border-b border-zinc-800/50 last:border-0">
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-[10px] font-bold ${ev.phrases_added > 0 ? 'text-emerald-400' : 'text-zinc-600'}`}>
                  +{ev.phrases_added}
                </span>
                {ev.correct.length > 0 && (
                  <div className="flex gap-1 flex-wrap">
                    {ev.correct.map(id => (
                      <span key={id} className="text-[9px] font-mono px-1 py-0.5 rounded bg-emerald-900/20 text-emerald-500 border border-emerald-800/40">{id}</span>
                    ))}
                  </div>
                )}
                {ev.wrong.length > 0 && (
                  <div className="flex gap-1 flex-wrap">
                    {ev.wrong.map(id => (
                      <span key={id} className="text-[9px] font-mono px-1 py-0.5 rounded bg-red-900/20 text-red-500 border border-red-800/40 line-through">{id}</span>
                    ))}
                  </div>
                )}
              </div>
              {ev.summary && <div className="text-[10px] text-zinc-600 italic">{ev.summary.slice(0, 80)}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Review Detail (right panel — manual analysis + fix) ─────────────────────

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
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis,  setAnalysis]  = useState<ReviewAnalyzeResult | null>(null);
  const [blocks,    setBlocks]    = useState<IntentBlock[]>([]);
  const { settings: studioSettings } = useAppStore();
  const enabledLangs = studioSettings.languages;

  useEffect(() => { setAnalysis(null); setBlocks([]); }, [item.id]);

  const runAnalysis = async () => {
    setAnalyzing(true);
    try {
      const result = await api.reviewAnalyze(item.id);
      setAnalysis(result);
      setBlocks(Object.entries(result.phrases_to_add).map(([intentId, phraseList]) => ({
        intentId,
        phrases: phraseList.map(s => ({ phrase: s, lang: result.languages[0] || 'en' })),
      })));
    } catch (e) {
      alert('Analysis failed: ' + (e instanceof Error ? e.message : String(e)));
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
    const result = await api.reviewFix(item.id, toApply);
    const msgs = [`Applied ${result.added} phrases.`];
    if (result.resolved_count > 0) msgs.push(`${result.resolved_count} failures resolved.`);
    if (result.blocked.length > 0) msgs.push(`Blocked: ${result.blocked.map((b: any) => `"${b.phrase}"`).join(', ')}`);
    alert(msgs.join(' '));
    onFixed();
  };

  const totalPhrases = blocks.flatMap(b => b.phrases).filter(s => s.phrase.trim()).length;
  const usedIntents  = new Set(blocks.map(b => b.intentId).filter(Boolean));

  return (
    <div className="p-5 space-y-4">
      <div className="flex items-center gap-2">
        <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${
          item.flag === 'miss' ? 'bg-red-900/30 text-red-400' :
          item.flag === 'low_confidence' ? 'bg-amber-900/30 text-amber-400' : 'bg-blue-900/30 text-blue-400'
        }`}>{item.flag.replace('_', ' ')}</span>
        {item.detected.length > 0 && <span className="text-xs text-zinc-500">detected: {item.detected.join(', ')}</span>}
      </div>
      <div className="bg-zinc-800 rounded-lg p-3">
        <div className="text-[10px] text-zinc-500 mb-1">Query</div>
        <div className="text-white font-mono text-sm">"{item.query}"</div>
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
                  className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-white font-mono focus:border-violet-500 focus:outline-none" />
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

// ─── Shared rendering helpers ─────────────────────────────────────────────────

function IntentRow({ intent, index, bestScore, isMulti }: {
  intent: { id: string; score: number; intent_type: string; span: [number, number]; confidence?: string };
  index: number; bestScore: number; isMulti: boolean;
}) {
  const rel    = intent.score / bestScore;
  const isWeak = isMulti && rel < 0.3;
  const color  = INTENT_COLORS[index % INTENT_COLORS.length];
  const bg     = INTENT_BG_COLORS[index % INTENT_BG_COLORS.length];
  const conf   = intent.confidence || 'low';
  const confStyle = conf === 'high' ? 'text-emerald-400 border-emerald-400/40'
    : conf === 'medium' ? 'text-amber-400 border-amber-400/40'
    : 'text-zinc-400 border-zinc-500/40';
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
