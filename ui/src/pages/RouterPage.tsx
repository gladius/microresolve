import { useState, useRef, useEffect } from 'react';
import { api, type ResolveOutput, type ReviewAnalysis } from '@/api/client';
import Page from '@/components/Page';
import { useAppStore } from '@/store';

const INTENT_COLORS = [
  'text-emerald-400', 'text-blue-400', 'text-amber-400', 'text-pink-400',
  'text-cyan-400', 'text-emerald-400', 'text-orange-400', 'text-lime-400',
];
const INTENT_BG_COLORS = [
  'bg-emerald-400/20', 'bg-blue-400/20', 'bg-amber-400/20', 'bg-pink-400/20',
  'bg-cyan-400/20', 'bg-emerald-400/20', 'bg-orange-400/20', 'bg-lime-400/20',
];

type Message =
  | { type: 'query'; text: string }
  | { type: 'result'; result: ResolveOutput; latency: number; query: string; review?: ReviewAnalysis; reviewing?: boolean }
  | { type: 'training'; query: string }
  | { type: 'trained'; query: string; phrases_added: number; summary: string }
  | { type: 'error'; text: string };

const DEBUG_KEY = 'resolve.debug';

export default function RouterPage() {
  const { settings } = useAppStore();

  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [intentCount, setIntentCount] = useState<number | null>(null);
  const [debug, setDebug] = useState<boolean>(() => localStorage.getItem(DEBUG_KEY) === 'true');
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const toggleDebug = () => setDebug(prev => {
    const next = !prev;
    localStorage.setItem(DEBUG_KEY, String(next));
    return next;
  });

  useEffect(() => {
    api.listIntents().then(list => setIntentCount(list.length)).catch(() => setIntentCount(0));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const push = (...msgs: Message[]) => setMessages(prev => [...prev, ...msgs]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const raw = input.trim();
    if (!raw) return;
    setInput('');
    await handleInput(raw);
    inputRef.current?.focus();
  };

  const handleInput = async (raw: string) => {
    // Regular query
    push({ type: 'query', text: raw });
    const t0 = performance.now();
    try {
      const result = await api.resolve(raw, 0.3, true);
      const latency = performance.now() - t0;
      push({ type: 'result', result, latency, query: raw });
    } catch (err) {
      push({ type: 'error', text: String(err) });
    }
  };

  const handleTrain = async (query: string, detected: string[]) => {
    push({ type: 'training', query });
    try {
      const r = await api.learnNow(query, detected);
      setMessages(prev => prev.map(m =>
        m.type === 'training' && m.query === query
          ? { type: 'trained', query, phrases_added: r.phrases_added, summary: r.summary }
          : m
      ));
    } catch (err) {
      push({ type: 'error', text: `Train failed: ${err}` });
    }
  };

  const applySuggestion = async (suggestion: ReviewAnalysis['suggestions'][0]) => {
    try {
      await api.learnNow(suggestion.phrase, [suggestion.intent_id]);
    } catch (err) {
      push({ type: 'error', text: `Failed: ${err}` });
    }
  };

  return (
    <Page
      title="Resolve"
      subtitle="Resolve queries to intents"
      fullscreen
      actions={
        <button
          onClick={toggleDebug}
          title={debug ? 'Hide trace' : 'Show trace'}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-mono transition-colors ${
            debug
              ? 'bg-emerald-500/15 border border-emerald-500/40 text-emerald-400'
              : 'border border-zinc-700 text-zinc-500 hover:text-zinc-300'
          }`}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${debug ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
          debug
        </button>
      }
    >
    <div className="flex flex-col h-full min-h-0 px-6 py-4">
        <div className="flex-1 overflow-y-auto space-y-3 pb-4 min-h-0">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8">
              <div className="space-y-1">
                <div className="text-zinc-300 text-sm font-medium">Test your router</div>
                <div className="text-zinc-600 text-xs max-w-sm leading-relaxed">
                  Type any natural language query. The router detects which intents it matches — in under 1ms.
                  Weak or missing results show a <span className="text-emerald-400">Train →</span> button to improve instantly.
                </div>
              </div>
              <div className="space-y-1.5 text-left">
                <div className="text-[10px] text-zinc-600 uppercase tracking-wide font-semibold mb-2">Try these</div>
                {[
                  'cancel my subscription',
                  'I need to refund my last order and update my address',
                  'how do I reset my password',
                ].map(q => (
                  <button key={q} onClick={() => handleInput(q)}
                    className="block w-full text-left text-xs font-mono text-emerald-400 hover:text-emerald-300 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 rounded px-3 py-1.5 transition-colors">
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <MessageBubble
              key={i}
              msg={msg}
              onApplySuggestion={applySuggestion}
              onTrain={handleTrain}
              intentCount={intentCount}
              debug={debug}
            />
          ))}
          <div ref={bottomRef} />
        </div>

        <form onSubmit={handleSubmit} className="flex gap-2 pt-3 border-t border-zinc-800">
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type a natural language query…"
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-zinc-100 font-mono text-sm placeholder-zinc-500 focus:outline-none focus:border-emerald-500 transition-colors"
            autoFocus
          />
          <button
            type="submit"
            className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition-colors text-sm"
          >
            Resolve
          </button>
        </form>
    </div>
    </Page>
  );
}

function MessageBubble({ msg, onApplySuggestion, onTrain, intentCount, debug }: {
  msg: Message;
  onApplySuggestion: (s: ReviewAnalysis['suggestions'][0]) => void;
  onTrain: (query: string, detected: string[]) => void;
  intentCount: number | null;
  debug: boolean;
}) {
  if (msg.type === 'query') {
    return (
      <div className="font-mono text-sm text-zinc-100 mb-1">
        <span className="text-emerald-400">{'> '}</span>
        {msg.text}
      </div>
    );
  }

  if (msg.type === 'error') {
    return <div className="text-red-400 text-sm pl-5">{msg.text}</div>;
  }

  if (msg.type === 'training') {
    return (
      <div className="text-blue-400 text-xs pl-5 flex items-center gap-2">
        <div className="w-2.5 h-2.5 border border-blue-400 border-t-transparent rounded-full animate-spin" />
        Training from "{msg.query.slice(0, 50)}"...
      </div>
    );
  }

  if (msg.type === 'trained') {
    return (
      <div className="pl-5 text-xs">
        {msg.phrases_added > 0
          ? <span className="text-emerald-400">✓ +{msg.phrases_added} phrases learned{msg.summary ? ` — ${msg.summary.slice(0, 80)}` : ''}</span>
          : <span className="text-zinc-500">No new phrases (already well-trained or no clear fix)</span>
        }
      </div>
    );
  }

  // Result card
  const { result, latency, query, review, reviewing } = msg;
  const allIntents = result?.intents || [];
  if (!result || allIntents.length === 0) {
    const noIntents = intentCount !== null && intentCount === 0;
    return (
      <div className="pl-5 space-y-1">
        <div className="flex items-center gap-3">
          <span className="text-zinc-500 text-sm">No match found.</span>
          {!noIntents && (
            <button onClick={() => onTrain(query, [])}
              className="text-[10px] px-2 py-0.5 border border-emerald-500/40 text-emerald-400 rounded hover:bg-emerald-500/10 transition-colors">
              Train →
            </button>
          )}
        </div>
        {noIntents && (
          <div className="text-[11px] text-zinc-600">
            No intents exist yet — <a href="/l2" className="text-emerald-400 hover:underline">create some</a> or <a href="/import" className="text-emerald-400 hover:underline">import from a spec</a>.
          </div>
        )}
      </div>
    );
  }

  const highIntents = allIntents.filter(i => i.band === 'High');
  const mediumIntents = allIntents.filter(i => i.band === 'Medium');
  const bestScore = Math.max(...allIntents.map(i => i.score));
  const isWeak = result.disposition !== 'Confident';

  return (
    <div className="pl-5 mb-3">
      {/* Query card */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 mb-2">
        <HighlightedQuery query={query} />
      </div>

      {/* Intent list */}
      {highIntents.length > 0 && (
        <div className="mb-1">
          {mediumIntents.length > 0 && (
            <div className="text-[10px] text-emerald-400/60 uppercase font-semibold tracking-wide mb-0.5 pl-1">Matched</div>
          )}
          {highIntents.map((intent, i) => (
            <IntentRow key={intent.id} intent={intent} index={i} bestScore={bestScore} isMulti={allIntents.length > 1} />
          ))}
        </div>
      )}

      {/* Medium-band intents */}
      {mediumIntents.length > 0 && (
        <div className="mb-1">
          <div className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wide mb-0.5 pl-1 mt-1">Low confidence</div>
          {mediumIntents.map((intent, i) => (
            <IntentRow key={intent.id} intent={intent} index={highIntents.length + i} bestScore={bestScore} isMulti={allIntents.length > 1} />
          ))}
        </div>
      )}

      {/* Timing + Train */}
      <div className="flex items-center gap-3 pl-2">
        <div className="text-zinc-600 text-xs">
          {allIntents.length} intent{allIntents.length !== 1 ? 's' : ''}{' '}
          <span className="text-zinc-700">·</span>{' '}
          <span className="text-emerald-400 font-semibold" title="library routing time">resolved in {result.routing_us != null ? (result.routing_us < 1000 ? `${result.routing_us}µs` : `${(result.routing_us / 1000).toFixed(1)}ms`) : '—'}</span>
          <span className="text-zinc-700"> · </span>
          <span title="HTTP round trip">http {latency.toFixed(0)}ms</span>
        </div>
        {isWeak && (
          <button onClick={() => onTrain(query, allIntents.map(i => i.id))}
            className="text-[10px] px-2 py-0.5 border border-emerald-500/40 text-emerald-400 rounded hover:bg-emerald-500/10 transition-colors">
            Train →
          </button>
        )}
      </div>



      {/* E1: LLM Review card */}
      {reviewing && (
        <div className="mt-2 bg-amber-400/5 border border-amber-400/20 rounded-lg p-3">
          <div className="flex items-center gap-2 text-amber-400 text-xs">
            <span className="animate-pulse">●</span>
            LLM reviewing...
          </div>
        </div>
      )}
      {review && <ReviewCard review={review} onApply={onApplySuggestion} />}

      {/* Trace — opt-in via header debug toggle (persisted in localStorage) */}
      {debug && result.trace && (
        <div className="mt-3 text-xs font-mono text-zinc-500">
          tokens: {result.trace.tokens.join(' ')}
        </div>
      )}
    </div>
  );
}

// --- Intent row ---

const BAND_STYLES: Record<string, string> = {
  High: 'text-emerald-400 border-emerald-400/40 bg-emerald-400/10',
  Medium: 'text-amber-400 border-amber-400/40 bg-amber-400/10',
  Low: 'text-zinc-400 border-zinc-500/40 bg-zinc-500/10',
};

function IntentRow({ intent, index, bestScore, isMulti }: {
  intent: { id: string; score: number; confidence: number; band: string };
  index: number;
  bestScore: number;
  isMulti: boolean;
}) {
  const relativeScore = intent.score / bestScore;
  const isWeak = isMulti && relativeScore < 0.3;
  const color = INTENT_COLORS[index % INTENT_COLORS.length];
  const bgColor = INTENT_BG_COLORS[index % INTENT_BG_COLORS.length];
  const band = intent.band || 'Low';
  const bandStyle = BAND_STYLES[band] || BAND_STYLES.Low;

  return (
    <div className={`flex items-center gap-2.5 font-mono text-sm px-2 py-1 rounded ${bgColor} ${isWeak ? 'opacity-40' : ''}`}>
      <span className={`text-[9px] px-1.5 py-0.5 rounded border font-bold uppercase ${bandStyle}`}>
        {band.toLowerCase()}
      </span>
      <span className={`font-semibold ${color}`}>{intent.id}</span>
      <span className="text-amber-400">{intent.score.toFixed(2)}</span>
      {isWeak && <span className="text-zinc-600 text-[10px]">weak</span>}
    </div>
  );
}

// --- E1: Review card ---

function ReviewCard({ review, onApply }: {
  review: ReviewAnalysis;
  onApply: (s: ReviewAnalysis['suggestions'][0]) => void;
}) {
  const [applied, setApplied] = useState<Set<number>>(new Set());

  const confidenceColor = {
    high: 'text-emerald-400',
    medium: 'text-amber-400',
    low: 'text-red-400',
  }[review.confidence];

  return (
    <div className="mt-2 bg-zinc-900 border border-zinc-800 rounded-lg p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded font-semibold uppercase">LLM Review</span>
          <span className={`text-[10px] ${confidenceColor} font-semibold`}>{review.confidence}</span>
        </div>
      </div>

      <div className="text-xs text-zinc-400">{review.summary}</div>

      {/* False positives */}
      {review.false_positives.length > 0 && (
        <div>
          <div className="text-[10px] text-red-400/70 uppercase font-semibold mb-0.5">False Positives</div>
          {review.false_positives.map((fp, i) => (
            <div key={i} className="text-xs text-zinc-500 pl-2">
              <span className="text-red-400 font-mono">{fp.id}</span> — {fp.reason}
            </div>
          ))}
        </div>
      )}

      {/* Missed */}
      {review.missed.length > 0 && (
        <div>
          <div className="text-[10px] text-amber-400/70 uppercase font-semibold mb-0.5">Missed Intents</div>
          {review.missed.map((m, i) => (
            <div key={i} className="text-xs text-zinc-500 pl-2">
              <span className="text-amber-400 font-mono">{m.id}</span> — {m.reason}
            </div>
          ))}
        </div>
      )}

      {/* Suggestions with approve buttons */}
      {review.suggestions.length > 0 && (
        <div>
          <div className="text-[10px] text-emerald-400/70 uppercase font-semibold mb-1">Suggestions</div>
          {review.suggestions.map((s, i) => (
            <div key={i} className="flex items-start gap-2 text-xs pl-2 py-1">
              <div className="flex-1">
                <span className="text-emerald-400 font-mono">add_phrase</span>
                <span className="text-zinc-400"> "{s.phrase}" → <span className="text-emerald-400">{s.intent_id}</span></span>
                <div className="text-zinc-600 mt-0.5">{s.reason}</div>
              </div>
              <button
                onClick={() => {
                  onApply(s);
                  setApplied(prev => new Set(prev).add(i));
                }}
                disabled={applied.has(i)}
                className={`px-2 py-1 rounded text-[10px] font-semibold transition-colors flex-shrink-0 ${
                  applied.has(i)
                    ? 'bg-emerald-400/10 text-emerald-400 border border-emerald-400/30'
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                }`}
              >
                {applied.has(i) ? 'Applied' : 'Approve'}
              </button>
            </div>
          ))}
        </div>
      )}

      {review.correct.length > 0 && review.false_positives.length === 0 && review.suggestions.length === 0 && (
        <div className="text-emerald-400 text-xs">All routing correct.</div>
      )}
    </div>
  );
}


// --- Query display ---

function HighlightedQuery({ query }: { query: string }) {
  return (
    <div className="font-mono text-sm leading-relaxed text-zinc-300">{query}</div>
  );
}
