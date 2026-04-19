import { useState, useRef, useEffect } from 'react';
import { api, type MultiRouteOutput, type ReviewAnalysis } from '@/api/client';
import Page from '@/components/Page';

const INTENT_COLORS = [
  'text-emerald-400', 'text-blue-400', 'text-amber-400', 'text-pink-400',
  'text-cyan-400', 'text-violet-400', 'text-orange-400', 'text-lime-400',
];
const INTENT_BG_COLORS = [
  'bg-emerald-400/20', 'bg-blue-400/20', 'bg-amber-400/20', 'bg-pink-400/20',
  'bg-cyan-400/20', 'bg-violet-400/20', 'bg-orange-400/20', 'bg-lime-400/20',
];

type Message =
  | { type: 'query'; text: string }
  | { type: 'result'; result: MultiRouteOutput; latency: number; query: string; review?: ReviewAnalysis; reviewing?: boolean }
  | { type: 'learn'; text: string }
  | { type: 'training'; query: string }
  | { type: 'trained'; query: string; phrases_added: number; summary: string }
  | { type: 'system'; html: string }
  | { type: 'error'; text: string }
  | { type: 'help' };

export default function RouterPage() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [intentCount, setIntentCount] = useState<number | null>(null);
  const [debugMode, setDebugMode] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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
    if (raw === '/help') {
      push({ type: 'query', text: raw }, { type: 'help' });
      return;
    }

    if (raw === '/reset') {
      push({ type: 'query', text: raw });
      if (!window.confirm('Reset router to defaults? This clears all learned state for this workspace.')) {
        push({ type: 'system', html: 'Reset cancelled.' });
        return;
      }
      try {
        await api.reset();
        await api.loadDefaults();
        push({ type: 'learn', text: 'Router reset to defaults.' });
      } catch (err) {
        push({ type: 'error', text: String(err) });
      }
      return;
    }

    const learnMatch = raw.match(/^\/learn\s+(.+?)\s*(?:->|→)\s*(\S+)$/i);
    if (learnMatch) {
      const [, query, intent] = learnMatch;
      push({ type: 'query', text: raw });
      try {
        await api.learn(query, intent);
        push({ type: 'learn', text: `Learned: "${query}" → ${intent}` });
      } catch (err) {
        push({ type: 'error', text: String(err) });
      }
      return;
    }

    const correctMatch = raw.match(/^\/correct\s+(.+?)\s*(?:->|→)\s*(\S+)\s*(?:->|→)\s*(\S+)$/i);
    if (correctMatch) {
      const [, query, wrong, right] = correctMatch;
      push({ type: 'query', text: raw });
      try {
        await api.correct(query, wrong, right);
        push({ type: 'learn', text: `Corrected: "${query}" moved from ${wrong} → ${right}` });
      } catch (err) {
        push({ type: 'error', text: String(err) });
      }
      return;
    }

    // Regular query
    push({ type: 'query', text: raw });
    const t0 = performance.now();
    try {
      const result = await api.routeMulti(raw, 0.3, true, debugMode);
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
      await api.learn(suggestion.phrase, suggestion.intent_id);
      push({ type: 'learn', text: `Applied: phrase "${suggestion.phrase}" → ${suggestion.intent_id}` });
    } catch (err) {
      push({ type: 'error', text: `Failed: ${err}` });
    }
  };

  return (
    <Page title="Route" subtitle="Test queries against the router" fullscreen>
    <div className="flex flex-col h-full px-6 py-4">
      <div className="flex-1 overflow-y-auto space-y-3 pb-4 min-h-0">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8">
            <div className="space-y-1">
              <div className="text-zinc-300 text-sm font-medium">Test your router</div>
              <div className="text-zinc-600 text-xs max-w-sm leading-relaxed">
                Type any natural language query. The router detects which intents it matches — in under 1ms.
                Weak or missing results show a <span className="text-violet-400">Train →</span> button to improve instantly.
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
                  className="block w-full text-left text-xs font-mono text-violet-400 hover:text-violet-300 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 rounded px-3 py-1.5 transition-colors">
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} onApplySuggestion={applySuggestion} onTrain={handleTrain} intentCount={intentCount} debugMode={debugMode} />
        ))}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 pt-3 border-t border-zinc-800">
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type a query... or /help for commands"
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-white font-mono text-sm placeholder-zinc-500 focus:outline-none focus:border-violet-500 transition-colors"
          autoFocus
        />
        <button
          type="button"
          onClick={() => setDebugMode(d => !d)}
          title="Toggle layer trace"
          className={`px-3 py-2.5 rounded-lg border text-xs font-mono transition-colors ${
            debugMode
              ? 'bg-amber-500/10 border-amber-500/40 text-amber-400'
              : 'bg-zinc-900 border-zinc-700 text-zinc-500 hover:text-zinc-300'
          }`}
        >
          trace
        </button>
        <button
          type="submit"
          className="px-5 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-medium transition-colors text-sm"
        >
          Route
        </button>
      </form>
    </div>
    </Page>
  );
}

function MessageBubble({ msg, onApplySuggestion, onTrain, intentCount, debugMode }: {
  msg: Message;
  onApplySuggestion: (s: ReviewAnalysis['suggestions'][0]) => void;
  onTrain: (query: string, detected: string[]) => void;
  intentCount: number | null;
  debugMode: boolean;
}) {
  if (msg.type === 'query') {
    return (
      <div className="font-mono text-sm text-white mb-1">
        <span className="text-violet-400">{'> '}</span>
        {msg.text}
      </div>
    );
  }

  if (msg.type === 'error') {
    return <div className="text-red-400 text-sm pl-5">{msg.text}</div>;
  }

  if (msg.type === 'learn') {
    return <div className="text-emerald-400 text-sm pl-5">{msg.text}</div>;
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

  if (msg.type === 'system') {
    return <div className="text-zinc-500 text-sm pl-5" dangerouslySetInnerHTML={{ __html: msg.html }} />;
  }

  if (msg.type === 'help') {
    return (
      <div className="text-zinc-400 text-sm pl-5 leading-relaxed">
        <strong>Commands:</strong><br />
        <code className="text-violet-400">any text</code> — route query (auto-detects multi-intent)<br />
        <code className="text-violet-400">/learn &lt;query&gt; -&gt; &lt;intent&gt;</code> — teach the router<br />
        <code className="text-violet-400">/correct &lt;query&gt; -&gt; &lt;wrong&gt; -&gt; &lt;right&gt;</code> — fix misroute<br />
        <code className="text-violet-400">/reset</code> — reset to default demo intents<br />
        <code className="text-violet-400">/help</code> — show this message
      </div>
    );
  }

  // Result card
  const { result, latency, query, review, reviewing } = msg;
  const allIntents = [...(result?.confirmed || []), ...(result?.candidates || [])];
  if (!result || allIntents.length === 0) {
    const noIntents = intentCount !== null && intentCount === 0;
    return (
      <div className="pl-5 space-y-1">
        <div className="flex items-center gap-3">
          <span className="text-zinc-500 text-sm">No match found.</span>
          {!noIntents && (
            <button onClick={() => onTrain(query, [])}
              className="text-[10px] px-2 py-0.5 border border-violet-500/40 text-violet-400 rounded hover:bg-violet-500/10 transition-colors">
              Train →
            </button>
          )}
        </div>
        {noIntents && (
          <div className="text-[11px] text-zinc-600">
            No intents exist yet — <a href="/intents" className="text-violet-400 hover:underline">create some</a> or <a href="/import" className="text-violet-400 hover:underline">import from a spec</a>.
          </div>
        )}
      </div>
    );
  }

  const confirmed = result.confirmed || [];
  const candidates = result.candidates || [];
  const bestScore = Math.max(...allIntents.map(i => i.score));
  const isWeak = confirmed.length > 0 && confirmed[0].confidence !== 'high';

  return (
    <div className="pl-5 mb-3">
      {/* Highlighted query card */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 mb-2">
        <HighlightedQuery query={query} intents={allIntents} />
      </div>

      {/* Confirmed intents */}
      {confirmed.length > 0 && (
        <div className="mb-1">
          {candidates.length > 0 && (
            <div className="text-[10px] text-emerald-400/60 uppercase font-semibold tracking-wide mb-0.5 pl-1">Confirmed</div>
          )}
          {confirmed.map((intent, i) => (
            <IntentRow key={intent.id} intent={intent} index={i} bestScore={bestScore} isMulti={allIntents.length > 1} />
          ))}
        </div>
      )}

      {/* Candidate intents */}
      {candidates.length > 0 && (
        <div className="mb-1">
          <div className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wide mb-0.5 pl-1 mt-1">Candidates</div>
          {candidates.map((intent, i) => (
            <IntentRow key={intent.id} intent={intent} index={confirmed.length + i} bestScore={bestScore} isMulti={allIntents.length > 1} />
          ))}
        </div>
      )}

      {/* Timing + Train */}
      <div className="flex items-center gap-3 pl-2">
        <div className="text-zinc-600 text-xs">
          {confirmed.length} confirmed{candidates.length > 0 ? `, ${candidates.length} candidates` : ''}{' '}
          <span className="text-zinc-700">·</span>{' '}
          <span className="text-emerald-700" title="library routing time">router {result.routing_us != null ? (result.routing_us < 1000 ? `${result.routing_us}µs` : `${(result.routing_us / 1000).toFixed(1)}ms`) : '—'}</span>
          <span className="text-zinc-700"> · </span>
          <span title="HTTP round trip">http {latency.toFixed(0)}ms</span>
        </div>
        {isWeak && (
          <button onClick={() => onTrain(query, allIntents.map(i => i.id))}
            className="text-[10px] px-2 py-0.5 border border-violet-500/40 text-violet-400 rounded hover:bg-violet-500/10 transition-colors">
            Train →
          </button>
        )}
      </div>

      {/* Relations */}
      {result.relations.length > 0 && (
        <div className="pl-2 mt-1 flex flex-wrap gap-1.5">
          {result.relations.map((rel, i) => {
            const labels: Record<string, string> = {
              sequential: 'do in order',
              conditional: 'if/then',
              negation: 'negated',
              parallel: 'at the same time',
            };
            const hint: Record<string, string> = {
              sequential: 'Intents should be handled in sequence',
              conditional: 'One intent is conditional on another',
              negation: 'User is negating or cancelling an intent',
              parallel: 'Intents can be handled simultaneously',
            };
            return (
              <span key={i} title={hint[rel.type] || rel.type}
                className="text-[10px] text-violet-400 bg-violet-400/10 border border-violet-400/20 px-1.5 py-0.5 rounded font-mono cursor-default">
                {labels[rel.type] || rel.type}
              </span>
            );
          })}
        </div>
      )}



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

      {/* Layer trace panel */}
      {debugMode && result.debug && result.debug !== null && (
        <LayerTrace debug={result.debug} query={query} />
      )}
    </div>
  );
}

// --- Intent row ---

const CONFIDENCE_STYLES: Record<string, string> = {
  high: 'text-emerald-400 border-emerald-400/40 bg-emerald-400/10',
  medium: 'text-amber-400 border-amber-400/40 bg-amber-400/10',
  low: 'text-zinc-400 border-zinc-500/40 bg-zinc-500/10',
};

const SOURCE_LABELS: Record<string, string> = {
  dual: 'dual',
  paraphrase: 'phrase',
  routing: 'route',
};

function IntentRow({ intent, index, bestScore, isMulti }: {
  intent: { id: string; score: number; intent_type: string; span: [number, number]; confidence?: string; source?: string };
  index: number;
  bestScore: number;
  isMulti: boolean;
}) {
  const relativeScore = intent.score / bestScore;
  const isWeak = isMulti && relativeScore < 0.3;
  const color = INTENT_COLORS[index % INTENT_COLORS.length];
  const bgColor = INTENT_BG_COLORS[index % INTENT_BG_COLORS.length];
  const confidence = intent.confidence || 'low';
  const source = intent.source || 'routing';
  const confStyle = CONFIDENCE_STYLES[confidence] || CONFIDENCE_STYLES.low;

  return (
    <div className={`flex items-center gap-2.5 font-mono text-sm px-2 py-1 rounded ${bgColor} ${isWeak ? 'opacity-40' : ''}`}>
      <span className={`text-[9px] px-1.5 py-0.5 rounded border font-bold uppercase ${confStyle}`}>
        {confidence}
      </span>
      <span className={`font-semibold ${color}`}>{intent.id}</span>
      <span className="text-amber-400">{intent.score.toFixed(2)}</span>
      <span className={`text-[9px] px-1 py-0.5 rounded border font-semibold uppercase ${
        intent.intent_type === 'context'
          ? 'text-cyan-400 border-cyan-400/30'
          : 'text-emerald-400 border-emerald-400/30'
      }`}>
        {intent.intent_type}
      </span>
      <span className="text-zinc-500 text-[10px]">{SOURCE_LABELS[source] || source}</span>
      {isMulti && (
        <span className="text-zinc-600 text-xs">[{intent.span[0]},{intent.span[1]}]</span>
      )}
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
          <div className="text-[10px] text-violet-400/70 uppercase font-semibold mb-1">Suggestions</div>
          {review.suggestions.map((s, i) => (
            <div key={i} className="flex items-start gap-2 text-xs pl-2 py-1">
              <div className="flex-1">
                <span className="text-violet-400 font-mono">add_phrase</span>
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
                    : 'bg-violet-600 hover:bg-violet-500 text-white'
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

// --- Layer trace panel ---

function LayerTrace({ debug, query }: { debug: any; query: string }) {
  const l0 = debug.l0_corrected as string | undefined;
  const l1 = debug.l1_normalized as string | undefined;
  const injected = (debug.l1_injected as string[]) || [];
  const tokens = (debug.l2_tokens as string[]) || [];
  const scores = (debug.l2_all_scores as { id: string; score: number }[]) || [];
  const top5 = scores.slice(0, 5);
  const maxScore = top5[0]?.score ?? 1;

  const changed = (a?: string, b?: string) => a && b && a !== b;

  return (
    <div className="mt-2 bg-zinc-950 border border-zinc-800 rounded-lg p-3 space-y-2 font-mono text-xs">
      <div className="text-[10px] text-zinc-600 uppercase font-semibold tracking-wide">Layer trace</div>

      {/* L0 */}
      <div className="flex items-start gap-2">
        <span className="text-zinc-600 w-6 shrink-0">L0</span>
        <span className="text-zinc-500">typo</span>
        {changed(query.toLowerCase(), l0)
          ? <><span className="text-zinc-600 line-through">{query}</span><span className="text-amber-400 ml-1">{l0}</span></>
          : <span className="text-zinc-600">no change</span>
        }
      </div>

      {/* L1 */}
      <div className="flex items-start gap-2">
        <span className="text-zinc-600 w-6 shrink-0">L1</span>
        <span className="text-zinc-500">morph</span>
        {changed(l0 ?? query, l1)
          ? <span className="text-amber-400">{l1}</span>
          : <span className="text-zinc-600">no change</span>
        }
      </div>
      {injected.length > 0 && (
        <div className="flex items-start gap-2 pl-8">
          <span className="text-zinc-500">inject</span>
          <span className="text-violet-400">{injected.join(', ')}</span>
        </div>
      )}

      {/* L2 tokens */}
      <div className="flex items-start gap-2">
        <span className="text-zinc-600 w-6 shrink-0">L2</span>
        <span className="text-zinc-500">tokens</span>
        <span className="text-zinc-400">{tokens.join(' · ')}</span>
      </div>

      {/* L2 scores */}
      {top5.length > 0 && (
        <div className="pl-8 space-y-1">
          {top5.map(({ id, score }) => {
            const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;
            const isTop = score === maxScore;
            return (
              <div key={id} className="flex items-center gap-2">
                <div className="w-24 truncate text-right text-[10px] text-zinc-500">{id.split(':').pop()}</div>
                <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${isTop ? 'bg-emerald-500' : 'bg-zinc-600'}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className={`text-[10px] w-8 text-right ${isTop ? 'text-emerald-400' : 'text-zinc-600'}`}>
                  {score.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// --- Highlighted query with colored spans ---

function HighlightedQuery({
  query, intents,
}: {
  query: string;
  intents: { id: string; span: [number, number] }[];
}) {
  const charMap = new Array(query.length).fill(-1);
  intents.forEach((intent, idx) => {
    const [start, end] = intent.span;
    for (let i = Math.max(0, start); i < Math.min(query.length, end); i++) {
      charMap[i] = idx;
    }
  });

  const segments: { text: string; intentIdx: number }[] = [];
  let i = 0;
  while (i < query.length) {
    const currentIdx = charMap[i];
    let j = i + 1;
    while (j < query.length && charMap[j] === currentIdx) j++;
    segments.push({ text: query.slice(i, j), intentIdx: currentIdx });
    i = j;
  }

  return (
    <div className="font-mono text-sm leading-relaxed">
      {segments.map((seg, i) => {
        if (seg.intentIdx === -1) {
          return <span key={i} className="text-zinc-500">{seg.text}</span>;
        }
        const color = INTENT_COLORS[seg.intentIdx % INTENT_COLORS.length];
        const intent = intents[seg.intentIdx];
        return (
          <span key={i} className={`${color} font-semibold`} title={intent.id}>
            {seg.text}
          </span>
        );
      })}
    </div>
  );
}
