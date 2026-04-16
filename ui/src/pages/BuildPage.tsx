import { useState, useRef, useEffect, useCallback } from 'react';
import { useAppStore } from '@/store';
import { api, type IntentInfo } from '@/api/client';

// ─── Types ───────────────────────────────────────────────────────────────

type BuildAction = { tool: string; result: string };

type BuildMessage = {
  role: 'user' | 'assistant';
  content: string;
  actions?: BuildAction[];
};

type TestRouting = {
  intent: string | null;
  is_transition: boolean;
  disposition: string;
  routing_us?: number;
  total_us?: number;
};

type TestLookup = { intent: string; value: string | null; error?: string };

type TestMessage = {
  role: 'user' | 'assistant';
  content: string;
  intent?: string;            // on user turn when transition
  remark?: string;            // on assistant turn
  routing?: TestRouting;      // on assistant turn
  next_intent?: string;       // on assistant turn if handoff fired
  context?: string;           // on assistant turn if context briefing emitted
  lookups?: TestLookup[];     // on assistant turn if any fact intents were looked up
};

// ─── Starter prompts ─────────────────────────────────────────────────────

const BUILD_STARTERS = [
  'I run a small coffee shop. Create intents for menu pricing, placing an order, and hours.',
  'Build a support agent for a SaaS product: diagnose_issue, escalate_to_human.',
  'Create a reservation agent for a small restaurant: check_hours, make_reservation.',
  'Build a pet grooming shop agent: booking, cancellation (24hr/$25 fee), pricing.',
];

const TEST_STARTERS = [
  'Hi there',
  'I need help with something',
  'How much does it cost?',
  'Can I book an appointment?',
];

// ─── Main ────────────────────────────────────────────────────────────────

export default function BuildPage() {
  const { settings } = useAppStore();
  const namespace = settings.selectedNamespaceId || 'default';

  const [buildMessages, setBuildMessages] = useState<BuildMessage[]>([]);
  const [testMessages, setTestMessages] = useState<TestMessage[]>([]);
  const [buildInput, setBuildInput] = useState('');
  const [testInput, setTestInput] = useState('');
  const [buildLoading, setBuildLoading] = useState(false);
  const [testLoading, setTestLoading] = useState(false);
  const [intents, setIntents] = useState<IntentInfo[]>([]);
  const [expandedIntent, setExpandedIntent] = useState<string | null>(null);

  const refreshIntents = useCallback(async () => {
    try {
      const list = await api.listIntents();
      setIntents(list);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { refreshIntents(); }, [refreshIntents, namespace]);

  const sendBuild = async () => {
    const userMsg = buildInput.trim();
    if (!userMsg || buildLoading) return;
    setBuildInput('');

    const newMessages: BuildMessage[] = [...buildMessages, { role: 'user', content: userMsg }];
    setBuildMessages(newMessages);
    setBuildLoading(true);
    try {
      const history = buildMessages.map(m => ({ role: m.role, content: m.content }));
      const res = await fetch('/api/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Namespace-ID': namespace },
        body: JSON.stringify({ message: userMsg, history }),
      });
      const data = await res.json();
      setBuildMessages([...newMessages, {
        role: 'assistant',
        content: data.response || data.error || 'No response',
        actions: data.actions || [],
      }]);
      if (data.actions && data.actions.length > 0) {
        refreshIntents();
      }
    } catch (e) {
      setBuildMessages([...newMessages, { role: 'assistant', content: `Error: ${e}` }]);
    }
    setBuildLoading(false);
  };

  const sendTest = async () => {
    const userMsg = testInput.trim();
    if (!userMsg || testLoading) return;
    setTestInput('');

    const newMessages: TestMessage[] = [...testMessages, { role: 'user', content: userMsg }];
    setTestMessages(newMessages);
    setTestLoading(true);
    try {
      const history = testMessages.map(m => {
        const entry: Record<string, unknown> = { role: m.role, content: m.content };
        if (m.intent) entry.intent = m.intent;
        if (m.remark) entry.remark = m.remark;
        if (m.next_intent) entry.next_intent = m.next_intent;
        if (m.context) entry.context = m.context;
        return entry;
      });
      const res = await fetch('/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Namespace-ID': namespace },
        body: JSON.stringify({ query: userMsg, history }),
      });
      const data = await res.json();

      const routing: TestRouting | undefined = data.routing ? {
        intent: data.routing.intent,
        is_transition: !!data.routing.is_transition,
        disposition: data.routing.disposition || 'unknown',
        routing_us: data.routing.routing_us,
        total_us: data.routing.total_us,
      } : undefined;

      // Tag user message with intent if this was a transition
      if (routing?.is_transition && routing.intent) {
        newMessages[newMessages.length - 1] = {
          ...newMessages[newMessages.length - 1],
          intent: routing.intent,
        };
      }

      const assistant: TestMessage = {
        role: 'assistant',
        content: data.response || data.error || '(empty response)',
        remark: data.remark || undefined,
        routing,
        next_intent: data.next_intent || undefined,
        context: data.context || undefined,
        lookups: data.lookups && data.lookups.length > 0 ? data.lookups : undefined,
      };

      setTestMessages([...newMessages, assistant]);
    } catch (e) {
      setTestMessages([...newMessages, { role: 'assistant', content: `Error: ${e}` }]);
    }
    setTestLoading(false);
  };

  const intentCount = intents.length;
  const hasIntents = intentCount > 0;

  return (
    <div className="h-full flex flex-col">
      {/* Top bar: namespace, model, intent count */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-zinc-800 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="text-zinc-500">Namespace</span>
          <span className="font-mono text-violet-400">{namespace}</span>
        </div>
        <div className="text-zinc-600">·</div>
        <div className="flex items-center gap-1.5">
          <span className="text-zinc-500">Intents</span>
          <span className="font-mono text-white">{intentCount}</span>
        </div>
        <div className="text-zinc-600">·</div>
        <div className="flex items-center gap-1.5">
          <span className="text-zinc-500">Model</span>
          <span className="font-mono text-emerald-400/80">llama-4-scout (groq)</span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => { setBuildMessages([]); setTestMessages([]); }}
            className="text-zinc-500 hover:text-red-400 text-[11px] px-2 py-1 rounded border border-zinc-800 hover:border-red-400/40 transition-colors"
          >
            Clear all
          </button>
        </div>
      </div>

      {/* Main: Build (left) | Test (right) */}
      <div className="flex-1 grid grid-cols-[1fr_1fr] min-h-0 divide-x divide-zinc-800">
        <BuildPanel
          messages={buildMessages}
          input={buildInput}
          setInput={setBuildInput}
          loading={buildLoading}
          onSend={sendBuild}
          onClear={() => setBuildMessages([])}
          onStarter={(s) => { setBuildInput(s); }}
          hasIntents={hasIntents}
        />
        <TestPanel
          messages={testMessages}
          input={testInput}
          setInput={setTestInput}
          loading={testLoading}
          onSend={sendTest}
          onClear={() => setTestMessages([])}
          onStarter={(s) => { setTestInput(s); }}
          hasIntents={hasIntents}
          intents={intents}
          expandedIntent={expandedIntent}
          setExpandedIntent={setExpandedIntent}
        />
      </div>
    </div>
  );
}

// ─── Build panel ─────────────────────────────────────────────────────────

function BuildPanel(props: {
  messages: BuildMessage[];
  input: string;
  setInput: (s: string) => void;
  loading: boolean;
  onSend: () => void;
  onClear: () => void;
  onStarter: (s: string) => void;
  hasIntents: boolean;
}) {
  const { messages, input, setInput, loading, onSend, onClear, onStarter } = props;
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div className="flex flex-col min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800 bg-zinc-950/40">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-400" />
          <span className="text-sm font-medium text-zinc-200">Build</span>
          <span className="text-[11px] text-zinc-500">— describe what you want; the AI creates and wires up intents</span>
        </div>
        <button
          onClick={onClear}
          disabled={messages.length === 0}
          className="text-[11px] text-zinc-500 hover:text-zinc-300 disabled:opacity-30"
        >
          Clear
        </button>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <EmptyBuild onStarter={onStarter} />
        )}
        {messages.map((msg, i) => <BuildMessageView key={i} msg={msg} />)}
        {loading && <Thinking label="Building" />}
      </div>

      {/* Input */}
      <InputBar
        value={input}
        onChange={setInput}
        onSend={onSend}
        loading={loading}
        placeholder="Describe what you want your agent to do…"
        accent="blue"
      />
    </div>
  );
}

function EmptyBuild({ onStarter }: { onStarter: (s: string) => void }) {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center px-6">
      <div className="text-zinc-500 text-sm mb-2">Start by describing your business</div>
      <div className="text-zinc-600 text-xs mb-6">The agent will create intents, phrases, and flow logic as you chat.</div>
      <div className="grid grid-cols-1 gap-2 w-full max-w-md">
        {BUILD_STARTERS.map((s, i) => (
          <button
            key={i}
            onClick={() => onStarter(s)}
            className="text-left text-xs px-3 py-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 rounded transition-colors text-zinc-300"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function BuildMessageView({ msg }: { msg: BuildMessage }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[88%] space-y-2 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div className={`rounded-lg px-3.5 py-2 text-sm ${
          isUser ? 'bg-blue-600/20 text-blue-100 border border-blue-500/20' : 'bg-zinc-800 text-zinc-200'
        }`}>
          <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
        </div>
        {msg.actions && msg.actions.length > 0 && (
          <div className="space-y-1.5 w-full">
            {msg.actions.map((a, j) => <ActionCard key={j} action={a} />)}
          </div>
        )}
      </div>
    </div>
  );
}

function ActionCard({ action }: { action: BuildAction }) {
  // Parse "Created intent 'X' with N phrases." → extract intent id
  const created = action.result.match(/Created intent '([^']+)' with (\d+) phrases?/);
  const updated = action.result.match(/Updated '([^']+)' field '([^']+)'/);

  if (created) {
    const [, id, count] = created;
    return (
      <div className="flex items-center gap-2 text-[11px] bg-emerald-950/30 border border-emerald-400/20 rounded px-2.5 py-1.5">
        <span className="text-emerald-400">+</span>
        <span className="font-mono text-emerald-300">{id}</span>
        <span className="text-emerald-400/50">created</span>
        <span className="text-zinc-600">·</span>
        <span className="text-zinc-500">{count} phrases</span>
      </div>
    );
  }
  if (updated) {
    const [, id, field] = updated;
    return (
      <div className="flex items-center gap-2 text-[11px] bg-amber-950/30 border border-amber-400/20 rounded px-2.5 py-1.5">
        <span className="text-amber-400">~</span>
        <span className="font-mono text-amber-300">{id}</span>
        <span className="text-amber-400/50">updated</span>
        <span className="text-zinc-600">·</span>
        <span className="text-zinc-500">{field}</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 text-[11px] bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-400">
      <span className="text-zinc-500">⚡</span>
      <span className="font-mono">{action.tool}</span>
      <span className="text-zinc-600">·</span>
      <span className="text-zinc-500">{action.result}</span>
    </div>
  );
}

// ─── Test panel ──────────────────────────────────────────────────────────

function TestPanel(props: {
  messages: TestMessage[];
  input: string;
  setInput: (s: string) => void;
  loading: boolean;
  onSend: () => void;
  onClear: () => void;
  onStarter: (s: string) => void;
  hasIntents: boolean;
  intents: IntentInfo[];
  expandedIntent: string | null;
  setExpandedIntent: (s: string | null) => void;
}) {
  const {
    messages, input, setInput, loading, onSend, onClear, onStarter,
    hasIntents, intents, expandedIntent, setExpandedIntent,
  } = props;
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div className="flex flex-col min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800 bg-zinc-950/40">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          <span className="text-sm font-medium text-zinc-200">Test</span>
          <span className="text-[11px] text-zinc-500">— talk to your agent; see routing, handoffs, and context</span>
        </div>
        <button
          onClick={onClear}
          disabled={messages.length === 0}
          className="text-[11px] text-zinc-500 hover:text-zinc-300 disabled:opacity-30"
        >
          Clear
        </button>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.length === 0 && (
          <EmptyTest
            onStarter={onStarter}
            hasIntents={hasIntents}
            intents={intents}
            expandedIntent={expandedIntent}
            setExpandedIntent={setExpandedIntent}
          />
        )}
        {messages.map((msg, i) => <TestMessageView key={i} msg={msg} />)}
        {loading && <Thinking label="Thinking" />}
      </div>

      {/* Input */}
      <InputBar
        value={input}
        onChange={setInput}
        onSend={onSend}
        loading={loading}
        placeholder={hasIntents ? 'Say something to test your agent…' : 'Build at least one intent first.'}
        disabled={!hasIntents}
        accent="emerald"
      />
    </div>
  );
}

function EmptyTest(props: {
  onStarter: (s: string) => void;
  hasIntents: boolean;
  intents: IntentInfo[];
  expandedIntent: string | null;
  setExpandedIntent: (s: string | null) => void;
}) {
  const { onStarter, hasIntents, intents, expandedIntent, setExpandedIntent } = props;

  if (!hasIntents) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center px-6">
        <div className="text-zinc-500 text-sm mb-1">No agent yet</div>
        <div className="text-zinc-600 text-xs">Use the Build panel to create your first intents, then come back here to test them.</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col px-1 py-2">
      <div className="text-[11px] text-zinc-500 mb-2 uppercase tracking-wider">Your agent's intents</div>
      <div className="space-y-1.5 mb-6">
        {intents.map(i => {
          const isOpen = expandedIntent === i.id;
          const body = (i.metadata?.instructions?.[0] || i.description || '').trim();
          const phrasesPreview = (i.phrases || []).slice(0, 3).join(' · ');
          return (
            <div key={i.id} className="bg-zinc-900/50 border border-zinc-800 rounded">
              <button
                onClick={() => setExpandedIntent(isOpen ? null : i.id)}
                className="w-full text-left px-3 py-2 flex items-center gap-2 hover:bg-zinc-900"
              >
                <span className={`text-[10px] ${isOpen ? 'rotate-90' : ''} transition-transform text-zinc-600`}>▶</span>
                <span className="font-mono text-sm text-emerald-400/90">{i.id}</span>
                <span className="text-[10px] text-zinc-600">{i.phrases?.length ?? 0} phrases</span>
              </button>
              {isOpen && (
                <div className="px-3 pb-3 space-y-2 text-xs border-t border-zinc-800 pt-2">
                  {body && (
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1">Instructions</div>
                      <div className="text-zinc-300 leading-relaxed whitespace-pre-wrap">{body}</div>
                    </div>
                  )}
                  {phrasesPreview && (
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1">Phrases</div>
                      <div className="text-zinc-400 font-mono">{phrasesPreview}{(i.phrases?.length ?? 0) > 3 ? ' …' : ''}</div>
                    </div>
                  )}
                  {i.metadata?.guardrails && i.metadata.guardrails.length > 0 && (
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1">Guardrails</div>
                      <ul className="text-zinc-400 space-y-0.5">
                        {i.metadata.guardrails.map((g, j) => (
                          <li key={j}>⚠ {g}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="text-[11px] text-zinc-500 mb-2 uppercase tracking-wider">Try asking</div>
      <div className="space-y-1.5">
        {TEST_STARTERS.map((s, i) => (
          <button
            key={i}
            onClick={() => onStarter(s)}
            className="block w-full text-left text-xs px-3 py-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 rounded transition-colors text-zinc-300"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function TestMessageView({ msg }: { msg: TestMessage }) {
  const isUser = msg.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[88%] space-y-1 flex flex-col items-end">
          {msg.intent && (
            <div className="flex items-center gap-1 text-[10px] font-mono text-emerald-400/80">
              <span>↪</span><span>{msg.intent}</span>
            </div>
          )}
          <div className="bg-emerald-600/15 text-emerald-100 border border-emerald-500/20 rounded-lg px-3.5 py-2 text-sm">
            <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[95%] space-y-1.5 flex flex-col w-full">
        {msg.routing && (
          <RoutingBadge routing={msg.routing} />
        )}
        <div className="bg-zinc-800 text-zinc-200 rounded-lg px-3.5 py-2 text-sm">
          <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
        </div>
        {msg.lookups && msg.lookups.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pl-1">
            {msg.lookups.map((l, i) => (
              <span key={i} className="inline-flex items-center gap-1 text-[10px] bg-cyan-950/30 border border-cyan-400/20 text-cyan-200 rounded px-2 py-0.5 font-mono">
                <span className="text-cyan-400">🔍</span>
                {l.intent}
                {l.value !== null && l.value !== undefined && <span className="text-cyan-400/70">= {l.value}</span>}
                {l.error && <span className="text-red-400">· {l.error}</span>}
              </span>
            ))}
          </div>
        )}
        {msg.remark && (
          <div className="text-[10px] text-amber-400/60 italic pl-1">💭 {msg.remark}</div>
        )}
        {msg.next_intent && (
          <div className="flex items-start gap-2 text-[11px] bg-violet-950/30 border border-violet-400/20 rounded px-3 py-2 mt-1">
            <span className="text-violet-400 font-mono shrink-0">→ {msg.next_intent}</span>
            {msg.context && (
              <span className="text-zinc-400 leading-relaxed">
                <span className="text-violet-400/50">context: </span>
                {msg.context}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function RoutingBadge({ routing }: { routing: TestRouting }) {
  if (!routing.intent) {
    return (
      <div className="text-[10px] text-red-400/70 font-mono">· no match</div>
    );
  }
  const arrow = routing.is_transition ? '→' : '·';
  const label = routing.is_transition ? 'enter' : 'continue';
  const color = routing.is_transition ? 'text-emerald-400/90' : 'text-zinc-500';
  const tFmt = (us?: number) => us === undefined ? '' : us >= 1000 ? `${(us/1000).toFixed(0)}ms` : `${us}µs`;
  return (
    <div className={`text-[10px] font-mono ${color} flex items-center gap-2`}>
      <span>{arrow} {routing.intent}</span>
      <span className="text-zinc-600">·</span>
      <span className="text-zinc-500">{label}</span>
      {routing.disposition && routing.disposition !== 'continued' && (
        <>
          <span className="text-zinc-600">·</span>
          <span className="text-zinc-500">{routing.disposition}</span>
        </>
      )}
      {routing.total_us !== undefined && (
        <>
          <span className="text-zinc-600">·</span>
          <span className="text-zinc-600">{tFmt(routing.total_us)}</span>
        </>
      )}
    </div>
  );
}

// ─── Shared bits ─────────────────────────────────────────────────────────

function InputBar(props: {
  value: string;
  onChange: (s: string) => void;
  onSend: () => void;
  loading: boolean;
  placeholder: string;
  disabled?: boolean;
  accent: 'blue' | 'emerald';
}) {
  const { value, onChange, onSend, loading, placeholder, disabled, accent } = props;
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Auto-size textarea
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [value]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const btnColor = accent === 'blue'
    ? 'bg-blue-600 hover:bg-blue-500'
    : 'bg-emerald-600 hover:bg-emerald-500';

  return (
    <div className="border-t border-zinc-800 px-3 py-2.5 bg-zinc-950/40">
      <div className="flex items-end gap-2">
        <textarea
          ref={taRef}
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          disabled={disabled || loading}
          rows={1}
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-zinc-500 resize-none min-h-[38px] max-h-[200px]"
        />
        <button
          onClick={onSend}
          disabled={loading || !value.trim() || disabled}
          className={`shrink-0 ${btnColor} text-white rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors`}
        >
          Send
        </button>
      </div>
      <div className="text-[10px] text-zinc-600 mt-1 pl-1">Enter to send · Shift+Enter for newline</div>
    </div>
  );
}

function Thinking({ label }: { label: string }) {
  return (
    <div className="flex justify-start">
      <div className="bg-zinc-800 rounded-lg px-3.5 py-2 text-sm text-zinc-500 italic flex items-center gap-2">
        <span className="inline-flex gap-0.5">
          <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-pulse" style={{ animationDelay: '0ms' }} />
          <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-pulse" style={{ animationDelay: '200ms' }} />
          <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-pulse" style={{ animationDelay: '400ms' }} />
        </span>
        <span>{label}…</span>
      </div>
    </div>
  );
}
