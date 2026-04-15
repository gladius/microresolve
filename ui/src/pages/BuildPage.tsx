import { useState, useRef, useEffect } from 'react';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  actions?: { tool: string; result: string }[];
};

type Mode = 'build' | 'test';

export default function BuildPage() {
  const [mode, setMode] = useState<Mode>('build');
  const [messages, setMessages] = useState<Message[]>([]);
  const [testMessages, setTestMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const currentMessages = mode === 'build' ? messages : testMessages;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [currentMessages]);

  const sendBuild = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');

    const newMessages = [...messages, { role: 'user' as const, content: userMsg }];
    setMessages(newMessages);
    setLoading(true);

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const res = await fetch('/api/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Namespace-ID': getNamespace() },
        body: JSON.stringify({ message: userMsg, history }),
      });
      const data = await res.json();
      setMessages([...newMessages, {
        role: 'assistant',
        content: data.response || data.error || 'No response',
        actions: data.actions,
      }]);
    } catch (e) {
      setMessages([...newMessages, { role: 'assistant', content: `Error: ${e}` }]);
    }
    setLoading(false);
  };

  const sendTest = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');

    const newMessages = [...testMessages, { role: 'user' as const, content: userMsg }];
    setTestMessages(newMessages);
    setLoading(true);

    try {
      const history = testMessages.map(m => {
        const entry: any = { role: m.role, content: m.content };
        if ((m as any).intent) entry.intent = (m as any).intent;
        if ((m as any).remark) entry.remark = (m as any).remark;
        return entry;
      });
      const res = await fetch('/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Namespace-ID': getNamespace() },
        body: JSON.stringify({ query: userMsg, history }),
      });
      const data = await res.json();
      const assistantMsg: any = {
        role: 'assistant',
        content: data.response || data.error || 'No response',
      };
      if (data.remark) assistantMsg.remark = data.remark;
      if (data.routing?.intent) {
        // Tag the user message with intent if it was a transition
        if (data.routing.is_transition) {
          newMessages[newMessages.length - 1] = {
            ...newMessages[newMessages.length - 1],
            ...({ intent: data.routing.intent } as any),
          };
        }
        assistantMsg.routing = data.routing;
      }
      setTestMessages([...newMessages, assistantMsg]);
    } catch (e) {
      setTestMessages([...newMessages, { role: 'assistant', content: `Error: ${e}` }]);
    }
    setLoading(false);
  };

  const send = mode === 'build' ? sendBuild : sendTest;

  return (
    <div className="flex flex-col h-full">
      {/* Mode tabs */}
      <div className="flex gap-2 p-3 border-b border-zinc-800">
        <button
          onClick={() => setMode('build')}
          className={`px-4 py-1.5 rounded text-sm font-medium ${
            mode === 'build' ? 'bg-blue-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
          }`}
        >
          Build
        </button>
        <button
          onClick={() => setMode('test')}
          className={`px-4 py-1.5 rounded text-sm font-medium ${
            mode === 'test' ? 'bg-green-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
          }`}
        >
          Test
        </button>
        <button
          onClick={() => { setTestMessages([]); setMessages([]); }}
          className="ml-auto px-3 py-1.5 rounded text-xs bg-zinc-800 text-zinc-500 hover:bg-zinc-700"
        >
          Clear
        </button>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {currentMessages.length === 0 && (
          <div className="text-zinc-500 text-sm text-center mt-8">
            {mode === 'build'
              ? 'Describe your business to start building an agent.'
              : 'Talk to your agent to test it.'}
          </div>
        )}
        {currentMessages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-lg px-4 py-2.5 text-sm ${
              msg.role === 'user'
                ? 'bg-blue-600/20 text-blue-100'
                : 'bg-zinc-800 text-zinc-200'
            }`}>
              {/* Intent routing info (test mode) */}
              {(msg as any).routing && (
                <div className="text-[10px] text-zinc-500 mb-1 font-mono">
                  {(msg as any).routing.is_transition ? '→' : '·'}{' '}
                  {(msg as any).routing.intent}{' '}
                  ({(msg as any).routing.routing_us}µs)
                </div>
              )}

              {/* Message content */}
              <div className="whitespace-pre-wrap">{msg.content}</div>

              {/* Remark (test mode) */}
              {(msg as any).remark && (
                <div className="mt-1.5 text-[10px] text-amber-400/60 italic">
                  💭 {(msg as any).remark}
                </div>
              )}

              {/* Actions (build mode) */}
              {msg.actions && msg.actions.length > 0 && (
                <div className="mt-2 space-y-1">
                  {msg.actions.map((a, j) => (
                    <div key={j} className="text-[10px] bg-zinc-900 rounded px-2 py-1 font-mono text-green-400/70">
                      ⚡ {a.tool}: {a.result}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-zinc-800 rounded-lg px-4 py-2 text-sm text-zinc-500">
              Thinking...
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-3 border-t border-zinc-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
            placeholder={mode === 'build' ? 'Describe your business...' : 'Talk to your agent...'}
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-500"
            disabled={loading}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className="px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function getNamespace(): string {
  // Read from URL params or localStorage
  const params = new URLSearchParams(window.location.search);
  return params.get('ns') || localStorage.getItem('asv_namespace') || 'default';
}
