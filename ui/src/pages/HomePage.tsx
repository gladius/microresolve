import { useState, useCallback, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { api } from '@/api/client';
import { useAppStore } from '@/store';
import { useFetch } from '@/hooks/useFetch';

interface NamespaceInfo {
  id: string;
  name: string;
  description: string;
  auto_learn: boolean;
  default_threshold: number | null;
}

// Six representative shapes that link to docs/use-case walkthroughs.
// These are NOT "templates" in a setup sense — they're examples of what
// the engine handles, each linking to a docs page with a setup recipe.
const COMMON_SHAPES: { title: string; desc: string; href: string }[] = [
  { title: 'MCP tool routing',          desc: 'Pre-select tools for an LLM agent before calling',    href: 'https://github.com/gladius/microresolve#use-cases' },
  { title: 'Customer support triage',   desc: 'Route incoming tickets to the right team',            href: 'https://github.com/gladius/microresolve#use-cases' },
  { title: 'Multilingual classification', desc: 'Same intent across English, Spanish, Mandarin, …', href: 'https://github.com/gladius/microresolve#use-cases' },
  { title: 'Voice assistant intent',    desc: 'Transcribed audio → structured command',              href: 'https://github.com/gladius/microresolve#use-cases' },
  { title: 'API request triage',        desc: 'Free-text "what they want" → endpoint',               href: 'https://github.com/gladius/microresolve#use-cases' },
  { title: 'Slash-command dispatch',    desc: 'Slack/CLI natural-language → DSL command',            href: 'https://github.com/gladius/microresolve#use-cases' },
];

export default function HomePage() {
  const navigate = useNavigate();
  const { setSelectedNamespaceId } = useAppStore();

  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([]);
  const [loading, setLoading]       = useState(true);
  const [showCreate, setShowCreate] = useState(false);

  const refresh = useCallback(async () => {
    try { setNamespaces(await api.listNamespaces()); } catch { /* */ }
    setLoading(false);
  }, []);

  useFetch(refresh, [refresh]);

  const openNamespace = (id: string) => {
    setSelectedNamespaceId(id);
    navigate('/resolve');
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Top bar */}
      <header className="h-12 flex items-center px-6 border-b border-zinc-900">
        <Link to="/" className="flex items-baseline gap-1.5 hover:opacity-80 transition-opacity">
          <span className="text-emerald-400 font-bold text-lg leading-none">μ</span>
          <span className="text-zinc-100 font-bold text-base tracking-tight leading-none">Resolve</span>
          <span className="text-[9px] font-semibold text-zinc-500 uppercase tracking-[0.15em] leading-none">Studio</span>
        </Link>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12 space-y-12">

        {/* Hero */}
        <section className="space-y-3">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">
            Pre-LLM intent routing in 50µs.
          </h1>
          <p className="text-sm text-zinc-400 leading-relaxed max-w-2xl">
            Route natural-language queries to intents at $0 per call.
            Self-improving from production traffic. One library — Rust · Python · Node · HTTP.
          </p>
        </section>

        {/* Live demo */}
        <TryItWidget namespaces={namespaces} loading={loading} />

        {/* Existing namespaces */}
        <section className="space-y-3">
          <div className="flex items-baseline justify-between">
            <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.15em]">
              Your namespaces
            </div>
            <button
              onClick={() => setShowCreate(true)}
              className="text-xs px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded transition-colors"
            >
              + New namespace
            </button>
          </div>

          {loading ? (
            <div className="text-xs text-zinc-500 py-6">Loading…</div>
          ) : namespaces.length === 0 ? (
            <div className="text-xs text-zinc-600 py-6 border border-dashed border-zinc-800 rounded-lg text-center">
              No namespaces yet. Click <span className="text-zinc-300">+ New namespace</span> to create your first.
            </div>
          ) : (
            <>
              <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/60">
                {namespaces.slice(0, 5).map(ns => (
                  <NamespaceRow
                    key={ns.id}
                    ns={ns}
                    onOpen={() => openNamespace(ns.id)}
                  />
                ))}
              </div>
              <div className="flex justify-end pt-1">
                <button
                  onClick={() => navigate('/namespaces')}
                  className="text-[11px] text-zinc-500 hover:text-emerald-300 transition-colors"
                >
                  {namespaces.length > 5
                    ? `Manage all ${namespaces.length} namespaces →`
                    : 'Manage all namespaces →'}
                </button>
              </div>
            </>
          )}
        </section>

        {/* Common shapes — examples linking to docs */}
        <section className="space-y-3">
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.15em]">
            Common shapes
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {COMMON_SHAPES.map(s => (
              <a
                key={s.title}
                href={s.href}
                target="_blank"
                rel="noopener noreferrer"
                className="block p-3 rounded-lg border border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/50 transition-colors"
              >
                <div className="text-sm text-zinc-200 mb-0.5">{s.title}</div>
                <div className="text-xs text-zinc-500">{s.desc}</div>
              </a>
            ))}
          </div>
          <p className="text-[11px] text-zinc-600 italic">
            One engine handles all of these — different setup paths, same Resolver underneath.
          </p>
        </section>
      </main>

      {showCreate && (
        <CreateNamespaceModal
          onClose={() => setShowCreate(false)}
          onCreated={refresh}
        />
      )}
    </div>
  );
}

// ─── Try-it live demo widget ─────────────────────────────────────────────────

function TryItWidget({ namespaces, loading }: { namespaces: NamespaceInfo[]; loading: boolean }) {
  const [nsId, setNsId]       = useState<string>('');
  const [query, setQuery]     = useState<string>('I want to cancel my order');
  const [running, setRunning] = useState(false);
  const [result, setResult]   = useState<{
    confirmed: { id: string; score: number; confidence: string }[];
    disposition: string;
    routing_us: number;
  } | null>(null);
  const [err, setErr]         = useState<string | null>(null);

  // Auto-pick first non-default namespace once they load
  useEffect(() => {
    if (!nsId && namespaces.length > 0) {
      const useful = namespaces.find(n => n.id !== 'default') ?? namespaces[0];
      setNsId(useful.id);
    }
  }, [namespaces, nsId]);

  const run = async () => {
    if (!query.trim() || !nsId) return;
    setRunning(true);
    setErr(null);
    try {
      // Direct fetch with explicit ns header — don't mutate the global ns state.
      const res = await fetch('/api/route_multi', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Namespace-ID': nsId,
        },
        body: JSON.stringify({ query, log: false }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const data = await res.json();
      setResult({
        confirmed: data.confirmed ?? [],
        disposition: data.disposition ?? 'unknown',
        routing_us: data.routing_us ?? 0,
      });
    } catch (e) {
      setErr('Request failed: ' + (e instanceof Error ? e.message : 'unknown'));
    } finally {
      setRunning(false);
    }
  };

  if (loading) return null;

  if (namespaces.length === 0) {
    return (
      <section className="rounded-lg border border-zinc-800 p-5 bg-zinc-900/30">
        <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.15em] mb-2">
          Try it
        </div>
        <p className="text-xs text-zinc-500">
          Create a namespace below to try a routing call live.
        </p>
      </section>
    );
  }

  const dispoColor = result?.disposition === 'confident'
    ? 'text-emerald-400'
    : result?.disposition === 'low_confidence'
      ? 'text-amber-400'
      : 'text-zinc-500';

  return (
    <section className="rounded-lg border border-zinc-800 p-5 bg-zinc-900/30 space-y-3">
      <div className="flex items-baseline justify-between">
        <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.15em]">
          Try it
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[10px] text-zinc-600 uppercase tracking-wide">Namespace</label>
          <select
            value={nsId}
            onChange={e => { setNsId(e.target.value); setResult(null); }}
            className="bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs rounded px-2 py-1 font-mono focus:outline-none focus:border-emerald-500"
          >
            {namespaces.map(n => <option key={n.id} value={n.id}>{n.id}</option>)}
          </select>
        </div>
      </div>

      <div className="flex gap-2">
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && run()}
          placeholder="type a query — e.g. cancel my subscription"
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-emerald-500"
        />
        <button
          onClick={run}
          disabled={running || !query.trim() || !nsId}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded disabled:opacity-40 transition-colors"
        >
          {running ? 'Routing…' : 'Route'}
        </button>
      </div>

      {err && <div className="text-xs text-red-400">{err}</div>}

      {result && !err && (
        <div className="bg-zinc-950 border border-zinc-800 rounded p-3 font-mono text-xs space-y-1.5">
          {result.confirmed.length === 0 ? (
            <div className="text-zinc-500">no match · {result.routing_us}µs</div>
          ) : (
            result.confirmed.slice(0, 3).map((c, i) => (
              <div key={i} className="flex items-baseline gap-3">
                <span className="text-zinc-300">→</span>
                <span className="text-zinc-100 min-w-[12rem]">{c.id}</span>
                <span className="text-zinc-500">score {c.score.toFixed(2)}</span>
                <span className="text-zinc-600">·</span>
                <span className="text-zinc-500">{c.confidence}</span>
              </div>
            ))
          )}
          <div className="pt-1 mt-1 border-t border-zinc-800/50 flex gap-3 text-[10px]">
            <span className={`${dispoColor} uppercase`}>{result.disposition}</span>
            <span className="text-zinc-600">·</span>
            <span className="text-zinc-500">{result.routing_us}µs</span>
          </div>
        </div>
      )}
    </section>
  );
}

// ─── Create namespace modal ──────────────────────────────────────────────────

function CreateNamespaceModal({
  onClose, onCreated,
}: {
  onClose: () => void;
  onCreated: () => void;
}) {
  const navigate = useNavigate();
  const { setSelectedNamespaceId } = useAppStore();

  const [id, setId]         = useState('');
  const [desc, setDesc]     = useState('');
  const [start, setStart]   = useState<'blank' | 'import' | 'intents'>('blank');
  const [busy, setBusy]     = useState(false);
  const [err, setErr]       = useState('');

  const create = async () => {
    const trimmed = id.trim();
    if (!trimmed) { setErr('Namespace ID is required.'); return; }
    if (!/^[a-z0-9_-]+$/.test(trimmed)) {
      setErr('Lowercase letters, digits, hyphens, underscores only.');
      return;
    }
    if (trimmed.length > 40) { setErr('Max 40 characters.'); return; }
    setBusy(true);
    setErr('');
    try {
      await api.createNamespace(trimmed, '', desc.trim());
      setSelectedNamespaceId(trimmed);
      onCreated();
      const dest = start === 'import' ? '/import' : start === 'intents' ? '/l2' : '/resolve';
      navigate(dest);
    } catch (e) {
      setErr('Failed: ' + (e instanceof Error ? e.message : 'unknown'));
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4"
      onClick={() => !busy && onClose()}
    >
      <div
        className="bg-zinc-900 rounded-lg p-6 max-w-md w-full border border-zinc-800 space-y-4"
        onClick={e => e.stopPropagation()}
      >
        <div className="text-lg text-zinc-100">New namespace</div>

        <div className="space-y-3">
          <div>
            <label className="text-[10px] text-zinc-500 uppercase tracking-wide block mb-1">
              ID <span className="text-zinc-600 normal-case">(immutable, lowercase)</span>
            </label>
            <input
              autoFocus
              value={id}
              onChange={e => { setId(e.target.value); setErr(''); }}
              placeholder="my-namespace"
              className="w-full bg-zinc-950 border border-zinc-700 rounded px-3 py-2 text-sm font-mono focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="text-[10px] text-zinc-500 uppercase tracking-wide block mb-1">Description (optional)</label>
            <input
              value={desc}
              onChange={e => setDesc(e.target.value)}
              placeholder="What this namespace routes"
              className="w-full bg-zinc-950 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="text-[10px] text-zinc-500 uppercase tracking-wide block mb-1">Start with</label>
            <select
              value={start}
              onChange={e => setStart(e.target.value as typeof start)}
              className="w-full bg-zinc-950 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
            >
              <option value="blank">Blank — go to Resolve</option>
              <option value="intents">Hand-craft intents — go to Intents</option>
              <option value="import">Import from spec — go to Import (OpenAPI / MCP / Postman)</option>
            </select>
          </div>
        </div>

        {err && <div className="text-xs text-red-400">{err}</div>}

        <div className="flex justify-end gap-2 pt-2">
          <button onClick={onClose} disabled={busy} className="px-3 py-1.5 text-xs text-zinc-500 hover:text-zinc-100">Cancel</button>
          <button
            onClick={create}
            disabled={busy || !id.trim()}
            className="px-4 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-white rounded disabled:opacity-40"
          >
            {busy ? 'Creating…' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Namespace row + embed snippet ───────────────────────────────────────────

function NamespaceRow({ ns, onOpen }: { ns: NamespaceInfo; onOpen: () => void }) {
  return (
    <button
      onClick={onOpen}
      className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-zinc-900/50 transition-colors"
      title="Open this namespace"
    >
      <span className="font-mono text-sm text-zinc-200 group-hover:text-emerald-300">
        {ns.id}
      </span>
      {ns.auto_learn && (
        <span className="text-[9px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded uppercase tracking-wide">
          auto-learn
        </span>
      )}
      <span className="text-xs text-zinc-500 truncate flex-1">
        {ns.description || <span className="italic text-zinc-700">no description</span>}
      </span>
      <span className="text-[11px] text-zinc-600 shrink-0">→</span>
    </button>
  );
}

