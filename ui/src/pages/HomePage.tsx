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
  intent_count?: number;
}

const USE_CASES_URL = 'https://github.com/gladius/microresolve#use-cases';

export default function HomePage() {
  const navigate = useNavigate();
  const { setSelectedNamespaceId } = useAppStore();

  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([]);
  const [loading, setLoading]       = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [stats, setStats] = useState<{ clients: number; reviewPending: number }>({
    clients: 0,
    reviewPending: 0,
  });

  const refresh = useCallback(async () => {
    try { setNamespaces(await api.listNamespaces()); } catch { /* */ }
    setLoading(false);
  }, []);

  useFetch(refresh, [refresh]);

  // Stats strip — single fetch, no polling. The home page is short-lived;
  // accuracy on visit is what matters, not real-time updates.
  useEffect(() => {
    Promise.all([
      fetch('/api/connected_clients').then(r => r.ok ? r.json() : { clients: [] }).catch(() => ({ clients: [] })),
      api.getReviewStats().catch(() => ({ pending: 0 })),
    ]).then(([conn, review]) => {
      setStats({
        clients: (conn.clients ?? []).length,
        reviewPending: review.pending ?? 0,
      });
    });
  }, []);

  const totalIntents = namespaces.reduce((sum, ns) => sum + (ns.intent_count ?? 0), 0);

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
            The pre-LLM reflex layer.
          </h1>
          <p className="text-sm text-zinc-400 leading-relaxed max-w-2xl">
            The decisions your LLM keeps making — which tool, what guardrail, who to refuse — in 50µs at $0 per call.
            Learns from production traffic. One library — Rust · Python · Node · HTTP.
          </p>
          {/* Stats strip — at-a-glance state for repeat visitors. */}
          {!loading && namespaces.length > 0 && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-zinc-500 pt-2">
              <Stat n={namespaces.length} label="namespace" plural="namespaces" />
              <span className="text-zinc-700">·</span>
              <Stat n={totalIntents} label="intent" plural="intents" />
              <span className="text-zinc-700">·</span>
              <Link to="/connected" className="hover:text-emerald-300">
                <Stat n={stats.clients} label="connected client" plural="connected clients" />
              </Link>
              {stats.reviewPending > 0 && (
                <>
                  <span className="text-zinc-700">·</span>
                  <Link to="/review" className="text-amber-400 hover:text-amber-300">
                    <Stat n={stats.reviewPending} label="review item" plural="review items" />
                  </Link>
                </>
              )}
            </div>
          )}
        </section>

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
            <div className="border border-dashed border-zinc-800 rounded-lg p-6 space-y-3 text-center">
              <div className="text-sm text-zinc-300">
                No namespaces yet — click <span className="text-emerald-400">+ New namespace</span> to create your first.
              </div>
              <div className="text-xs text-zinc-500">
                Quickest path: paste an OpenAPI / MCP spec on the{' '}
                <button onClick={() => navigate('/import')} className="text-emerald-400 hover:underline">Import</button>{' '}
                page to get a populated namespace in 30 seconds.
              </div>
              <div className="text-[11px] text-zinc-600 pt-1">
                Not sure what to use this for?{' '}
                <a href={USE_CASES_URL} target="_blank" rel="noopener noreferrer" className="text-zinc-400 hover:text-emerald-300 underline">
                  See use cases →
                </a>
              </div>
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

// ─── Create namespace modal ──────────────────────────────────────────────────

function CreateNamespaceModal({
  onClose, onCreated,
}: {
  onClose: () => void;
  onCreated: () => void;
}) {
  const { setSelectedNamespaceId } = useAppStore();

  const [id, setId]     = useState('');
  const [desc, setDesc] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr]   = useState('');

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
      // Select it but stay on Home — the new namespace shows up in the list
      // and the user picks Import / Intents / Resolve from the sidebar at
      // their own pace. No forced redirect.
      setSelectedNamespaceId(trimmed);
      onCreated();
      onClose();
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

function Stat({ n, label, plural }: { n: number; label: string; plural: string }) {
  return (
    <span><span className="text-zinc-300 font-medium tabular-nums">{n}</span> {n === 1 ? label : plural}</span>
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

