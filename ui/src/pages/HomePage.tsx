import { useState, useCallback, useMemo } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { api } from '@/api/client';
import { useAppStore } from '@/store';
import { useFetch } from '@/hooks/useFetch';

interface Template {
  id: 'pii' | 'tools' | 'intents';
  icon: string;
  title: string;
  hint: string;
  defaultId: string;
  defaultDescription: string;
  threshold: number | null;
  destinationPath: string;
  destinationLabel: string;
}

const TEMPLATES: Template[] = [
  {
    id: 'pii',
    icon: '🛡',
    title: 'Process user input',
    hint: 'Strip PII, redact credentials, mask identifiers before they reach your LLM.',
    defaultId: 'pii',
    defaultDescription: 'Detect PII, credentials, and sensitive identifiers in user input.',
    threshold: null,
    destinationPath: '/entities',
    destinationLabel: 'configure entity patterns',
  },
  {
    id: 'tools',
    icon: '🔧',
    title: 'Dispatch to tools',
    hint: 'Route a query to the right MCP tool, OpenAPI endpoint, or LangChain function.',
    defaultId: 'tools',
    defaultDescription: 'Route user queries to the correct tool from imported MCP/OpenAPI specs.',
    threshold: 0.3,
    destinationPath: '/import',
    destinationLabel: 'import your tools',
  },
  {
    id: 'intents',
    icon: '💬',
    title: 'Classify user goals',
    hint: 'Categorize free-form messages into your application’s intents.',
    defaultId: 'intents',
    defaultDescription: 'Classify free-form user queries into application intents.',
    threshold: 1.0,
    destinationPath: '/intents',
    destinationLabel: 'add your intents',
  },
];

interface NamespaceInfo {
  id: string;
  name: string;
  description: string;
  auto_learn: boolean;
  default_threshold: number | null;
}

export default function HomePage() {
  const navigate = useNavigate();
  const { setSelectedNamespaceId } = useAppStore();

  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [picked, setPicked] = useState<Template | null>(null);
  const [draftId, setDraftId] = useState('');
  const [draftDesc, setDraftDesc] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const [snippetFor, setSnippetFor] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try { setNamespaces(await api.listNamespaces()); } catch { /* */ }
    setLoading(false);
  }, []);

  useFetch(refresh, [refresh]);

  const pick = (t: Template) => {
    setPicked(t);
    setDraftId(t.defaultId);
    setDraftDesc(t.defaultDescription);
    setErr('');
  };

  const create = async () => {
    if (!picked) return;
    const id = draftId.trim();
    if (!id) { setErr('Namespace ID is required.'); return; }
    if (!/^[a-z0-9_-]+$/.test(id)) { setErr('Lowercase letters, digits, hyphens, underscores only.'); return; }
    if (id.length > 40) { setErr('Max 40 characters.'); return; }
    setBusy(true);
    setErr('');
    try {
      await api.createNamespace(id, '', draftDesc.trim());
      if (picked.threshold !== null) {
        await api.updateNamespace(id, { default_threshold: picked.threshold });
      }
      setSelectedNamespaceId(id);
      navigate(picked.destinationPath);
    } catch (e) {
      setErr('Failed: ' + (e instanceof Error ? e.message : 'unknown'));
      setBusy(false);
    }
  };

  const openNamespace = (id: string) => {
    setSelectedNamespaceId(id);
    navigate('/resolve');
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Minimal top bar — just the brand and a way back to Manage */}
      <header className="h-12 flex items-center px-6 border-b border-zinc-900">
        <Link to="/" className="flex items-baseline gap-1.5 hover:opacity-80 transition-opacity">
          <span className="text-violet-400 font-bold text-lg leading-none">μ</span>
          <span className="text-zinc-100 font-bold text-base tracking-tight leading-none">Resolve</span>
          <span className="text-[9px] font-semibold text-zinc-500 uppercase tracking-[0.15em] leading-none">Studio</span>
        </Link>
        <button
          onClick={() => navigate('/namespaces')}
          className="ml-auto text-xs text-zinc-500 hover:text-zinc-200 transition-colors"
        >
          Manage all namespaces →
        </button>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12 space-y-12">

        {/* Hero */}
        <section className="space-y-3">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">
            Train here. Run anywhere.
          </h1>
          <p className="text-sm text-zinc-400 leading-relaxed max-w-2xl">
            Studio is where you configure routing — define intents, distill entity
            patterns, and watch the system learn from real queries. Your application loads
            the trained namespace via the embedded <span className="text-zinc-300 font-medium">microresolve</span> library.
            No server, no LLM at runtime, ~30 µs per query.
          </p>
        </section>

        {/* Templates */}
        <section className="space-y-4">
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.15em]">
            Start a new namespace
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {TEMPLATES.map(t => (
              <button
                key={t.id}
                onClick={() => pick(t)}
                className={`text-left p-4 rounded-lg border transition-colors ${
                  picked?.id === t.id
                    ? 'border-violet-500/60 bg-violet-500/5'
                    : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/50'
                }`}
              >
                <div className="flex items-baseline gap-2 mb-1.5">
                  <span className="text-base">{t.icon}</span>
                  <span className="text-sm font-medium text-zinc-100">{t.title}</span>
                </div>
                <div className="text-xs text-zinc-500 leading-relaxed">{t.hint}</div>
              </button>
            ))}
          </div>

          <p className="text-[11px] text-zinc-600 italic">
            One engine, layered configuration. You can mix all three concerns inside one namespace once you’re set up.
          </p>

          {/* Inline create form */}
          {picked && (
            <div className="p-5 rounded-lg border border-violet-500/30 bg-violet-500/5 space-y-3">
              <div className="flex items-baseline gap-2">
                <span>{picked.icon}</span>
                <h3 className="text-sm font-medium text-zinc-100">
                  Create — <span className="text-zinc-400">{picked.title.toLowerCase()}</span>
                </h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] text-zinc-500 uppercase tracking-wide block mb-1">
                    Namespace ID <span className="text-zinc-600 normal-case">(immutable)</span>
                  </label>
                  <input
                    autoFocus
                    value={draftId}
                    onChange={e => { setDraftId(e.target.value); setErr(''); }}
                    onKeyDown={e => e.key === 'Enter' && create()}
                    className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 font-mono focus:outline-none focus:border-violet-500"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-zinc-500 uppercase tracking-wide block mb-1">Description</label>
                  <input
                    value={draftDesc}
                    onChange={e => setDraftDesc(e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-violet-500"
                  />
                </div>
              </div>

              {err && <p className="text-xs text-red-400">{err}</p>}

              <div className="flex items-center justify-between pt-1">
                <p className="text-[11px] text-zinc-600">
                  {picked.threshold !== null ? <>Threshold preset: <span className="font-mono text-zinc-400">{picked.threshold}</span></> : 'Uses system default threshold.'}
                </p>
                <div className="flex gap-2">
                  <button onClick={() => setPicked(null)} className="px-3 py-1.5 text-xs text-zinc-500 hover:text-zinc-100">Cancel</button>
                  <button
                    onClick={create}
                    disabled={busy || !draftId.trim()}
                    className="px-4 py-1.5 text-xs bg-violet-600 hover:bg-violet-500 text-white rounded disabled:opacity-40"
                  >
                    {busy ? 'Creating…' : `Create & ${picked.destinationLabel}`}
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Existing namespaces */}
        <section className="space-y-3">
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.15em]">
            Your trained namespaces
          </div>

          {loading ? (
            <div className="text-xs text-zinc-500 py-6">Loading…</div>
          ) : namespaces.length === 0 ? (
            <div className="text-xs text-zinc-600 py-6 border border-dashed border-zinc-800 rounded-lg text-center">
              Pick a template above to create your first namespace.
            </div>
          ) : (
            <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/60">
              {namespaces.map(ns => (
                <NamespaceRow
                  key={ns.id}
                  ns={ns}
                  expanded={snippetFor === ns.id}
                  onToggle={() => setSnippetFor(snippetFor === ns.id ? null : ns.id)}
                  onOpen={() => openNamespace(ns.id)}
                />
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Namespace row with collapsible "Use in your app" snippets

function NamespaceRow({
  ns, expanded, onToggle, onOpen,
}: {
  ns: NamespaceInfo;
  expanded: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  const [lang, setLang] = useState<'rust' | 'python' | 'node'>('rust');

  return (
    <div className="px-4 py-3">
      <div className="flex items-center gap-3">
        <button
          onClick={onOpen}
          className="font-mono text-sm text-zinc-200 hover:text-violet-300 transition-colors"
          title="Open this namespace"
        >
          {ns.id}
        </button>
        {ns.auto_learn && (
          <span className="text-[9px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded uppercase tracking-wide">
            auto-learn
          </span>
        )}
        {ns.default_threshold != null && (
          <span className="text-[9px] text-zinc-500 bg-zinc-800/60 px-1.5 py-0.5 rounded font-mono">
            thr {ns.default_threshold}
          </span>
        )}
        <span className="text-xs text-zinc-500 truncate">
          {ns.description || <span className="italic text-zinc-700">no description</span>}
        </span>
        <button
          onClick={onToggle}
          className="ml-auto text-[11px] text-zinc-500 hover:text-violet-300 transition-colors flex items-center gap-1"
        >
          Use in your app
          <span className={`transition-transform ${expanded ? 'rotate-180' : ''}`}>▾</span>
        </button>
      </div>

      {expanded && (
        <div className="mt-3 ml-0 border border-zinc-800 rounded-lg overflow-hidden bg-zinc-950">
          {/* Lang tabs */}
          <div className="flex border-b border-zinc-800 text-[11px] font-medium">
            {(['rust', 'python', 'node'] as const).map(l => (
              <button
                key={l}
                onClick={() => setLang(l)}
                className={`px-3 py-2 transition-colors ${
                  lang === l
                    ? 'text-violet-300 border-b border-violet-500'
                    : 'text-zinc-500 hover:text-zinc-200'
                }`}
              >
                {l === 'rust' ? 'Rust' : l === 'python' ? 'Python' : 'Node'}
              </button>
            ))}
            <span className="ml-auto text-[10px] text-zinc-600 self-center pr-3">
              embedded library · zero network
            </span>
          </div>

          <SnippetBlock lang={lang} nsId={ns.id} />
        </div>
      )}
    </div>
  );
}

function SnippetBlock({ lang, nsId }: { lang: 'rust' | 'python' | 'node'; nsId: string }) {
  // Actual data location written by the studio
  const dataPath = `~/.local/share/microresolve/${nsId}`;

  const snippet = useMemo(() => {
    if (lang === 'rust') {
      return [
        `use microresolve::Router;`,
        `use std::path::Path;`,
        ``,
        `let router = Router::load_from_dir(Path::new("${dataPath}"))?;`,
        `let result = router.resolve("cancel my subscription", 0.3, 1.5);`,
      ].join('\n');
    }
    if (lang === 'python') {
      return [
        `# Export the namespace from Studio first:`,
        `#   curl -H "X-Namespace-ID: ${nsId}" http://localhost:3001/api/export > ${nsId}.json`,
        ``,
        `from microresolve import Router`,
        ``,
        `with open("${nsId}.json") as f:`,
        `    router = Router.import_json(f.read())`,
        ``,
        `result = router.resolve("cancel my subscription")`,
      ].join('\n');
    }
    // node
    return [
      `// Export the namespace from Studio first:`,
      `//   curl -H "X-Namespace-ID: ${nsId}" http://localhost:3001/api/export > ${nsId}.json`,
      ``,
      `import { readFileSync } from 'node:fs';`,
      `import { Router } from '@microresolve/core';`,
      ``,
      `const router = Router.importJson(readFileSync('${nsId}.json', 'utf8'));`,
      `const result = router.resolve('cancel my subscription');`,
    ].join('\n');
  }, [lang, nsId, dataPath]);

  const copy = () => { navigator.clipboard.writeText(snippet).catch(() => {}); };

  return (
    <div className="relative">
      <pre className="text-[12px] leading-relaxed text-zinc-300 font-mono p-4 overflow-x-auto whitespace-pre">
{snippet}
      </pre>
      <button
        onClick={copy}
        className="absolute top-2 right-2 text-[10px] text-zinc-500 hover:text-zinc-200 bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5"
      >
        Copy
      </button>
    </div>
  );
}
