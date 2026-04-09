import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '@/api/client';
import { setApiAppId } from '@/api/client';

interface ParsedOperation {
  id: string;
  name: string;
  method: string;
  path: string;
  summary: string | null;
  description: string;
  tags: string[];
  parameters: { name: string; in: string; required: boolean }[];
  has_body: boolean;
}

interface ParseResult {
  title: string;
  version: string;
  description: string | null;
  total_operations: number;
  tags: string[];
  operations: ParsedOperation[];
}

interface ImportResult {
  title: string;
  imported: number;
  seeds_added: number;
  seeds_blocked: number;
  intents: string[];
}

export default function ImportPage() {
  const navigate = useNavigate();
  const [specUrl, setSpecUrl] = useState('');
  const [rawSpec, setRawSpec] = useState('');
  const [appName, setAppName] = useState('');
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [parsed, setParsed] = useState<ParseResult | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [tagFilter, setTagFilter] = useState('');
  const [result, setResult] = useState<ImportResult | null>(null);
  const [error, setError] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const handleParse = async (content: string) => {
    setLoading(true);
    setError('');
    setParsed(null);
    setResult(null);
    setRawSpec(content);
    try {
      const res = await fetch('/api/import/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: content }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: ParseResult = await res.json();
      setParsed(data);
      setSelected(new Set(data.operations.map(op => op.id)));
      // Suggest app name from spec title
      setAppName(data.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Parse failed');
    } finally {
      setLoading(false);
    }
  };

  const handleUrlFetch = async () => {
    if (!specUrl.trim()) return;
    setLoading(true);
    setError('');
    try {
      const res = await fetch(specUrl.trim());
      if (!res.ok) throw new Error(`Failed to fetch: HTTP ${res.status}`);
      await handleParse(await res.text());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Fetch failed');
      setLoading(false);
    }
  };

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await handleParse(await file.text());
    e.target.value = '';
  };

  const handleImport = async () => {
    if (!rawSpec || selected.size === 0 || !appName.trim()) return;
    setImporting(true);
    setError('');
    try {
      // Create the app first
      await api.createApp(appName.trim());

      const res = await fetch('/api/import/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-App-ID': appName.trim() },
        body: JSON.stringify({ spec: rawSpec, selected: Array.from(selected) }),
      });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  const goToApp = () => {
    setApiAppId(appName.trim());
    localStorage.setItem('asv_app_id', appName.trim());
    // Force page reload to pick up new app
    window.location.href = '/intents';
  };

  const toggleOp = (id: string) => setSelected(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  const filteredOps = () => {
    if (!parsed) return [];
    if (!tagFilter) return parsed.operations;
    return parsed.operations.filter(op => op.tags.includes(tagFilter));
  };

  const methodColor = (m: string) => ({
    GET: 'bg-emerald-900/30 text-emerald-400',
    POST: 'bg-blue-900/30 text-blue-400',
    PUT: 'bg-amber-900/30 text-amber-400',
    PATCH: 'bg-amber-900/30 text-amber-400',
    DELETE: 'bg-red-900/30 text-red-400',
  }[m] || 'bg-zinc-700 text-zinc-400');

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-white mb-1">Import API Spec</h2>
        <p className="text-xs text-zinc-500">
          OpenAPI 3.x, Swagger 2.0, or Postman Collection. Each operation becomes a routable intent.
        </p>
      </div>

      {/* Step 1: Upload / URL */}
      {!parsed && !result && (
        <>
          <div className="flex gap-2">
            <input
              value={specUrl}
              onChange={e => setSpecUrl(e.target.value)}
              placeholder="https://api.example.com/openapi.json"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-white focus:border-violet-500 focus:outline-none"
              onKeyDown={e => { if (e.key === 'Enter') handleUrlFetch(); }}
              disabled={loading}
            />
            <button
              onClick={handleUrlFetch}
              disabled={loading || !specUrl.trim()}
              className="px-4 py-2 text-sm bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-30"
            >
              {loading ? 'Parsing...' : 'Fetch'}
            </button>
          </div>
          <div className="text-center">
            <input ref={fileRef} type="file" accept=".json,.yaml,.yml" onChange={handleFile} disabled={loading} className="hidden" />
            <button onClick={() => fileRef.current?.click()} disabled={loading} className="text-sm text-zinc-500 hover:text-violet-400">
              or upload a file
            </button>
          </div>
        </>
      )}

      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded px-4 py-3 text-sm text-red-400">
          {error}
          <button onClick={() => { setError(''); setParsed(null); setResult(null); }} className="ml-3 text-zinc-500 hover:text-white">Try again</button>
        </div>
      )}

      {/* Step 2: Select operations + name app */}
      {parsed && !result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white font-semibold">{parsed.title} <span className="text-zinc-500 text-xs">v{parsed.version}</span></div>
              {parsed.description && <div className="text-xs text-zinc-500 mt-0.5">{parsed.description.slice(0, 120)}</div>}
            </div>
            <button onClick={() => { setParsed(null); setRawSpec(''); }} className="text-xs text-zinc-500 hover:text-white">Change spec</button>
          </div>

          {/* App name */}
          <div className="flex items-center gap-3">
            <label className="text-xs text-zinc-500 shrink-0">App name</label>
            <input
              value={appName}
              onChange={e => setAppName(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ''))}
              placeholder="my-api"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-white font-mono focus:border-violet-500 focus:outline-none max-w-xs"
            />
          </div>

          {/* Tag filter + select controls */}
          <div className="flex items-center gap-3 flex-wrap">
            {parsed.tags.length > 0 && (
              <select
                value={tagFilter}
                onChange={e => setTagFilter(e.target.value)}
                className="bg-zinc-800 border border-zinc-700 text-xs text-zinc-300 rounded px-2 py-1 focus:outline-none"
              >
                <option value="">All ({parsed.total_operations})</option>
                {parsed.tags.map(tag => {
                  const count = parsed.operations.filter(op => op.tags.includes(tag)).length;
                  return <option key={tag} value={tag}>{tag} ({count})</option>;
                })}
              </select>
            )}
            <button onClick={() => setSelected(new Set(filteredOps().map(op => op.id)))} className="text-[10px] text-zinc-500 hover:text-violet-400">Select all</button>
            <button onClick={() => setSelected(new Set())} className="text-[10px] text-zinc-500 hover:text-violet-400">Select none</button>
            <span className="text-[10px] text-zinc-600 ml-auto">{selected.size} of {parsed.total_operations} selected</span>
          </div>

          {/* Operations list */}
          <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/50 max-h-96 overflow-y-auto">
            {filteredOps().map(op => (
              <label key={op.id} className={`flex items-start gap-3 px-3 py-2 cursor-pointer hover:bg-zinc-800/40 ${selected.has(op.id) ? '' : 'opacity-40'}`}>
                <input
                  type="checkbox"
                  checked={selected.has(op.id)}
                  onChange={() => toggleOp(op.id)}
                  className="mt-1 accent-violet-500"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded shrink-0 ${methodColor(op.method)}`}>{op.method}</span>
                    <span className="text-xs text-zinc-300 font-mono truncate">{op.path}</span>
                  </div>
                  <div className="text-xs text-zinc-500 mt-0.5">{op.summary || op.name}</div>
                </div>
                {op.tags.length > 0 && (
                  <span className="text-[10px] text-zinc-600 shrink-0">{op.tags[0]}</span>
                )}
              </label>
            ))}
          </div>

          {/* Import button + progress */}
          <div className="flex items-center justify-between pt-2">
            {importing ? (
              <div className="flex items-center gap-3 text-xs text-violet-400">
                <div className="w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                Generating seeds with AI... This takes 10-30 seconds.
              </div>
            ) : (
              <div className="text-xs text-zinc-500">
                Seeds generated with AI • Collision guard active
              </div>
            )}
            <button
              onClick={handleImport}
              disabled={importing || selected.size === 0 || !appName.trim()}
              className="px-5 py-2 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30 shrink-0"
            >
              {importing ? 'Importing...' : `Import ${selected.size} Operations`}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Result */}
      {result && (
        <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white font-semibold">{result.title}</div>
              <div className="text-xs text-zinc-500 mt-0.5">
                Imported into app: <span className="text-violet-400 font-mono">{appName}</span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-emerald-400">{result.imported}</div>
              <div className="text-xs text-zinc-500">intents created</div>
            </div>
          </div>

          <div className="flex gap-4 text-xs">
            <div>
              <span className="text-emerald-400 font-semibold">{result.seeds_added}</span>
              <span className="text-zinc-500 ml-1">seeds added</span>
            </div>
            {result.seeds_blocked > 0 && (
              <div>
                <span className="text-amber-400 font-semibold">{result.seeds_blocked}</span>
                <span className="text-zinc-500 ml-1">blocked by guard</span>
              </div>
            )}
          </div>

          <div className="flex gap-3 pt-3 border-t border-zinc-700">
            <button
              onClick={goToApp}
              className="px-4 py-2 text-sm bg-violet-600 text-white rounded hover:bg-violet-500"
            >
              Go to {appName} →
            </button>
            <button
              onClick={() => { setParsed(null); setResult(null); setRawSpec(''); setSpecUrl(''); setAppName(''); }}
              className="px-4 py-2 text-sm border border-zinc-700 text-zinc-400 rounded hover:text-white"
            >
              Import another
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
