import { useState, useRef } from 'react';
import { api } from '@/api/client';

interface ImportResult {
  title: string;
  version: string;
  total_operations: number;
  created: number;
  skipped: number;
  seeds_added?: number;
  seeds_blocked?: number;
  intents: any[];
}

export default function ImportPage() {
  const [specUrl, setSpecUrl] = useState('');
  const [specContent, setSpecContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [useLLM, setUseLLM] = useState(true);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [error, setError] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const handleImport = async (content: string) => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const endpoint = useLLM ? '/import/spec/llm' : '/import/spec';
      const res = await fetch(`/api${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-App-ID': localStorage.getItem('asv_app_id') || 'default' },
        body: JSON.stringify({ spec: content }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Import failed');
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
      const text = await res.text();
      setSpecContent(text);
      await handleImport(text);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Fetch failed');
      setLoading(false);
    }
  };

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setSpecContent(text);
    await handleImport(text);
    e.target.value = '';
  };

  const handlePaste = async () => {
    if (!specContent.trim()) return;
    await handleImport(specContent);
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-white mb-1">Import API Spec</h2>
        <p className="text-xs text-zinc-500">
          Import OpenAPI (3.x / Swagger 2.0) or Postman Collection. Each operation becomes a routable intent.
        </p>
      </div>

      {/* LLM toggle */}
      <div className="flex items-center gap-3">
        <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
          <input
            type="checkbox"
            checked={useLLM}
            onChange={e => setUseLLM(e.target.checked)}
            className="rounded"
          />
          Generate seeds with AI (recommended — better vocabulary coverage)
        </label>
      </div>

      {/* URL input */}
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
          {loading ? 'Importing...' : 'Fetch & Import'}
        </button>
      </div>

      {/* File upload */}
      <div className="text-center">
        <input
          ref={fileRef}
          type="file"
          accept=".json,.yaml,.yml"
          onChange={handleFile}
          disabled={loading}
          className="hidden"
        />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={loading}
          className="text-sm text-zinc-500 hover:text-violet-400 transition-colors"
        >
          or upload a file (.json, .yaml)
        </button>
      </div>

      {/* Paste area */}
      <div>
        <textarea
          value={specContent}
          onChange={e => setSpecContent(e.target.value)}
          placeholder="or paste spec content here (JSON or YAML)..."
          className="w-full h-32 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-xs text-zinc-300 font-mono focus:border-violet-500 focus:outline-none resize-y"
          disabled={loading}
        />
        {specContent.trim() && (
          <button
            onClick={handlePaste}
            disabled={loading}
            className="mt-2 px-4 py-1.5 text-xs bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30"
          >
            {loading ? 'Importing...' : 'Import Pasted Spec'}
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white font-semibold">{result.title}</div>
              <div className="text-xs text-zinc-500">v{result.version}</div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-emerald-400">{result.created}</div>
              <div className="text-xs text-zinc-500">intents created</div>
            </div>
          </div>

          {result.seeds_added !== undefined && (
            <div className="flex gap-4 text-xs">
              <div>
                <span className="text-emerald-400 font-semibold">{result.seeds_added}</span>
                <span className="text-zinc-500 ml-1">seeds added</span>
              </div>
              {result.seeds_blocked !== undefined && result.seeds_blocked > 0 && (
                <div>
                  <span className="text-amber-400 font-semibold">{result.seeds_blocked}</span>
                  <span className="text-zinc-500 ml-1">seeds blocked (collision guard)</span>
                </div>
              )}
            </div>
          )}

          {result.skipped > 0 && (
            <div className="text-xs text-zinc-500">
              {result.skipped} operations skipped (no description)
            </div>
          )}

          {/* Intent list */}
          <div className="space-y-1 max-h-80 overflow-y-auto">
            <div className="text-[10px] text-zinc-500 uppercase font-semibold mb-2">Created Intents</div>
            {result.intents.map((intent: any, i: number) => (
              <div key={i} className="flex items-center gap-2 text-xs py-1 border-b border-zinc-800/50">
                {intent.method && (
                  <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded ${
                    intent.method === 'GET' ? 'bg-emerald-900/30 text-emerald-400' :
                    intent.method === 'POST' ? 'bg-blue-900/30 text-blue-400' :
                    intent.method === 'DELETE' ? 'bg-red-900/30 text-red-400' :
                    'bg-zinc-700 text-zinc-400'
                  }`}>{intent.method}</span>
                )}
                <span className="text-violet-400 font-mono">
                  {intent.intent_id || intent}
                </span>
                {intent.endpoint && (
                  <span className="text-zinc-600 font-mono text-[10px] ml-auto">
                    {intent.endpoint}
                  </span>
                )}
                {intent.seeds !== undefined && (
                  <span className="text-zinc-600 text-[10px]">
                    {intent.seeds} seeds
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
