import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';
import Page from '@/components/Page';
import GenerationPlan from './GenerationPlan';
import { ImportReport } from './ImportReport';
import type { ImportResult } from './ImportReport';
import DomainPicker from './DomainPicker';

interface ParsedOperation {
  id: string; name: string; method: string; path: string;
  summary: string | null; description: string; tags: string[];
  parameters: { name: string; in: string; required: boolean }[];
  has_body: boolean;
}

interface ParseResult {
  title: string; version: string; description: string | null;
  total_operations: number; tags: string[]; operations: ParsedOperation[];
}


export default function OpenApiImport() {
  const navigate = useNavigate();
  const [specUrl, setSpecUrl] = useState('');
  const [rawSpec, setRawSpec] = useState('');
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [parsed, setParsed] = useState<ParseResult | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [tagFilter, setTagFilter] = useState('');
  const [result, setResult] = useState<ImportResult | null>(null);
  const [error, setError] = useState('');
  // domain: null = picker not ready / invalid (Apply disabled); string = resolved domain
  const [domain, setDomain] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const { settings } = useAppStore();
  const currentApp = settings.selectedNamespaceId;
  const languages = settings.languages;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (currentApp && currentApp !== 'default') headers['X-Namespace-ID'] = currentApp;

  const handleParse = async (content: string) => {
    setLoading(true); setError(''); setParsed(null); setResult(null); setRawSpec(content);
    try {
      const res = await fetch('/api/import/parse', { method: 'POST', headers, body: JSON.stringify({ spec: content }) });
      if (!res.ok) throw new Error(await res.text());
      const data: ParseResult = await res.json();
      setParsed(data);
      setSelected(new Set(data.operations.map(op => op.id)));
    } catch (e) { setError(e instanceof Error ? e.message : 'Parse failed'); }
    finally { setLoading(false); }
  };

  const handleImport = async () => {
    if (!rawSpec || selected.size === 0 || domain === null) return;
    setImporting(true); setError('');
    try {
      const res = await fetch('/api/import/apply', { method: 'POST', headers, body: JSON.stringify({ spec: rawSpec, selected: Array.from(selected), domain }) });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (e) { setError(e instanceof Error ? e.message : 'Import failed'); }
    finally { setImporting(false); }
  };

  const filteredOps = () => !parsed ? [] : !tagFilter ? parsed.operations : parsed.operations.filter(op => op.tags.includes(tagFilter));

  const methodColor = (m: string) => ({ GET: 'bg-emerald-900/30 text-emerald-400', POST: 'bg-blue-900/30 text-blue-400', PUT: 'bg-amber-900/30 text-amber-400', PATCH: 'bg-amber-900/30 text-amber-400', DELETE: 'bg-red-900/30 text-red-400' }[m] || 'bg-zinc-700 text-zinc-400');

  const subtitle = (
    <>
      into <span className="text-emerald-400 font-mono">{currentApp}</span>
    </>
  );
  const backAction = (
    <button onClick={() => navigate('/import')} className="text-xs text-zinc-500 hover:text-zinc-100 transition-colors">← Back</button>
  );

  return (
    <Page title="Import OpenAPI / Swagger" subtitle={subtitle} actions={backAction} size="lg">
      <div className="space-y-6">
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded px-4 py-3 text-sm text-red-400">
          {error}
          <button onClick={() => { setError(''); setParsed(null); setResult(null); }} className="ml-3 text-zinc-500 hover:text-zinc-100">Try again</button>
        </div>
      )}

      {!parsed && !result && (
        <>
          <div className="flex gap-2">
            <input value={specUrl} onChange={e => setSpecUrl(e.target.value)} placeholder="https://api.example.com/openapi.json"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500 focus:outline-none"
              onKeyDown={e => { if (e.key === 'Enter') { fetch(specUrl.trim()).then(r => r.text()).then(handleParse).catch(e => setError(String(e))); }}}
              disabled={loading} />
            <button onClick={() => { fetch(specUrl.trim()).then(r => r.text()).then(handleParse).catch(e => setError(String(e))); }}
              disabled={loading || !specUrl.trim()} className="px-4 py-2 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30">
              {loading ? 'Parsing...' : 'Fetch'}
            </button>
          </div>
          <div className="text-center">
            <input ref={fileRef} type="file" accept=".json,.yaml,.yml" onChange={async e => { const f = e.target.files?.[0]; if (f) await handleParse(await f.text()); e.target.value = ''; }} disabled={loading} className="hidden" />
            <button onClick={() => fileRef.current?.click()} disabled={loading} className="text-sm text-zinc-500 hover:text-emerald-400">or upload a file</button>
          </div>
        </>
      )}

      {parsed && !result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-zinc-100 font-semibold">{parsed.title} <span className="text-zinc-500 text-xs">v{parsed.version}</span></div>
              {parsed.description && <div className="text-xs text-zinc-500 mt-0.5">{parsed.description.slice(0, 120)}</div>}
            </div>
            <button onClick={() => { setParsed(null); setRawSpec(''); setDomain(null); }} className="text-xs text-zinc-500 hover:text-zinc-100">Change spec</button>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            {parsed.tags.length > 0 && (
              <select value={tagFilter} onChange={e => setTagFilter(e.target.value)} className="bg-zinc-800 border border-zinc-700 text-xs text-zinc-300 rounded px-2 py-1 focus:outline-none">
                <option value="">All ({parsed.total_operations})</option>
                {parsed.tags.map(tag => <option key={tag} value={tag}>{tag} ({parsed.operations.filter(op => op.tags.includes(tag)).length})</option>)}
              </select>
            )}
            <button onClick={() => setSelected(new Set(filteredOps().map(op => op.id)))} className="text-[10px] text-zinc-500 hover:text-emerald-400">All</button>
            <button onClick={() => setSelected(new Set())} className="text-[10px] text-zinc-500 hover:text-emerald-400">None</button>
            <span className="text-[10px] text-zinc-600 ml-auto">{selected.size} of {parsed.total_operations} selected</span>
          </div>
          <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/50 max-h-96 overflow-y-auto">
            {filteredOps().map(op => (
              <label key={op.id} className={`flex items-start gap-3 px-3 py-2 cursor-pointer hover:bg-zinc-800/40 ${selected.has(op.id) ? '' : 'opacity-40'}`}>
                <input type="checkbox" checked={selected.has(op.id)} onChange={() => setSelected(prev => { const n = new Set(prev); n.has(op.id) ? n.delete(op.id) : n.add(op.id); return n; })} className="mt-1 accent-emerald-500" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded shrink-0 ${methodColor(op.method)}`}>{op.method}</span>
                    <span className="text-xs text-zinc-300 font-mono truncate">{op.path}</span>
                  </div>
                  <div className="text-xs text-zinc-500 mt-0.5">{op.summary || op.name}</div>
                </div>
                {op.tags.length > 0 && <span className="text-[10px] text-zinc-600 shrink-0">{op.tags[0]}</span>}
              </label>
            ))}
          </div>
          {selected.size > 0 && (
            <GenerationPlan numTools={selected.size} languages={languages} importing={importing} />
          )}

          {/* Domain picker — slug derived from API spec title */}
          <DomainPicker
            suggestedSlug={parsed.title}
            namespaceId={currentApp}
            onChange={setDomain}
          />

          <div className="flex items-center justify-between pt-2">
            {!importing && <div className="text-xs text-zinc-500">Collision guard active</div>}
            <button onClick={handleImport} disabled={importing || selected.size === 0 || domain === null}
              className="px-5 py-2 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30 shrink-0">
              {importing ? 'Importing...' : `Import ${selected.size} Operations`}
            </button>
          </div>
        </div>
      )}

      {result && (
        <ImportReport
          result={result}
          onViewIntents={() => navigate('/intents')}
          onImportMore={() => { setParsed(null); setResult(null); setRawSpec(''); setSpecUrl(''); setDomain(null); }}
        />
      )}
      </div>
    </Page>
  );
}
