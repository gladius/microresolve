import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';
import Page from '@/components/Page';
import GenerationPlan from './GenerationPlan';
import { ImportReport } from './ImportReport';
import type { ImportResult } from './ImportReport';
import DomainPicker from './DomainPicker';

const EXAMPLE = `[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string", "description": "City name" },
          "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
        },
        "required": ["location"]
      }
    }
  }
]`;

interface Tool {
  name: string;
  description: string;
  has_input: boolean;
  params: string[];
  required_params: string[];
}

/**
 * Derive a slug suggestion from parsed tools.
 * If all tools share a common snake_case prefix (e.g. "stripe_"), use that.
 * If a "namespace" field is present in the raw JSON, use it.
 * Otherwise fall back to "tools".
 */
function deriveSlugFromTools(tools: Tool[], sourceName: string): string {
  if (sourceName) return sourceName;
  if (tools.length === 0) return 'tools';
  // Try common prefix from tool names (split by _)
  const prefixes = tools.map(t => t.name.split('_')[0]).filter(Boolean);
  const first = prefixes[0];
  if (first && prefixes.every(p => p === first)) return first;
  return 'tools';
}

export default function OpenAIFunctionsImport() {
  const navigate = useNavigate();
  const { settings } = useAppStore();
  const ns = settings.selectedNamespaceId;

  const [json, setJson] = useState('');
  const [sourceName, setSourceName] = useState(''); // derived from filename on upload
  const [tools, setTools] = useState<Tool[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ImportResult | null>(null);
  const [languages] = useState(['en']);
  // domain: null = picker not ready / invalid (Apply disabled); string = resolved domain
  const [domain, setDomain] = useState<string | null>(null);

  const slugSuggestion = deriveSlugFromTools(tools, sourceName);

  const headers = () => {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (ns && ns !== 'default') h['X-Namespace-ID'] = ns;
    return h;
  };

  const parse = async (raw: string, name?: string) => {
    setError(''); setTools([]); setSelected(new Set()); setResult(null); setDomain(null);
    if (name !== undefined) setSourceName(name);
    setLoading(true);
    try {
      const r = await fetch('/api/import/mcp/parse', {
        method: 'POST', headers: headers(),
        body: JSON.stringify({ tools_json: raw }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setTools(data.tools);
      setSelected(new Set(data.tools.map((t: Tool) => t.name)));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Parse failed');
    }
    setLoading(false);
  };

  const apply = async () => {
    if (domain === null) return;
    setImporting(true); setError('');
    try {
      const r = await fetch('/api/import/mcp/apply', {
        method: 'POST', headers: headers(),
        body: JSON.stringify({ tools_json: json, selected: [...selected], domain }),
      });
      if (!r.ok) throw new Error(await r.text());
      setResult(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Import failed');
    }
    setImporting(false);
  };

  const toggleAll = () => {
    setSelected(selected.size === tools.length ? new Set() : new Set(tools.map(t => t.name)));
  };

  const backAction = (
    <button onClick={() => navigate('/import')}
      className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">
      ← Import
    </button>
  );

  const subtitle = (
    <>
      into <span className="text-emerald-400 font-mono">{ns}</span>
    </>
  );

  if (result) return (
    <Page title="OpenAI Functions — Import Complete" subtitle={subtitle} actions={backAction} size="lg">
      <ImportReport result={result} onViewIntents={() => navigate('/l2')} onImportMore={() => { setResult(null); setTools([]); setSourceName(''); setDomain(null); }} />
    </Page>
  );

  return (
    <Page title="Import OpenAI Functions" subtitle={subtitle} actions={backAction} size="lg">
      <div className="space-y-5">
        <p className="text-xs text-zinc-500 leading-relaxed">
          Paste your OpenAI <code className="text-zinc-300 bg-zinc-800 px-1 py-0.5 rounded">functions</code> or{' '}
          <code className="text-zinc-300 bg-zinc-800 px-1 py-0.5 rounded">tools</code> array.
          Each function becomes a routable intent — MicroResolve pre-selects the right function before your LLM call,
          reducing context tokens by up to 90%.
        </p>

        <div className="space-y-2">
          <textarea
            value={json}
            onChange={e => setJson(e.target.value)}
            placeholder={EXAMPLE}
            rows={8}
            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2.5 text-xs text-zinc-300 font-mono focus:border-emerald-500 focus:outline-none resize-y"
          />
          <div className="flex gap-2 items-center">
            <button onClick={() => parse(json)} disabled={loading || !json.trim()}
              className="px-3 py-1.5 text-xs bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30 transition-colors">
              {loading ? 'Parsing...' : 'Parse'}
            </button>
            <button onClick={() => { setJson(EXAMPLE); }}
              className="px-3 py-1.5 text-xs border border-zinc-700 text-zinc-400 rounded hover:text-zinc-100 hover:border-zinc-500 transition-colors">
              Load example
            </button>
            <label className="text-xs text-zinc-500 hover:text-emerald-400 cursor-pointer">
              Upload .json
              <input type="file" accept=".json" className="hidden" onChange={async e => {
                const f = e.target.files?.[0];
                if (!f) return;
                const raw = await f.text();
                const name = f.name.replace(/\.json$/i, '');
                setJson(raw);
                await parse(raw, name);
                e.target.value = '';
              }} />
            </label>
          </div>
        </div>

        {error && <p className="text-xs text-red-400">{error}</p>}

        {tools.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-zinc-400">{tools.length} functions found</span>
              <button onClick={toggleAll} className="text-xs text-emerald-400 hover:text-emerald-300">
                {selected.size === tools.length ? 'Deselect all' : 'Select all'}
              </button>
            </div>

            <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/50 max-h-80 overflow-y-auto">
              {tools.map(tool => (
                <label key={tool.name} className="flex items-start gap-3 px-4 py-3 hover:bg-zinc-800/40 cursor-pointer">
                  <input type="checkbox" checked={selected.has(tool.name)}
                    onChange={e => {
                      const next = new Set(selected);
                      e.target.checked ? next.add(tool.name) : next.delete(tool.name);
                      setSelected(next);
                    }}
                    className="mt-0.5 accent-emerald-500 shrink-0" />
                  <div className="min-w-0">
                    <div className="text-xs font-mono text-zinc-100">{tool.name}</div>
                    {tool.description && <div className="text-xs text-zinc-500 mt-0.5 line-clamp-2">{tool.description}</div>}
                    {tool.params.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {tool.params.map(p => (
                          <span key={p} className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                            tool.required_params.includes(p)
                              ? 'bg-emerald-500/15 text-emerald-300'
                              : 'bg-zinc-800 text-zinc-500'
                          }`}>{p}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </label>
              ))}
            </div>

            <GenerationPlan numTools={selected.size} languages={languages} importing={importing} />

            {/* Domain picker — slug derived from filename or common tool prefix */}
            <DomainPicker
              suggestedSlug={slugSuggestion}
              namespaceId={ns}
              onChange={setDomain}
            />

            <button onClick={apply} disabled={importing || selected.size === 0 || domain === null}
              className="w-full py-2 text-sm bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 disabled:opacity-30 transition-colors">
              {importing ? 'Importing...' : `Import ${selected.size} Functions`}
            </button>
          </div>
        )}
      </div>
    </Page>
  );
}
