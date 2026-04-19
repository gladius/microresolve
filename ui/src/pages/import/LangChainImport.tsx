import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';
import Page from '@/components/Page';
import GenerationPlan from './GenerationPlan';
import { ImportReport } from './ImportReport';
import type { ImportResult } from './ImportReport';

const EXAMPLE = `[
  {
    "name": "search_web",
    "description": "Search the web for current information on a topic",
    "args_schema": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "Search query" },
        "num_results": { "type": "integer", "default": 5 }
      },
      "required": ["query"]
    }
  },
  {
    "name": "read_file",
    "description": "Read the contents of a file from disk",
    "args_schema": {
      "type": "object",
      "properties": {
        "path": { "type": "string", "description": "File path to read" }
      },
      "required": ["path"]
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

export default function LangChainImport() {
  const navigate = useNavigate();
  const { settings } = useAppStore();
  const ns = settings.selectedNamespaceId;
  const domain = settings.selectedDomain;

  const [json, setJson] = useState('');
  const [tools, setTools] = useState<Tool[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ImportResult | null>(null);
  const [languages] = useState(['en']);

  const headers = () => {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (ns && ns !== 'default') h['X-Namespace-ID'] = ns;
    return h;
  };

  const parse = async (raw: string) => {
    setError(''); setTools([]); setSelected(new Set()); setResult(null);
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
      into <span className="text-violet-400 font-mono">{ns}</span>
      {domain && <><span className="text-zinc-600 mx-1">/</span><span className="text-violet-400 font-mono">{domain}</span></>}
    </>
  );

  if (result) return (
    <Page title="LangChain Tools — Import Complete" subtitle={subtitle} actions={backAction} size="lg">
      <ImportReport result={result} onViewIntents={() => navigate('/intents')} onImportMore={() => setResult(null)} />
    </Page>
  );

  return (
    <Page title="Import LangChain Tools" subtitle={subtitle} actions={backAction} size="lg">
      <div className="space-y-5">
        <p className="text-xs text-zinc-500 leading-relaxed">
          Paste your LangChain tool definitions — a JSON array where each tool has{' '}
          <code className="text-zinc-300 bg-zinc-800 px-1 py-0.5 rounded">name</code>,{' '}
          <code className="text-zinc-300 bg-zinc-800 px-1 py-0.5 rounded">description</code>, and{' '}
          <code className="text-zinc-300 bg-zinc-800 px-1 py-0.5 rounded">args_schema</code>.
          MicroResolve pre-selects the right tool before your agent's reasoning loop — cutting tool context tokens significantly.
        </p>

        <div className="space-y-2">
          <textarea
            value={json}
            onChange={e => setJson(e.target.value)}
            placeholder={EXAMPLE}
            rows={8}
            className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2.5 text-xs text-zinc-300 font-mono focus:border-violet-500 focus:outline-none resize-y"
          />
          <div className="flex gap-2">
            <button onClick={() => parse(json)} disabled={loading || !json.trim()}
              className="px-3 py-1.5 text-xs bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-30 transition-colors">
              {loading ? 'Parsing...' : 'Parse'}
            </button>
            <button onClick={() => { setJson(EXAMPLE); }}
              className="px-3 py-1.5 text-xs border border-zinc-700 text-zinc-400 rounded hover:text-white hover:border-zinc-500 transition-colors">
              Load example
            </button>
          </div>
        </div>

        {error && <p className="text-xs text-red-400">{error}</p>}

        {tools.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-zinc-400">{tools.length} tools found</span>
              <button onClick={toggleAll} className="text-xs text-violet-400 hover:text-violet-300">
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
                    className="mt-0.5 accent-violet-500 shrink-0" />
                  <div className="min-w-0">
                    <div className="text-xs font-mono text-white">{tool.name}</div>
                    {tool.description && <div className="text-xs text-zinc-500 mt-0.5 line-clamp-2">{tool.description}</div>}
                    {tool.params.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {tool.params.map(p => (
                          <span key={p} className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                            tool.required_params.includes(p)
                              ? 'bg-violet-500/15 text-violet-300'
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

            <button onClick={apply} disabled={importing || selected.size === 0}
              className="w-full py-2 text-sm bg-violet-600 text-white rounded-lg hover:bg-violet-500 disabled:opacity-30 transition-colors">
              {importing ? 'Importing...' : `Import ${selected.size} Tools`}
            </button>
          </div>
        )}
      </div>
    </Page>
  );
}
