import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';
import Page from '@/components/Page';
import GenerationPlan from './GenerationPlan';
import { ImportReport } from './ImportReport';
import type { ImportResult } from './ImportReport';

interface McpTool {
  name: string;
  description: string;
  has_input: boolean;
  params: string[];
  required_params: string[];
  read_only: boolean;
}

interface SearchResult {
  qualifiedName: string;
  displayName: string;
  description: string;
  homepage: string;
}

export default function McpImport() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [selectedServer, setSelectedServer] = useState('');
  const [mcpJson, setMcpJson] = useState('');
  const [tools, setTools] = useState<McpTool[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [result, setResult] = useState<ImportResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');
  const [showPaste, setShowPaste] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const { settings } = useAppStore();
  const currentApp = settings.selectedNamespaceId;
  const currentDomain = settings.selectedDomain;
  const languages = settings.languages;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (currentApp && currentApp !== 'default') headers['X-Namespace-ID'] = currentApp;

  // Search Smithery registry
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    setError('');
    setSearchResults([]);
    try {
      const res = await fetch(`/api/import/mcp/search?q=${encodeURIComponent(searchQuery)}&limit=10`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const servers = (data.servers || data || []).map((s: any) => ({
        qualifiedName: s.qualifiedName || s.name || '',
        displayName: s.displayName || s.name || s.qualifiedName || '',
        description: s.description || '',
        homepage: s.homepage || s.repository || '',
      }));
      setSearchResults(servers);
      if (servers.length === 0) setError('No MCP servers found for that search.');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setSearching(false);
    }
  };

  // Fetch tools from a specific server via Smithery
  const fetchServer = async (name: string) => {
    setLoading(true);
    setError('');
    setSelectedServer(name);
    try {
      const res = await fetch(`/api/import/mcp/fetch?name=${encodeURIComponent(name)}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      // Extract tools from Smithery response
      const toolsData = data.tools || [];
      const json = JSON.stringify({ tools: toolsData });
      await parseMcp(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Fetch failed');
      setLoading(false);
    }
  };

  const parseMcp = async (json: string) => {
    setLoading(true);
    setError('');
    setTools([]);
    setResult(null);
    setMcpJson(json);
    try {
      const res = await fetch('/api/import/mcp/parse', {
        method: 'POST', headers,
        body: JSON.stringify({ tools_json: json }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setTools(data.tools);
      setSelected(new Set(data.tools.map((t: McpTool) => t.name)));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Parse failed');
    } finally {
      setLoading(false);
    }
  };

  const applyImport = async () => {
    if (!mcpJson || selected.size === 0) return;
    setImporting(true);
    setError('');
    try {
      const res = await fetch('/api/import/mcp/apply', {
        method: 'POST', headers,
        body: JSON.stringify({ tools_json: mcpJson, selected: Array.from(selected), domain: currentDomain }),
      });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  const toggleTool = (name: string) => setSelected(prev => {
    const n = new Set(prev);
    n.has(name) ? n.delete(name) : n.add(name);
    return n;
  });

  const reset = () => {
    setTools([]);
    setResult(null);
    setMcpJson('');
    setSearchResults([]);
    setSelectedServer('');
    setShowPaste(false);
  };

  const subtitle = (
    <>
      into <span className="text-violet-400 font-mono">{currentApp}</span>
      {currentDomain && (
        <>
          <span className="text-zinc-600 mx-1">/</span>
          <span className="text-violet-400 font-mono">{currentDomain}</span>
        </>
      )}
    </>
  );
  const backAction = (
    <button onClick={() => navigate('/import')} className="text-xs text-zinc-500 hover:text-zinc-100 transition-colors">← Back</button>
  );

  return (
    <Page title="Import MCP Tools" subtitle={subtitle} actions={backAction} size="lg">
      <div className="space-y-6">
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded px-4 py-3 text-sm text-red-400">
          {error}
          <button onClick={() => { setError(''); reset(); }} className="ml-3 text-zinc-500 hover:text-zinc-100">Try again</button>
        </div>
      )}

      {/* Step 1: Search or paste */}
      {tools.length === 0 && !result && (
        <div className="space-y-4">
          {/* Search */}
          <div className="flex gap-2">
            <input
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search MCP servers... (e.g. stripe, github, slack)"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 focus:border-violet-500 focus:outline-none"
              onKeyDown={e => { if (e.key === 'Enter') handleSearch(); }}
              disabled={searching || loading}
            />
            <button onClick={handleSearch} disabled={searching || !searchQuery.trim()}
              className="px-4 py-2 text-sm bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-30">
              {searching ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* Search results */}
          {searchResults.length > 0 && (
            <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/50">
              {searchResults.map(server => (
                <div key={server.qualifiedName}
                  onClick={() => fetchServer(server.qualifiedName)}
                  className="px-4 py-3 cursor-pointer hover:bg-zinc-800/40 transition-colors">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm text-zinc-100 font-medium">{server.displayName}</div>
                      <div className="text-[10px] text-zinc-600 font-mono">{server.qualifiedName}</div>
                    </div>
                    <span className="text-xs text-violet-400 shrink-0">Select →</span>
                  </div>
                  {server.description && (
                    <div className="text-xs text-zinc-500 mt-1">{server.description.slice(0, 150)}</div>
                  )}
                </div>
              ))}
            </div>
          )}

          {loading && (
            <div className="flex items-center gap-2 text-xs text-violet-400">
              <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
              Fetching tools from {selectedServer}...
            </div>
          )}

          {/* Paste / upload fallback */}
          <div className="flex items-center gap-3">
            <div className="flex-1 border-t border-zinc-800" />
            <button onClick={() => setShowPaste(!showPaste)} className="text-[10px] text-zinc-600 hover:text-zinc-400">
              {showPaste ? 'Hide' : 'Or paste tools/list JSON manually'}
            </button>
            <div className="flex-1 border-t border-zinc-800" />
          </div>

          {showPaste && (
            <div className="space-y-2">
              <textarea
                value={mcpJson}
                onChange={e => setMcpJson(e.target.value)}
                placeholder='Paste MCP tools/list JSON here...'
                rows={5}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-xs text-zinc-300 font-mono focus:border-violet-500 focus:outline-none resize-y"
              />
              <div className="flex gap-2">
                <button onClick={() => parseMcp(mcpJson)} disabled={loading || !mcpJson.trim()}
                  className="px-3 py-1.5 text-xs bg-zinc-700 text-zinc-100 rounded hover:bg-zinc-600 disabled:opacity-30">
                  Parse
                </button>
                <input ref={fileRef} type="file" accept=".json" onChange={async e => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  await parseMcp(await file.text());
                  e.target.value = '';
                }} className="hidden" />
                <button onClick={() => fileRef.current?.click()} className="text-xs text-zinc-500 hover:text-violet-400">Upload .json</button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Step 2: Select tools */}
      {tools.length > 0 && !result && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm text-zinc-100 font-semibold">{tools.length} tools</span>
              {selectedServer && <span className="text-xs text-zinc-500 ml-2">from {selectedServer}</span>}
            </div>
            <div className="flex gap-2 text-[10px]">
              <button onClick={() => setSelected(new Set(tools.map(t => t.name)))} className="text-zinc-500 hover:text-violet-400">All</button>
              <button onClick={() => setSelected(new Set())} className="text-zinc-500 hover:text-violet-400">None</button>
              <button onClick={reset} className="text-zinc-500 hover:text-zinc-100">Change source</button>
              <span className="text-zinc-600 ml-2">{selected.size} selected</span>
            </div>
          </div>

          <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/50 max-h-96 overflow-y-auto">
            {tools.map(tool => (
              <label key={tool.name} className={`flex items-start gap-3 px-3 py-2.5 cursor-pointer hover:bg-zinc-800/40 ${selected.has(tool.name) ? '' : 'opacity-40'}`}>
                <input type="checkbox" checked={selected.has(tool.name)} onChange={() => toggleTool(tool.name)} className="mt-1 accent-violet-500" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={`font-mono text-[10px] px-1.5 py-0.5 rounded shrink-0 ${tool.read_only ? 'bg-emerald-900/30 text-emerald-400' : 'bg-blue-900/30 text-blue-400'}`}>
                      {tool.read_only ? 'READ' : 'WRITE'}
                    </span>
                    <span className="text-xs text-zinc-100 font-mono">{tool.name}</span>
                  </div>
                  {tool.description && <div className="text-[10px] text-zinc-500 mt-0.5">{tool.description.slice(0, 120)}</div>}
                  {tool.params.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {tool.params.map(p => (
                        <span key={p} className={`text-[9px] px-1 rounded ${tool.required_params.includes(p) ? 'bg-amber-900/30 text-amber-400' : 'bg-zinc-800 text-zinc-600'}`}>{p}</span>
                      ))}
                    </div>
                  )}
                </div>
              </label>
            ))}
          </div>

          {selected.size > 0 && (
            <GenerationPlan numTools={selected.size} languages={languages} importing={importing} />
          )}

          <div className="flex items-center justify-between pt-2">
            {!importing && <div className="text-xs text-zinc-500">Collision guard active</div>}
            <button onClick={applyImport} disabled={importing || selected.size === 0}
              className="px-5 py-2 text-sm bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-30 shrink-0">
              {importing ? 'Importing...' : `Import ${selected.size} Tools`}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Result */}
      {result && <ImportReport result={result} onViewIntents={() => navigate('/intents')} onImportMore={reset}
        onFixCollisions={() => navigate(`/collisions${currentDomain ? `?domain=${currentDomain}` : ''}`)} />}
      </div>
    </Page>
  );
}

