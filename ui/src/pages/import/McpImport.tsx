import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';

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

interface ImportResult {
  imported: number;
  seeds_added: number;
  seeds_blocked: number;
  intents: string[];
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
  const currentApp = settings.selectedAppId;
  const headers = { 'Content-Type': 'application/json', 'X-App-ID': currentApp };

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
        body: JSON.stringify({ tools_json: mcpJson, selected: Array.from(selected) }),
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

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate('/import')} className="text-xs text-zinc-500 hover:text-white">← Back</button>
        <div>
          <h2 className="text-lg font-semibold text-white">Import MCP Tools</h2>
          <p className="text-xs text-zinc-500">Into: <span className="text-violet-400 font-mono">{currentApp}</span></p>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded px-4 py-3 text-sm text-red-400">
          {error}
          <button onClick={() => { setError(''); reset(); }} className="ml-3 text-zinc-500 hover:text-white">Try again</button>
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
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-white focus:border-violet-500 focus:outline-none"
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
                      <div className="text-sm text-white font-medium">{server.displayName}</div>
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
              {showPaste ? 'Hide' : 'Or paste JSON manually'}
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
                  className="px-3 py-1.5 text-xs bg-zinc-700 text-white rounded hover:bg-zinc-600 disabled:opacity-30">
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
              <span className="text-sm text-white font-semibold">{tools.length} tools</span>
              {selectedServer && <span className="text-xs text-zinc-500 ml-2">from {selectedServer}</span>}
            </div>
            <div className="flex gap-2 text-[10px]">
              <button onClick={() => setSelected(new Set(tools.map(t => t.name)))} className="text-zinc-500 hover:text-violet-400">All</button>
              <button onClick={() => setSelected(new Set())} className="text-zinc-500 hover:text-violet-400">None</button>
              <button onClick={reset} className="text-zinc-500 hover:text-white">Change source</button>
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
                    <span className="text-xs text-white font-mono">{tool.name}</span>
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

          {/* Token estimate + language info */}
          {!importing && selected.size > 0 && (() => {
            const langs = (() => { try { return JSON.parse(localStorage.getItem('asv_languages') || '["en"]'); } catch { return ['en']; } })() as string[];
            const batches = Math.ceil(selected.size / 10);
            const tokensPerBatch = 300 * selected.size + 150 * selected.size; // ~300 input + ~150 output per tool
            const totalTokens = tokensPerBatch * langs.length;
            const retryTokens = Math.round(totalTokens * 0.3); // ~30% retry overhead
            return (
              <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-500">Estimated tokens</span>
                  <span className="text-white font-mono">~{Math.round((totalTokens + retryTokens) / 1000)}K tokens</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-500">Languages</span>
                  <span className="text-violet-400">{langs.map((l: string) => l.toUpperCase()).join(', ')}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-500">Seeds per tool</span>
                  <span className="text-zinc-400">10 × {langs.length} lang{langs.length > 1 ? 's' : ''} = {10 * langs.length}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-500">LLM batches</span>
                  <span className="text-zinc-400">{batches} batch{batches > 1 ? 'es' : ''} + guard retries</span>
                </div>
                {langs.length > 1 && (
                  <div className="text-[10px] text-amber-400">Multi-language: {langs.length}x seeds = {langs.length}x tokens</div>
                )}
                <div className="text-[10px] text-zinc-600">Language settings can be changed in Settings → Languages</div>
              </div>
            );
          })()}

          <div className="flex items-center justify-between pt-2">
            {importing ? (
              <div className="flex items-center gap-3 text-xs text-violet-400">
                <div className="w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                Generating seeds with AI... {Math.ceil(selected.size / 10) * 10}-{Math.ceil(selected.size / 10) * 20}s estimated.
              </div>
            ) : (
              <div className="text-xs text-zinc-500">Collision guard active</div>
            )}
            <button onClick={applyImport} disabled={importing || selected.size === 0}
              className="px-5 py-2 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30 shrink-0">
              {importing ? 'Importing...' : `Import ${selected.size} Tools`}
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Result */}
      {result && (
        <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-white font-semibold">Import Complete</div>
            <div className="text-right">
              <div className="text-2xl font-bold text-emerald-400">{result.imported}</div>
              <div className="text-xs text-zinc-500">intents created</div>
            </div>
          </div>
          <div className="flex gap-4 text-xs">
            <div><span className="text-emerald-400 font-semibold">{result.seeds_added}</span> <span className="text-zinc-500">seeds</span></div>
            {result.seeds_blocked > 0 && <div><span className="text-amber-400 font-semibold">{result.seeds_blocked}</span> <span className="text-zinc-500">blocked</span></div>}
          </div>
          <div className="flex gap-3 pt-3 border-t border-zinc-700">
            <button onClick={() => navigate('/intents')} className="px-4 py-2 text-sm bg-violet-600 text-white rounded hover:bg-violet-500">View Intents →</button>
            <button onClick={reset} className="px-4 py-2 text-sm border border-zinc-700 text-zinc-400 rounded hover:text-white">Import more</button>
          </div>
        </div>
      )}
    </div>
  );
}
