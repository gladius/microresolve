import { useState, useRef } from 'react';
import { api, type DiscoveredCluster } from '@/api/client';
import { useAppStore } from '@/store';

export default function Discovery() {
  const { settings } = useAppStore();
  const [input, setInput] = useState('');
  const [clusters, setClusters] = useState<(DiscoveredCluster & { selected: boolean; editName: string })[]>([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<{ total_queries: number; total_assigned: number } | null>(null);
  const [applied, setApplied] = useState<string[] | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleDiscover = async () => {
    const queries = input.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    if (queries.length < 10) { alert('Need at least 10 queries.'); return; }
    setLoading(true);
    setApplied(null);
    try {
      const result = await api.discover(queries);
      setClusters(result.clusters.map(c => ({ ...c, selected: c.confidence >= 0.2, editName: c.suggested_name })));
      setStats({ total_queries: result.total_queries, total_assigned: result.total_assigned });
    } catch (e) { alert(e instanceof Error ? e.message : 'Discovery failed'); }
    finally { setLoading(false); }
  };

  const handleApply = async () => {
    const sel = clusters.filter(c => c.selected);
    if (sel.length === 0) return;
    try {
      const result = await api.discoverApply(sel.map(c => ({ name: c.editName, representative_queries: c.representative_queries })));
      setApplied(result.created);
    } catch (e) { alert(e instanceof Error ? e.message : 'Apply failed'); }
  };

  const selectedCount = clusters.filter(c => c.selected).length;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Intent Discovery</h2>
        <p className="text-xs text-zinc-500 mt-1">
          Upload raw queries to discover intent clusters. Creates intents in: <span className="text-violet-400">{settings.selectedNamespaceId}</span>
        </p>
      </div>

      {clusters.length === 0 && !loading && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <input ref={fileRef} type="file" accept=".csv,.txt,.tsv" onChange={e => {
              const file = e.target.files?.[0];
              if (!file) return;
              file.text().then(setInput);
              e.target.value = '';
            }} className="hidden" />
            <button onClick={() => fileRef.current?.click()} className="text-xs text-zinc-400 hover:text-violet-400 border border-zinc-700 rounded px-2 py-1">
              Upload file
            </button>
            <span className="text-[10px] text-zinc-600">{input.split('\n').filter(l => l.trim()).length} queries</span>
          </div>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={"Paste queries, one per line...\n\ncancel my order\nwhere is my package\nI want a refund\n..."}
            rows={8}
            className="w-full bg-zinc-900 text-white text-xs border border-zinc-700 rounded-lg p-3 font-mono focus:outline-none focus:border-violet-500 resize-y"
          />
          <button onClick={handleDiscover} disabled={loading} className="px-4 py-1.5 text-sm bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-50">
            Discover Intents
          </button>
        </div>
      )}

      {loading && (
        <div className="text-center py-8">
          <div className="inline-block w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs text-zinc-400 mt-2">Analyzing queries...</p>
        </div>
      )}

      {clusters.length > 0 && !loading && (
        <>
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-3 text-zinc-400">
              <span><span className="text-white font-medium">{clusters.length}</span> clusters</span>
              {stats && <span>{Math.round((stats.total_assigned / stats.total_queries) * 100)}% assigned</span>}
              <span>{selectedCount} selected</span>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => setClusters(p => p.map(c => ({ ...c, selected: true })))} className="text-zinc-500 hover:text-white">All</button>
              <button onClick={() => setClusters(p => p.map(c => ({ ...c, selected: false })))} className="text-zinc-500 hover:text-white">None</button>
              <button onClick={() => { setClusters([]); setStats(null); setApplied(null); }} className="text-zinc-500 hover:text-white">Reset</button>
            </div>
          </div>

          {applied && (
            <div className="bg-emerald-900/20 border border-emerald-800 rounded px-3 py-2 text-xs text-emerald-400">
              Created {applied.length} intents: {applied.join(', ')}
            </div>
          )}

          <div className="space-y-2">
            {clusters.map((cluster, i) => (
              <div key={i} className={`border rounded-lg p-3 transition-colors ${cluster.selected ? 'bg-zinc-900 border-zinc-700' : 'bg-zinc-950 border-zinc-800 opacity-40'}`}>
                <div className="flex items-start gap-2">
                  <input type="checkbox" checked={cluster.selected} onChange={() => setClusters(p => p.map((c, j) => j === i ? { ...c, selected: !c.selected } : c))} className="mt-1 accent-violet-500" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <input
                        value={cluster.editName}
                        onChange={e => setClusters(p => p.map((c, j) => j === i ? { ...c, editName: e.target.value } : c))}
                        className="bg-transparent text-white text-sm font-medium border-b border-transparent hover:border-zinc-600 focus:border-violet-500 focus:outline-none"
                      />
                      <span className="text-[10px] text-zinc-500">{cluster.size} queries</span>
                      <span className={`text-[10px] px-1 rounded ${cluster.confidence >= 0.7 ? 'bg-emerald-900/50 text-emerald-400' : cluster.confidence >= 0.4 ? 'bg-amber-900/50 text-amber-400' : 'bg-red-900/50 text-red-400'}`}>
                        {Math.round(cluster.confidence * 100)}%
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {cluster.top_terms.slice(0, 6).map((t, ti) => (
                        <span key={ti} className="text-[10px] bg-zinc-800 text-zinc-400 px-1.5 py-0.5 rounded">{t}</span>
                      ))}
                    </div>
                    <div className="mt-1">
                      {cluster.representative_queries.slice(0, 3).map((q, qi) => (
                        <div key={qi} className="text-[10px] text-zinc-500 truncate">{q}</div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <button onClick={handleApply} disabled={selectedCount === 0 || applied !== null} className="px-4 py-1.5 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30">
            Apply Selected ({selectedCount})
          </button>
        </>
      )}
    </div>
  );
}
