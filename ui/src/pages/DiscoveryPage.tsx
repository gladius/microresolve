import { useState } from 'react';
import { api, type DiscoveredCluster } from '@/api/client';
import { useAppStore } from '@/store';

export default function DiscoveryPage() {
  const { settings } = useAppStore();
  const [input, setInput] = useState('');
  const [clusters, setClusters] = useState<(DiscoveredCluster & { selected: boolean; editName: string })[]>([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<{ total_queries: number; total_assigned: number } | null>(null);
  const [applied, setApplied] = useState<string[] | null>(null);

  const handleDiscover = async () => {
    const queries = input
      .split('\n')
      .map(l => l.trim())
      .filter(l => l.length > 0);

    if (queries.length < 10) {
      alert('Need at least 10 queries for discovery.');
      return;
    }

    setLoading(true);
    setApplied(null);
    try {
      const result = await api.discover(queries);
      setClusters(
        result.clusters.map(c => ({ ...c, selected: true, editName: c.suggested_name }))
      );
      setStats({ total_queries: result.total_queries, total_assigned: result.total_assigned });
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Discovery failed');
    } finally {
      setLoading(false);
    }
  };

  const handleApply = async () => {
    const selected = clusters.filter(c => c.selected);
    if (selected.length === 0) return;

    try {
      const result = await api.discoverApply(
        selected.map(c => ({
          name: c.editName,
          representative_queries: c.representative_queries,
        }))
      );
      setApplied(result.created);
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Apply failed');
    }
  };

  const toggleAll = (val: boolean) => {
    setClusters(prev => prev.map(c => ({ ...c, selected: val })));
  };

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setInput(reader.result);
      }
    };
    reader.readAsText(file);
  };

  const selectedCount = clusters.filter(c => c.selected).length;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Intent Discovery</h1>
        <p className="text-sm text-zinc-400 mt-1">
          Upload raw queries to automatically discover intent clusters.
          Results will be created in app: <span className="text-blue-400 font-medium">{settings.selectedAppId}</span>
        </p>
      </div>

      {/* Input section */}
      {clusters.length === 0 && !loading && (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <label className="text-sm text-zinc-400">
              Upload CSV/TXT:
              <input
                type="file"
                accept=".csv,.txt,.tsv"
                onChange={handleFile}
                className="ml-2 text-xs text-zinc-500"
              />
            </label>
          </div>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={"Paste queries here, one per line...\n\ncancel my order\nwhere is my package\nI want a refund\ntrack my delivery\n..."}
            rows={14}
            className="w-full bg-zinc-900 text-white text-sm border border-zinc-700 rounded-lg p-3 font-mono focus:outline-none focus:border-blue-500 resize-y"
          />
          <div className="flex items-center gap-3">
            <button
              onClick={handleDiscover}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-500 disabled:opacity-50"
            >
              Discover Intents
            </button>
            <span className="text-xs text-zinc-500">
              {input.split('\n').filter(l => l.trim()).length} queries
            </span>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="text-center py-12">
          <div className="inline-block w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-zinc-400 mt-3">Analyzing queries...</p>
        </div>
      )}

      {/* Results */}
      {clusters.length > 0 && !loading && (
        <>
          {/* Stats bar */}
          <div className="flex items-center justify-between bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-2">
            <div className="flex items-center gap-4 text-sm">
              <span className="text-zinc-400">
                <span className="text-white font-medium">{clusters.length}</span> clusters found
              </span>
              {stats && (
                <span className="text-zinc-500">
                  {Math.round((stats.total_assigned / stats.total_queries) * 100)}% queries assigned
                </span>
              )}
              <span className="text-zinc-500">
                {selectedCount} selected
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => toggleAll(true)} className="text-xs text-zinc-400 hover:text-white">Select All</button>
              <span className="text-zinc-600">|</span>
              <button onClick={() => toggleAll(false)} className="text-xs text-zinc-400 hover:text-white">Deselect All</button>
              <span className="text-zinc-600">|</span>
              <button
                onClick={() => { setClusters([]); setStats(null); setApplied(null); }}
                className="text-xs text-zinc-400 hover:text-white"
              >
                Start Over
              </button>
            </div>
          </div>

          {/* Applied success */}
          {applied && (
            <div className="bg-emerald-900/30 border border-emerald-700 rounded-lg px-4 py-3">
              <p className="text-sm text-emerald-400 font-medium">
                Created {applied.length} intents in "{settings.selectedAppId}"
              </p>
              <p className="text-xs text-emerald-500 mt-1">
                {applied.join(', ')}
              </p>
            </div>
          )}

          {/* Cluster cards */}
          <div className="grid gap-3">
            {clusters.map((cluster, i) => (
              <div
                key={i}
                className={`border rounded-lg p-4 transition-colors ${
                  cluster.selected
                    ? 'bg-zinc-900 border-zinc-700'
                    : 'bg-zinc-950 border-zinc-800 opacity-50'
                }`}
              >
                <div className="flex items-start gap-3">
                  {/* Checkbox */}
                  <input
                    type="checkbox"
                    checked={cluster.selected}
                    onChange={() =>
                      setClusters(prev =>
                        prev.map((c, j) => j === i ? { ...c, selected: !c.selected } : c)
                      )
                    }
                    className="mt-1 accent-blue-500"
                  />

                  <div className="flex-1 min-w-0">
                    {/* Name (editable) */}
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={cluster.editName}
                        onChange={e =>
                          setClusters(prev =>
                            prev.map((c, j) => j === i ? { ...c, editName: e.target.value } : c)
                          )
                        }
                        className="bg-transparent text-white font-medium text-sm border-b border-transparent hover:border-zinc-600 focus:border-blue-500 focus:outline-none px-0 py-0.5"
                      />
                      <span className="text-xs text-zinc-500">{cluster.size} queries</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        cluster.confidence >= 0.7 ? 'bg-emerald-900/50 text-emerald-400' :
                        cluster.confidence >= 0.4 ? 'bg-amber-900/50 text-amber-400' :
                        'bg-red-900/50 text-red-400'
                      }`}>
                        {Math.round(cluster.confidence * 100)}%
                      </span>
                    </div>

                    {/* Top terms */}
                    <div className="flex flex-wrap gap-1 mt-2">
                      {cluster.top_terms.slice(0, 8).map((term, ti) => (
                        <span key={ti} className="text-xs bg-zinc-800 text-zinc-400 px-1.5 py-0.5 rounded">
                          {term}
                        </span>
                      ))}
                    </div>

                    {/* Sample queries */}
                    <div className="mt-2 space-y-0.5">
                      {cluster.representative_queries.slice(0, 4).map((q, qi) => (
                        <p key={qi} className="text-xs text-zinc-500 truncate">{q}</p>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Apply bar */}
          <div className="sticky bottom-0 bg-zinc-950/90 backdrop-blur border-t border-zinc-800 -mx-4 px-4 py-3 flex items-center justify-between">
            <span className="text-sm text-zinc-400">
              Creating in: <span className="text-blue-400 font-medium">{settings.selectedAppId}</span>
            </span>
            <button
              onClick={handleApply}
              disabled={selectedCount === 0 || applied !== null}
              className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-500 disabled:opacity-50"
            >
              Apply Selected ({selectedCount})
            </button>
          </div>
        </>
      )}
    </div>
  );
}
