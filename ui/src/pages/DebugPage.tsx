import { useState, useEffect, useCallback } from 'react';
import { api, type LogEntry } from '@/api/client';

export default function DebugPage() {
  return (
    <div className="space-y-8">
      <h1 className="text-lg font-semibold text-white">Debug & Tuning</h1>
      <CoOccurrenceSection />
      <QueryLogSection />
    </div>
  );
}

// --- Co-occurrence matrix (E2) ---

function CoOccurrenceSection() {
  const [pairs, setPairs] = useState<{ a: string; b: string; count: number }[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getCoOccurrence();
      setPairs(data);
    } catch { /* */ }
    setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Build unique intent set and matrix
  const intentSet = new Set<string>();
  pairs.forEach(p => { intentSet.add(p.a); intentSet.add(p.b); });
  const intents = Array.from(intentSet).sort();

  const getCount = (a: string, b: string) => {
    if (a === b) return null;
    const [x, y] = a < b ? [a, b] : [b, a];
    return pairs.find(p => p.a === x && p.b === y)?.count || 0;
  };

  const maxCount = Math.max(1, ...pairs.map(p => p.count));

  return (
    <section>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Intent Co-occurrence</h2>
        <button onClick={refresh} disabled={loading} className="text-xs text-violet-400 hover:text-violet-300 transition-colors">
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {intents.length === 0 ? (
        <div className="text-zinc-600 text-xs bg-zinc-900 border border-zinc-800 rounded-lg p-6 text-center">
          No co-occurrence data yet. Send multi-intent queries to build the matrix.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="text-xs font-mono">
            <thead>
              <tr>
                <th className="px-2 py-1 text-zinc-500 text-left" />
                {intents.map(id => (
                  <th key={id} className="px-2 py-1 text-zinc-400 font-normal" style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}>
                    {id}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {intents.map(row => (
                <tr key={row}>
                  <td className="px-2 py-1 text-zinc-400 text-right">{row}</td>
                  {intents.map(col => {
                    const count = getCount(row, col);
                    if (count === null) {
                      return <td key={col} className="px-2 py-1 text-center text-zinc-700">-</td>;
                    }
                    const intensity = count / maxCount;
                    return (
                      <td
                        key={col}
                        className="px-2 py-1 text-center"
                        style={{
                          backgroundColor: count > 0 ? `rgba(139, 92, 246, ${intensity * 0.4})` : undefined,
                          color: count > 0 ? 'rgb(196, 181, 253)' : 'rgb(63, 63, 70)',
                        }}
                      >
                        {count}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Top pairs list */}
      {pairs.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {pairs.slice(0, 10).map((p, i) => (
            <span key={i} className="text-xs bg-zinc-800 border border-zinc-700 rounded px-2 py-1">
              <span className="text-violet-400">{p.a}</span>
              <span className="text-zinc-600"> + </span>
              <span className="text-violet-400">{p.b}</span>
              <span className="text-zinc-500 ml-1.5">×{p.count}</span>
            </span>
          ))}
        </div>
      )}
    </section>
  );
}

// --- Query log viewer (E3) ---

function QueryLogSection() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getLogs(50, 0);
      setEntries(data.entries);
      setTotal(data.total);
    } catch { /* */ }
    setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const handleClear = async () => {
    if (!confirm(`Clear all ${total} log entries?`)) return;
    await api.clearLogs();
    refresh();
  };

  return (
    <section>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">
          Query Log ({total} entries)
        </h2>
        <div className="flex gap-2">
          <button onClick={refresh} disabled={loading} className="text-xs text-violet-400 hover:text-violet-300 transition-colors">
            {loading ? 'Loading...' : 'Refresh'}
          </button>
          {total > 0 && (
            <button onClick={handleClear} className="text-xs text-red-400/70 hover:text-red-400 transition-colors">
              Clear
            </button>
          )}
        </div>
      </div>

      {entries.length === 0 ? (
        <div className="text-zinc-600 text-xs bg-zinc-900 border border-zinc-800 rounded-lg p-6 text-center">
          No queries logged yet. Route some queries to see them here.
        </div>
      ) : (
        <div className="space-y-1 max-h-[60vh] overflow-y-auto">
          {entries.map((entry, i) => (
            <LogEntryRow key={i} entry={entry} />
          ))}
        </div>
      )}
    </section>
  );
}

function LogEntryRow({ entry }: { entry: LogEntry }) {
  const [expanded, setExpanded] = useState(false);
  const date = new Date(entry.ts);
  const time = date.toLocaleTimeString();
  const bestScore = Math.max(0, ...entry.results.map(r => r.score));

  return (
    <div
      className="bg-zinc-900 border border-zinc-800 rounded px-3 py-2 cursor-pointer hover:border-zinc-700 transition-colors"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-3">
        <span className="text-zinc-600 text-[10px] w-16 flex-shrink-0">{time}</span>
        <span className="text-white text-sm font-mono truncate flex-1">{entry.query}</span>
        <span className="text-zinc-500 text-[10px]">{entry.latency_us}μs</span>
        <div className="flex gap-1">
          {entry.results.map((r, i) => {
            const isWeak = entry.results.length > 1 && r.score / bestScore < 0.3;
            return (
              <span
                key={i}
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                  r.intent_type === 'context'
                    ? 'text-cyan-400 bg-cyan-400/10'
                    : 'text-emerald-400 bg-emerald-400/10'
                } ${isWeak ? 'opacity-40' : ''}`}
              >
                {r.id}
              </span>
            );
          })}
        </div>
      </div>

      {expanded && (
        <div className="mt-2 pl-16 space-y-0.5">
          {entry.results.map((r, i) => (
            <div key={i} className="text-xs font-mono flex gap-3">
              <span className="text-zinc-400 w-28 truncate">{r.id}</span>
              <span className="text-amber-400 w-10">{r.score.toFixed(2)}</span>
              <span className={`text-[10px] ${r.intent_type === 'context' ? 'text-cyan-400' : 'text-emerald-400'}`}>
                {r.intent_type}
              </span>
              <span className="text-zinc-600">[{r.span[0]},{r.span[1]}]</span>
            </div>
          ))}
          <div className="text-[10px] text-zinc-600 mt-1">
            threshold: {entry.threshold.toFixed(2)} | latency: {entry.latency_us}μs
          </div>
        </div>
      )}
    </div>
  );
}
