import { useState, useEffect } from 'react';

interface TemporalEdge { first: string; second: string; count: number; probability: number }

export default function Temporal() {
  const [temporal, setTemporal] = useState<TemporalEdge[]>([]);
  useEffect(() => {
    fetch('/api/temporal_order').then(r => r.ok ? r.json() : []).then(d => setTemporal(Array.isArray(d) ? d : [])).catch(() => {});
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Temporal Ordering</h2>
        <p className="text-xs text-zinc-500 mt-1">Which intents typically follow which — directional flow from usage patterns.</p>
      </div>
      {temporal.length === 0 ? (
        <div className="text-zinc-600 text-sm bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">No temporal data yet.</div>
      ) : (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg divide-y divide-zinc-800">
          {temporal.slice(0, 25).map((edge, i) => (
            <div key={i} className="flex items-center gap-3 px-4 py-2">
              <span className="text-sm font-mono text-emerald-400 w-36 truncate">{edge.first}</span>
              <span className="text-zinc-600">→</span>
              <span className="text-sm font-mono text-cyan-400 w-36 truncate">{edge.second}</span>
              <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div className="h-full bg-violet-500/40 rounded-full" style={{ width: `${edge.probability * 100}%` }} />
              </div>
              <span className="text-xs text-zinc-500 w-20 text-right">{Math.round(edge.probability * 100)}% ({edge.count}x)</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
