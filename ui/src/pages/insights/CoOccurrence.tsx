import { useState, useEffect } from 'react';
import { api } from '@/api/client';

interface CoOc { a: string; b: string; count: number }

export default function CoOccurrence() {
  const [cooc, setCooc] = useState<CoOc[]>([]);
  useEffect(() => { api.getCoOccurrence().then(d => setCooc(Array.isArray(d) ? d : [])).catch(() => {}); }, []);

  const intents = Array.from(new Set(cooc.flatMap(c => [c.a, c.b]))).sort();
  const maxCount = Math.max(1, ...cooc.map(c => c.count));
  const sorted = [...cooc].sort((a, b) => b.count - a.count);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-white">Co-occurrence Matrix</h2>
        <p className="text-xs text-zinc-500 mt-1">How often intent pairs fire together. Darker = more frequent.</p>
      </div>
      {intents.length === 0 ? (
        <div className="text-zinc-600 text-sm bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">No co-occurrence data yet.</div>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="text-xs">
              <thead>
                <tr>
                  <th className="p-1" />
                  {intents.map(id => (
                    <th key={id} className="p-1 text-zinc-500 font-mono font-normal" style={{ writingMode: 'vertical-rl', maxHeight: 100 }}>{id}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {intents.map(row => (
                  <tr key={row}>
                    <td className="p-1 text-zinc-500 font-mono text-right pr-2 whitespace-nowrap">{row}</td>
                    {intents.map(col => {
                      const pair = cooc.find(c => (c.a === row && c.b === col) || (c.a === col && c.b === row));
                      const count = pair?.count || 0;
                      const intensity = count > 0 ? Math.max(0.1, count / maxCount) : 0;
                      return (
                        <td key={col} className="p-0.5">
                          {row === col ? <div className="w-6 h-6 bg-zinc-800 rounded" /> : (
                            <div className="w-6 h-6 rounded flex items-center justify-center text-[8px]"
                              style={{ backgroundColor: count > 0 ? `rgba(139, 92, 246, ${intensity})` : 'rgb(39,39,42)' }}
                              title={`${row} + ${col}: ${count}`}>
                              {count > 0 ? count : ''}
                            </div>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h3 className="text-xs text-zinc-500 font-semibold uppercase mb-2">Top Co-occurring Pairs</h3>
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg divide-y divide-zinc-800">
              {sorted.slice(0, 15).map((pair, i) => (
                <div key={i} className="flex items-center gap-3 px-4 py-2">
                  <span className="text-sm font-mono text-emerald-400 w-36 truncate">{pair.a}</span>
                  <span className="text-zinc-600">+</span>
                  <span className="text-sm font-mono text-cyan-400 w-36 truncate">{pair.b}</span>
                  <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div className="h-full bg-violet-500/40 rounded-full" style={{ width: `${(pair.count / maxCount) * 100}%` }} />
                  </div>
                  <span className="text-xs text-zinc-500 w-12 text-right">{pair.count}x</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
