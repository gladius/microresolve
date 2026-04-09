import { useState, useEffect } from 'react';
import { api } from '@/api/client';

interface Projection { action: string; total_co_occurrences: number; projected_context: { id: string; count: number; strength: number }[] }

export default function Projections() {
  const [projections, setProjections] = useState<Projection[]>([]);
  useEffect(() => { api.getProjections().then(setProjections).catch(() => {}); }, []);
  const maxStrength = Math.max(0.01, ...projections.flatMap(p => p.projected_context.map(c => c.strength)));

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Projected Context</h2>
        <p className="text-xs text-zinc-500 mt-1">Context intents that co-occur with each action — discovered from usage.</p>
      </div>
      {projections.length === 0 ? (
        <div className="text-zinc-600 text-sm bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">No projections yet. Route some queries to build data.</div>
      ) : projections.map(proj => (
        <div key={proj.action} className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-[9px] w-4 h-4 flex items-center justify-center rounded border font-bold text-emerald-400 bg-emerald-400/10 border-emerald-400/30">A</span>
            <span className="text-sm font-semibold text-emerald-400 font-mono">{proj.action}</span>
            <span className="text-zinc-600 text-xs ml-auto">{proj.total_co_occurrences} obs</span>
          </div>
          <div className="space-y-1.5">
            {proj.projected_context.map(ctx => (
              <div key={ctx.id} className="flex items-center gap-2">
                <span className="text-xs font-mono text-cyan-400 w-32 truncate">{ctx.id}</span>
                <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-500/40 rounded-full" style={{ width: `${(ctx.strength / maxStrength) * 100}%` }} />
                </div>
                <span className="text-[10px] text-zinc-500 w-16 text-right">{Math.round(ctx.strength * 100)}% ({ctx.count}x)</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
