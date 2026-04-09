import { useState, useEffect } from 'react';

interface EscalationPattern { sequence: string[]; occurrences: number; frequency: number }

export default function Escalations() {
  const [escalations, setEscalations] = useState<EscalationPattern[]>([]);
  useEffect(() => {
    fetch('/api/escalation_patterns').then(r => r.ok ? r.json() : { patterns: [] }).then(d => setEscalations(Array.isArray(d) ? d : (d.patterns || []))).catch(() => {});
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Escalation Patterns</h2>
        <p className="text-xs text-zinc-500 mt-1">Recurring sequences that indicate customer frustration or process gaps.</p>
      </div>
      {escalations.length === 0 ? (
        <div className="text-zinc-600 text-sm bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">No escalation patterns detected.</div>
      ) : escalations.map((e, i) => (
        <div key={i} className={`bg-zinc-900 border rounded-lg p-4 ${e.frequency > 0.1 ? 'border-red-800' : 'border-zinc-800'}`}>
          <div className="flex items-center gap-2 mb-2">
            <span className={`text-xs font-bold ${e.frequency > 0.1 ? 'text-red-400' : 'text-amber-400'}`}>
              {e.frequency > 0.1 ? 'HIGH' : 'MODERATE'}
            </span>
            <span className="text-xs text-zinc-500">{e.occurrences} occurrences ({Math.round(e.frequency * 100)}%)</span>
          </div>
          <div className="flex items-center gap-2">
            {e.sequence.map((s, j) => (
              <div key={j} className="flex items-center gap-2">
                {j > 0 && <span className="text-red-400/50">→</span>}
                <span className="text-sm font-mono text-zinc-300 bg-zinc-800 px-2 py-0.5 rounded">{s}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
