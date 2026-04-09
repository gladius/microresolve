import { useState, useEffect } from 'react';

interface Workflow { intents: { id: string }[] }

export default function Workflows() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  useEffect(() => {
    fetch('/api/workflows').then(r => r.ok ? r.json() : { workflows: [] }).then(d => setWorkflows(Array.isArray(d) ? d : (d.workflows || []))).catch(() => {});
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Discovered Workflows</h2>
        <p className="text-xs text-zinc-500 mt-1">Intent clusters that form business processes — discovered from query patterns.</p>
      </div>
      {workflows.length === 0 ? (
        <div className="text-zinc-600 text-sm bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">No workflows discovered yet.</div>
      ) : workflows.map((wf, i) => (
        <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-xs text-zinc-500 mb-2">Workflow {i + 1} ({wf.intents.length} intents)</div>
          <div className="flex flex-wrap gap-2">
            {wf.intents.map((intent, j) => (
              <div key={intent.id} className="flex items-center gap-1">
                {j > 0 && <span className="text-zinc-600 text-xs mr-1">→</span>}
                <span className="text-sm font-mono text-violet-400 bg-violet-400/10 border border-violet-400/20 px-2 py-0.5 rounded">{intent.id}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
