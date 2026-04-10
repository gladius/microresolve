import { api } from '@/api/client';
import { useState, useCallback, useRef } from 'react';
import { useFetch } from '@/hooks/useFetch';

const SIMULATION_QUERIES = [
  "I want a refund, how much was I charged for that order",
  "cancel my order and tell me where the package is",
  "I was charged twice, show me my payment history",
  "my account was hacked, reset my password immediately",
  "upgrade my plan, what am I paying now",
  "change my shipping address, when will it arrive",
  "I want to return this, what's your return policy",
  "file a complaint, I want to talk to a manager",
  "cancel my order, get a refund, and check my balance",
  "where is my package and can I change the delivery address",
];

export default function Overview() {
  const [coocCount, setCoocCount] = useState(0);
  const [projCount, setProjCount] = useState(0);
  const [simulating, setSimulating] = useState(false);
  const [progress, setProgress] = useState({ sent: 0, total: 0 });
  const stopRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const [p, c] = await Promise.all([api.getProjections(), api.getCoOccurrence()]);
      setProjCount(p.length);
      setCoocCount(Array.isArray(c) ? c.reduce((s: number, p: any) => s + p.count, 0) : 0);
    } catch { /* */ }
  }, []);

  useFetch(refresh, [refresh]);

  const simulate = async () => {
    stopRef.current = false;
    setSimulating(true);
    setProgress({ sent: 0, total: SIMULATION_QUERIES.length });
    for (let i = 0; i < SIMULATION_QUERIES.length; i++) {
      if (stopRef.current) break;
      await api.routeMulti(SIMULATION_QUERIES[i], 0.3);
      setProgress({ sent: i + 1, total: SIMULATION_QUERIES.length });
    }
    await refresh();
    setSimulating(false);
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Overview</h2>
        {simulating ? (
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-amber-400 font-mono">{progress.sent}/{progress.total}</span>
            <button onClick={() => { stopRef.current = true; }} className="text-[10px] text-red-400">Stop</button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <button onClick={refresh} className="text-[10px] text-zinc-500 hover:text-white">Refresh</button>
            <button onClick={simulate} className="text-[10px] bg-violet-600 hover:bg-violet-500 text-white px-2 py-0.5 rounded">Simulate</button>
          </div>
        )}
      </div>
      <p className="text-xs text-zinc-500">Intelligence from routing patterns. Click Simulate to populate.</p>
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3">
          <div className="text-xs text-zinc-500">Observations</div>
          <div className="text-2xl font-mono text-white mt-0.5">{coocCount}</div>
        </div>
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3">
          <div className="text-xs text-zinc-500">Projections</div>
          <div className="text-2xl font-mono text-white mt-0.5">{projCount}</div>
        </div>
      </div>
    </div>
  );
}
