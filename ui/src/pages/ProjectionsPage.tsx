import { useState, useCallback, useRef } from 'react';
import { useFetch } from '@/hooks/useFetch';
import { api } from '@/api/client';

const SIMULATION_QUERIES = [
  "I want a refund, how much was I charged for that order",
  "get my money back and show me the transaction details",
  "refund my purchase and show me my payment history",
  "I need a refund, what does my account balance look like",
  "refund the damaged item and check if it's under warranty",
  "process my return and check if the credit went through",
  "cancel my order and tell me where the package is",
  "cancel the purchase and show me my order history",
  "stop my order from shipping, what's the delivery estimate",
  "I changed my mind, cancel it and show me what I ordered",
  "I was charged twice, show me my payment history",
  "wrong charge on my account, pull up the transaction details",
  "billing error, what is my current balance",
  "dispute this charge, show me the breakdown of payments",
  "unauthorized charge on my card, check my account status",
  "my account was hacked, reset my password immediately",
  "someone made unauthorized purchases, check my account status",
  "report fraud and check my account limits",
  "reset my password and tell me if my account is compromised",
  "upgrade my plan, what am I paying now",
  "change my plan and tell me what features I'll lose",
  "pause my subscription, when is my next billing date",
  "switch to annual billing, what's my current plan",
  "reorder my last purchase, is the item still in stock",
  "buy the same thing again, what's the current price",
  "repeat my order, check product availability",
  "order this again and tell me the shipping options",
  "reorder and apply my loyalty points",
  "change my shipping address, when will it arrive",
  "update delivery address and tell me the tracking status",
  "upgrade to express shipping, what's the estimated delivery",
  "expedite my order, what delivery methods are available",
  "add a new credit card and check my balance",
  "update my payment method, what do I currently owe",
  "link my bank account, what are my account limits",
  "I want to return this, what's your return policy",
  "is this still under warranty, I need a replacement",
  "does warranty cover shipping damage, I need a refund",
  "file a complaint, I want to talk to a manager",
  "terrible service, schedule a callback for me",
  "redeem my gift card and check the remaining balance",
  "apply a coupon code, what's the total price",
  "apply discount and show me the final cost",
  "cancel my order, get a refund, and check my balance",
  "update address, upgrade shipping, and track my order",
  "report fraud, reset password, and check account status",
  "return item, check warranty, and get refund status",
  "reorder, apply loyalty points, and check delivery estimate",
  "close my account and transfer remaining funds",
  "change plan, apply coupon, and check if I'm eligible",
];

type Projection = {
  action: string;
  total_co_occurrences: number;
  projected_context: { id: string; count: number; strength: number }[];
};

export default function ProjectionsPage() {
  const [projections, setProjections] = useState<Projection[]>([]);
  const [simulating, setSimulating] = useState(false);
  const [progress, setProgress] = useState({ sent: 0, total: 0 });
  const [coCount, setCoCount] = useState(0);
  const stopRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const data = await api.getProjections();
      setProjections(data);
      const co = await api.getCoOccurrence();
      setCoCount(co.reduce((sum, p) => sum + p.count, 0));
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
      // Refresh projections every 5 queries so user sees it build up
      if ((i + 1) % 5 === 0 || i === SIMULATION_QUERIES.length - 1) {
        await refresh();
      }
    }
    setSimulating(false);
  };

  const maxStrength = Math.max(
    0.01,
    ...projections.flatMap(p => p.projected_context.map(c => c.strength))
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-white">Projected Context</h1>
          <p className="text-xs text-zinc-500 mt-1">
            Context intents that historically co-occur with each action — discovered from usage patterns, not configured.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-zinc-600">{coCount} co-occurrences tracked</span>
          <button
            onClick={refresh}
            className="text-xs text-violet-400 hover:text-violet-300 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Simulate button */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-white font-medium">Simulation</div>
            <p className="text-xs text-zinc-500 mt-0.5">
              Send {SIMULATION_QUERIES.length} multi-intent queries to build co-occurrence data. Watch projections emerge in real-time.
            </p>
          </div>
          {simulating ? (
            <div className="flex items-center gap-3">
              <div className="text-xs text-amber-400 font-mono">
                {progress.sent}/{progress.total}
              </div>
              <div className="w-32 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-violet-500 transition-all duration-300"
                  style={{ width: `${(progress.sent / progress.total) * 100}%` }}
                />
              </div>
              <button
                onClick={() => { stopRef.current = true; }}
                className="text-xs text-red-400 hover:text-red-300"
              >
                Stop
              </button>
            </div>
          ) : (
            <button
              onClick={simulate}
              className="px-4 py-2 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-medium transition-colors"
            >
              Run Simulation
            </button>
          )}
        </div>
      </div>

      {/* Projection map */}
      {projections.length === 0 ? (
        <div className="text-zinc-600 text-sm bg-zinc-900 border border-zinc-800 rounded-lg p-12 text-center">
          No projections yet. Run the simulation or send multi-intent queries through the Router to build co-occurrence data.
        </div>
      ) : (
        <div className="space-y-3">
          {projections.map(proj => (
            <div key={proj.action} className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-sm font-semibold text-emerald-400 font-mono">{proj.action}</span>
                <span className="text-[9px] px-1.5 py-0.5 rounded border text-emerald-400 border-emerald-400/30 font-semibold uppercase">action</span>
                <span className="text-zinc-600 text-xs ml-auto">{proj.total_co_occurrences} total observations</span>
              </div>
              <div className="space-y-1.5">
                {proj.projected_context.map(ctx => {
                  const pct = Math.round(ctx.strength * 100);
                  const barWidth = (ctx.strength / maxStrength) * 100;
                  return (
                    <div key={ctx.id} className="flex items-center gap-3">
                      <span className="text-xs font-mono text-cyan-400 w-32 truncate">{ctx.id}</span>
                      <div className="flex-1 h-4 bg-zinc-800 rounded overflow-hidden relative">
                        <div
                          className="h-full bg-cyan-400/20 border-r border-cyan-400/50 transition-all duration-500"
                          style={{ width: `${barWidth}%` }}
                        />
                        <span className="absolute inset-0 flex items-center px-2 text-[10px] text-cyan-400/80 font-mono">
                          {pct}% ({ctx.count}x)
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
