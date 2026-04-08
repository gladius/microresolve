import { useState, useCallback, useEffect, useRef } from 'react';
import { api } from '@/api/client';
import SidebarLayout, { type SidebarItem } from '@/components/SidebarLayout';

const SIMULATION_QUERIES = [
  "I want a refund, how much was I charged for that order",
  "get my money back and show me the transaction details",
  "refund my purchase and show me my payment history",
  "cancel my order and tell me where the package is",
  "cancel the purchase and show me my order history",
  "stop my order from shipping, what's the delivery estimate",
  "I was charged twice, show me my payment history",
  "wrong charge on my account, pull up the transaction details",
  "billing error, what is my current balance",
  "dispute this charge, show me the breakdown of payments",
  "my account was hacked, reset my password immediately",
  "someone made unauthorized purchases, check my account status",
  "upgrade my plan, what am I paying now",
  "change my plan and tell me what features I'll lose",
  "reorder my last purchase, is the item still in stock",
  "buy the same thing again, what's the current price",
  "change my shipping address, when will it arrive",
  "update delivery address and tell me the tracking status",
  "add a new credit card and check my balance",
  "I want to return this, what's your return policy",
  "is this still under warranty, I need a replacement",
  "file a complaint, I want to talk to a manager",
  "terrible service, schedule a callback for me",
  "redeem my gift card and check the remaining balance",
  "cancel my order, get a refund, and check my balance",
  "update address, upgrade shipping, and track my order",
  "report fraud, reset password, and check account status",
  "return item, check warranty, and get refund status",
  "close my account and transfer remaining funds",
  "refund the damaged item and check if it's under warranty",
];

type Projection = {
  action: string;
  total_co_occurrences: number;
  projected_context: { id: string; count: number; strength: number }[];
};

type Workflow = {
  intents: { id: string; connections: number; neighbors: string[] }[];
  size: number;
};

type TemporalEdge = { first: string; second: string; probability: number; count: number };
type EscalationPattern = { sequence: string[]; occurrences: number; frequency: number };
type CoOccurrence = { a: string; b: string; count: number };

export default function DashboardPage() {
  const [section, setSection] = useState('overview');
  const [projections, setProjections] = useState<Projection[]>([]);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [temporal, setTemporal] = useState<TemporalEdge[]>([]);
  const [escalations, setEscalations] = useState<EscalationPattern[]>([]);
  const [cooc, setCooc] = useState<CoOccurrence[]>([]);
  const [simulating, setSimulating] = useState(false);
  const [progress, setProgress] = useState({ sent: 0, total: 0 });
  const stopRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const [p, wRaw, t, eRaw, c] = await Promise.all([
        api.getProjections(),
        fetch('/api/workflows').then(r => r.ok ? r.json() : { workflows: [] }),
        fetch('/api/temporal_order').then(r => r.ok ? r.json() : []),
        fetch('/api/escalation_patterns').then(r => r.ok ? r.json() : { patterns: [] }),
        api.getCoOccurrence(),
      ]);
      setProjections(p);
      setWorkflows(Array.isArray(wRaw) ? wRaw : (wRaw.workflows || []));
      setTemporal(t);
      setEscalations(Array.isArray(eRaw) ? eRaw : (eRaw.patterns || []));
      setCooc(c);
    } catch { /* */ }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const simulate = async () => {
    stopRef.current = false;
    setSimulating(true);
    setProgress({ sent: 0, total: SIMULATION_QUERIES.length });
    for (let i = 0; i < SIMULATION_QUERIES.length; i++) {
      if (stopRef.current) break;
      await api.routeMulti(SIMULATION_QUERIES[i], 0.3);
      setProgress({ sent: i + 1, total: SIMULATION_QUERIES.length });
      if ((i + 1) % 5 === 0 || i === SIMULATION_QUERIES.length - 1) {
        await refresh();
      }
    }
    setSimulating(false);
  };

  const totalObs = cooc.reduce((s, p) => s + p.count, 0);

  const sidebarItems: SidebarItem[] = [
    { id: 'overview', label: 'Overview', badge: `${totalObs} obs` },
    { id: 'projections', label: 'Projections', badge: projections.length || undefined },
    { id: 'workflows', label: 'Workflows', badge: workflows.length || undefined },
    { id: 'temporal', label: 'Temporal Flow', badge: temporal.length || undefined },
    { id: 'escalations', label: 'Escalations', badge: escalations.length || undefined, color: escalations.length > 0 ? 'text-red-400' : undefined },
    { id: 'cooccurrence', label: 'Co-occurrence', badge: cooc.length || undefined },
  ];

  const simButton = (
    <div className="flex items-center gap-1.5">
      {simulating ? (
        <>
          <span className="text-[10px] text-amber-400 font-mono">{progress.sent}/{progress.total}</span>
          <button onClick={() => { stopRef.current = true; }} className="text-[10px] text-red-400">Stop</button>
        </>
      ) : (
        <>
          <button onClick={refresh} className="text-[10px] text-zinc-500 hover:text-white">Refresh</button>
          <button onClick={simulate} className="text-[10px] bg-violet-600 hover:bg-violet-500 text-white px-2 py-0.5 rounded">Simulate</button>
        </>
      )}
    </div>
  );

  return (
    <SidebarLayout
      title="Dashboard"
      items={sidebarItems}
      selected={section}
      onSelect={setSection}
      headerActions={simButton}
    >
      <div className="p-5">
        {section === 'overview' && <OverviewSection totalObs={totalObs} projections={projections.length} workflows={workflows.length} escalations={escalations.length} cooc={cooc.length} />}
        {section === 'projections' && <ProjectionsSection projections={projections} />}
        {section === 'workflows' && <WorkflowsSection workflows={workflows} />}
        {section === 'temporal' && <TemporalSection temporal={temporal} />}
        {section === 'escalations' && <EscalationsSection escalations={escalations} />}
        {section === 'cooccurrence' && <CoOccurrenceSection cooc={cooc} />}
      </div>
    </SidebarLayout>
  );
}

// --- Overview ---

function OverviewSection({ totalObs, projections, workflows, escalations, cooc }: {
  totalObs: number; projections: number; workflows: number; escalations: number; cooc: number;
}) {
  return (
    <div className="space-y-5">
      <h2 className="text-lg font-semibold text-white">Overview</h2>
      <p className="text-xs text-zinc-500">Intelligence emerging from routing patterns. Click Simulate to populate.</p>
      <div className="grid grid-cols-4 gap-3">
        <StatCard label="Observations" value={totalObs} />
        <StatCard label="Projections" value={projections} />
        <StatCard label="Workflows" value={workflows} />
        <StatCard label="Escalations" value={escalations} color={escalations > 0 ? 'text-red-400' : undefined} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Co-occurrence Pairs" value={cooc} />
        <StatCard label="Status" value={totalObs > 0 ? 'Active' : 'No data'} isText />
      </div>
    </div>
  );
}

function StatCard({ label, value, color, isText }: { label: string; value: number | string; color?: string; isText?: boolean }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className={`${isText ? 'text-sm' : 'text-2xl'} font-mono ${color || 'text-white'} mt-0.5`}>{value}</div>
    </div>
  );
}

// --- Projections ---

function ProjectionsSection({ projections }: { projections: Projection[] }) {
  const maxStrength = Math.max(0.01, ...projections.flatMap(p => p.projected_context.map(c => c.strength)));

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Projected Context</h2>
        <p className="text-xs text-zinc-500 mt-1">Context intents that co-occur with each action — discovered from usage, not configured.</p>
      </div>
      {projections.length === 0 ? (
        <EmptyState text="No projections yet. Run a simulation to build co-occurrence data." />
      ) : (
        projections.map(proj => (
          <div key={proj.action} className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-[9px] w-4 h-4 flex items-center justify-center rounded border font-bold text-emerald-400 bg-emerald-400/10 border-emerald-400/30">A</span>
              <span className="text-sm font-semibold text-emerald-400 font-mono">{proj.action}</span>
              <span className="text-zinc-600 text-xs ml-auto">{proj.total_co_occurrences} obs</span>
            </div>
            <div className="space-y-1.5">
              {proj.projected_context.map(ctx => {
                const pct = Math.round(ctx.strength * 100);
                const barWidth = (ctx.strength / maxStrength) * 100;
                return (
                  <div key={ctx.id} className="flex items-center gap-3">
                    <span className="text-xs font-mono text-cyan-400 w-32 truncate">{ctx.id}</span>
                    <div className="flex-1 h-4 bg-zinc-800 rounded overflow-hidden relative">
                      <div className="h-full bg-cyan-400/20 border-r border-cyan-400/50 transition-all duration-500" style={{ width: `${barWidth}%` }} />
                      <span className="absolute inset-0 flex items-center px-2 text-[10px] text-cyan-400/80 font-mono">{pct}% ({ctx.count}x)</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))
      )}
    </div>
  );
}

// --- Workflows ---

function WorkflowsSection({ workflows }: { workflows: Workflow[] }) {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Discovered Workflows</h2>
        <p className="text-xs text-zinc-500 mt-1">Intent clusters that form business processes — discovered from query patterns.</p>
      </div>
      {workflows.length === 0 ? (
        <EmptyState text="No workflows discovered yet. Run a simulation to build sequence data." />
      ) : (
        workflows.map((wf, i) => (
          <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
            <div className="text-xs text-zinc-500 mb-2">Workflow {i + 1} ({wf.intents.length} intents)</div>
            <div className="flex flex-wrap gap-2">
              {wf.intents.map((intent, j) => (
                <div key={intent.id} className="flex items-center gap-1">
                  {j > 0 && <span className="text-zinc-600 text-xs mr-1">→</span>}
                  <span className="text-sm font-mono text-violet-400 bg-violet-400/10 border border-violet-400/20 px-2 py-0.5 rounded">
                    {intent.id}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  );
}

// --- Temporal ---

function TemporalSection({ temporal }: { temporal: TemporalEdge[] }) {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Temporal Ordering</h2>
        <p className="text-xs text-zinc-500 mt-1">Which intents typically follow which — directional flow from usage patterns.</p>
      </div>
      {temporal.length === 0 ? (
        <EmptyState text="No temporal data yet." />
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

// --- Escalations ---

function EscalationsSection({ escalations }: { escalations: EscalationPattern[] }) {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Escalation Patterns</h2>
        <p className="text-xs text-zinc-500 mt-1">Recurring sequences that indicate customer frustration or process gaps.</p>
      </div>
      {escalations.length === 0 ? (
        <EmptyState text="No escalation patterns detected." />
      ) : (
        escalations.map((e, i) => (
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
        ))
      )}
    </div>
  );
}

// --- Co-occurrence ---

function CoOccurrenceSection({ cooc }: { cooc: CoOccurrence[] }) {
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
        <EmptyState text="No co-occurrence data yet. Run a simulation." />
      ) : (
        <>
          {/* Heatmap */}
          <div className="overflow-x-auto">
            <table className="text-xs">
              <thead>
                <tr>
                  <th className="p-1" />
                  {intents.map(id => (
                    <th key={id} className="p-1 text-zinc-500 font-mono font-normal" style={{ writingMode: 'vertical-rl', maxHeight: 100 }}>
                      {id}
                    </th>
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
                          {row === col ? (
                            <div className="w-6 h-6 bg-zinc-800 rounded" />
                          ) : (
                            <div
                              className="w-6 h-6 rounded flex items-center justify-center text-[8px]"
                              style={{ backgroundColor: count > 0 ? `rgba(139, 92, 246, ${intensity})` : 'rgb(39,39,42)' }}
                              title={`${row} + ${col}: ${count}`}
                            >
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

          {/* Top pairs list */}
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

function EmptyState({ text }: { text: string }) {
  return (
    <div className="text-zinc-600 text-sm bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">
      {text}
    </div>
  );
}
