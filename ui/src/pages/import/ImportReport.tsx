export interface IntentResult {
  name: string;
  phrases_added: number;
  blocked: number;
  recovered: number;
}

export interface ImportResult {
  imported: number;
  phrases_added: number;
  phrases_blocked: number;
  intents: string[];
  per_intent: IntentResult[];
}

interface ImportReportProps {
  result: ImportResult;
  onViewIntents: () => void;
  onImportMore: () => void;
  onFixCollisions?: () => void;
}

export function ImportReport({ result, onViewIntents, onImportMore, onFixCollisions }: ImportReportProps) {
  const totalRecovered = result.per_intent?.reduce((s, i) => s + (i.recovered ?? 0), 0) ?? 0;
  const emptyIntents = result.per_intent?.filter(i => i.phrases_added === 0) ?? [];
  const guardFired = result.phrases_blocked > 0 || totalRecovered > 0;

  return (
    <div className="border border-zinc-700 rounded-lg p-5 space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-white font-semibold">Import Complete</div>
          <div className="text-xs text-zinc-500 mt-0.5">{result.imported} intents created</div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-emerald-400">{result.phrases_added}</div>
          <div className="text-[10px] text-zinc-500">phrases added</div>
        </div>
      </div>

      {/* Collision guard summary */}
      {guardFired && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2.5 space-y-1">
          <div className="text-[10px] font-medium text-zinc-400">Collision Guard</div>
          <div className="flex gap-4 text-xs">
            {totalRecovered > 0 && (
              <span className="text-emerald-400/80">✓ {totalRecovered} recovered by retry</span>
            )}
            {result.phrases_blocked > 0 && (
              <span className="text-amber-400/80">✗ {result.phrases_blocked} permanently blocked</span>
            )}
          </div>
          {result.phrases_blocked > 0 && totalRecovered === 0 && (
            <div className="text-[10px] text-zinc-600">
              Guard blocked phrases whose terms exclusively match a competing intent. Retry did not find alternatives.
            </div>
          )}
          {result.phrases_blocked > 0 && totalRecovered > 0 && (
            <div className="text-[10px] text-zinc-600">
              Retry recovered {totalRecovered} phrases with alternative vocabulary. {result.phrases_blocked} remained unresolvable.
            </div>
          )}
        </div>
      )}

      {/* Per-intent table */}
      {result.per_intent && result.per_intent.length > 0 && (
        <div className="border border-zinc-800 rounded-lg overflow-hidden">
          <div className="max-h-64 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-zinc-900 border-b border-zinc-800">
                <tr>
                  <th className="text-left px-3 py-2 text-zinc-500 font-normal">Intent</th>
                  <th className="text-right px-3 py-2 text-zinc-500 font-normal w-20">Added</th>
                  {guardFired && <th className="text-right px-3 py-2 text-zinc-500 font-normal w-20">Blocked</th>}
                  {totalRecovered > 0 && <th className="text-right px-3 py-2 text-zinc-500 font-normal w-24">Recovered</th>}
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/50">
                {result.per_intent.map(i => (
                  <tr key={i.name} className={i.phrases_added === 0 ? 'bg-red-900/10' : ''}>
                    <td className="px-3 py-1.5 font-mono text-zinc-300 truncate max-w-xs">{i.name}</td>
                    <td className="px-3 py-1.5 text-right">
                      <span className={i.phrases_added === 0 ? 'text-red-400' : 'text-emerald-400'}>{i.phrases_added}</span>
                    </td>
                    {guardFired && (
                      <td className="px-3 py-1.5 text-right">
                        {i.blocked > 0 ? <span className="text-amber-400/70">{i.blocked}</span> : <span className="text-zinc-700">—</span>}
                      </td>
                    )}
                    {totalRecovered > 0 && (
                      <td className="px-3 py-1.5 text-right">
                        {(i.recovered ?? 0) > 0 ? <span className="text-emerald-400/70">{i.recovered}</span> : <span className="text-zinc-700">—</span>}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {emptyIntents.length > 0 && (
        <div className="text-[10px] text-zinc-500 border border-red-900/30 rounded px-3 py-2 bg-red-900/5">
          {emptyIntents.length} intent{emptyIntents.length !== 1 ? 's' : ''} have 0 phrases — every generated phrase conflicted with a competing intent and retry couldn't find alternatives. Add phrases manually from the Intents page.
        </div>
      )}

      <div className="flex gap-3 pt-1 border-t border-zinc-800">
        <button onClick={onViewIntents} className="px-4 py-2 text-sm bg-violet-600 text-white rounded hover:bg-violet-500">View Intents →</button>
        {onFixCollisions && (
          <button onClick={onFixCollisions} className="px-4 py-2 text-sm border border-amber-600/50 text-amber-400 rounded hover:bg-amber-500/10 transition-colors">
            Fix Collisions →
          </button>
        )}
        <button onClick={onImportMore} className="px-4 py-2 text-sm border border-zinc-700 text-zinc-400 rounded hover:text-white">Import more</button>
      </div>
    </div>
  );
}
