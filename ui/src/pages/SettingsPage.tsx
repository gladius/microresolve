import { useAppStore, type AppMode } from '@/store';

export default function SettingsPage() {
  const { settings, setMode, setThreshold } = useAppStore();

  return (
    <div className="max-w-2xl space-y-8">
      <h1 className="text-lg font-semibold text-white">Settings</h1>

      {/* Mode */}
      <section className="space-y-3">
        <h2 className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Router Mode</h2>
        <div className="flex gap-3">
          {([
            { mode: 'production' as AppMode, label: 'Production', desc: 'Fast routing only. Queries are logged but not reviewed.', color: 'emerald' },
            { mode: 'learn' as AppMode, label: 'Learn', desc: 'Every query is reviewed by LLM. Suggestions appear inline for approval.', color: 'amber' },
          ]).map(({ mode, label, desc, color }) => (
            <button
              key={mode}
              onClick={() => setMode(mode)}
              className={`flex-1 p-4 rounded-lg border text-left transition-colors ${
                settings.mode === mode
                  ? `border-${color}-400/50 bg-${color}-400/10`
                  : 'border-zinc-800 bg-zinc-900 hover:border-zinc-700'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={`w-2 h-2 rounded-full ${settings.mode === mode ? `bg-${color}-400` : 'bg-zinc-600'}`} />
                <span className={`font-semibold text-sm ${settings.mode === mode ? `text-${color}-400` : 'text-zinc-400'}`}>
                  {label}
                </span>
              </div>
              <p className="text-xs text-zinc-500">{desc}</p>
            </button>
          ))}
        </div>
        <p className="text-xs text-zinc-600">
          LLM features require <code className="text-violet-400">ANTHROPIC_API_KEY</code> in the server's <code className="text-violet-400">.env</code> file.
        </p>
      </section>

      {/* Threshold */}
      <section className="space-y-3">
        <h2 className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Default Threshold</h2>
        <p className="text-xs text-zinc-600">
          Minimum score for multi-intent detection. Lower catches more but risks false positives.
        </p>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min="0"
            max="5"
            step="0.1"
            value={settings.threshold}
            onChange={e => setThreshold(parseFloat(e.target.value))}
            className="w-48 accent-violet-500"
          />
          <span className="text-white font-mono text-sm w-10">{settings.threshold.toFixed(1)}</span>
        </div>
      </section>

      {/* Query Log */}
      <section className="space-y-3">
        <h2 className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Query Log</h2>
        <p className="text-xs text-zinc-600">
          All queries are logged server-side to <code className="text-violet-400">asv_queries.jsonl</code> for review and tuning.
        </p>
        <LogStats />
      </section>
    </div>
  );
}

function LogStats() {
  return (
    <div className="flex gap-4">
      <button
        onClick={async () => {
          try {
            const res = await fetch('/api/logs/stats');
            if (res.ok) {
              const data = await res.json();
              alert(`Log entries: ${data.count}\nFile: ${data.file}`);
            } else {
              alert('Log endpoint not available yet');
            }
          } catch {
            alert('Server not reachable');
          }
        }}
        className="text-xs text-violet-400 hover:text-violet-300 px-3 py-1.5 border border-violet-400/30 rounded transition-colors"
      >
        View Log Stats
      </button>
      <button
        onClick={async () => {
          if (!confirm('Clear all query logs?')) return;
          try {
            await fetch('/api/logs', { method: 'DELETE' });
          } catch { /* */ }
        }}
        className="text-xs text-red-400/70 hover:text-red-400 px-3 py-1.5 border border-red-400/20 rounded transition-colors"
      >
        Clear Logs
      </button>
    </div>
  );
}
