import { useEffect, useState } from 'react';
import Page from '@/components/Page';

interface ConnectedClient {
  name: string;
  namespaces: string[];
  tick_interval_secs: number;
  library_version: string | null;
  last_seen_ms: number;
  age_ms: number;
  expires_in_ms: number;
}

interface RosterResponse {
  count: number;
  clients: ConnectedClient[];
}

const REFRESH_MS = 3000;

function formatAge(ms: number): string {
  if (ms < 1000) return 'just now';
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
  return `${Math.floor(ms / 3_600_000)}h ago`;
}

function formatExpiry(ms: number): string {
  return `${Math.max(0, Math.floor(ms / 1000))}s`;
}

export default function ConnectedClientsPage() {
  const [clients, setClients] = useState<ConnectedClient[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const fetchClients = async () => {
      try {
        const r = await fetch('/api/connected_clients');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d: RosterResponse = await r.json();
        if (alive) {
          setClients(d.clients ?? []);
          setError(null);
          setLoaded(true);
        }
      } catch (e) {
        if (alive) setError(String(e));
      }
    };
    fetchClients();
    const t = setInterval(fetchClients, REFRESH_MS);
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <Page title="Connected Clients" fullscreen>
      <div className="h-full overflow-auto">
        <div className="max-w-5xl mx-auto p-6">
          {/* Header */}
          <div className="flex items-baseline justify-between mb-1">
            <h1 className="text-xl font-semibold text-zinc-100">Connected Clients</h1>
            <div className="flex items-center gap-2 text-xs">
              <span className={`w-2 h-2 rounded-full ${clients.length > 0 ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-600'}`} />
              <span className={clients.length > 0 ? 'text-emerald-300' : 'text-zinc-500'}>
                {clients.length} active
              </span>
            </div>
          </div>
          <p className="text-sm text-zinc-500 leading-relaxed mb-6">
            Library clients (Python / Node / Rust) currently syncing with this Studio via <code className="text-zinc-400">POST /api/sync</code>.
            An entry is considered active until <code className="text-zinc-400">2 × tick_interval_secs</code> have elapsed
            since its last sync. Entries are tracked only for authenticated requests — open mode (no API keys) shows nothing here.
          </p>

          {error && (
            <div className="mb-4 px-4 py-2 rounded-md bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
              Failed to load roster: {error}
            </div>
          )}

          {/* Empty state */}
          {loaded && clients.length === 0 && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-8 text-center">
              <div className="text-4xl mb-2 text-zinc-700">⚡</div>
              <div className="text-zinc-300 font-medium">No library clients connected</div>
              <div className="text-zinc-500 text-sm mt-2 max-w-md mx-auto leading-relaxed">
                Configure auth keys in <a href="/auth" className="text-violet-400 hover:underline">Manage → Auth Keys</a>,
                then point your library at this Studio's URL with the generated key.
                Open-mode clients are not tracked here — by design, since there's no identity to attribute.
              </div>
              <div className="text-zinc-600 text-xs mt-4">
                Auto-refreshes every {REFRESH_MS / 1000}s
              </div>
            </div>
          )}

          {/* Table of clients */}
          {clients.length > 0 && (
            <div className="rounded-lg border border-zinc-800 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-zinc-900/60 text-zinc-500 text-xs uppercase tracking-wider">
                  <tr>
                    <th className="text-left px-4 py-2 font-medium">Client</th>
                    <th className="text-left px-4 py-2 font-medium">Library</th>
                    <th className="text-left px-4 py-2 font-medium">Subscribed</th>
                    <th className="text-left px-4 py-2 font-medium">Tick</th>
                    <th className="text-left px-4 py-2 font-medium">Last sync</th>
                    <th className="text-right px-4 py-2 font-medium">Expires in</th>
                  </tr>
                </thead>
                <tbody>
                  {clients.map(c => (
                    <tr key={c.name} className="border-t border-zinc-800 hover:bg-zinc-900/40">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                          <span className="font-mono text-emerald-300 font-medium">{c.name}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 font-mono text-zinc-400 text-xs">
                        {c.library_version ?? <span className="text-zinc-600 italic">unknown</span>}
                      </td>
                      <td className="px-4 py-3 text-zinc-400">
                        {c.namespaces.length === 0 ? (
                          <span className="text-zinc-600 italic">—</span>
                        ) : (
                          <span className="font-mono text-xs">{c.namespaces.join(', ')}</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-zinc-400 text-xs">
                        {c.tick_interval_secs}s
                      </td>
                      <td className="px-4 py-3 text-zinc-400 text-xs">
                        {formatAge(c.age_ms)}
                      </td>
                      <td className="px-4 py-3 text-right text-zinc-400 text-xs font-mono">
                        {formatExpiry(c.expires_in_ms)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {clients.length > 0 && (
            <div className="text-zinc-600 text-xs mt-3 text-right">
              Auto-refreshes every {REFRESH_MS / 1000}s
            </div>
          )}
        </div>
      </div>
    </Page>
  );
}
