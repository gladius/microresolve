import { useState, useCallback, useEffect } from 'react';
import Page from '@/components/Page';
import { api } from '@/api/client';

interface KeyRow {
  name: string;
  prefix: string;
  created_at: number;
  last_used_at: number | null;
}

export default function AuthKeysPage() {
  const [keys, setKeys] = useState<KeyRow[]>([]);
  const [enabled, setEnabled] = useState(false);
  const [newName, setNewName] = useState('');
  const [generating, setGenerating] = useState(false);
  const [showKey, setShowKey] = useState<{ key: string; name: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await api.listAuthKeys();
      setKeys(r.keys);
      setEnabled(r.enabled);
    } catch (e) {
      setErr((e as Error).message);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setGenerating(true); setErr(null);
    try {
      const r = await api.createAuthKey(name);
      setShowKey({ key: r.key, name: r.name });
      setNewName('');
      refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  const handleRevoke = async (name: string) => {
    if (!confirm(`Revoke key "${name}"? Any library using it will get 401 errors.`)) return;
    try {
      await api.revokeAuthKey(name);
      refresh();
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const formatTime = (t: number | null) => {
    if (!t) return '—';
    const d = new Date(t * 1000);
    return d.toLocaleString();
  };

  return (
    <Page title="Auth Keys" subtitle="API keys for connected libraries (sync mode)" size="md">
      <div className="space-y-6">

        {/* Status banner */}
        <div className={`p-4 rounded-lg border ${enabled
          ? 'bg-emerald-500/5 border-emerald-500/30 text-emerald-400'
          : 'bg-amber-500/5 border-amber-500/30 text-amber-400'}`}>
          <div className="text-sm font-semibold mb-1">
            {enabled
              ? `Auth required: ${keys.length} key${keys.length === 1 ? '' : 's'} configured`
              : 'Open mode: no keys configured'}
          </div>
          <div className="text-xs text-zinc-400">
            {enabled
              ? 'Connected libraries must send X-Api-Key header. Local UI is unaffected.'
              : 'Connected libraries can call /api/sync, /api/ingest, /api/correct without auth. Generate a key below to enable auth (production deployments should always set keys).'}
          </div>
        </div>

        {/* Create form */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-sm font-medium text-zinc-100 mb-2">Generate new key</div>
          <div className="flex gap-2 items-end">
            <div className="flex-1">
              <label className="text-[10px] text-zinc-500 uppercase block mb-1">Name (e.g. "prod-api", "staging-bot")</label>
              <input
                value={newName}
                onChange={e => setNewName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleCreate()}
                placeholder="library instance name"
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-violet-500" />
            </div>
            <button
              onClick={handleCreate}
              disabled={!newName.trim() || generating}
              className="px-4 py-2 text-sm bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white rounded">
              {generating ? 'Generating…' : 'Generate'}
            </button>
          </div>
          {err && <div className="text-xs text-red-400 mt-2">{err}</div>}
        </div>

        {/* Generated key modal */}
        {showKey && (
          <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
            <div className="max-w-2xl w-full bg-zinc-900 border border-amber-500/40 rounded-lg p-6 space-y-4">
              <div className="text-lg font-semibold text-amber-400">Save this key now</div>
              <div className="text-sm text-zinc-300">
                Key for <span className="font-mono text-violet-400">{showKey.name}</span>:
              </div>
              <div className="bg-zinc-950 border border-zinc-800 rounded p-3 font-mono text-xs break-all text-emerald-400 select-all">
                {showKey.key}
              </div>
              <div className="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded p-2">
                ⚠ This key will not be shown again. Copy it now and store it securely (env var, secret manager, etc).
              </div>
              <div className="flex gap-2 justify-end">
                <button
                  onClick={() => navigator.clipboard.writeText(showKey.key)}
                  className="px-3 py-1.5 text-xs border border-zinc-700 text-zinc-300 hover:bg-zinc-800 rounded">
                  Copy
                </button>
                <button
                  onClick={() => setShowKey(null)}
                  className="px-3 py-1.5 text-xs bg-violet-600 hover:bg-violet-500 text-white rounded">
                  I've saved it
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Keys table */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg">
          <div className="px-4 py-2 border-b border-zinc-800 text-xs uppercase tracking-wide text-zinc-500">
            Configured keys ({keys.length})
          </div>
          {keys.length === 0 ? (
            <div className="p-6 text-center text-sm text-zinc-500">
              No keys yet. Generate one above to enable auth on connected-mode endpoints.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-xs text-zinc-500 uppercase">
                <tr className="border-b border-zinc-800">
                  <th className="text-left px-4 py-2 font-normal">Name</th>
                  <th className="text-left px-4 py-2 font-normal">Prefix</th>
                  <th className="text-left px-4 py-2 font-normal">Created</th>
                  <th className="text-left px-4 py-2 font-normal">Last used</th>
                  <th className="text-right px-4 py-2 font-normal w-24"></th>
                </tr>
              </thead>
              <tbody>
                {keys.map(k => (
                  <tr key={k.name} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
                    <td className="px-4 py-2 text-zinc-100 font-medium">{k.name}</td>
                    <td className="px-4 py-2 font-mono text-xs text-zinc-400">{k.prefix}</td>
                    <td className="px-4 py-2 text-xs text-zinc-500">{formatTime(k.created_at)}</td>
                    <td className="px-4 py-2 text-xs text-zinc-500">{formatTime(k.last_used_at)}</td>
                    <td className="px-4 py-2 text-right">
                      <button
                        onClick={() => handleRevoke(k.name)}
                        className="text-xs text-red-400 hover:text-red-300">
                        Revoke
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="text-xs text-zinc-500 space-y-1">
          <div>Keys are stored at <span className="font-mono text-zinc-400">~/.config/microresolve/keys.json</span> (separate from data dir, never git-tracked).</div>
          <div>Last-used tracking is in-memory only (resets on server restart).</div>
        </div>

      </div>
    </Page>
  );
}
