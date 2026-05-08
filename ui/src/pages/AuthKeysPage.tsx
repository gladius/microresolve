import { useState, useCallback, useEffect } from 'react';
import Page from '@/components/Page';
import { api, type AuditMode } from '@/api/client';

type KeyScope = 'admin' | 'app';

interface KeyRow {
  name: string;
  prefix: string;
  scope: KeyScope;
  created_at: number;
  last_used_at: number | null;
}

interface ChainHead {
  kid: string;
  head_hash: string;
  count: number;
}

interface VerifyChain {
  kid: string;
  entries: number;
  head_hash: string;
  ok: boolean;
  error: string | null;
}

interface VerifyReport {
  ok: boolean;
  total_entries: number;
  chains_verified: number;
  chains_with_errors: number;
  chains: VerifyChain[];
}

export default function AuthKeysPage() {
  const [keys, setKeys] = useState<KeyRow[]>([]);
  const [enabled, setEnabled] = useState(false);
  const [newName, setNewName] = useState('');
  const [newScope, setNewScope] = useState<KeyScope>('app');
  const [generating, setGenerating] = useState(false);
  const [showKey, setShowKey] = useState<{ key: string; name: string; scope: KeyScope } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // Audit state — per-key chain summaries live alongside the keys
  // table since "audit chain identity = key name (kid)".
  const [auditMode, setAuditMode] = useState<AuditMode>('default');
  const [heads, setHeads] = useState<ChainHead[]>([]);
  const [verifying, setVerifying] = useState(false);
  const [verifyReport, setVerifyReport] = useState<VerifyReport | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [keyList, auditCfg, headsRes] = await Promise.all([
        api.listAuthKeys(),
        api.auditConfig(),
        api.auditHeads(),
      ]);
      setKeys(keyList.keys);
      setEnabled(keyList.enabled);
      setAuditMode(auditCfg.mode);
      setHeads(headsRes.heads);
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
      const r = await api.createAuthKey(name, newScope);
      setShowKey({ key: r.key, name: r.name, scope: r.scope });
      setNewName('');
      setNewScope('app');
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

  const handleVerify = async () => {
    setVerifying(true); setVerifyReport(null);
    try {
      const r = await api.auditVerify();
      setVerifyReport(r);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setVerifying(false);
    }
  };

  const formatTime = (t: number | null) => {
    if (!t) return '—';
    const d = new Date(t * 1000);
    return d.toLocaleString();
  };

  const headFor = (name: string): ChainHead | undefined => heads.find(h => h.kid === name);
  const verifyFor = (name: string): VerifyChain | undefined => verifyReport?.chains.find(c => c.kid === name);

  return (
    <Page
      title="Auth & Audit"
      subtitle="API keys for connected libraries — and the tamper-evident audit chain each key writes"
      size="md"
    >
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
          <div className="text-xs mt-2">
            <span className="text-zinc-500">Audit:</span>{' '}
            <span className={`font-mono ${auditMode === 'default' ? 'text-emerald-400' : 'text-zinc-500'}`}>
              {auditMode}
            </span>
            {auditMode === 'default' && (
              <span className="text-zinc-500"> — every routing decision and mutation is recorded in the per-key chain.</span>
            )}
            {auditMode === 'off' && (
              <span className="text-zinc-500"> — set <span className="font-mono">[audit] mode = "default"</span> in config.toml to enable.</span>
            )}
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
                placeholder="app / workload name"
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-emerald-500" />
            </div>
            <div>
              <label className="text-[10px] text-zinc-500 uppercase block mb-1">Scope</label>
              <select
                value={newScope}
                onChange={e => setNewScope(e.target.value as KeyScope)}
                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:border-emerald-500"
              >
                <option value="app">app</option>
                <option value="admin">admin</option>
              </select>
            </div>
            <button
              onClick={handleCreate}
              disabled={!newName.trim() || generating}
              className="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded">
              {generating ? 'Generating…' : 'Generate'}
            </button>
          </div>
          <div className="text-[11px] text-zinc-500 mt-2">
            <span className="text-zinc-400 font-medium">App</span> — one key per <em className="text-zinc-400 not-italic">workload</em> (a Deployment / service / tenant — not per pod or per process). All replicas of one workload share the key. The key name identifies the chain in the audit log.{' '}
            <span className="text-zinc-400 font-medium">Admin</span> — for operators / teammates who manage the Studio itself.
          </div>
          {err && <div className="text-xs text-red-400 mt-2">{err}</div>}
        </div>

        {/* Generated key modal */}
        {showKey && (
          <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
            <div className="max-w-2xl w-full bg-zinc-900 border border-amber-500/40 rounded-lg p-6 space-y-4">
              <div className="text-lg font-semibold text-amber-400">Save this key now</div>
              <div className="text-sm text-zinc-300">
                Key for <span className="font-mono text-emerald-400">{showKey.name}</span>{' '}
                <ScopeBadge scope={showKey.scope} />:
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
                  className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-white rounded">
                  I've saved it
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Keys table — with audit columns */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg">
          <div className="px-4 py-2 border-b border-zinc-800 flex items-center justify-between">
            <span className="text-xs uppercase tracking-wide text-zinc-500">
              Configured keys ({keys.length})
            </span>
            <button
              onClick={handleVerify}
              disabled={verifying || heads.length === 0}
              className="px-3 py-1 text-xs bg-emerald-600 hover:bg-emerald-500 disabled:opacity-30 text-white rounded font-mono"
              title="Walk every chain, recompute hashes, flag any tampering."
            >
              {verifying ? 'Verifying…' : 'Verify chains'}
            </button>
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
                  <th className="text-left px-4 py-2 font-normal">Scope</th>
                  <th className="text-left px-4 py-2 font-normal">Created</th>
                  <th className="text-left px-4 py-2 font-normal">Last used</th>
                  <th className="text-right px-4 py-2 font-normal" title="Audit chain entries written by this key">Chain</th>
                  <th className="text-center px-4 py-2 font-normal">Verify</th>
                  <th className="text-right px-4 py-2 font-normal w-24"></th>
                </tr>
              </thead>
              <tbody>
                {keys.map(k => {
                  const head = headFor(k.name);
                  const verify = verifyFor(k.name);
                  return (
                    <tr key={k.name} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
                      <td className="px-4 py-2 text-zinc-100 font-medium">
                        <div className="flex items-center gap-2">
                          {k.name}
                          <span className="font-mono text-[10px] text-zinc-600">{k.prefix}</span>
                        </div>
                      </td>
                      <td className="px-4 py-2"><ScopeBadge scope={k.scope} /></td>
                      <td className="px-4 py-2 text-xs text-zinc-500">{formatTime(k.created_at)}</td>
                      <td className="px-4 py-2 text-xs text-zinc-500">{formatTime(k.last_used_at)}</td>
                      <td className="px-4 py-2 text-right text-xs font-mono">
                        {head ? (
                          <span className="text-zinc-300">{head.count.toLocaleString()}</span>
                        ) : (
                          <span className="text-zinc-700">—</span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-center text-xs">
                        {verify ? (
                          verify.ok ? (
                            <span className="text-emerald-400">✓</span>
                          ) : (
                            <span className="text-red-400" title={verify.error || ''}>✗ break</span>
                          )
                        ) : (
                          <span className="text-zinc-700">—</span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-right">
                        <button
                          onClick={() => handleRevoke(k.name)}
                          className="text-xs text-red-400 hover:text-red-300">
                          Revoke
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          {verifyReport && (
            <div className={`px-4 py-2 border-t border-zinc-800 text-xs ${verifyReport.ok ? 'text-emerald-400' : 'text-red-400'}`}>
              {verifyReport.ok
                ? `OK: ${verifyReport.total_entries.toLocaleString()} entries across ${verifyReport.chains_verified} chains verified.`
                : `FAIL: ${verifyReport.chains_with_errors} chain(s) failed verification.`}
            </div>
          )}
        </div>

        <div className="text-[11px] text-zinc-500 space-y-1">
          <div>Keys are stored at <span className="font-mono text-zinc-400">~/.config/microresolve/keys.json</span> (separate from data dir, never git-tracked).</div>
          <div>
            Audit chains live at <span className="font-mono text-zinc-400">{`{data_dir}/_audit/{key-name}.log`}</span> —
            append-only, hash-chained à la Certificate Transparency. Verify from the CLI with{' '}
            <span className="font-mono text-zinc-400">microresolve-studio verify-log</span>.
          </div>
          <div>Last-used tracking is in-memory only (resets on server restart).</div>
        </div>

      </div>
    </Page>
  );
}

function ScopeBadge({ scope }: { scope: KeyScope }) {
  const cls = scope === 'admin'
    ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
    : 'bg-zinc-800 text-zinc-300 border-zinc-700';
  return (
    <span className={`inline-block text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border ${cls}`}>
      {scope}
    </span>
  );
}
