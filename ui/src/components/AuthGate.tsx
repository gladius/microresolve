import { useState } from 'react';
import { setApiKey } from '@/api/client';

/// First-mount paste screen for the Studio API key.
///
/// The server auto-mints `studio-admin` on its first boot and prints
/// the full key to stdout (and persists it to `~/.config/microresolve/admin-key.txt`).
/// The operator pastes that key here once; it lives in localStorage and
/// every subsequent fetch carries `X-Api-Key`.
///
/// On 401 from any endpoint (revoked / rotated / wrong server) the api
/// client clears localStorage and reloads — that brings us right back here.
export default function AuthGate({ onAuthorized }: { onAuthorized: () => void }) {
  const [value, setValue] = useState('');
  const [busy,  setBusy]  = useState(false);
  const [err,   setErr]   = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const key = value.trim();
    if (!key) return;
    if (!key.startsWith('mr_')) {
      setErr('That doesn\'t look like a MicroResolve key — it should start with "mr_".');
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      // Probe a protected endpoint with this key. /api/version is public,
      // so we use /api/auth/keys (GET) which requires auth — its 200 vs 401
      // tells us if the key is valid before we persist.
      const res = await fetch('/api/auth/keys', { headers: { 'X-Api-Key': key } });
      if (res.status === 401) {
        setErr('Server rejected this key. Check the key matches what was printed on Studio boot.');
        setBusy(false);
        return;
      }
      if (!res.ok) {
        setErr(`Probe returned HTTP ${res.status}. Is the server running?`);
        setBusy(false);
        return;
      }
      setApiKey(key);
      onAuthorized();
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Network error');
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-zinc-950 px-4">
      <form
        onSubmit={submit}
        className="w-full max-w-md bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl p-6 space-y-4"
      >
        <div className="flex items-baseline gap-1.5">
          <span className="text-emerald-400 font-bold text-2xl leading-none">μ</span>
          <span className="text-zinc-100 font-bold text-xl tracking-tight">Resolve</span>
          <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-[0.15em] ml-1">Studio</span>
        </div>

        <div className="text-sm text-zinc-300 leading-relaxed space-y-2">
          <p>Paste your Studio API key to continue.</p>
          <p className="text-xs text-zinc-500">
            On first boot the server prints an admin key (<span className="font-mono">mr_studio-admin_…</span>)
            and persists it to <span className="font-mono">~/.config/microresolve/admin-key.txt</span>.
            Lost it? Run <span className="font-mono">cat ~/.config/microresolve/admin-key.txt</span>.
          </p>
        </div>

        <input
          type="password"
          autoFocus
          value={value}
          onChange={e => setValue(e.target.value)}
          placeholder="mr_studio-admin_…"
          className="w-full bg-zinc-950 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500 font-mono"
        />

        {err && <div className="text-xs text-red-400">{err}</div>}

        <button
          type="submit"
          disabled={busy || !value.trim()}
          className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium py-2 rounded transition-colors"
        >
          {busy ? 'Verifying…' : 'Continue'}
        </button>

        <div className="text-[10px] text-zinc-600 text-center">
          The key is stored in your browser's localStorage. Sign out by clicking it in the sidebar later.
        </div>
      </form>
    </div>
  );
}
