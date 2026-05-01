import { useState } from 'react';
import { useAppStore, type LayerStatus } from '@/store';
import { api } from '@/api/client';

/// Single source of truth for any "toggle a reflex-layer field" UI.
///
/// Flow: click → optimistic store update → PATCH /api/namespaces → on
/// failure, revert. Sidebar pills read from the same store so they
/// reflect the new state instantly (no extra GET).
export type LayerField = 'l0_enabled' | 'l1_morphology' | 'l1_synonym' | 'l1_abbreviation';

const FIELD_TO_KEY: Record<LayerField, keyof LayerStatus> = {
  l0_enabled:      'l0',
  l1_morphology:   'l1m',
  l1_synonym:      'l1s',
  l1_abbreviation: 'l1a',
};

export default function LayerToggle({ field, label, hint, compact }: {
  field: LayerField;
  label: string;
  hint?: string;
  compact?: boolean;
}) {
  const { settings, layerStatus, setLayerStatus } = useAppStore();
  const [busy, setBusy] = useState(false);
  const [err,  setErr]  = useState<string | null>(null);

  const key = FIELD_TO_KEY[field];
  const on  = layerStatus[key];

  const toggle = async () => {
    if (busy) return;
    const next = !on;
    setBusy(true);
    setErr(null);
    // Functional updates: rapid toggles on different fields would otherwise
    // race — each closure captures its own `layerStatus` snapshot, and the
    // last-to-resolve would clobber the others. `prev =>` always sees the
    // freshest store state, so an in-flight failure on field A only reverts
    // field A, leaving a concurrent field B toggle untouched.
    setLayerStatus(prev => ({ ...prev, [key]: next }));
    try {
      await api.updateNamespace(settings.selectedNamespaceId, { [field]: next });
    } catch (e) {
      setLayerStatus(prev => ({ ...prev, [key]: !next }));
      setErr(e instanceof Error ? e.message : 'failed');
    } finally {
      setBusy(false);
    }
  };

  if (compact) {
    // Compact mode is used inline in dense headers (e.g. L1 page column
    // headers) where there's no room for an inline error <div>. Surface
    // failures through the title + a red border instead of swallowing them.
    const titleText = err
      ? `${label} — failed: ${err}`
      : `${label}${hint ? ' — ' + hint : ''}`;
    return (
      <button
        type="button"
        onClick={toggle}
        disabled={busy}
        title={titleText}
        className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-medium border transition-colors ${
          err
            ? 'bg-red-500/15 border-red-500/40 text-red-300'
            : on
            ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
            : 'bg-zinc-800 border-zinc-700 text-zinc-500'
        } ${busy ? 'opacity-50' : 'hover:border-emerald-500/50 cursor-pointer'}`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${err ? 'bg-red-400' : on ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
        {label}
        <span className="font-mono uppercase tracking-wider opacity-60">{err ? 'err' : on ? 'on' : 'off'}</span>
      </button>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        role="switch"
        aria-checked={on}
        onClick={toggle}
        disabled={busy}
        className={`relative w-9 h-5 rounded-full transition-colors flex-shrink-0 ${on ? 'bg-emerald-500' : 'bg-zinc-700'} ${busy ? 'opacity-50' : ''}`}
      >
        <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${on ? 'translate-x-4' : 'translate-x-0'}`} />
      </button>
      <div className="flex-1 min-w-0">
        <div className="text-xs text-zinc-200 leading-tight">{label}</div>
        {hint && <div className="text-[10px] text-zinc-500 leading-tight font-mono">{hint}</div>}
        {err && <div className="text-[10px] text-red-400 leading-tight">{err}</div>}
      </div>
    </div>
  );
}
