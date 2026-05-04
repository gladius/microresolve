import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '@/api/client';

interface TuningPanelProps {
  namespaceId: string;
  compact?: boolean;
  onAfterUpdate?: () => void;
}

// Spinner shown while PATCH is in flight
function Spinner() {
  return (
    <span className="inline-block w-3 h-3 border border-zinc-500 border-t-emerald-400 rounded-full animate-spin" />
  );
}

// Checkmark shown briefly after a successful save
function Check() {
  return <span className="text-emerald-400 text-xs">✓</span>;
}

// Tooltip via title attribute — matches the existing NamespacesPage pattern
function HelpIcon({ text }: { text: string }) {
  return (
    <span
      title={text}
      className="text-zinc-600 cursor-help text-[10px] border border-zinc-700 rounded-full w-3.5 h-3.5 inline-flex items-center justify-center shrink-0"
    >
      ?
    </span>
  );
}

const THRESHOLD_TOOLTIP =
  'Score threshold for High-band match. Lower = more permissive (more matches, more false positives). ' +
  'Higher = stricter (fewer matches, fewer false positives). Most packs work well at 1.0–1.5.';

const VOTING_TOOLTIP =
  'Voting-token gate. Requires N distinct query words to vote for an intent before it can fire at full strength. ' +
  'Reduces single-word false positives like "let me speak from the heart" matching a clinical_urgent intent because ' +
  'of "heart" alone. Default 1 (off). Set 2 or 3 to suppress single-word matches.';

// Fetch current namespace values from the list endpoint
async function fetchNsValues(namespaceId: string): Promise<{ threshold: number | null; minVt: number }> {
  const list = await api.listNamespaces();
  const ns = list.find(n => n.id === namespaceId);
  if (!ns) return { threshold: null, minVt: 1 };
  return {
    threshold: ns.default_threshold,
    minVt: ns.default_min_voting_tokens ?? 1,
  };
}

// Save both values together
async function saveNsValues(namespaceId: string, threshold: number | null, minVt: number) {
  await api.updateNamespace(namespaceId, {
    default_threshold: threshold,
    default_min_voting_tokens: minVt,
  });
}

// ── Full (non-compact) panel ──────────────────────────────────────────────────

function FullPanel({ namespaceId, onAfterUpdate }: { namespaceId: string; onAfterUpdate?: () => void }) {
  const [threshold, setThreshold] = useState<number>(1.0);
  const [minVt,     setMinVt]     = useState<number>(1);
  const [loaded,    setLoaded]    = useState(false);
  const [saving,    setSaving]    = useState(false);
  const [saved,     setSaved]     = useState(false);
  const [error,     setError]     = useState<string | null>(null);

  // Optimistic snapshot for revert on failure
  const prevRef = useRef<{ threshold: number; minVt: number } | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    fetchNsValues(namespaceId).then(v => {
      setThreshold(v.threshold ?? 1.0);
      setMinVt(v.minVt);
      setLoaded(true);
    }).catch(() => setLoaded(true));
  }, [namespaceId]);

  const scheduleSave = useCallback((newThreshold: number, newMinVt: number) => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(async () => {
      setSaving(true);
      setSaved(false);
      setError(null);
      try {
        await saveNsValues(namespaceId, newThreshold, newMinVt);
        setSaving(false);
        setSaved(true);
        prevRef.current = null;
        onAfterUpdate?.();
        setTimeout(() => setSaved(false), 1500);
      } catch (e) {
        setSaving(false);
        setError('Save failed');
        // Revert optimistic update
        if (prevRef.current) {
          setThreshold(prevRef.current.threshold);
          setMinVt(prevRef.current.minVt);
          prevRef.current = null;
        }
        setTimeout(() => setError(null), 3000);
      }
    }, 300);
  }, [namespaceId, onAfterUpdate]);

  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!prevRef.current) prevRef.current = { threshold, minVt };
    const v = parseFloat(e.target.value);
    setThreshold(v);
    scheduleSave(v, minVt);
  };

  const handleMinVtChange = (v: number) => {
    if (!prevRef.current) prevRef.current = { threshold, minVt };
    setMinVt(v);
    scheduleSave(threshold, v);
  };

  if (!loaded) {
    return (
      <div className="flex items-center gap-2 py-2 text-xs text-zinc-600">
        <Spinner /> Loading tuning…
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">Tuning</span>
        <span className="flex items-center gap-1.5">
          {saving && <Spinner />}
          {saved && <Check />}
          {error && <span className="text-red-400 text-[10px]">{error}</span>}
        </span>
      </div>

      {/* Threshold slider */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-zinc-400">Threshold</label>
            <HelpIcon text={THRESHOLD_TOOLTIP} />
          </div>
          <span className="text-xs font-mono text-emerald-300">{threshold.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.1"
          max="4.0"
          step="0.05"
          value={threshold}
          onChange={handleThresholdChange}
          className="w-full h-1.5 appearance-none rounded bg-zinc-700 accent-emerald-500 cursor-pointer"
        />
        <div className="flex justify-between text-[9px] text-zinc-700 font-mono">
          <span>0.1</span>
          <span>4.0</span>
        </div>
      </div>

      {/* Voting-token gate button group */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-zinc-400">Min voting tokens</label>
          <HelpIcon text={VOTING_TOOLTIP} />
        </div>
        <div className="flex gap-1.5">
          {[1, 2, 3].map(v => (
            <button
              key={v}
              onClick={() => handleMinVtChange(v)}
              className={`flex-1 text-xs py-1.5 rounded border transition-colors font-mono ${
                minVt === v
                  ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                  : 'border-zinc-700 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300'
              }`}
              title={v === 1 ? 'Off — no gate' : `Require ${v} distinct tokens`}
            >
              {v === 1 ? '1 (off)' : v}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Compact (sidebar pill) panel ──────────────────────────────────────────────

function CompactPanel({ namespaceId, onAfterUpdate }: { namespaceId: string; onAfterUpdate?: () => void }) {
  const [threshold, setThreshold] = useState<number | null>(null);
  const [minVt,     setMinVt]     = useState<number>(1);
  const [expanded,  setExpanded]  = useState(false);
  const [loaded,    setLoaded]    = useState(false);

  const reload = useCallback(() => {
    fetchNsValues(namespaceId).then(v => {
      setThreshold(v.threshold);
      setMinVt(v.minVt);
      setLoaded(true);
    }).catch(() => setLoaded(true));
  }, [namespaceId]);

  useEffect(() => { reload(); }, [reload]);

  if (!loaded) return null;

  const threshDisplay = threshold != null ? threshold.toFixed(2) : 'def';

  return (
    <div className="px-2 pb-1">
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 transition-colors font-mono"
        title="Tuning — click to edit threshold and voting gate"
      >
        <span className="text-zinc-500">thresh</span>
        <span className="text-zinc-200 font-semibold">{threshDisplay}</span>
        <span className="text-zinc-700">·</span>
        <span className="text-zinc-500">min_vt</span>
        <span className="text-zinc-200 font-semibold">{minVt}</span>
        <span className="ml-auto text-zinc-500">⚙</span>
      </button>

      {expanded && (
        <div className="mt-1 px-1 py-2 bg-zinc-900/80 border border-zinc-800 rounded-lg">
          <FullPanel
            namespaceId={namespaceId}
            onAfterUpdate={() => {
              reload();
              onAfterUpdate?.();
            }}
          />
        </div>
      )}
    </div>
  );
}

// ── Public export ─────────────────────────────────────────────────────────────

export default function TuningPanel({ namespaceId, compact = false, onAfterUpdate }: TuningPanelProps) {
  if (compact) {
    return <CompactPanel namespaceId={namespaceId} onAfterUpdate={onAfterUpdate} />;
  }
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <FullPanel namespaceId={namespaceId} onAfterUpdate={onAfterUpdate} />
    </div>
  );
}
