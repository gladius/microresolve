import { useState, useCallback } from 'react';
import { useFetch } from '@/hooks/useFetch';
import { useNavigate } from 'react-router-dom';
import { api } from '@/api/client';
import { useAppStore } from '@/store';
import Page from '@/components/Page';

interface NamespaceInfo {
  id: string;
  name: string;
  description: string;
  auto_learn: boolean;
  default_threshold: number | null;
  version?: number;
  intent_count?: number;
  l0_enabled?: boolean;
  l1_morphology?: boolean;
  l1_synonym?: boolean;
  l1_abbreviation?: boolean;
}

export default function NamespacesPage() {
  const { settings, setSelectedNamespaceId, setLayerStatus } = useAppStore();
  const navigate = useNavigate();
  const current = settings.selectedNamespaceId;

  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Edit modal state — single namespace at a time.
  const [editingNs, setEditingNs] = useState<NamespaceInfo | null>(null);
  // Per-row auto-learn toggle (so on-row toggle doesn't need the full edit modal).
  const [togglingId, setTogglingId] = useState<string | null>(null);

  // Create modal
  const [showModal, setShowModal] = useState(false);
  const [modalId, setModalId] = useState('');
  const [modalDesc, setModalDesc] = useState('');
  const [modalError, setModalError] = useState('');
  const [modalBusy, setModalBusy] = useState(false);

  const refresh = useCallback(async () => {
    try { setNamespaces(await api.listNamespaces()); } catch { /* */ }
    setLoading(false);
  }, []);

  useFetch(refresh, [refresh]);

  const toggleAutoLearn = async (ns: NamespaceInfo) => {
    setTogglingId(ns.id);
    try {
      await api.updateNamespace(ns.id, { auto_learn: !ns.auto_learn });
      refresh();
    } catch { /* */ }
    setTogglingId(null);
  };

  const deleteNs = async (id: string) => {
    if (id === 'default') return;
    if (!confirm(`Delete namespace "${id}" and all its intents? This cannot be undone.`)) return;
    try {
      await api.deleteNamespace(id);
      if (settings.selectedNamespaceId === id) setSelectedNamespaceId('default');
      refresh();
    } catch (e) {
      alert('Delete failed: ' + (e instanceof Error ? e.message : 'unknown'));
    }
  };

  const openModal = () => { setShowModal(true); setModalId(''); setModalDesc(''); setModalError(''); };
  const closeModal = () => { setShowModal(false); setModalId(''); setModalDesc(''); setModalError(''); };

  const submitModal = async () => {
    const id = modalId.trim();
    if (!id) return;
    if (id.length > 40) { setModalError('Max 40 characters.'); return; }
    if (!/^[a-z0-9_-]+$/.test(id)) { setModalError('Lowercase letters, digits, hyphens and underscores only.'); return; }
    setModalBusy(true);
    setModalError('');
    try {
      await api.createNamespace(id, '', modalDesc.trim());
      closeModal();
      refresh();
    } catch (e) {
      setModalError('Failed: ' + (e instanceof Error ? e.message : 'unknown'));
    }
    setModalBusy(false);
  };

  return (
    <Page
      size="sm"
      className="space-y-6"
      title="Namespaces"
      subtitle="Isolated routing namespaces"
      actions={
        <button
          onClick={openModal}
          className="flex items-center gap-1.5 px-3 py-1 bg-emerald-600 text-white text-xs rounded hover:bg-emerald-500 transition-colors"
        >
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New
        </button>
      }
    >

      {loading ? (
        <div className="text-xs text-zinc-500 text-center py-12">Loading…</div>
      ) : (
        <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/50">
          {namespaces.map(ns => {
            const isActive = ns.id === current;
            const isToggling = togglingId === ns.id;
            return (
              <div key={ns.id} className={`px-4 py-3 ${isActive ? 'bg-emerald-500/5' : ''}`}>
                {/* Top line: id · meta · pills */}
                <div className="flex items-center gap-3 flex-wrap">
                  <button
                    onClick={() => navigate(`/namespaces/${ns.id}`)}
                    className="text-sm font-mono text-zinc-200 hover:text-emerald-400 transition-colors"
                  >
                    {ns.id}
                  </button>
                  <span className="text-[10px] text-zinc-500 font-mono">v{ns.version ?? 0}</span>
                  <span className="text-[10px] text-zinc-500">·</span>
                  <span className="text-[10px] text-zinc-500">{ns.intent_count ?? 0} intent{ns.intent_count !== 1 ? 's' : ''}</span>
                  {isActive && (
                    <span className="text-[9px] text-emerald-400 bg-emerald-500/20 px-1.5 py-0.5 rounded uppercase tracking-wide">active</span>
                  )}
                </div>

                {/* Description (static) */}
                <div className="text-xs text-zinc-500 mt-1 mb-2.5">
                  {ns.description || <span className="italic text-zinc-700">no description</span>}
                </div>

                {/* Action row: prominent auto-learn + edit + delete */}
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-2 cursor-pointer select-none" title={ns.auto_learn ? 'LLM-judges every routed query and applies fixes' : 'Queries pile up for manual review'}>
                    <button
                      type="button"
                      role="switch"
                      aria-checked={ns.auto_learn}
                      disabled={isToggling}
                      onClick={() => toggleAutoLearn(ns)}
                      className={`relative w-9 h-5 rounded-full transition-colors disabled:opacity-50 ${ns.auto_learn ? 'bg-emerald-500' : 'bg-zinc-700'}`}
                    >
                      <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${ns.auto_learn ? 'translate-x-4' : 'translate-x-0'}`} />
                    </button>
                    <span className={`text-xs ${ns.auto_learn ? 'text-emerald-300 font-medium' : 'text-zinc-500'}`}>
                      Auto-learn {ns.auto_learn ? 'on' : 'off'}
                    </span>
                  </label>

                  <div className="ml-auto flex items-center gap-2">
                    <button
                      onClick={() => setEditingNs(ns)}
                      className="text-xs text-zinc-400 hover:text-emerald-300 transition-colors px-2 py-1 rounded hover:bg-zinc-800/60"
                    >
                      Edit
                    </button>
                    {ns.id !== 'default' && (
                      <button
                        onClick={() => deleteNs(ns.id)}
                        className="text-xs text-zinc-600 hover:text-red-400 transition-colors px-2 py-1"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Edit modal */}
      {editingNs && (
        <EditNamespaceModal
          ns={editingNs}
          onClose={() => setEditingNs(null)}
          onSaved={(updated) => {
            setEditingNs(null);
            // Push the new layer-toggle state into the global store if the
            // edited namespace is the one currently selected in the sidebar.
            // Keeps sidebar pills in sync without a roundtrip.
            if (updated.id === settings.selectedNamespaceId) {
              setLayerStatus({
                l0:  updated.l0_enabled       ?? true,
                l1m: updated.l1_morphology    ?? true,
                l1s: updated.l1_synonym       ?? true,
                l1a: updated.l1_abbreviation  ?? true,
              });
            }
            refresh();
          }}
        />
      )}

      {/* Create modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-zinc-100">New Namespace</h3>
              <button onClick={closeModal} className="text-zinc-500 hover:text-zinc-300 text-xl leading-none">×</button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Namespace ID <span className="text-zinc-600">(immutable)</span></label>
                <input
                  autoFocus
                  value={modalId}
                  onChange={e => { setModalId(e.target.value); setModalError(''); }}
                  onKeyDown={e => e.key === 'Enter' && submitModal()}
                  placeholder="billing-bot"
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-zinc-100 font-mono placeholder-zinc-500 focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Description <span className="text-zinc-600">(optional)</span></label>
                <input
                  value={modalDesc}
                  onChange={e => setModalDesc(e.target.value)}
                  placeholder="What does this namespace handle?"
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>
            {modalError && <p className="text-xs text-red-400">{modalError}</p>}
            <p className="text-[11px] text-zinc-600">Lowercase letters, digits, hyphens, underscores · max 40 chars.</p>
            <div className="flex gap-2 justify-end pt-1">
              <button onClick={closeModal} className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-100">Cancel</button>
              <button
                onClick={submitModal}
                disabled={modalBusy || !modalId.trim()}
                className="px-5 py-2 text-sm bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 disabled:opacity-40"
              >
                {modalBusy ? 'Creating…' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </Page>
  );
}

// ─── Edit modal ───────────────────────────────────────────────────────────────

function EditNamespaceModal({
  ns, onClose, onSaved,
}: {
  ns: NamespaceInfo;
  onClose: () => void;
  onSaved: (updated: NamespaceInfo) => void;
}) {
  const [description, setDescription] = useState(ns.description);
  const [autoLearn,   setAutoLearn]   = useState(ns.auto_learn);
  const [threshold,   setThreshold]   = useState(ns.default_threshold == null ? '' : String(ns.default_threshold));
  const [l0,   setL0]   = useState(ns.l0_enabled       ?? true);
  const [l1m,  setL1m]  = useState(ns.l1_morphology    ?? true);
  const [l1s,  setL1s]  = useState(ns.l1_synonym       ?? true);
  const [l1a,  setL1a]  = useState(ns.l1_abbreviation  ?? true);
  const [busy, setBusy] = useState(false);
  const [err,  setErr]  = useState<string | null>(null);

  const save = async () => {
    setBusy(true);
    setErr(null);
    let thresholdValue: number | null | undefined;
    const raw = threshold.trim();
    if (raw === '') thresholdValue = null;
    else {
      const n = Number(raw);
      if (Number.isNaN(n) || n < 0) { setErr('Threshold must be a non-negative number.'); setBusy(false); return; }
      thresholdValue = n;
    }
    try {
      await api.updateNamespace(ns.id, {
        description,
        auto_learn: autoLearn,
        ...(thresholdValue !== undefined ? { default_threshold: thresholdValue } : {}),
        l0_enabled: l0,
        l1_morphology: l1m,
        l1_synonym: l1s,
        l1_abbreviation: l1a,
      });
      onSaved({
        ...ns,
        description,
        auto_learn: autoLearn,
        default_threshold: thresholdValue ?? null,
        l0_enabled: l0,
        l1_morphology: l1m,
        l1_synonym: l1s,
        l1_abbreviation: l1a,
      });
    } catch (e) {
      setErr('Failed: ' + (e instanceof Error ? e.message : 'unknown'));
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={() => !busy && onClose()}
    >
      <div
        className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-md p-6 space-y-4"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold text-zinc-100">Edit namespace</h3>
          <span className="font-mono text-xs text-zinc-500">{ns.id}</span>
        </div>

        <div>
          <label className="text-[10px] text-zinc-500 uppercase tracking-wide block mb-1">Description</label>
          <input
            autoFocus
            value={description}
            onChange={e => setDescription(e.target.value)}
            placeholder="What does this namespace handle?"
            className="w-full bg-zinc-950 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div>
          <label className="text-[10px] text-zinc-500 uppercase tracking-wide block mb-1 flex items-center gap-1.5">
            Default routing threshold
            <span
              title={
                'Minimum score required for a route to be confirmed. Leave empty to use the system default (0.3). ' +
                'Lower values for tool-routing where vocabulary is unique. Higher values for safety/classification ' +
                'where attack vocabulary overlaps with normal English.'
              }
              className="text-zinc-600 cursor-help text-[10px] border border-zinc-700 rounded-full w-3.5 h-3.5 inline-flex items-center justify-center"
            >?</span>
          </label>
          <input
            type="number"
            step="0.05"
            min="0"
            value={threshold}
            onChange={e => setThreshold(e.target.value)}
            placeholder="empty = system default (0.3)"
            className="w-full bg-zinc-950 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div className="flex items-center gap-3 py-2">
          <button
            type="button"
            role="switch"
            aria-checked={autoLearn}
            onClick={() => setAutoLearn(!autoLearn)}
            className={`relative w-10 h-6 rounded-full transition-colors ${autoLearn ? 'bg-emerald-500' : 'bg-zinc-600'}`}
          >
            <span className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white shadow transition-transform ${autoLearn ? 'translate-x-4' : 'translate-x-0'}`} />
          </button>
          <div className="flex-1">
            <div className="text-sm text-zinc-200">Auto-learn</div>
            <div className="text-[11px] text-zinc-500">
              {autoLearn ? 'LLM-judges every routed query and applies fixes' : 'Queries pile up for manual review'}
            </div>
          </div>
        </div>

        <div className="border-t border-zinc-800 pt-3 space-y-2">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide flex items-center gap-1.5">
            Reflex layers
            <span
              title="Disable per layer when defaults are wrong for your content. Examples: turn L0 off for medical/legal namespaces; turn L1 abbreviations off for code search."
              className="text-zinc-600 cursor-help text-[10px] border border-zinc-700 rounded-full w-3.5 h-3.5 inline-flex items-center justify-center"
            >?</span>
          </div>
          <LayerToggle label="L0 — Spelling"           hint="Char n-gram typo correction" on={l0}  set={setL0}  />
          <LayerToggle label="L1 — Morphology"         hint="canceling → cancel"          on={l1m} set={setL1m} />
          <LayerToggle label="L1 — Synonyms"           hint="OOV-only synonym substitution" on={l1s} set={setL1s} />
          <LayerToggle label="L1 — Abbreviations"      hint="pr → pull request"           on={l1a} set={setL1a} />
        </div>

        {err && <p className="text-xs text-red-400">{err}</p>}

        <div className="flex justify-end gap-2 pt-1">
          <button
            onClick={onClose}
            disabled={busy}
            className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-100 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={save}
            disabled={busy}
            className="px-5 py-2 text-sm bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 disabled:opacity-40"
          >
            {busy ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}

function LayerToggle({ label, hint, on, set }: {
  label: string; hint: string; on: boolean; set: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        role="switch"
        aria-checked={on}
        onClick={() => set(!on)}
        className={`relative w-9 h-5 rounded-full transition-colors flex-shrink-0 ${on ? 'bg-emerald-500' : 'bg-zinc-700'}`}
      >
        <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${on ? 'translate-x-4' : 'translate-x-0'}`} />
      </button>
      <div className="flex-1 min-w-0">
        <div className="text-xs text-zinc-200 leading-tight">{label}</div>
        <div className="text-[10px] text-zinc-500 leading-tight font-mono">{hint}</div>
      </div>
    </div>
  );
}

