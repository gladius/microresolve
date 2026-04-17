import { useState, useCallback, useEffect } from 'react';
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
}

export default function NamespacesPage() {
  const { settings, setSelectedNamespaceId } = useAppStore();
  const navigate = useNavigate();
  const current = settings.selectedNamespaceId;

  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Per-row edit state
  const [editing, setEditing] = useState<Record<string, { description: string; auto_learn: boolean }>>({});
  const [saving, setSaving] = useState<Record<string, boolean>>({});

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

  const startEditing = (ns: NamespaceInfo) => {
    setEditing(prev => ({
      ...prev,
      [ns.id]: { description: ns.description, auto_learn: ns.auto_learn },
    }));
  };

  const cancelEditing = (id: string) => {
    setEditing(prev => { const next = { ...prev }; delete next[id]; return next; });
  };

  const saveEditing = async (id: string) => {
    const patch = editing[id];
    if (!patch) return;
    setSaving(prev => ({ ...prev, [id]: true }));
    try {
      await api.updateNamespace(id, patch);
      cancelEditing(id);
      refresh();
    } catch { /* */ }
    setSaving(prev => ({ ...prev, [id]: false }));
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
          className="flex items-center gap-1.5 px-3 py-1 bg-violet-600 text-white text-xs rounded hover:bg-violet-500 transition-colors"
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
            const draft = editing[ns.id];
            const isSaving = saving[ns.id];

            return (
              <div key={ns.id} className={`px-4 py-4 ${isActive ? 'bg-violet-500/5' : ''}`}>
                {/* Header row */}
                <div className="flex items-center gap-3 mb-3">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <button
                      onClick={() => navigate(`/namespaces/${ns.id}`)}
                      className="text-sm font-mono text-zinc-400 hover:text-violet-400 transition-colors"
                    >
                      {ns.id}
                    </button>
                    {isActive && (
                      <span className="text-[9px] text-violet-400 bg-violet-500/20 px-1.5 py-0.5 rounded uppercase tracking-wide">active</span>
                    )}
                    {ns.auto_learn && (
                      <span className="text-[9px] text-emerald-400 bg-emerald-500/15 px-1.5 py-0.5 rounded uppercase tracking-wide flex items-center gap-1">
                        <span className="w-1 h-1 rounded-full bg-emerald-400 inline-block" />
                        auto-learn
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {!draft && (
                      <button
                        onClick={() => startEditing(ns)}
                        className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                      >
                        Edit
                      </button>
                    )}
                    {ns.id !== 'default' && (
                      <button
                        onClick={() => deleteNs(ns.id)}
                        className="text-xs text-zinc-600 hover:text-red-400 transition-colors px-1"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>

                {draft ? (
                  /* Edit form */
                  <div className="space-y-2.5 border border-zinc-700/50 rounded-lg p-3 bg-zinc-900/40">
                    <div>
                      <label className="text-[10px] text-zinc-500 uppercase tracking-wide block mb-1">Description</label>
                      <input
                        autoFocus
                        value={draft.description}
                        onChange={e => setEditing(prev => ({ ...prev, [ns.id]: { ...draft, description: e.target.value } }))}
                        placeholder="What does this namespace handle?"
                        className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500"
                      />
                    </div>

                    <div className="flex items-center justify-between pt-1">
                      {/* Auto-learn toggle */}
                      <label className="flex items-center gap-2.5 cursor-pointer select-none">
                        <button
                          type="button"
                          role="switch"
                          aria-checked={draft.auto_learn}
                          onClick={() => setEditing(prev => ({ ...prev, [ns.id]: { ...draft, auto_learn: !draft.auto_learn } }))}
                          className={`relative w-10 h-6 rounded-full transition-colors ${draft.auto_learn ? 'bg-emerald-500' : 'bg-zinc-600'}`}
                        >
                          <span className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white shadow transition-transform ${draft.auto_learn ? 'translate-x-4' : 'translate-x-0'}`} />
                        </button>
                        <span className="text-xs text-zinc-300">Auto-learn</span>
                        <span className="text-xs text-zinc-600">
                          {draft.auto_learn ? 'LLM reviews every flagged query automatically' : 'Manual review only'}
                        </span>
                      </label>

                      <div className="flex items-center gap-2">
                        <button onClick={() => cancelEditing(ns.id)} className="text-xs text-zinc-500 hover:text-zinc-300 px-2 py-1">Cancel</button>
                        <button
                          onClick={() => saveEditing(ns.id)}
                          disabled={isSaving}
                          className="text-xs px-3 py-1.5 bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-50 transition-colors"
                        >
                          {isSaving ? 'Saving…' : 'Save'}
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Read-only view */
                  <div className="ml-0.5 space-y-3">
                    <div className="text-xs text-zinc-500">
                      {ns.description || <span className="italic text-zinc-600">No description — click Edit to add one</span>}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Create modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">New Namespace</h3>
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
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white font-mono placeholder-zinc-500 focus:outline-none focus:border-violet-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Description <span className="text-zinc-600">(optional)</span></label>
                <input
                  value={modalDesc}
                  onChange={e => setModalDesc(e.target.value)}
                  placeholder="What does this namespace handle?"
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-violet-500"
                />
              </div>
            </div>
            {modalError && <p className="text-xs text-red-400">{modalError}</p>}
            <p className="text-[11px] text-zinc-600">Lowercase letters, digits, hyphens, underscores · max 40 chars.</p>
            <div className="flex gap-2 justify-end pt-1">
              <button onClick={closeModal} className="px-4 py-2 text-sm text-zinc-400 hover:text-white">Cancel</button>
              <button
                onClick={submitModal}
                disabled={modalBusy || !modalId.trim()}
                className="px-5 py-2 text-sm bg-violet-600 text-white rounded-lg hover:bg-violet-500 disabled:opacity-40"
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

