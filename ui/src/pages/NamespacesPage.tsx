import { useState, useCallback } from 'react';
import { useFetch } from '@/hooks/useFetch';
import { useNavigate } from 'react-router-dom';
import { api, setApiNamespaceId } from '@/api/client';
import { useAppStore } from '@/store';

interface NamespaceInfo {
  id: string;
  description: string;
}

export default function NamespacesPage() {
  const { settings, setSelectedNamespaceId } = useAppStore();
  const navigate = useNavigate();
  const current = settings.selectedNamespaceId;

  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Inline description editing
  const [editingNs, setEditingNs] = useState<string | null>(null);
  const [editNsDesc, setEditNsDesc] = useState('');

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

  const switchTo = (id: string) => {
    setSelectedNamespaceId(id);
    setApiNamespaceId(id);
    window.location.href = '/intents';
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

  const saveNsDesc = async (id: string) => {
    try {
      await api.updateNamespace(id, editNsDesc);
      setEditingNs(null);
      refresh();
    } catch { /* */ }
  };

  const openModal = () => {
    setShowModal(true);
    setModalId('');
    setModalDesc('');
    setModalError('');
  };

  const closeModal = () => {
    setShowModal(false);
    setModalId('');
    setModalDesc('');
    setModalError('');
  };

  const submitModal = async () => {
    const id = modalId.trim();
    if (!id) return;
    if (!/^[a-z0-9_-]+$/.test(id)) {
      setModalError('Lowercase letters, numbers, hyphens and underscores only.');
      return;
    }
    setModalBusy(true);
    setModalError('');
    try {
      await api.createNamespace(id, modalDesc.trim());
      closeModal();
      refresh();
    } catch (e) {
      setModalError('Failed: ' + (e instanceof Error ? e.message : 'unknown'));
    }
    setModalBusy(false);
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Namespaces</h2>
          <p className="text-xs text-zinc-500 mt-0.5">Isolated routing workspaces. Click a namespace to manage its domains.</p>
        </div>
        <button
          onClick={openModal}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-violet-600 text-white text-sm rounded hover:bg-violet-500 transition-colors"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Namespace
        </button>
      </div>

      {loading ? (
        <div className="text-xs text-zinc-500 text-center py-12">Loading…</div>
      ) : (
        <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/50">
          {namespaces.map(ns => {
            const isActive = ns.id === current;
            return (
              <div key={ns.id} className={`px-4 py-3 ${isActive ? 'bg-violet-500/5' : ''}`}>
                <div className="flex items-start gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <button
                        onClick={() => navigate(`/namespaces/${ns.id}`)}
                        className="text-sm font-mono text-white hover:text-violet-400 transition-colors"
                      >
                        {ns.id}
                      </button>
                      {isActive && (
                        <span className="text-[9px] text-violet-400 bg-violet-500/20 px-1.5 py-0.5 rounded uppercase tracking-wide">active</span>
                      )}
                    </div>

                    {editingNs === ns.id ? (
                      <div className="flex items-center gap-2 mt-1">
                        <input
                          autoFocus
                          value={editNsDesc}
                          onChange={e => setEditNsDesc(e.target.value)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') saveNsDesc(ns.id);
                            if (e.key === 'Escape') setEditingNs(null);
                          }}
                          className="flex-1 bg-zinc-900 border border-violet-500/50 rounded px-2 py-1 text-xs text-white focus:outline-none"
                        />
                        <button onClick={() => saveNsDesc(ns.id)} className="text-xs text-violet-400 hover:text-violet-300">Save</button>
                        <button onClick={() => setEditingNs(null)} className="text-xs text-zinc-500 hover:text-zinc-300">Cancel</button>
                      </div>
                    ) : (
                      <button
                        onClick={() => { setEditingNs(ns.id); setEditNsDesc(ns.description); }}
                        className="text-xs text-zinc-500 hover:text-zinc-300 text-left block"
                      >
                        {ns.description || <span className="italic text-zinc-600">Add description…</span>}
                      </button>
                    )}
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {!isActive && (
                      <button
                        onClick={() => switchTo(ns.id)}
                        className="text-xs px-3 py-1.5 border border-zinc-700 text-zinc-400 rounded hover:text-white hover:border-violet-500 transition-colors"
                      >
                        Switch
                      </button>
                    )}
                    {ns.id !== 'default' && (
                      <button
                        onClick={() => deleteNs(ns.id)}
                        className="text-xs px-2 py-1.5 text-zinc-600 hover:text-red-400 transition-colors"
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

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">New Namespace</h3>
              <button onClick={closeModal} className="text-zinc-500 hover:text-zinc-300 text-xl leading-none">×</button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Namespace ID</label>
                <input
                  autoFocus
                  value={modalId}
                  onChange={e => { setModalId(e.target.value); setModalError(''); }}
                  onKeyDown={e => e.key === 'Enter' && submitModal()}
                  placeholder="my-namespace"
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white font-mono placeholder-zinc-500 focus:outline-none focus:border-violet-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Description <span className="text-zinc-600">(optional)</span></label>
                <input
                  value={modalDesc}
                  onChange={e => setModalDesc(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && submitModal()}
                  placeholder="What is this namespace for?"
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-violet-500"
                />
              </div>
            </div>
            {modalError && <p className="text-xs text-red-400">{modalError}</p>}
            <p className="text-[11px] text-zinc-600">Lowercase letters, numbers, hyphens and underscores only.</p>
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
    </div>
  );
}
