import { useState, useCallback } from 'react';
import { useFetch } from '@/hooks/useFetch';
import { useNavigate, useParams } from 'react-router-dom';
import { api, setApiNamespaceId } from '@/api/client';
import { useAppStore } from '@/store';
import Page from '@/components/Page';

interface DomainInfo {
  name: string;
  description: string;
  intent_count: number;
}

export default function NamespacePage() {
  const { nsId } = useParams<{ nsId: string }>();
  const { settings, setSelectedNamespaceId, setSelectedDomain } = useAppStore();
  const navigate = useNavigate();
  const namespaceId = nsId ?? 'default';
  const isActive = settings.selectedNamespaceId === namespaceId;

  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const [editingDomain, setEditingDomain] = useState<string | null>(null);
  const [editDesc, setEditDesc] = useState('');

  const [showModal, setShowModal] = useState(false);
  const [modalName, setModalName] = useState('');
  const [modalDesc, setModalDesc] = useState('');
  const [modalError, setModalError] = useState('');
  const [modalBusy, setModalBusy] = useState(false);

  const refresh = useCallback(async () => {
    try { setDomains(await api.listDomainsFor(namespaceId)); } catch { /* */ }
    setLoading(false);
  }, [namespaceId]);

  useFetch(refresh, [refresh]);

  const openModal = () => { setShowModal(true); setModalName(''); setModalDesc(''); setModalError(''); };
  const closeModal = () => { setShowModal(false); setModalName(''); setModalDesc(''); setModalError(''); };

  const submitModal = async () => {
    const name = modalName.trim();
    if (!name) return;
    if (name.length > 40 || !/^[a-z0-9_-]+$/.test(name)) {
      setModalError('Lowercase letters, digits, hyphens, underscores · max 40 chars.');
      return;
    }
    setModalBusy(true);
    setModalError('');
    try {
      await api.createDomainFor(namespaceId, name, modalDesc.trim());
      closeModal();
      refresh();
    } catch (e) {
      setModalError('Failed: ' + (e instanceof Error ? e.message : 'unknown'));
    }
    setModalBusy(false);
  };

  const saveDesc = async (domain: string) => {
    try {
      await api.updateDomainFor(namespaceId, domain, editDesc);
      setEditingDomain(null);
      refresh();
    } catch { /* */ }
  };

  const deleteDomain = async (domain: string) => {
    if (!confirm(`Delete domain "${domain}"? Intents with this prefix are unaffected.`)) return;
    try {
      await api.deleteDomainFor(namespaceId, domain);
      refresh();
    } catch (e) {
      alert('Delete failed: ' + (e instanceof Error ? e.message : 'unknown'));
    }
  };

  const viewIntents = (domain: string) => {
    setSelectedDomain(domain);
    if (!isActive) {
      setSelectedNamespaceId(namespaceId);
      setApiNamespaceId(namespaceId);
      window.location.href = '/l2';
    } else {
      navigate('/l2');
    }
  };

  return (
    <Page
      size="sm"
      className="space-y-6"
      title={`Namespace: ${namespaceId}`}
      subtitle={isActive ? 'active namespace' : undefined}
      actions={
        <>
          <button onClick={() => navigate('/namespaces')} className="text-xs text-zinc-500 hover:text-zinc-300">
            ← Namespaces
          </button>
          <button
            onClick={openModal}
            className="flex items-center gap-1.5 px-3 py-1 bg-emerald-600 text-white text-xs rounded hover:bg-emerald-500 transition-colors"
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New
          </button>
        </>
      }
    >
      <div className="text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-2">Domains</div>
      <p className="text-xs text-zinc-500 mb-4">
        Domains group intents via <span className="font-mono text-zinc-400">domain:intent_id</span> prefixes.
      </p>

      {loading ? (
        <div className="text-xs text-zinc-500 text-center py-12">Loading…</div>
      ) : domains.length === 0 ? (
        <div className="border border-zinc-800 rounded-lg p-8 text-center space-y-2">
          <p className="text-sm text-zinc-400">No domains yet</p>
          <p className="text-xs text-zinc-600">
            Create a domain or import intents using the{' '}
            <span className="font-mono text-zinc-500">domain:intent_id</span> format.
          </p>
        </div>
      ) : (
        <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/50">
          {domains.map(domain => (
            <div key={domain.name} className="px-4 py-3">
              <div className="flex items-start gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <button
                      onClick={() => viewIntents(domain.name)}
                      className="text-sm font-mono text-emerald-400 hover:text-emerald-300 hover:underline transition-colors"
                    >
                      {domain.name}
                    </button>
                    <span className="text-[10px] text-zinc-500 bg-zinc-800 px-1.5 py-0.5 rounded">
                      {domain.intent_count} intent{domain.intent_count !== 1 ? 's' : ''}
                    </span>
                  </div>

                  {editingDomain === domain.name ? (
                    <div className="flex items-center gap-2 mt-1">
                      <input
                        autoFocus
                        value={editDesc}
                        onChange={e => setEditDesc(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') saveDesc(domain.name);
                          if (e.key === 'Escape') setEditingDomain(null);
                        }}
                        placeholder="What does this domain handle?"
                        className="flex-1 bg-zinc-900 border border-emerald-500/50 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none"
                      />
                      <button onClick={() => saveDesc(domain.name)} className="text-xs text-emerald-400 hover:text-emerald-300">Save</button>
                      <button onClick={() => setEditingDomain(null)} className="text-xs text-zinc-500 hover:text-zinc-300">Cancel</button>
                    </div>
                  ) : (
                    <button
                      onClick={() => { setEditingDomain(domain.name); setEditDesc(domain.description); }}
                      className="text-xs text-zinc-500 hover:text-zinc-300 text-left block"
                    >
                      {domain.description || <span className="italic text-zinc-600">Add description…</span>}
                    </button>
                  )}
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => {
                      setSelectedDomain(domain.name);
                      if (!isActive) { setSelectedNamespaceId(namespaceId); setApiNamespaceId(namespaceId); }
                      navigate('/import');
                    }}
                    className="text-xs px-3 py-1.5 border border-zinc-700 text-zinc-400 rounded hover:text-zinc-100 hover:border-zinc-500 transition-colors"
                  >
                    Import
                  </button>
                  <button
                    onClick={() => deleteDomain(domain.name)}
                    className="text-xs px-2 py-1.5 text-zinc-600 hover:text-red-400 transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-zinc-100">
                New Domain <span className="text-zinc-500 font-normal">in {namespaceId}</span>
              </h3>
              <button onClick={closeModal} className="text-zinc-500 hover:text-zinc-300 text-xl leading-none">×</button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Domain name</label>
                <input
                  autoFocus
                  value={modalName}
                  onChange={e => { setModalName(e.target.value); setModalError(''); }}
                  onKeyDown={e => e.key === 'Enter' && submitModal()}
                  placeholder="billing"
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-zinc-100 font-mono placeholder-zinc-500 focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Description <span className="text-zinc-600">(optional)</span></label>
                <input
                  value={modalDesc}
                  onChange={e => setModalDesc(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && submitModal()}
                  placeholder="What does this domain cover?"
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>
            {modalError && <p className="text-xs text-red-400">{modalError}</p>}
            <p className="text-[11px] text-zinc-600">
              Intents in this domain use <span className="font-mono">{modalName || 'name'}:intent_id</span> format.
            </p>
            <div className="flex gap-2 justify-end pt-1">
              <button onClick={closeModal} className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-100">Cancel</button>
              <button
                onClick={submitModal}
                disabled={modalBusy || !modalName.trim()}
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
