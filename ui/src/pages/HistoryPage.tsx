import { useState, useEffect } from 'react';
import { api } from '@/api/client';
import { useAppStore } from '@/store';
import Page from '@/components/Page';

// ── Types ─────────────────────────────────────────────────────────────────────

interface CommitInfo {
  sha: string;
  ts: number;
  message: string;
  author: string;
}

interface PhraseChange {
  intent_id: string;
  lang: string;
  phrase: string;
}

interface MetadataChange {
  intent_id: string;
  field: string;
  from: string;
  to: string;
}

interface IntentWithPhrases {
  id: string;
  phrases_sample: string[];
  total_phrases: number;
}

interface DiffResult {
  from: string;
  to: string;
  intents_added: IntentWithPhrases[];
  intents_removed: IntentWithPhrases[];
  phrases_added: PhraseChange[];
  phrases_removed: PhraseChange[];
  metadata_changes: MetadataChange[];
  l2_edges_changed: number;
  l1_edges_changed: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function relativeTime(unixSeconds: number): string {
  const diff = Math.floor(Date.now() / 1000) - unixSeconds;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`;
  const d = new Date(unixSeconds * 1000);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function affectedIntents(diff: DiffResult): string[] {
  const ids = new Set<string>();
  diff.intents_added.forEach(i => { if (i.id !== '_ns') ids.add(i.id); });
  diff.intents_removed.forEach(i => { if (i.id !== '_ns') ids.add(i.id); });
  diff.phrases_added.forEach(p => { if (p.intent_id !== '_ns') ids.add(p.intent_id); });
  diff.phrases_removed.forEach(p => { if (p.intent_id !== '_ns') ids.add(p.intent_id); });
  diff.metadata_changes.forEach(m => { if (m.intent_id !== '_ns') ids.add(m.intent_id); });
  return Array.from(ids).sort();
}

function groupByIntent(phrases: PhraseChange[]): Map<string, PhraseChange[]> {
  const map = new Map<string, PhraseChange[]>();
  for (const p of phrases) {
    const list = map.get(p.intent_id) ?? [];
    list.push(p);
    map.set(p.intent_id, list);
  }
  return map;
}

// ── DiffPanel ─────────────────────────────────────────────────────────────────

function DiffPanel({ namespaceId, from, to }: { namespaceId: string; from: string; to: string }) {
  const [diff, setDiff] = useState<DiffResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.getNamespaceDiff(namespaceId, from, to)
      .then(d => { setDiff(d); setLoading(false); })
      .catch(e => { setError(e instanceof Error ? e.message : 'Unknown error'); setLoading(false); });
  }, [namespaceId, from, to]);

  if (loading) return <div className="text-xs text-zinc-500 py-4">Loading diff…</div>;
  if (error) return <div className="text-xs text-red-400 py-4">Could not load diff: {error}</div>;
  if (!diff) return null;

  const empty =
    diff.intents_added.length === 0 &&
    diff.intents_removed.length === 0 &&
    diff.phrases_added.length === 0 &&
    diff.phrases_removed.length === 0 &&
    diff.metadata_changes.length === 0 &&
    diff.l2_edges_changed === 0 &&
    diff.l1_edges_changed === 0;

  if (empty) {
    return <div className="text-xs text-zinc-500 py-4">No changes between these commits.</div>;
  }

  const touched = affectedIntents(diff);

  return (
    <div className="space-y-4 text-xs font-mono">
      {/* Affects-N summary */}
      <div className="text-zinc-400 font-sans">
        Affects {touched.length} intent{touched.length !== 1 ? 's' : ''}
        {touched.length > 0 && (
          <span className="text-zinc-500">: {touched.join(', ')}</span>
        )}
      </div>

      {diff.intents_added.length > 0 && (
        <div>
          <div className="font-semibold text-[11px] text-zinc-400 font-sans mb-1">Intents added ({diff.intents_added.length})</div>
          <div className="space-y-1">
            {diff.intents_added.map(intent => (
              <div key={intent.id}>
                <div className="text-emerald-400">+ {intent.id}</div>
                {intent.phrases_sample.length > 0 && (
                  <div className="ml-4 text-zinc-500 font-mono text-[11px]">
                    {intent.phrases_sample.map((ph, i) => (
                      <span key={i}>"{ph}"{i < intent.phrases_sample.length - 1 ? ' · ' : ''}</span>
                    ))}
                    {intent.total_phrases > intent.phrases_sample.length && (
                      <span> · +{intent.total_phrases - intent.phrases_sample.length} more</span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {diff.intents_removed.length > 0 && (
        <div>
          <div className="font-semibold text-[11px] text-zinc-400 font-sans mb-1">Intents removed ({diff.intents_removed.length})</div>
          <div className="space-y-1">
            {diff.intents_removed.map(intent => (
              <div key={intent.id}>
                <div className="text-red-400">− {intent.id}</div>
                {intent.phrases_sample.length > 0 && (
                  <div className="ml-4 text-zinc-500 font-mono text-[11px]">
                    {intent.phrases_sample.map((ph, i) => (
                      <span key={i}>"{ph}"{i < intent.phrases_sample.length - 1 ? ' · ' : ''}</span>
                    ))}
                    {intent.total_phrases > intent.phrases_sample.length && (
                      <span> · +{intent.total_phrases - intent.phrases_sample.length} more</span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {diff.phrases_added.length > 0 && (() => {
        const groups = groupByIntent(diff.phrases_added);
        return (
          <div>
            <div className="font-semibold text-[11px] text-zinc-400 font-sans mb-1">Phrases added ({diff.phrases_added.length})</div>
            <div className="space-y-1">
              {Array.from(groups.entries()).map(([intentId, phrases]) => (
                <div key={intentId}>
                  <div className="text-emerald-400">+ {intentId} ({phrases.length} phrase{phrases.length !== 1 ? 's' : ''} added)</div>
                  <div className="ml-4 space-y-0.5">
                    {phrases.map((p, i) => (
                      <div key={i} className="text-emerald-400/70">
                        <span className="text-zinc-500">{p.lang}</span>{'  '}"{p.phrase}"
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })()}

      {diff.phrases_removed.length > 0 && (() => {
        const groups = groupByIntent(diff.phrases_removed);
        return (
          <div>
            <div className="font-semibold text-[11px] text-zinc-400 font-sans mb-1">Phrases removed ({diff.phrases_removed.length})</div>
            <div className="space-y-1">
              {Array.from(groups.entries()).map(([intentId, phrases]) => (
                <div key={intentId}>
                  <div className="text-red-400">− {intentId} ({phrases.length} phrase{phrases.length !== 1 ? 's' : ''} removed)</div>
                  <div className="ml-4 space-y-0.5">
                    {phrases.map((p, i) => (
                      <div key={i} className="text-red-400/70">
                        <span className="text-zinc-500">{p.lang}</span>{'  '}"{p.phrase}"
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })()}

      {diff.metadata_changes.length > 0 && (
        <div>
          <div className="font-semibold text-[11px] text-zinc-400 font-sans mb-1">Metadata changed ({diff.metadata_changes.length})</div>
          <div className="space-y-2">
            {diff.metadata_changes.map((m, i) => (
              <div key={i}>
                <div className="text-zinc-400 font-sans mb-1">{m.intent_id}.{m.field}</div>
                <div className="flex gap-2">
                  <div className="flex-1 border border-zinc-700 rounded p-2 bg-zinc-900/60">
                    <div className="text-[10px] text-zinc-500 font-sans mb-1">Before</div>
                    <div className="text-red-300 break-words">"{m.from}"</div>
                  </div>
                  <div className="flex-1 border border-zinc-700 rounded p-2 bg-zinc-900/60">
                    <div className="text-[10px] text-zinc-500 font-sans mb-1">After</div>
                    <div className="text-emerald-300 break-words">"{m.to}"</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(diff.l2_edges_changed > 0 || diff.l1_edges_changed > 0) && (
        <div className="text-zinc-500 font-sans text-[11px]">
          {diff.l2_edges_changed > 0 && `${diff.l2_edges_changed} routing weight${diff.l2_edges_changed !== 1 ? 's' : ''} updated by training.`}
          {diff.l1_edges_changed > 0 && ` ${diff.l1_edges_changed} morphology edge${diff.l1_edges_changed !== 1 ? 's' : ''} changed.`}
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function HistoryPage() {
  const { settings, setSelectedNamespaceId } = useAppStore();
  const namespaceId = settings.selectedNamespaceId || 'default';

  const [commits, setCommits] = useState<CommitInfo[]>([]);
  const [namespaces, setNamespaces] = useState<string[]>(['default']);
  const [loading, setLoading] = useState(true);
  const [selectedSha, setSelectedSha] = useState<string | null>(null);
  const [confirmSha, setConfirmSha] = useState<string | null>(null);
  const [rolling, setRolling] = useState(false);
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);

  const loadHistory = (nsId: string) => {
    setLoading(true);
    api.getNamespaceHistory(nsId, 20)
      .then(r => {
        const list: CommitInfo[] = r.commits ?? [];
        setCommits(list);
        setSelectedSha(prev => {
          if (prev && list.find(c => c.sha === prev)) return prev;
          return list.length > 0 ? list[0].sha : null;
        });
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  useEffect(() => {
    api.listNamespaces().then(ns => setNamespaces(ns.map((n: { id: string }) => n.id))).catch(() => {});
  }, []);

  useEffect(() => {
    loadHistory(namespaceId);
  }, [namespaceId]);

  const showToast = (msg: string, ok: boolean) => {
    setToast({ msg, ok });
    setTimeout(() => setToast(null), 3500);
  };

  const doRollback = async (sha: string) => {
    setRolling(true);
    try {
      await api.rollbackNamespace(namespaceId, sha);
      setConfirmSha(null);
      showToast(`Rolled back to ${sha.slice(0, 7)}`, true);
      loadHistory(namespaceId);
    } catch (e) {
      showToast('Rollback failed: ' + (e instanceof Error ? e.message : 'unknown'), false);
    }
    setRolling(false);
  };

  const selectedIdx = selectedSha ? commits.findIndex(c => c.sha === selectedSha) : -1;
  const selectedCommit = selectedIdx >= 0 ? commits[selectedIdx] : null;
  // commits[0] is newest; commits[i-1] is more recent than commits[i]
  const newerCommit = selectedIdx > 0 ? commits[selectedIdx - 1] : null;
  const commitsAhead = selectedIdx; // number of commits newer than selected

  return (
    <Page title="History" subtitle="Every change is a git commit — roll back to any state." fullscreen>
      {toast && (
        <div className={`fixed bottom-4 right-4 z-50 px-4 py-2 rounded-lg text-sm shadow-lg ${toast.ok ? 'bg-emerald-700 text-white' : 'bg-red-700 text-white'}`}>
          {toast.msg}
        </div>
      )}

      <div className="flex gap-0 h-full">
        {/* LEFT: commit list */}
        <div className="w-[300px] min-w-[300px] border-r border-zinc-800 flex flex-col">
          {/* Sidebar header */}
          <div className="h-12 px-4 border-b border-zinc-800 flex items-center justify-between flex-shrink-0 gap-2">
            <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide whitespace-nowrap">
              Commits ({commits.length})
            </span>
            <div className="flex items-center gap-2 min-w-0">
              <select
                value={namespaceId}
                onChange={e => setSelectedNamespaceId(e.target.value)}
                className="text-xs bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-zinc-300 min-w-0 max-w-[120px] truncate"
              >
                {namespaces.map(ns => (
                  <option key={ns} value={ns}>{ns}</option>
                ))}
              </select>
              <button
                onClick={() => loadHistory(namespaceId)}
                className="text-zinc-500 hover:text-zinc-200 text-xs px-1"
                title="Refresh"
              >↺</button>
            </div>
          </div>

          {/* Commit list */}
          <div className="flex-1 overflow-y-auto">
            {loading ? (
              <div className="text-xs text-zinc-500 py-8 text-center">Loading…</div>
            ) : commits.length === 0 ? (
              <div className="text-xs text-zinc-500 py-8 text-center px-4 leading-relaxed">
                No history available. The data directory may not be a git repository.
              </div>
            ) : (
              commits.map(c => {
                const isActive = c.sha === selectedSha;
                return (
                  <button
                    key={c.sha}
                    onClick={() => setSelectedSha(c.sha)}
                    className={`w-full text-left px-4 py-3 border-b border-zinc-800/50 flex flex-col gap-0.5 transition-colors ${
                      isActive ? 'bg-zinc-800' : 'hover:bg-zinc-800/40'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] text-zinc-500">{relativeTime(c.ts)}</span>
                      <span className="font-mono text-[11px] text-zinc-600">{c.sha.slice(0, 7)}</span>
                    </div>
                    <div className="text-xs text-zinc-300 truncate">
                      {c.message.length > 42 ? c.message.slice(0, 42) + '…' : c.message}
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>

        {/* RIGHT: detail pane */}
        <div className="flex-1 flex flex-col overflow-y-auto">
          {!selectedCommit ? (
            <div className="flex-1 flex items-center justify-center text-xs text-zinc-500 px-8 text-center">
              Select a commit on the left to see what changed and roll back if needed.
            </div>
          ) : (
            <div className="p-6 space-y-5">
              {/* Commit header */}
              <div className="space-y-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-mono text-sm text-zinc-400">{selectedCommit.sha.slice(0, 7)}</span>
                  <span className="text-zinc-600 text-xs">·</span>
                  <span className="text-xs text-zinc-500">{relativeTime(selectedCommit.ts)}</span>
                  <span className="text-zinc-600 text-xs">·</span>
                  <span className="text-sm text-zinc-200">{selectedCommit.message}</span>
                </div>
                {selectedCommit.author && (
                  <div className="text-xs text-zinc-500">Author: {selectedCommit.author}</div>
                )}
                {commitsAhead > 0 && (
                  <div className="text-xs text-amber-400 mt-1">
                    ⚠ Rolling back undoes this commit and discards {commitsAhead} newer commit{commitsAhead !== 1 ? 's' : ''}
                  </div>
                )}
              </div>

              {/* Diff section */}
              <div>
                <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wide mb-3">Changes</div>
                {newerCommit ? (
                  <DiffPanel namespaceId={namespaceId} from={selectedCommit.sha} to={newerCommit.sha} />
                ) : (
                  <div className="text-xs text-zinc-500">This is the latest commit; nothing newer to compare against.</div>
                )}
              </div>

              {/* Rollback action */}
              <div className="pt-2 border-t border-zinc-800">
                <button
                  onClick={() => setConfirmSha(selectedCommit.sha)}
                  className="text-sm px-4 py-2 border border-zinc-700 text-zinc-400 rounded hover:border-red-500 hover:text-red-400 transition-colors"
                >
                  Rollback to this commit
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Confirm modal */}
      {confirmSha && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <h3 className="text-base font-semibold text-zinc-100">
              Roll back to {confirmSha.slice(0, 7)}?
            </h3>
            <p className="text-sm text-zinc-400">
              This resets the entire data directory to that commit. All namespaces will reload from the
              post-rollback state — changes after this commit will be discarded.
            </p>
            <div className="flex gap-2 justify-end pt-1">
              <button
                onClick={() => setConfirmSha(null)}
                disabled={rolling}
                className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-100"
              >
                Cancel
              </button>
              <button
                onClick={() => doRollback(confirmSha)}
                disabled={rolling}
                className="px-5 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-500 disabled:opacity-50"
              >
                {rolling ? 'Rolling back…' : 'Roll back'}
              </button>
            </div>
          </div>
        </div>
      )}
    </Page>
  );
}
