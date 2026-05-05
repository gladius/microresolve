import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFetch } from '@/hooks/useFetch';
import { api, type IntentInfo, type NamespaceModel } from '@/api/client';
import { useAppStore } from '@/store';
import Page from '@/components/Page';

const UNCATEGORIZED = '(uncategorized)';

export default function IntentsPage() {
  const [intents, setIntents] = useState<IntentInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [filter, setFilter] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const navigate = useNavigate();

  const refresh = useCallback(async () => {
    try {
      const data = await api.listIntents();
      setIntents(data);
      if (selectedId && !data.find(i => i.id === selectedId)) {
        setSelectedId(data.length > 0 ? data[0].id : null);
      }
    } catch { /* server not running */ }
  }, [selectedId]);

  useFetch(refresh, [refresh]);

  const filteredIntents = useMemo(() => intents.filter(i => {
    if (filter && !i.id.toLowerCase().includes(filter.toLowerCase())) return false;
    return true;
  }), [intents, filter]);

  const grouped = useMemo(() => {
    const byDomain = new Map<string, IntentInfo[]>();
    for (const i of filteredIntents) {
      const colon = i.id.indexOf(':');
      const domain = colon > 0 ? i.id.slice(0, colon) : UNCATEGORIZED;
      if (!byDomain.has(domain)) byDomain.set(domain, []);
      byDomain.get(domain)!.push(i);
    }
    return Array.from(byDomain.entries()).sort(([a], [b]) => {
      if (a === UNCATEGORIZED) return 1;
      if (b === UNCATEGORIZED) return -1;
      return a.localeCompare(b);
    });
  }, [filteredIntents]);

  const toggleDomain = useCallback((d: string) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      if (next.has(d)) next.delete(d); else next.add(d);
      return next;
    });
  }, []);

  const selected = intents.find(i => i.id === selectedId) || null;
  const allIntentIds = useMemo(() => intents.map(i => i.id), [intents]);

  return (
    <Page title="Intents" subtitle={`${intents.length} intents`} fullscreen>
    <div className="flex gap-0 h-full">
      {/* Left: Intent list */}
      <div className="w-72 min-w-[18rem] border-r border-zinc-800 flex flex-col">
        <div className="h-12 px-4 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
          <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">
            Intents ({filteredIntents.length}{filteredIntents.length !== intents.length ? `/${intents.length}` : ''})
          </span>
          <div className="flex gap-1.5 items-center">
            <button
              onClick={() => {
                setShowSearch(v => !v);
                if (showSearch) setFilter('');
              }}
              className={`w-6 h-6 flex items-center justify-center rounded transition-colors ${showSearch || filter ? 'text-emerald-400 bg-emerald-500/10' : 'text-zinc-500 hover:text-white'}`}
              title="Search intents"
              aria-label="Search intents"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="7" />
                <path d="m20 20-3.5-3.5" />
              </svg>
            </button>
            <button
              onClick={() => setShowAdd(true)}
              className="text-[11px] bg-emerald-600 hover:bg-emerald-500 text-white px-2 py-0.5 rounded transition-colors"
            >
              + New
            </button>
          </div>
        </div>

        {showSearch && (
          <div className="px-3 py-2 border-b border-zinc-800 flex-shrink-0">
            <input
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder="Search intents..."
              autoFocus
              className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none"
            />
          </div>
        )}

        <div className="flex-1 overflow-y-auto">
          {grouped.map(([domain, items]) => {
            const isCollapsed = collapsed.has(domain);
            return (
              <div key={domain}>
                <button
                  onClick={() => toggleDomain(domain)}
                  className="w-full flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/40 border-b border-zinc-800/50 transition-colors"
                >
                  <span className="w-3 text-zinc-600">{isCollapsed ? '▸' : '▾'}</span>
                  <span className="flex-1 text-left">{domain}</span>
                  <span className="text-zinc-600">{items.length}</span>
                </button>
                {!isCollapsed && items.map(intent => (
                  <IntentListItem
                    key={intent.id}
                    intent={intent}
                    selected={selectedId === intent.id}
                    onClick={() => { setSelectedId(intent.id); setShowAdd(false); }}
                  />
                ))}
              </div>
            );
          })}
          {filteredIntents.length === 0 && (
            <div className="text-zinc-600 text-xs text-center py-8 px-4">
              {intents.length === 0 ? (
                <div className="space-y-1.5">
                  <div>No intents yet.</div>
                  <div>
                    <button onClick={() => setShowAdd(true)} className="text-emerald-400 hover:text-emerald-300">Create one</button>
                    <span className="mx-1.5 text-zinc-700">or</span>
                    <button onClick={() => navigate('/import')} className="text-emerald-400 hover:text-emerald-300">import</button>
                  </div>
                </div>
              ) : (
                <span>No intents match filter</span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right: Detail panel */}
      <div className="flex-1 overflow-y-auto">
        {showAdd ? (
          <div className="p-5">
            <AddIntentPanel
              onDone={(newId) => {
                setShowAdd(false);
                refresh().then(() => setSelectedId(newId));
              }}
              onCancel={() => setShowAdd(false)}
            />
          </div>
        ) : selected ? (
          <IntentDetailPanel
            intent={selected}
            allIntentIds={allIntentIds}
            onRefresh={refresh}
            onDeleted={() => {
              setSelectedId(null);
              refresh();
            }}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-zinc-600 text-sm">
            Select an intent or create a new one
          </div>
        )}
      </div>
    </div>
    </Page>
  );
}

// --- Left sidebar item ---

function IntentListItem({
  intent, selected, onClick,
}: {
  intent: IntentInfo; selected: boolean; onClick: () => void;
}) {
  const colon = intent.id.indexOf(':');
  const shortName = colon > 0 ? intent.id.slice(colon + 1) : intent.id;
  return (
    <div
      onClick={onClick}
      className={`pl-7 pr-3 py-2 cursor-pointer border-b border-zinc-800/50 transition-colors ${
        selected ? 'bg-zinc-800/80' : 'hover:bg-zinc-800/40'
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="text-emerald-400 font-mono text-sm font-semibold truncate flex-1" title={intent.id}>{shortName}</span>
        <span className="text-zinc-600 text-[11px]">{intent.phrases.length}</span>
        {intent.learned_count > 0 && (
          <span className="text-emerald-400/40 text-[10px]">+{intent.learned_count}</span>
        )}
        {intent.source && (
          <span className="text-[9px] text-amber-400/70 border border-amber-400/20 rounded px-1" title={`Source: ${intent.source.type}`}>
            {intent.source.type}
          </span>
        )}
      </div>
      {intent.description && intent.description !== intent.id && intent.description !== shortName && (
        <div className="text-[10px] text-zinc-500 mt-0.5 pl-6 truncate">{intent.description}</div>
      )}
    </div>
  );
}

// --- Right detail panel with tabs ---

type DetailTab = 'details' | 'phrases';

function IntentDetailPanel({
  intent, onRefresh, onDeleted,
}: {
  intent: IntentInfo; allIntentIds?: string[]; onRefresh: () => void; onDeleted: () => void;
}) {
  const [activeTab, setActiveTab] = useState<DetailTab>('details');
  const [phraseSearch, setPhraseSearch] = useState('');

  const handleDelete = async () => {
    if (!confirm(`Delete intent "${intent.id}"?`)) return;
    await api.deleteIntent(intent.id);
    onDeleted();
  };

  const langKeys = Object.keys(intent.phrases_by_lang).filter(k => k !== '_learned');

  const tabs: { id: DetailTab; label: string; count?: number }[] = [
    { id: 'details', label: 'Details' },
    { id: 'phrases', label: 'Learnt phrases', count: intent.phrases.length },
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-5 pt-5 pb-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-semibold text-emerald-400 font-mono">{intent.id}</h2>
            {intent.source && (
              <span className="text-[10px] font-semibold text-amber-400 bg-amber-400/10 border border-amber-400/30 rounded px-1.5 py-0.5 uppercase">
                {intent.source.label || intent.source.type}
              </span>
            )}
            {langKeys.length > 1 && (
              <div className="flex gap-1">
                {langKeys.map(lang => (
                  <span key={lang} className="text-[10px] font-semibold text-emerald-400 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 uppercase">
                    {lang}
                  </span>
                ))}
              </div>
            )}
          </div>
          <button onClick={handleDelete} className="text-xs text-red-400 hover:text-red-300 px-2 py-1 border border-red-400/20 rounded hover:border-red-400/50 transition-colors">
            Delete
          </button>
        </div>
        {intent.description && (
          <div className="text-xs text-zinc-500 mb-3">{intent.description}</div>
        )}

        {/* Tabs + search (search in tab bar area) */}
        <div className="flex items-center border-b border-zinc-800">
          <div className="flex gap-0">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'text-zinc-100 border-emerald-500'
                    : 'text-zinc-500 border-transparent hover:text-zinc-300'
                }`}
              >
                {tab.label}
                {tab.count !== undefined && (
                  <span className="ml-1.5 text-xs text-zinc-600">{tab.count}</span>
                )}
              </button>
            ))}
          </div>
          {activeTab === 'phrases' && (
            <div className="ml-auto relative">
              <input
                value={phraseSearch}
                onChange={e => setPhraseSearch(e.target.value)}
                placeholder="Search..."
                autoComplete="off"
                className="w-40 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 focus:border-emerald-500 focus:outline-none"
              />
              {phraseSearch && (
                <button onClick={() => setPhraseSearch('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-100 text-xs">×</button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto px-5 py-4">
        {activeTab === 'phrases' && (
          <PhrasesTab intent={intent} onRefresh={onRefresh} phraseSearch={phraseSearch} />
        )}
        {activeTab === 'details' && (
          <DetailsTab intent={intent} onRefresh={onRefresh} />
        )}
      </div>
    </div>
  );
}

// --- Phrases Tab ---

function PhrasesTab({ intent, onRefresh, phraseSearch }: { intent: IntentInfo; onRefresh: () => void; phraseSearch: string }) {
  const [newPhrase, setNewPhrase] = useState('');
  const [showBulk, setShowBulk] = useState(false);
  const [bulkText, setBulkText] = useState('');
  const [showAI, setShowAI] = useState(true);
  const [aiDescription, setAIDescription] = useState(intent.description || '');
  const [languages, setLanguages] = useState<Record<string, string>>({});
  const [generating, setGenerating] = useState(false);
  const [genStatus, setGenStatus] = useState('');
  const { settings } = useAppStore();
  const enabledLangs = new Set(settings.languages);
  const [aiLangs, setAILangs] = useState<Set<string>>(new Set(settings.languages));

  useEffect(() => {
    api.getLanguages().then(setLanguages).catch(() => {});
  }, []);

  const handleRemovePhrase = async (phrase: string) => {
    await api.removePhrase(intent.id, phrase);
    onRefresh();
  };

  const langKeys = Object.keys(intent.phrases_by_lang).filter(k => k !== '_learned');

  // Build flat list with language tags
  const allPhrases = useMemo(() => {
    const result: { lang: string; phrase: string }[] = [];
    for (const lang of langKeys) {
      for (const phrase of intent.phrases_by_lang[lang] || []) {
        result.push({ lang, phrase });
      }
    }
    return result;
  }, [intent.phrases_by_lang, langKeys]);

  const filtered = useMemo(() => {
    if (!phraseSearch.trim()) return allPhrases;
    const q = phraseSearch.toLowerCase();
    return allPhrases.filter(s => s.phrase.toLowerCase().includes(q));
  }, [allPhrases, phraseSearch]);

  const [phraseWarning, setPhraseWarning] = useState('');

  const handleAddPhrase = async () => {
    if (!newPhrase.trim()) return;
    setPhraseWarning('');
    const result = await api.addPhrase(intent.id, newPhrase.trim());
    if (result.added) {
      setNewPhrase('');
      onRefresh();
    } else if (result.reason) {
      setPhraseWarning(result.reason);
    } else if (result.redundant) {
      setPhraseWarning('All terms already covered by existing phrases');
    }
  };

  const handleBulkAdd = async () => {
    const lines = bulkText.split('\n').map(s => s.trim()).filter(Boolean);
    if (lines.length === 0) return;
    const warnings: string[] = [];
    for (const line of lines) {
      const result = await api.addPhrase(intent.id, line);
      if (!result.added && result.reason) {
        warnings.push(`"${line}": ${result.reason}`);
      }
    }
    if (warnings.length > 0) {
      setPhraseWarning(`${lines.length - warnings.length} added, ${warnings.length} blocked:\n${warnings.join('\n')}`);
    }
    setBulkText('');
    setShowBulk(false);
    onRefresh();
  };

  const [collapsedLangs, setCollapsedLangs] = useState<Set<string>>(new Set());
  const toggleLang = (lang: string) => setCollapsedLangs(prev => {
    const next = new Set(prev);
    next.has(lang) ? next.delete(lang) : next.add(lang);
    return next;
  });

  // When searching, group filtered results by lang for display
  const groupedFiltered = useMemo(() => {
    const groups: Record<string, string[]> = {};
    for (const { lang, phrase } of filtered) {
      (groups[lang] ??= []).push(phrase);
    }
    return groups;
  }, [filtered]);

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Inline stats — total · langs · learned (replaces the dropped Stats tab) */}
      <div className="text-xs text-zinc-500 font-mono">
        <span className="text-zinc-300">{intent.phrases.length}</span> example{intent.phrases.length === 1 ? '' : 's'}
        {langKeys.length > 0 && (
          <>
            <span className="text-zinc-700 mx-2">·</span>
            <span className="text-zinc-300">{langKeys.length}</span> lang{langKeys.length === 1 ? '' : 's'}
            <span className="text-zinc-600 ml-1">({langKeys.join(', ')})</span>
          </>
        )}
        {intent.learned_count > 0 && (
          <>
            <span className="text-zinc-700 mx-2">·</span>
            <span className="text-emerald-400">{intent.learned_count}</span> learned from corrections
          </>
        )}
      </div>

      {/* Phrase list */}
      <div className="flex-1 border border-zinc-800 rounded-lg bg-zinc-900/50 overflow-y-auto">
        {filtered.length === 0 && (
          <div className="text-zinc-600 text-xs text-center py-6">
            {phraseSearch ? 'No phrases match search' : 'No phrases yet'}
          </div>
        )}
        {Object.entries(groupedFiltered).map(([lang, phrases]) => {
          const isCollapsed = collapsedLangs.has(lang);
          return (
            <div key={lang}>
              {/* Language section header */}
              <button
                onClick={() => toggleLang(lang)}
                className="w-full flex items-center gap-2 px-3 py-2 bg-zinc-800/60 hover:bg-zinc-800 transition-colors text-left"
              >
                <svg className={`w-3 h-3 text-zinc-500 transition-transform ${isCollapsed ? '-rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
                <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wide">{lang}</span>
                <span className="text-[10px] text-zinc-600">{phrases.length} phrase{phrases.length !== 1 ? 's' : ''}</span>
              </button>
              {!isCollapsed && (
                <div className="divide-y divide-zinc-800/50">
                  {phrases.map((phrase, i) => (
                    <div key={i} className="flex items-center gap-2 px-3 py-1.5 hover:bg-zinc-800/50 transition-colors group">
                      <span className="text-sm text-zinc-300 flex-1 truncate">{phrase}</span>
                      <button
                        onClick={() => handleRemovePhrase(phrase)}
                        className="text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all text-xs flex-shrink-0"
                        title="Remove phrase"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Add area: single input OR bulk paste (toggle) */}
      <div className="flex-shrink-0 space-y-2">
        {showBulk ? (
          <>
            <textarea
              value={bulkText}
              onChange={e => setBulkText(e.target.value)}
              placeholder="Paste multiple phrases, one per line..."
              rows={4}
              autoFocus
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 font-mono resize-y focus:border-emerald-500 focus:outline-none"
            />
            <div className="flex items-center gap-2">
              <button
                onClick={handleBulkAdd}
                disabled={!bulkText.trim()}
                className="text-xs px-3 py-1.5 bg-emerald-600 text-white rounded hover:bg-emerald-500 disabled:opacity-30"
              >
                Add {bulkText.split('\n').filter(s => s.trim()).length} phrases
              </button>
              <button onClick={() => setShowBulk(false)} className="text-xs text-zinc-500 hover:text-zinc-100">Cancel</button>
            </div>
          </>
        ) : (
          <div className="flex gap-2">
            <input
              value={newPhrase}
              onChange={e => setNewPhrase(e.target.value)}
              placeholder="Type a phrase and press Enter..."
              autoComplete="off"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-zinc-100 font-mono focus:border-emerald-500 focus:outline-none"
              onKeyDown={e => { if (e.key === 'Enter') handleAddPhrase(); }}
            />
            <button
              onClick={handleAddPhrase}
              disabled={!newPhrase.trim()}
              className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 text-emerald-400 rounded hover:bg-zinc-700 disabled:opacity-30 transition-colors"
            >
              + Add
            </button>
            <button
              onClick={() => setShowBulk(true)}
              className="px-2 py-1.5 text-xs text-zinc-500 hover:text-zinc-100 border border-zinc-700 rounded transition-colors"
              title="Bulk paste"
            >
              Bulk
            </button>
          </div>
        )}
        {/* Guard warning */}
        {phraseWarning && (
          <div className="bg-amber-900/20 border border-amber-800/50 rounded px-3 py-2 text-xs text-amber-400 flex items-start gap-2">
            <span className="shrink-0">⚠</span>
            <span className="whitespace-pre-wrap">{phraseWarning}</span>
            <button onClick={() => setPhraseWarning('')} className="shrink-0 text-zinc-500 hover:text-zinc-100 ml-auto">×</button>
          </div>
        )}
        {/* AI Generate — prominent panel, defaults to expanded */}
        <div className="border border-emerald-500/30 bg-emerald-500/5 rounded-lg p-3">
          <button
            onClick={() => setShowAI(!showAI)}
            className="w-full flex items-center justify-between text-sm font-semibold text-emerald-400 hover:text-emerald-300"
          >
            <span className="flex items-center gap-2">
              <span>✨</span>
              <span>Generate examples with AI</span>
            </span>
            <svg className={`w-4 h-4 transition-transform ${showAI ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>

          {showAI && (
            <div className="mt-3 space-y-2">
              <textarea
                value={aiDescription}
                onChange={e => setAIDescription(e.target.value)}
                placeholder={"AI guidance — what should the LLM know? Examples, tone, edge cases all welcome.\n\ne.g.\n  Customer wants to cancel an active subscription.\n  Sample phrases: \"cancel my plan\", \"stop billing\"\n  Be empathetic — these users are usually frustrated."}
                rows={2}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-zinc-100 text-sm resize-y focus:border-emerald-500 focus:outline-none"
              />
              <div className="flex flex-wrap gap-2">
                {Object.entries(languages).filter(([code]) => enabledLangs.has(code)).map(([code, name]) => (
                  <label key={code} className="inline-flex items-center gap-1 text-xs text-zinc-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={aiLangs.has(code)}
                      onChange={() => {
                        setAILangs(prev => {
                          const next = new Set(prev);
                          next.has(code) ? next.delete(code) : next.add(code);
                          return next;
                        });
                      }}
                      className="accent-emerald-500"
                    />
                    {name}
                  </label>
                ))}
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={async () => {
                    if (!aiDescription.trim()) { setGenStatus('Enter a description'); return; }
                    setGenerating(true);
                    setGenStatus('Generating...');
                    try {
                      const langs = Array.from(aiLangs);
                      // Auto-anchor with the top 5 existing learnt phrases — gives the
                      // LLM context on the intent's actual phrasing without dumping all
                      // 100+ phrases (which would defeat the variation prompt).
                      const anchors = (intent.phrases || []).slice(0, 5);
                      const parsed = await api.generatePhrases(intent.id, aiDescription, langs, anchors);
                      // Add generated phrases through guard
                      let added = 0;
                      const blocked: string[] = [];
                      for (const lang of langs) {
                        for (const phrase of parsed.phrases_by_lang[lang] || []) {
                          const r = await api.addPhrase(intent.id, phrase, lang);
                          if (r.added) { added++; }
                          else if (r.reason) { blocked.push(`"${phrase}": ${r.reason}`); }
                        }
                      }
                      let msg = `Added ${added} phrases`;
                      if (blocked.length > 0) msg += `. ${blocked.length} blocked by guard.`;
                      setGenStatus(msg);
                      if (blocked.length > 0) setPhraseWarning(blocked.join('\n'));
                      onRefresh();
                    } catch (e) {
                      setGenStatus('Error: ' + (e as Error).message);
                    } finally {
                      setGenerating(false);
                    }
                  }}
                  disabled={generating}
                  className="text-xs px-3 py-1.5 border border-emerald-500 text-emerald-400 rounded hover:bg-emerald-500 hover:text-white disabled:opacity-50"
                >
                  {generating ? 'Generating...' : 'Generate'}
                </button>
                {genStatus && <span className="text-xs text-zinc-500">{genStatus}</span>}
              </div>
            </div>
          )}
        </div>

        {intent.learned_count > 0 && (
          <p className="text-xs text-emerald-400/50">+{intent.learned_count} terms learned from corrections</p>
        )}
      </div>
    </div>
  );
}

// --- Details Tab (source, target, schema, guardrails, instructions, persona) ---


function DetailsTab({
  intent, onRefresh,
}: {
  intent: IntentInfo; onRefresh: () => void;
}) {
  const [description, setDescription] = useState<string>(intent.description || '');
  const [instructions, setInstructions] = useState<string>(intent.instructions || '');
  const [guardrails, setGuardrails] = useState<string[]>(intent.guardrails || []);
  const [persona, setPersona] = useState<string>(intent.persona || '');
  const [model, setModel] = useState<string>(intent.target?.model || '');
  const [nsModels, setNsModels] = useState<NamespaceModel[]>([]);
  const [newGuardrail, setNewGuardrail] = useState('');
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api.getModels().then(setNsModels).catch(() => {});
  }, []);

  useEffect(() => {
    setDescription(intent.description || '');
    setInstructions(intent.instructions || '');
    setGuardrails(intent.guardrails || []);
    setPersona(intent.persona || '');
    setModel(intent.target?.model || '');
    setNewGuardrail('');
    setDirty(false);
  }, [intent.id, intent.description, intent.instructions, intent.guardrails, intent.persona, intent.target]);

  const addGuardrail = () => {
    const v = newGuardrail.trim();
    if (!v) return;
    setGuardrails([...guardrails, v]);
    setNewGuardrail('');
    setDirty(true);
  };
  const removeGuardrail = (i: number) => {
    setGuardrails(guardrails.filter((_, idx) => idx !== i));
    setDirty(true);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      // Single PATCH carrying every changed field — replaces the four
      // separate per-field POSTs we used to make.
      const fields: Parameters<typeof api.patchIntent>[1] = {
        description: description.trim(),
        instructions: instructions.trim(),
        persona: persona.trim(),
        guardrails: guardrails.filter(Boolean),
      };
      if (model !== (intent.target?.model || '')) {
        fields.target = { type: 'llm', model: model || undefined };
      }
      await api.patchIntent(intent.id, fields);
      setDirty(false);
      onRefresh();
    } finally {
      setSaving(false);
    }
  };

  // Determine if this is an imported tool (has source from import)
  const isImported = !!intent.source;
  const importedTarget = isImported ? intent.target : null;

  return (
    <div className="space-y-6">

      {/* Description — editable, primary metadata */}
      <div>
        <div className="flex items-baseline justify-between mb-1.5">
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Description</div>
          <span className="text-[10px] text-zinc-600">What this intent represents (used by LLM prompts + AI seed generation)</span>
        </div>
        <textarea
          value={description}
          onChange={e => { setDescription(e.target.value); setDirty(true); }}
          placeholder="e.g. Customer wants to cancel an active subscription"
          rows={2}
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-zinc-100 text-sm resize-y focus:border-emerald-500 focus:outline-none"
        />
      </div>

      {/* Source — read-only, import provenance */}
      {intent.source && (
        <div className="flex items-center gap-3 bg-zinc-900/60 border border-zinc-800 rounded px-3 py-2 text-xs">
          <span className="text-zinc-600">Imported from</span>
          <span className="text-amber-400 font-semibold">{intent.source.label || intent.source.type}</span>
          {importedTarget?.handler && (
            <>
              <span className="text-zinc-700">→</span>
              <span className="text-zinc-400 font-mono">{importedTarget.handler}</span>
            </>
          )}
          {importedTarget?.url && (
            <>
              <span className="text-zinc-700">→</span>
              <a href={importedTarget.url} target="_blank" rel="noreferrer" className="text-blue-400 hover:underline">{importedTarget.url}</a>
            </>
          )}
        </div>
      )}

      {/* Schema */}
      {intent.schema && (
        <div>
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide mb-1.5">Schema</div>
          <pre className="bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-xs text-zinc-300 overflow-auto max-h-48 font-mono">
            {JSON.stringify(intent.schema, null, 2)}
          </pre>
        </div>
      )}

      {/* Route to model */}
      <div>
        <div className="flex items-baseline justify-between mb-1.5">
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Route to Model</div>
          {nsModels.length === 0 ? (
            <a href="/models" className="text-[10px] text-emerald-400 hover:underline">
              Add models →
            </a>
          ) : (
            <span className="text-[10px] text-zinc-600">Which LLM handles this intent when it fires</span>
          )}
        </div>
        <select
          value={model}
          onChange={e => { setModel(e.target.value); setDirty(true); }}
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500 focus:outline-none"
        >
          <option value="">Default</option>
          {nsModels.map(m => (
            <option key={m.model_id} value={m.model_id}>{m.label} — {m.model_id}</option>
          ))}
        </select>
      </div>

      {/* Guardrails */}
      <div>
        <div className="flex items-baseline justify-between mb-1.5">
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Guardrails</div>
          <span className="text-[10px] text-zinc-600">Hard rules the LLM must not violate</span>
        </div>
        <div className="space-y-1.5">
          {guardrails.map((g, i) => (
            <div key={i} className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5">
              <span className="text-red-400/70 text-xs">⚠</span>
              <input
                value={g}
                onChange={e => { const n = [...guardrails]; n[i] = e.target.value; setGuardrails(n); setDirty(true); }}
                className="flex-1 bg-transparent text-sm text-zinc-200 focus:outline-none"
              />
              <button onClick={() => removeGuardrail(i)} className="text-zinc-600 hover:text-red-400 text-sm px-1">×</button>
            </div>
          ))}
          <div className="flex gap-2">
            <input
              value={newGuardrail}
              onChange={e => setNewGuardrail(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && addGuardrail()}
              placeholder="Add a guardrail..."
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-100 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none"
            />
            <button onClick={addGuardrail} disabled={!newGuardrail.trim()} className="px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded disabled:opacity-40">Add</button>
          </div>
        </div>
      </div>

      {/* Prompt */}
      <div>
        <div className="flex items-baseline justify-between mb-1.5">
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Prompt</div>
          <span className="text-[10px] text-zinc-600">Sent to the LLM when this intent fires</span>
        </div>
        <textarea
          value={instructions}
          onChange={e => { setInstructions(e.target.value); setDirty(true); }}
          rows={6}
          placeholder="You are helping the user cancel their subscription. Confirm their identity, check for an active plan, then call the cancel endpoint and reply with the refund timeline."
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none font-mono leading-relaxed"
        />
      </div>

      {/* Persona */}
      <div>
        <div className="flex items-baseline justify-between mb-1.5">
          <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Persona</div>
          <span className="text-[10px] text-zinc-600">Tone and voice for responses</span>
        </div>
        <input
          value={persona}
          onChange={e => { setPersona(e.target.value); setDirty(true); }}
          placeholder="e.g. professional but warm"
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none"
        />
      </div>

      {/* Save bar */}
      {dirty && (
        <div className="sticky bottom-0 -mx-5 px-5 py-3 bg-zinc-950/95 border-t border-zinc-800 flex items-center justify-between">
          <span className="text-xs text-amber-400">Unsaved changes</span>
          <button onClick={handleSave} disabled={saving} className="px-4 py-1.5 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded disabled:opacity-50">
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      )}
    </div>
  );
}

// --- Add Intent Panel (simplified two-step) ---

function AddIntentPanel({
  onDone, onCancel,
}: {
  onDone: (id: string) => void; onCancel: () => void;
}) {
  const { settings: appSettings } = useAppStore();
  const [id, setId] = useState('');
  const [phraseText, setPhraseText] = useState('');
  const [showAI, setShowAI] = useState(false);
  const [description, setDescription] = useState('');
  const [languages, setLanguages] = useState<Record<string, string>>({});
  const [selectedLangs, setSelectedLangs] = useState<Set<string>>(new Set(appSettings.languages));
  const [generating, setGenerating] = useState(false);
  const [genStatus, setGenStatus] = useState('');
  const enabledLangs = new Set(appSettings.languages);

  useEffect(() => {
    api.getLanguages().then(setLanguages).catch(() => {});
  }, []);

  const handleGenerate = async () => {
    if (!description.trim()) { setGenStatus('Enter a description first.'); return; }
    const langs = Array.from(selectedLangs);
    if (langs.length === 0) { setGenStatus('Select at least one language.'); return; }

    setGenerating(true);
    setGenStatus('Generating...');
    try {
      // Existing manual phrases (if any) become anchor examples for the LLM.
      const examples = phraseText.split('\n').map(s => s.trim()).filter(Boolean);
      const parsed = await api.generatePhrases(id || 'new_intent', description, langs, examples);
      const allPhrases: string[] = [];
      for (const lang of langs) {
        for (const phrase of parsed.phrases_by_lang[lang] || []) {
          allPhrases.push(phrase);
        }
      }
      const prev = phraseText.trim();
      setPhraseText(prev ? prev + '\n' + allPhrases.join('\n') : allPhrases.join('\n'));
      setGenStatus(`Generated ${parsed.total} phrases`);
    } catch (e) {
      setGenStatus('Error: ' + (e as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  const handleAdd = async () => {
    const intentId = id.trim();
    if (!intentId) return;
    const phrases = phraseText.split('\n').map(s => s.trim()).filter(Boolean);
    if (phrases.length === 0) return;
    const desc = description.trim();
    await api.addIntent(intentId, phrases, desc || undefined);
    onDone(intentId);
  };

  const phraseCount = phraseText.split('\n').filter(s => s.trim()).length;

  return (
    <div className="space-y-5 max-w-2xl">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-zinc-100">New Intent</h2>
        <button onClick={onCancel} className="text-sm text-zinc-500 hover:text-zinc-100">Cancel</button>
      </div>

      {/* Name */}
      <div>
        <label className="text-xs text-zinc-500 block mb-1">Intent ID</label>
        <input
          value={id}
          onChange={e => setId(e.target.value)}
          placeholder="e.g. cancel_order"
          autoComplete="off"
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-zinc-100 text-sm font-mono focus:border-emerald-500 focus:outline-none"
          autoFocus
        />
      </div>

      {/* Description — top-level, always visible. Used both as the saved
          description AND fed into the AI seed-gen prompt. */}
      <div>
        <label className="text-xs text-zinc-500 block mb-1">
          Description <span className="text-zinc-700">— what this intent represents</span>
        </label>
        <textarea
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder={"What is this intent? Examples, tone, edge cases all welcome — used by the LLM at runtime AND by the seed generator below.\n\ne.g.\n  Customer wants to cancel an active subscription.\n  Sample phrases: \"cancel my plan\", \"stop billing\"\n  Be empathetic — these users are usually frustrated."}
          rows={4}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-zinc-100 text-sm resize-y focus:border-emerald-500 focus:outline-none"
        />
      </div>

      {/* Example phrases */}
      <div>
        <label className="text-xs text-zinc-500 block mb-1">
          Example phrases {phraseCount > 0 && <span className="text-zinc-600">({phraseCount})</span>}
        </label>
        <textarea
          value={phraseText}
          onChange={e => setPhraseText(e.target.value)}
          placeholder={"Type 1–3 real user phrases — one per line. The AI uses these as anchors when generating more.\ne.g.\ncancel my order\nstop the order I placed"}
          rows={6}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 font-mono resize-y focus:border-emerald-500 focus:outline-none"
        />
      </div>

      {/* AI Generation (expandable) — uses the Description field above */}
      <div>
        <button
          onClick={() => setShowAI(!showAI)}
          className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
        >
          <svg className={`w-3 h-3 transition-transform ${showAI ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          Generate phrases with AI
        </button>

        {showAI && (
          <div className="mt-3 space-y-3 pl-4 border-l-2 border-emerald-500/20">
            <div className="text-[10px] text-zinc-500">
              Uses the Description field above as AI guidance. Add anchor phrases to <span className="text-zinc-400">Example phrases</span> to seed the variation pattern.
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(languages).filter(([code]) => enabledLangs.has(code)).map(([code, name]) => (
                <label key={code} className="inline-flex items-center gap-1 text-xs text-zinc-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedLangs.has(code)}
                    onChange={() => {
                      setSelectedLangs(prev => {
                        const next = new Set(prev);
                        next.has(code) ? next.delete(code) : next.add(code);
                        return next;
                      });
                    }}
                    className="accent-emerald-500"
                  />
                  {name}
                </label>
              ))}
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={handleGenerate}
                disabled={generating}
                className="text-xs px-3 py-1.5 border border-emerald-500 text-emerald-400 rounded hover:bg-emerald-500 hover:text-white disabled:opacity-50"
              >
                {generating ? 'Generating...' : 'Generate'}
              </button>
              {genStatus && <span className="text-xs text-zinc-500">{genStatus}</span>}
            </div>
          </div>
        )}
      </div>

      {/* Create */}
      <div className="flex justify-end pt-2 border-t border-zinc-800">
        <button
          onClick={handleAdd}
          disabled={!id.trim() || phraseCount === 0}
          className="px-5 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-colors font-medium disabled:opacity-30"
        >
          Create Intent
        </button>
      </div>
    </div>
  );
}

