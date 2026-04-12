import { useState, useEffect, useCallback, useMemo } from 'react';
import { useFetch } from '@/hooks/useFetch';
import { api, type IntentInfo, type IntentType } from '@/api/client';
import { useAppStore } from '@/store';

export default function IntentsPage() {
  const [intents, setIntents] = useState<IntentInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [filter, setFilter] = useState('');
  const [nsFilter, setNsFilter] = useState('');

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

  const handleReset = async () => {
    await api.reset();
    await api.loadDefaults();
    const data = await api.listIntents();
    setIntents(data);
    setSelectedId(data.length > 0 ? data[0].id : null);
  };

  const namespaces = useMemo(() => {
    const seen = new Set<string>();
    for (const i of intents) {
      const colon = i.id.indexOf(':');
      if (colon > 0) seen.add(i.id.slice(0, colon));
    }
    return Array.from(seen).sort();
  }, [intents]);

  const filteredIntents = useMemo(() => intents.filter(i => {
    if (nsFilter && !i.id.startsWith(nsFilter + ':')) return false;
    if (filter && !i.id.toLowerCase().includes(filter.toLowerCase())) return false;
    return true;
  }), [intents, filter, nsFilter]);

  const selected = intents.find(i => i.id === selectedId) || null;
  const allIntentIds = useMemo(() => intents.map(i => i.id), [intents]);

  return (
    <div className="flex gap-0 h-[calc(100vh-6rem)] -mx-4">
      {/* Left: Intent list */}
      <div className="w-72 min-w-[18rem] border-r border-zinc-800 flex flex-col">
        <div className="px-3 py-3 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
          <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">
            Intents ({filteredIntents.length}{filteredIntents.length !== intents.length ? `/${intents.length}` : ''})
          </span>
          <div className="flex gap-1.5">
            <button
              onClick={handleReset}
              className="text-[11px] text-zinc-500 hover:text-white px-1.5 py-0.5 rounded transition-colors"
              title="Reset to demo defaults"
            >
              Reset
            </button>
            <button
              onClick={() => setShowAdd(true)}
              className="text-[11px] bg-violet-600 hover:bg-violet-500 text-white px-2 py-0.5 rounded transition-colors"
            >
              + New
            </button>
          </div>
        </div>

        <div className="px-3 py-2 border-b border-zinc-800 space-y-1.5 flex-shrink-0">
          <input
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Search intents..."
            className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-xs text-white placeholder-zinc-600 focus:border-violet-500 focus:outline-none"
          />
          {namespaces.length > 0 && (
            <div className="flex flex-wrap gap-1">
              <button
                onClick={() => setNsFilter('')}
                className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${nsFilter === '' ? 'bg-violet-500/20 border-violet-500/50 text-violet-300' : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'}`}
              >
                all
              </button>
              {namespaces.map(ns => (
                <button
                  key={ns}
                  onClick={() => setNsFilter(nsFilter === ns ? '' : ns)}
                  className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${nsFilter === ns ? 'bg-violet-500/20 border-violet-500/50 text-violet-300' : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'}`}
                >
                  {ns}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto">
          {filteredIntents.map(intent => (
            <IntentListItem
              key={intent.id}
              intent={intent}
              selected={selectedId === intent.id}
              onClick={() => { setSelectedId(intent.id); setShowAdd(false); }}
            />
          ))}
          {filteredIntents.length === 0 && (
            <div className="text-zinc-600 text-xs text-center py-8 px-4">
              {intents.length === 0 ? (
                <>No intents.{' '}<button onClick={handleReset} className="text-violet-400 hover:text-violet-300">Load defaults</button></>
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
  );
}

// --- Left sidebar item ---

function IntentListItem({
  intent, selected, onClick,
}: {
  intent: IntentInfo; selected: boolean; onClick: () => void;
}) {
  const typeChar = intent.intent_type === 'action' ? 'A' : 'C';
  const typeColor = intent.intent_type === 'action' ? 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30' : 'text-cyan-400 bg-cyan-400/10 border-cyan-400/30';
  return (
    <div
      onClick={onClick}
      className={`px-3 py-2 cursor-pointer border-b border-zinc-800/50 transition-colors ${
        selected ? 'bg-zinc-800/80' : 'hover:bg-zinc-800/40'
      }`}
    >
      <div className="flex items-center gap-2">
        <span className={`text-[9px] w-4 h-4 flex items-center justify-center rounded border font-bold ${typeColor}`}>
          {typeChar}
        </span>
        <span className="text-emerald-400 font-mono text-sm font-semibold truncate flex-1">{intent.id}</span>
        <span className="text-zinc-600 text-[11px]">{intent.phrases.length}</span>
        {intent.learned_count > 0 && (
          <span className="text-emerald-400/40 text-[10px]">+{intent.learned_count}</span>
        )}
      </div>
      {intent.description && (
        <div className="text-[10px] text-zinc-500 mt-0.5 pl-6 truncate">{intent.description}</div>
      )}
    </div>
  );
}

// --- Right detail panel with tabs ---

type DetailTab = 'phrases' | 'situations' | 'metadata' | 'stats';

function IntentDetailPanel({
  intent, allIntentIds, onRefresh, onDeleted,
}: {
  intent: IntentInfo; allIntentIds: string[]; onRefresh: () => void; onDeleted: () => void;
}) {
  const [activeTab, setActiveTab] = useState<DetailTab>('phrases');
  const [phraseSearch, setPhraseSearch] = useState('');

  const handleTypeChange = async (newType: IntentType) => {
    await api.setIntentType(intent.id, newType);
    onRefresh();
  };

  const handleDelete = async () => {
    if (!confirm(`Delete intent "${intent.id}"?`)) return;
    await api.deleteIntent(intent.id);
    onDeleted();
  };

  const langKeys = Object.keys(intent.phrases_by_lang).filter(k => k !== '_learned');
  const metaKeyCount = Object.keys(intent.metadata || {}).length;

  const situationCount = (intent.situation_patterns || []).length;

  const tabs: { id: DetailTab; label: string; count?: number }[] = [
    { id: 'phrases', label: 'Phrases', count: intent.phrases.length },
    { id: 'situations', label: 'Situations', count: situationCount || undefined },
    { id: 'metadata', label: 'Metadata', count: metaKeyCount },
    { id: 'stats', label: 'Stats' },
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-5 pt-5 pb-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-semibold text-emerald-400 font-mono">{intent.id}</h2>
            <div className="flex rounded overflow-hidden border border-zinc-700">
              {(['action', 'context'] as IntentType[]).map(t => (
                <button
                  key={t}
                  onClick={() => handleTypeChange(t)}
                  className={`text-[10px] px-2 py-1 font-semibold uppercase transition-colors ${
                    intent.intent_type === t
                      ? t === 'action' ? 'bg-emerald-400/20 text-emerald-400' : 'bg-cyan-400/20 text-cyan-400'
                      : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
            {langKeys.length > 1 && (
              <div className="flex gap-1">
                {langKeys.map(lang => (
                  <span key={lang} className="text-[10px] font-semibold text-violet-400 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 uppercase">
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
                    ? 'text-white border-violet-500'
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
                className="w-40 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-white focus:border-violet-500 focus:outline-none"
              />
              {phraseSearch && (
                <button onClick={() => setPhraseSearch('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-white text-xs">×</button>
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
        {activeTab === 'situations' && (
          <SituationsTab intent={intent} onRefresh={onRefresh} />
        )}
        {activeTab === 'metadata' && (
          <MetadataTab intent={intent} allIntentIds={allIntentIds} onRefresh={onRefresh} />
        )}
        {activeTab === 'stats' && (
          <StatsTab intent={intent} />
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
  const [showAI, setShowAI] = useState(false);
  const [aiDescription, setAIDescription] = useState('');
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
    } else if (result.conflicts?.length) {
      setPhraseWarning(result.conflicts.map(c => `"${c.term}" conflicts with ${c.competing_intent}`).join('; '));
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
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-white font-mono resize-y focus:border-violet-500 focus:outline-none"
            />
            <div className="flex items-center gap-2">
              <button
                onClick={handleBulkAdd}
                disabled={!bulkText.trim()}
                className="text-xs px-3 py-1.5 bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-30"
              >
                Add {bulkText.split('\n').filter(s => s.trim()).length} phrases
              </button>
              <button onClick={() => setShowBulk(false)} className="text-xs text-zinc-500 hover:text-white">Cancel</button>
            </div>
          </>
        ) : (
          <div className="flex gap-2">
            <input
              value={newPhrase}
              onChange={e => setNewPhrase(e.target.value)}
              placeholder="Type a phrase and press Enter..."
              autoComplete="off"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-white font-mono focus:border-violet-500 focus:outline-none"
              onKeyDown={e => { if (e.key === 'Enter') handleAddPhrase(); }}
            />
            <button
              onClick={handleAddPhrase}
              disabled={!newPhrase.trim()}
              className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 text-violet-400 rounded hover:bg-zinc-700 disabled:opacity-30 transition-colors"
            >
              + Add
            </button>
            <button
              onClick={() => setShowBulk(true)}
              className="px-2 py-1.5 text-xs text-zinc-500 hover:text-white border border-zinc-700 rounded transition-colors"
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
            <button onClick={() => setPhraseWarning('')} className="shrink-0 text-zinc-500 hover:text-white ml-auto">×</button>
          </div>
        )}
        {/* AI Generate */}
        <div>
          <button
            onClick={() => setShowAI(!showAI)}
            className="text-xs text-violet-400 hover:text-violet-300 flex items-center gap-1"
          >
            <svg className={`w-3 h-3 transition-transform ${showAI ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            Generate phrases with AI
          </button>

          {showAI && (
            <div className="mt-2 space-y-2 pl-4 border-l-2 border-violet-500/20">
              <textarea
                value={aiDescription}
                onChange={e => setAIDescription(e.target.value)}
                placeholder="Describe the intent: e.g. Customer wants to cancel their order"
                rows={2}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-white text-sm resize-y focus:border-violet-500 focus:outline-none"
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
                      className="accent-violet-500"
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
                      const parsed = await api.generatePhrases(intent.id, aiDescription, langs);
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
                  className="text-xs px-3 py-1.5 border border-violet-500 text-violet-400 rounded hover:bg-violet-500 hover:text-white disabled:opacity-50"
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

// --- Metadata Tab ---

function MetadataTab({
  intent, allIntentIds, onRefresh,
}: {
  intent: IntentInfo; allIntentIds: string[]; onRefresh: () => void;
}) {
  const [contextIntents, setContextIntents] = useState<string[]>(intent.metadata?.context_intents || []);
  const [actionIntents, setActionIntents] = useState<string[]>(intent.metadata?.action_intents || []);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    setContextIntents(intent.metadata?.context_intents || []);
    setActionIntents(intent.metadata?.action_intents || []);
    setDirty(false);
  }, [intent.id, intent.metadata]);

  const handleSave = async () => {
    if (contextIntents.length > 0) {
      await api.setMetadata(intent.id, 'context_intents', contextIntents.filter(Boolean));
    }
    if (actionIntents.length > 0) {
      await api.setMetadata(intent.id, 'action_intents', actionIntents.filter(Boolean));
    }
    setDirty(false);
    onRefresh();
  };

  const availableIds = allIntentIds.filter(id => id !== intent.id);

  return (
    <div className="space-y-5">
      <p className="text-xs text-zinc-600">
        Opaque key-value data returned alongside routing results. Your app interprets it.
      </p>

      <MetadataListEditor
        label="Context Intents"
        description="Supporting intents that provide data when this intent fires"
        values={contextIntents}
        availableIds={availableIds}
        onChange={v => { setContextIntents(v); setDirty(true); }}
      />
      <MetadataListEditor
        label="Action Intents"
        description="Related action intents commonly needed alongside this one"
        values={actionIntents}
        availableIds={availableIds}
        onChange={v => { setActionIntents(v); setDirty(true); }}
      />

      {dirty && (
        <button
          onClick={handleSave}
          className="px-4 py-2 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded transition-colors"
        >
          Save Metadata
        </button>
      )}
    </div>
  );
}

// --- Stats Tab ---

function StatsTab({ intent }: { intent: IntentInfo }) {
  const langKeys = Object.keys(intent.phrases_by_lang).filter(k => k !== '_learned');
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-6">
        <div>
          <div className="text-zinc-500 text-xs mb-1">Total Phrases</div>
          <div className="text-white font-mono text-2xl">{intent.phrases.length}</div>
        </div>
        <div>
          <div className="text-zinc-500 text-xs mb-1">Languages</div>
          <div className="text-white font-mono text-2xl">{langKeys.length}</div>
          <div className="text-zinc-600 text-xs mt-1">{langKeys.join(', ')}</div>
        </div>
        <div>
          <div className="text-zinc-500 text-xs mb-1">Learned Terms</div>
          <div className="text-white font-mono text-2xl">{intent.learned_count}</div>
          <div className="text-zinc-600 text-xs mt-1">from corrections</div>
        </div>
      </div>

      <div className="border-t border-zinc-800 pt-4">
        <div className="text-zinc-500 text-xs mb-2">Phrases per Language</div>
        {langKeys.map(lang => {
          const count = (intent.phrases_by_lang[lang] || []).length;
          return (
            <div key={lang} className="flex items-center gap-3 py-1">
              <span className="text-xs text-violet-400 uppercase w-8">{lang}</span>
              <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-violet-500/50 rounded-full"
                  style={{ width: `${Math.min(100, (count / Math.max(1, intent.phrases.length)) * 100)}%` }}
                />
              </div>
              <span className="text-xs text-zinc-500 w-8 text-right">{count}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// --- Metadata list editor with autocomplete ---

function MetadataListEditor({
  label, description, values, availableIds, onChange,
}: {
  label: string;
  description: string;
  values: string[];
  availableIds: string[];
  onChange: (values: string[]) => void;
}) {
  const [inputValue, setInputValue] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  const filtered = availableIds.filter(id =>
    id.toLowerCase().includes(inputValue.toLowerCase()) && !values.includes(id)
  );

  const addValue = (v: string) => {
    if (v && !values.includes(v)) {
      onChange([...values, v]);
    }
    setInputValue('');
    setShowSuggestions(false);
  };

  const removeValue = (index: number) => {
    onChange(values.filter((_, i) => i !== index));
  };

  return (
    <div>
      <div className="text-[11px] text-zinc-400 font-semibold mb-1">{label}</div>
      <div className="text-[10px] text-zinc-600 mb-2">{description}</div>

      {values.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-2">
          {values.map((v, i) => (
            <span key={i} className="inline-flex items-center gap-1 text-xs font-mono text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 px-2 py-0.5 rounded">
              {v}
              <button onClick={() => removeValue(i)} className="text-cyan-400/50 hover:text-red-400 ml-0.5">×</button>
            </span>
          ))}
        </div>
      )}

      <div className="relative">
        <input
          value={inputValue}
          onChange={e => { setInputValue(e.target.value); setShowSuggestions(true); }}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
          onKeyDown={e => { if (e.key === 'Enter' && inputValue.trim()) addValue(inputValue.trim()); }}
          placeholder="Type intent ID..."
          autoComplete="off"
          className="w-full max-w-sm bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-sm text-white font-mono focus:border-violet-500 focus:outline-none"
        />
        {showSuggestions && filtered.length > 0 && (
          <div className="absolute top-full left-0 w-full max-w-sm mt-1 bg-zinc-800 border border-zinc-700 rounded shadow-xl z-10 max-h-32 overflow-y-auto">
            {filtered.map(id => (
              <div
                key={id}
                onMouseDown={() => addValue(id)}
                className="px-2.5 py-1.5 text-sm text-zinc-300 font-mono hover:bg-zinc-700 cursor-pointer"
              >
                {id}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// --- Add Intent Panel (simplified two-step) ---

function AddIntentPanel({
  onDone, onCancel,
}: {
  onDone: (id: string) => void; onCancel: () => void;
}) {
  const [id, setId] = useState('');
  const [intentType, setIntentType] = useState<IntentType>('action');
  const [phraseText, setPhraseText] = useState('');
  const [showAI, setShowAI] = useState(false);
  const [description, setDescription] = useState('');
  const [languages, setLanguages] = useState<Record<string, string>>({});
  const [selectedLangs, setSelectedLangs] = useState<Set<string>>(new Set(appSettings.languages));
  const [generating, setGenerating] = useState(false);
  const [genStatus, setGenStatus] = useState('');
  const { settings: appSettings } = useAppStore();
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
      const parsed = await api.generatePhrases(id || 'new_intent', description, langs);
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
    await api.addIntent(intentId, phrases, intentType);
    onDone(intentId);
  };

  const phraseCount = phraseText.split('\n').filter(s => s.trim()).length;

  return (
    <div className="space-y-5 max-w-2xl">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">New Intent</h2>
        <button onClick={onCancel} className="text-sm text-zinc-500 hover:text-white">Cancel</button>
      </div>

      {/* Name + Type */}
      <div className="flex gap-4 items-end">
        <div className="flex-1">
          <label className="text-xs text-zinc-500 block mb-1">Intent ID</label>
          <input
            value={id}
            onChange={e => setId(e.target.value)}
            placeholder="e.g. cancel_order"
            autoComplete="off"
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-white text-sm font-mono focus:border-violet-500 focus:outline-none"
            autoFocus
          />
        </div>
        <div>
          <label className="text-xs text-zinc-500 block mb-1">Type</label>
          <div className="flex rounded overflow-hidden border border-zinc-700">
            {(['action', 'context'] as IntentType[]).map(t => (
              <button
                key={t}
                onClick={() => setIntentType(t)}
                className={`text-xs px-3 py-2 font-semibold uppercase transition-colors ${
                  intentType === t
                    ? t === 'action' ? 'bg-emerald-400/20 text-emerald-400' : 'bg-cyan-400/20 text-cyan-400'
                    : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Phrases */}
      <div>
        <label className="text-xs text-zinc-500 block mb-1">
          Training Phrases {phraseCount > 0 && <span className="text-zinc-600">({phraseCount})</span>}
        </label>
        <textarea
          value={phraseText}
          onChange={e => setPhraseText(e.target.value)}
          placeholder={"One phrase per line:\ncancel my order\nI want to cancel\nstop my order"}
          rows={6}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-white font-mono resize-y focus:border-violet-500 focus:outline-none"
        />
      </div>

      {/* AI Generation (expandable) */}
      <div>
        <button
          onClick={() => setShowAI(!showAI)}
          className="text-xs text-violet-400 hover:text-violet-300 flex items-center gap-1"
        >
          <svg className={`w-3 h-3 transition-transform ${showAI ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          Generate phrases with AI
        </button>

        {showAI && (
          <div className="mt-3 space-y-3 pl-4 border-l-2 border-violet-500/20">
            <textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Describe the intent: e.g. Customer wants to cancel their order or subscription"
              rows={2}
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-white text-sm resize-y focus:border-violet-500 focus:outline-none"
            />
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
                    className="accent-violet-500"
                  />
                  {name}
                </label>
              ))}
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={handleGenerate}
                disabled={generating}
                className="text-xs px-3 py-1.5 border border-violet-500 text-violet-400 rounded hover:bg-violet-500 hover:text-white disabled:opacity-50"
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
          className="px-5 py-2 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded-lg transition-colors font-medium disabled:opacity-30"
        >
          Create Intent
        </button>
      </div>
    </div>
  );
}

// --- Situations Tab ---
// Situation patterns match state descriptions (not action vocabulary).
// e.g. "build failed", "OOM", "付款" → create_issue
// Score: weight × sqrt(char_len). Threshold: 0.8

const WEIGHT_PRESETS = [
  { label: 'Strong (1.0)', value: 1.0, description: 'Domain-specific — fires alone' },
  { label: 'Medium (0.7)', value: 0.7, description: 'Fairly specific — usually fires alone' },
  { label: 'Weak (0.4)', value: 0.4, description: 'Generic — needs a partner to fire' },
];

function SituationsTab({ intent, onRefresh }: { intent: IntentInfo; onRefresh: () => void }) {
  const [pattern, setPattern] = useState('');
  const [weight, setWeight] = useState(1.0);
  const [saving, setSaving] = useState(false);
  const [addError, setAddError] = useState('');

  // AI generation state
  const [showAI, setShowAI] = useState(false);
  const [aiDesc, setAiDesc] = useState(intent.description || '');
  const [languages, setLanguages] = useState<Record<string, string>>({});
  const [generating, setGenerating] = useState(false);
  const [genStatus, setGenStatus] = useState('');
  const { settings: sitSettings } = useAppStore();
  const enabledLangs = new Set(sitSettings.languages);
  const [aiLangs, setAiLangs] = useState<Set<string>>(new Set(sitSettings.languages));

  useEffect(() => {
    api.getLanguages().then(setLanguages).catch(() => {});
  }, []);

  // Keep aiDesc in sync when intent changes
  useEffect(() => { setAiDesc(intent.description || ''); }, [intent.id, intent.description]);

  const patterns: [string, number][] = intent.situation_patterns || [];

  const handleAdd = async () => {
    const p = pattern.trim();
    if (!p) return;
    setSaving(true);
    setAddError('');
    try {
      await api.addSituationPattern(intent.id, p, weight);
      setPattern('');
      onRefresh();
    } catch (e: unknown) {
      setAddError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const handleRemove = async (p: string) => {
    try {
      await api.removeSituationPattern(intent.id, p);
      onRefresh();
    } catch { /* ignore */ }
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setGenStatus('Generating…');
    try {
      const langs = Array.from(aiLangs);
      const result = await api.generateSituations(intent.id, aiDesc, langs);
      setGenStatus(`Added ${result.applied} patterns`);
      onRefresh();
    } catch (e: unknown) {
      setGenStatus('Error: ' + (e instanceof Error ? e.message : String(e)));
    } finally {
      setGenerating(false);
    }
  };

  // Group by weight tier
  const strong = patterns.filter(([, w]) => w >= 0.9);
  const medium = patterns.filter(([, w]) => w >= 0.6 && w < 0.9);
  const weak = patterns.filter(([, w]) => w < 0.6);

  return (
    <div className="flex flex-col gap-4">
      {/* Explanation strip */}
      <div className="text-xs text-zinc-500 leading-relaxed">
        <span className="text-emerald-400 font-medium">Phrases</span> = action vocabulary ("cancel my order").{' '}
        <span className="text-amber-400 font-medium">Situations</span> = state vocabulary ("payment declined", "OOM").
        Score: weight × √(chars). Threshold 0.8 — a Strong pattern fires alone; two Weak ones together also fire.
        Active learning: corrections automatically extract n-grams into this index.
      </div>

      {/* Pattern list */}
      {patterns.length === 0 ? (
        <div className="text-xs text-zinc-600 text-center py-4 border border-zinc-800 rounded-lg">
          No situation patterns yet. Generate with AI or add manually below.
        </div>
      ) : (
        <div className="space-y-3">
          {strong.length > 0 && (
            <PatternGroup label="Strong" color="text-emerald-400" dot="bg-emerald-400" patterns={strong} onRemove={handleRemove} />
          )}
          {medium.length > 0 && (
            <PatternGroup label="Medium" color="text-amber-400" dot="bg-amber-400" patterns={medium} onRemove={handleRemove} />
          )}
          {weak.length > 0 && (
            <PatternGroup label="Weak" color="text-zinc-500" dot="bg-zinc-600" patterns={weak} onRemove={handleRemove} />
          )}
        </div>
      )}

      {/* Manual add */}
      <div className="space-y-2">
        <div className="flex gap-2">
          <input
            value={pattern}
            onChange={e => { setPattern(e.target.value); setAddError(''); }}
            onKeyDown={e => { if (e.key === 'Enter') handleAdd(); }}
            placeholder='"build failed", "OOM", "付款", "declined"…'
            autoComplete="off"
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-white font-mono focus:border-amber-500 focus:outline-none placeholder-zinc-600"
          />
          <div className="flex items-center gap-1">
            {WEIGHT_PRESETS.map(p => (
              <button
                key={p.value}
                onClick={() => setWeight(p.value)}
                title={p.description}
                className={`text-[10px] px-2 py-1 rounded border transition-colors ${
                  weight === p.value
                    ? 'border-amber-500/60 bg-amber-500/10 text-amber-300'
                    : 'border-zinc-700 text-zinc-500 hover:text-zinc-300'
                }`}
              >
                {p.value}
              </button>
            ))}
          </div>
          <button
            onClick={handleAdd}
            disabled={saving || !pattern.trim()}
            className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 text-amber-400 rounded hover:bg-zinc-700 disabled:opacity-30 transition-colors"
          >
            + Add
          </button>
        </div>
        {addError && <div className="text-xs text-red-400">{addError}</div>}
      </div>

      {/* AI Generate — same expand pattern as Phrases tab */}
      <div>
        <button
          onClick={() => setShowAI(!showAI)}
          className="text-xs text-amber-400 hover:text-amber-300 flex items-center gap-1"
        >
          <svg className={`w-3 h-3 transition-transform ${showAI ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          Generate situation patterns with AI
        </button>

        {showAI && (
          <div className="mt-2 space-y-2 pl-4 border-l-2 border-amber-500/20">
            <textarea
              value={aiDesc}
              onChange={e => setAiDesc(e.target.value)}
              placeholder="Describe the intent: e.g. Report a bug or production incident"
              rows={2}
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-white text-sm resize-y focus:border-amber-500 focus:outline-none"
            />
            <div className="text-[10px] text-zinc-500">Include CJK languages if your users write in Chinese/Japanese/Korean.</div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(languages).filter(([code]) => enabledLangs.has(code)).map(([code, name]) => (
                <label key={code} className="inline-flex items-center gap-1 text-xs text-zinc-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={aiLangs.has(code)}
                    onChange={() => {
                      setAiLangs(prev => {
                        const next = new Set(prev);
                        next.has(code) ? next.delete(code) : next.add(code);
                        return next;
                      });
                    }}
                    className="accent-amber-500"
                  />
                  {name}
                </label>
              ))}
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={handleGenerate}
                disabled={generating || !aiDesc.trim()}
                className="text-xs px-3 py-1.5 border border-amber-500 text-amber-400 rounded hover:bg-amber-500 hover:text-white disabled:opacity-50 transition-colors"
              >
                {generating ? 'Generating…' : 'Generate'}
              </button>
              {genStatus && <span className="text-xs text-zinc-500">{genStatus}</span>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function PatternGroup({
  label, color, dot, patterns, onRemove,
}: {
  label: string; color: string; dot: string;
  patterns: [string, number][]; onRemove: (p: string) => void;
}) {
  return (
    <div>
      <div className={`text-[10px] font-semibold uppercase tracking-wide mb-2 flex items-center gap-1.5 ${color}`}>
        <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
        {label}
      </div>
      <div className="flex flex-wrap gap-2">
        {patterns.map(([p, w]) => (
          <span
            key={p}
            className="inline-flex items-center gap-1.5 text-sm font-mono text-zinc-200 bg-zinc-800 border border-zinc-700 px-2.5 py-1 rounded group"
          >
            {p}
            <span className="text-zinc-600 text-[10px]">{w}</span>
            <button
              onClick={() => onRemove(p)}
              className="text-zinc-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100 ml-0.5"
            >
              ×
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}
