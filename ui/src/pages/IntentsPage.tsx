import { useState, useEffect, useCallback, useMemo } from 'react';
import { api, type IntentInfo, type IntentType } from '@/api/client';

const TYPE_COLORS: Record<IntentType, string> = {
  action: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
  context: 'text-cyan-400 bg-cyan-400/10 border-cyan-400/30',
};

export default function IntentsPage() {
  const [intents, setIntents] = useState<IntentInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showAdd, setShowAdd] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const data = await api.listIntents();
      setIntents(data);
      if (selectedId && !data.find(i => i.id === selectedId)) {
        setSelectedId(data.length > 0 ? data[0].id : null);
      }
    } catch { /* server not running */ }
  }, [selectedId]);

  useEffect(() => { refresh(); }, [refresh]);

  const handleReset = async () => {
    await api.reset();
    await api.loadDefaults();
    const data = await api.listIntents();
    setIntents(data);
    setSelectedId(data.length > 0 ? data[0].id : null);
  };

  const selected = intents.find(i => i.id === selectedId) || null;
  const allIntentIds = useMemo(() => intents.map(i => i.id), [intents]);

  return (
    <div className="flex gap-0 h-[calc(100vh-6rem)] -mx-4">
      {/* Left: Intent list */}
      <div className="w-72 min-w-[18rem] border-r border-zinc-800 flex flex-col">
        <div className="px-3 py-3 border-b border-zinc-800 flex items-center justify-between flex-shrink-0">
          <span className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">
            Intents ({intents.length})
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

        <div className="flex-1 overflow-y-auto">
          {intents.map(intent => (
            <IntentListItem
              key={intent.id}
              intent={intent}
              selected={selectedId === intent.id}
              onClick={() => { setSelectedId(intent.id); setShowAdd(false); }}
            />
          ))}
          {intents.length === 0 && (
            <div className="text-zinc-600 text-xs text-center py-8 px-4">
              No intents.{' '}
              <button onClick={handleReset} className="text-violet-400 hover:text-violet-300">
                Load defaults
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Right: Detail panel */}
      <div className="flex-1 overflow-y-auto">
        {showAdd ? (
          <div className="p-5">
            <AddIntentPanel
              allIntentIds={allIntentIds}
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
  const langKeys = Object.keys(intent.seeds_by_lang).filter(k => k !== '_learned');
  return (
    <div
      onClick={onClick}
      className={`px-3 py-2.5 cursor-pointer border-b border-zinc-800/50 transition-colors ${
        selected ? 'bg-zinc-800/80' : 'hover:bg-zinc-800/40'
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="text-emerald-400 font-mono text-sm font-semibold truncate">{intent.id}</span>
        <span className={`text-[9px] px-1.5 py-0.5 rounded border font-semibold uppercase ${TYPE_COLORS[intent.intent_type]}`}>
          {intent.intent_type}
        </span>
      </div>
      <div className="flex items-center gap-2 mt-1">
        <span className="text-zinc-500 text-[11px]">{intent.seeds.length} seeds</span>
        {intent.learned_count > 0 && (
          <span className="text-emerald-400/50 text-[11px]">+{intent.learned_count}</span>
        )}
        <div className="flex gap-0.5 ml-auto">
          {langKeys.map(lang => (
            <span key={lang} className="text-[9px] text-violet-400/60 bg-zinc-800 rounded px-1 uppercase">
              {lang}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// --- Right detail panel with tabs ---

type DetailTab = 'seeds' | 'metadata' | 'stats';

function IntentDetailPanel({
  intent, allIntentIds, onRefresh, onDeleted,
}: {
  intent: IntentInfo; allIntentIds: string[]; onRefresh: () => void; onDeleted: () => void;
}) {
  const [activeTab, setActiveTab] = useState<DetailTab>('seeds');

  const handleTypeChange = async (newType: IntentType) => {
    await api.setIntentType(intent.id, newType);
    onRefresh();
  };

  const handleDelete = async () => {
    if (!confirm(`Delete intent "${intent.id}"?`)) return;
    await api.deleteIntent(intent.id);
    onDeleted();
  };

  const langKeys = Object.keys(intent.seeds_by_lang).filter(k => k !== '_learned');
  const metaKeyCount = Object.keys(intent.metadata || {}).length;

  const tabs: { id: DetailTab; label: string; count?: number }[] = [
    { id: 'seeds', label: 'Seeds', count: intent.seeds.length },
    { id: 'metadata', label: 'Metadata', count: metaKeyCount },
    { id: 'stats', label: 'Stats' },
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-5 pt-5 pb-0">
        <div className="flex items-center justify-between mb-4">
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

        {/* Tabs */}
        <div className="flex gap-0 border-b border-zinc-800">
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
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto px-5 py-4">
        {activeTab === 'seeds' && (
          <SeedsTab intent={intent} onRefresh={onRefresh} />
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

// --- Seeds Tab ---

function SeedsTab({ intent, onRefresh }: { intent: IntentInfo; onRefresh: () => void }) {
  const [newSeed, setNewSeed] = useState('');
  const [seedSearch, setSeedSearch] = useState('');
  const [showBulk, setShowBulk] = useState(false);
  const [bulkText, setBulkText] = useState('');

  const langKeys = Object.keys(intent.seeds_by_lang).filter(k => k !== '_learned');

  // Build flat list with language tags
  const allSeeds = useMemo(() => {
    const result: { lang: string; seed: string }[] = [];
    for (const lang of langKeys) {
      for (const seed of intent.seeds_by_lang[lang] || []) {
        result.push({ lang, seed });
      }
    }
    return result;
  }, [intent.seeds_by_lang, langKeys]);

  const filtered = useMemo(() => {
    if (!seedSearch.trim()) return allSeeds;
    const q = seedSearch.toLowerCase();
    return allSeeds.filter(s => s.seed.toLowerCase().includes(q));
  }, [allSeeds, seedSearch]);

  const handleAddSeed = async () => {
    if (!newSeed.trim()) return;
    await api.addSeed(intent.id, newSeed.trim());
    setNewSeed('');
    onRefresh();
  };

  const handleBulkAdd = async () => {
    const lines = bulkText.split('\n').map(s => s.trim()).filter(Boolean);
    if (lines.length === 0) return;
    for (const line of lines) {
      await api.addSeed(intent.id, line);
    }
    setBulkText('');
    setShowBulk(false);
    onRefresh();
  };

  return (
    <div className="space-y-3">
      {/* Search + actions */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1 max-w-xs">
          <input
            value={seedSearch}
            onChange={e => setSeedSearch(e.target.value)}
            placeholder="Search seeds..."
            autoComplete="off"
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1.5 text-xs text-white focus:border-violet-500 focus:outline-none"
          />
          {seedSearch && (
            <button onClick={() => setSeedSearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-white text-xs">×</button>
          )}
        </div>
        <button
          onClick={() => setShowBulk(!showBulk)}
          className="text-xs text-zinc-400 hover:text-white px-2 py-1.5 border border-zinc-700 rounded transition-colors"
        >
          {showBulk ? 'Cancel bulk' : 'Bulk paste'}
        </button>
      </div>

      {/* Bulk paste area */}
      {showBulk && (
        <div className="space-y-2">
          <textarea
            value={bulkText}
            onChange={e => setBulkText(e.target.value)}
            placeholder="Paste multiple seeds, one per line..."
            rows={5}
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-white font-mono resize-y focus:border-violet-500 focus:outline-none"
          />
          <button
            onClick={handleBulkAdd}
            disabled={!bulkText.trim()}
            className="text-xs px-3 py-1.5 bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-30"
          >
            Add {bulkText.split('\n').filter(s => s.trim()).length} seeds
          </button>
        </div>
      )}

      {/* Seed list */}
      <div className="border border-zinc-800 rounded-lg bg-zinc-900/50 divide-y divide-zinc-800/50">
        {filtered.length === 0 && (
          <div className="text-zinc-600 text-xs text-center py-6">
            {seedSearch ? 'No seeds match search' : 'No seeds yet'}
          </div>
        )}
        {filtered.map((s, i) => (
          <div key={`${s.lang}-${i}`} className="flex items-center gap-2 px-3 py-1.5 hover:bg-zinc-800/50 transition-colors group">
            {langKeys.length > 1 && (
              <span className="text-[9px] text-violet-400/60 bg-zinc-800 rounded px-1 uppercase w-6 text-center flex-shrink-0">
                {s.lang}
              </span>
            )}
            <span className="text-sm text-zinc-300 font-mono flex-1 truncate">{s.seed}</span>
          </div>
        ))}
      </div>

      {/* Inline add */}
      <div className="flex gap-2">
        <input
          value={newSeed}
          onChange={e => setNewSeed(e.target.value)}
          placeholder="Type a seed and press Enter..."
          autoComplete="off"
          className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-sm text-white font-mono focus:border-violet-500 focus:outline-none"
          onKeyDown={e => { if (e.key === 'Enter') handleAddSeed(); }}
        />
        <button
          onClick={handleAddSeed}
          disabled={!newSeed.trim()}
          className="px-3 py-1.5 text-sm bg-zinc-800 border border-zinc-700 text-violet-400 rounded hover:bg-zinc-700 disabled:opacity-30 transition-colors"
        >
          + Add
        </button>
      </div>

      {intent.learned_count > 0 && (
        <p className="text-xs text-emerald-400/50">+{intent.learned_count} terms learned from corrections</p>
      )}
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
  const langKeys = Object.keys(intent.seeds_by_lang).filter(k => k !== '_learned');
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-6">
        <div>
          <div className="text-zinc-500 text-xs mb-1">Total Seeds</div>
          <div className="text-white font-mono text-2xl">{intent.seeds.length}</div>
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
        <div className="text-zinc-500 text-xs mb-2">Seeds per Language</div>
        {langKeys.map(lang => {
          const count = (intent.seeds_by_lang[lang] || []).length;
          return (
            <div key={lang} className="flex items-center gap-3 py-1">
              <span className="text-xs text-violet-400 uppercase w-8">{lang}</span>
              <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-violet-500/50 rounded-full"
                  style={{ width: `${Math.min(100, (count / Math.max(1, intent.seeds.length)) * 100)}%` }}
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
  allIntentIds, onDone, onCancel,
}: {
  allIntentIds: string[]; onDone: (id: string) => void; onCancel: () => void;
}) {
  const [id, setId] = useState('');
  const [intentType, setIntentType] = useState<IntentType>('action');
  const [seedText, setSeedText] = useState('');
  const [showAI, setShowAI] = useState(false);
  const [description, setDescription] = useState('');
  const [languages, setLanguages] = useState<Record<string, string>>({});
  const [selectedLangs, setSelectedLangs] = useState<Set<string>>(new Set(['en']));
  const [generating, setGenerating] = useState(false);
  const [genStatus, setGenStatus] = useState('');

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
      const parsed = await api.generateSeeds(id || 'new_intent', description, langs);
      const allSeeds: string[] = [];
      for (const lang of langs) {
        for (const seed of parsed.seeds_by_lang[lang] || []) {
          allSeeds.push(seed);
        }
      }
      const prev = seedText.trim();
      setSeedText(prev ? prev + '\n' + allSeeds.join('\n') : allSeeds.join('\n'));
      setGenStatus(`Generated ${parsed.total} seeds`);
    } catch (e) {
      setGenStatus('Error: ' + (e as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  const handleAdd = async () => {
    const intentId = id.trim();
    if (!intentId) return;
    const seeds = seedText.split('\n').map(s => s.trim()).filter(Boolean);
    if (seeds.length === 0) return;
    await api.addIntent(intentId, seeds, intentType);
    onDone(intentId);
  };

  const seedCount = seedText.split('\n').filter(s => s.trim()).length;

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

      {/* Seeds */}
      <div>
        <label className="text-xs text-zinc-500 block mb-1">
          Seed Phrases {seedCount > 0 && <span className="text-zinc-600">({seedCount})</span>}
        </label>
        <textarea
          value={seedText}
          onChange={e => setSeedText(e.target.value)}
          placeholder={"One seed phrase per line:\ncancel my order\nI want to cancel\nstop my order"}
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
          Generate seeds with AI
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
              {Object.entries(languages).slice(0, 12).map(([code, name]) => (
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
          disabled={!id.trim() || seedCount === 0}
          className="px-5 py-2 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded-lg transition-colors font-medium disabled:opacity-30"
        >
          Create Intent
        </button>
      </div>
    </div>
  );
}
