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

// --- Right detail panel ---

function IntentDetailPanel({
  intent, allIntentIds, onRefresh, onDeleted,
}: {
  intent: IntentInfo; allIntentIds: string[]; onRefresh: () => void; onDeleted: () => void;
}) {
  const [newSeed, setNewSeed] = useState('');
  const [seedSearch, setSeedSearch] = useState('');

  // Metadata editing
  const [contextIntents, setContextIntents] = useState<string[]>(intent.metadata?.context_intents || []);
  const [actionIntents, setActionIntents] = useState<string[]>(intent.metadata?.action_intents || []);
  const [metaDirty, setMetaDirty] = useState(false);

  useEffect(() => {
    setContextIntents(intent.metadata?.context_intents || []);
    setActionIntents(intent.metadata?.action_intents || []);
    setMetaDirty(false);
    setNewSeed('');
    setSeedSearch('');
  }, [intent.id, intent.metadata]);

  const handleAddSeed = async () => {
    if (!newSeed.trim()) return;
    await api.addSeed(intent.id, newSeed.trim());
    setNewSeed('');
    onRefresh();
  };

  const handleTypeChange = async (newType: IntentType) => {
    await api.setIntentType(intent.id, newType);
    onRefresh();
  };

  const handleSaveMetadata = async () => {
    if (contextIntents.length > 0) {
      await api.setMetadata(intent.id, 'context_intents', contextIntents.filter(Boolean));
    }
    if (actionIntents.length > 0) {
      await api.setMetadata(intent.id, 'action_intents', actionIntents.filter(Boolean));
    }
    setMetaDirty(false);
    onRefresh();
  };

  const handleDelete = async () => {
    if (!confirm(`Delete intent "${intent.id}"?`)) return;
    await api.deleteIntent(intent.id);
    onDeleted();
  };

  const langKeys = Object.keys(intent.seeds_by_lang).filter(k => k !== '_learned');

  const filteredSeedsByLang = useMemo(() => {
    if (!seedSearch.trim()) return intent.seeds_by_lang;
    const q = seedSearch.toLowerCase();
    const result: Record<string, string[]> = {};
    for (const lang of langKeys) {
      const filtered = (intent.seeds_by_lang[lang] || []).filter(s => s.toLowerCase().includes(q));
      if (filtered.length > 0) result[lang] = filtered;
    }
    return result;
  }, [intent.seeds_by_lang, seedSearch, langKeys]);

  const filteredLangKeys = Object.keys(filteredSeedsByLang).filter(k => k !== '_learned');
  const availableIntentIds = allIntentIds.filter(id => id !== intent.id);

  return (
    <div className="p-5 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold text-emerald-400 font-mono">{intent.id}</h2>
          {/* Type toggle */}
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
          <div className="flex gap-1">
            {langKeys.map(lang => (
              <span key={lang} className="text-[10px] font-semibold text-violet-400 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 uppercase tracking-wide">
                {lang}
              </span>
            ))}
          </div>
          {intent.learned_count > 0 && (
            <span className="text-emerald-400/70 text-xs bg-emerald-400/10 px-2 py-0.5 rounded">
              +{intent.learned_count} learned
            </span>
          )}
        </div>
        <button onClick={handleDelete} className="text-xs text-red-400 hover:text-red-300 px-2 py-1 border border-red-400/20 rounded hover:border-red-400/50 transition-colors">
          Delete
        </button>
      </div>

      {/* Seeds section */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">
            Seed Phrases ({intent.seeds.length})
          </h3>
          <div className="relative w-48">
            <input
              value={seedSearch}
              onChange={e => setSeedSearch(e.target.value)}
              placeholder="Search seeds..."
              autoComplete="off"
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-1 text-xs text-white focus:border-violet-500 focus:outline-none"
            />
            {seedSearch && (
              <button
                onClick={() => setSeedSearch('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-white text-xs"
              >
                ×
              </button>
            )}
          </div>
        </div>

        <div className="max-h-60 overflow-y-auto space-y-3 border border-zinc-800 rounded-lg p-3 bg-zinc-900/50">
          {filteredLangKeys.map(lang => {
            const seeds = filteredSeedsByLang[lang] || [];
            return (
              <div key={lang}>
                {langKeys.length > 1 && (
                  <div className="text-[11px] text-violet-400/70 font-semibold uppercase tracking-wide mb-1.5">{lang}</div>
                )}
                <div className="space-y-0.5">
                  {seeds.map((seed, i) => (
                    <div key={i} className="text-sm text-zinc-300 font-mono px-2 py-1 rounded hover:bg-zinc-800/80 transition-colors">
                      {seed}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
          {filteredLangKeys.length === 0 && (
            <div className="text-zinc-600 text-xs text-center py-2">
              {seedSearch ? 'No seeds match search' : 'No seeds'}
            </div>
          )}
        </div>

        <div className="mt-3 flex gap-2">
          <input
            value={newSeed}
            onChange={e => setNewSeed(e.target.value)}
            placeholder="Add a seed phrase..."
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
      </section>

      {/* Metadata section — context & action intent suggestions */}
      <section>
        <h3 className="text-xs text-zinc-500 font-semibold uppercase tracking-wide mb-3">Metadata</h3>
        <p className="text-xs text-zinc-600 mb-3">
          Opaque data returned alongside routing results. ASV stores it, your app interprets it.
        </p>

        <div className="space-y-4">
          <MetadataListEditor
            label="Context Intents"
            description="Supporting intents that provide useful data when this intent fires"
            values={contextIntents}
            availableIds={availableIntentIds}
            onChange={v => { setContextIntents(v); setMetaDirty(true); }}
          />
          <MetadataListEditor
            label="Action Intents"
            description="Related action intents commonly needed alongside this one"
            values={actionIntents}
            availableIds={availableIntentIds}
            onChange={v => { setActionIntents(v); setMetaDirty(true); }}
          />
        </div>

        {metaDirty && (
          <button
            onClick={handleSaveMetadata}
            className="mt-3 px-4 py-2 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded transition-colors"
          >
            Save Metadata
          </button>
        )}
      </section>

      {/* Stats */}
      <section className="border-t border-zinc-800 pt-4">
        <h3 className="text-xs text-zinc-500 font-semibold uppercase tracking-wide mb-2">Info</h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-zinc-500 text-xs">Total Seeds</div>
            <div className="text-white font-mono">{intent.seeds.length}</div>
          </div>
          <div>
            <div className="text-zinc-500 text-xs">Languages</div>
            <div className="text-white font-mono">{langKeys.length}</div>
          </div>
          <div>
            <div className="text-zinc-500 text-xs">Learned Terms</div>
            <div className="text-white font-mono">{intent.learned_count}</div>
          </div>
        </div>
      </section>
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
              <button
                onClick={() => removeValue(i)}
                className="text-cyan-400/50 hover:text-red-400 ml-0.5"
              >
                ×
              </button>
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

// --- Add Intent Panel ---

function AddIntentPanel({
  allIntentIds, onDone, onCancel,
}: {
  allIntentIds: string[]; onDone: (id: string) => void; onCancel: () => void;
}) {
  const [id, setId] = useState('');
  const [intentType, setIntentType] = useState<IntentType>('action');
  const [description, setDescription] = useState('');
  const [languages, setLanguages] = useState<Record<string, string>>({});
  const [selectedLangs, setSelectedLangs] = useState<Set<string>>(new Set(['en']));
  const [seedsByLang, setSeedsByLang] = useState<Record<string, string>>({});
  const [generating, setGenerating] = useState(false);
  const [genStatus, setGenStatus] = useState('');
  const [genStatusColor, setGenStatusColor] = useState('text-zinc-500');

  useEffect(() => {
    api.getLanguages().then(setLanguages).catch(() => {});
  }, []);

  const toggleLang = (code: string) => {
    setSelectedLangs(prev => {
      const next = new Set(prev);
      if (next.has(code)) next.delete(code);
      else next.add(code);
      return next;
    });
  };

  const sortedLangs = Object.keys(languages).sort((a, b) => {
    if (a === 'en') return -1;
    if (b === 'en') return 1;
    return languages[a].localeCompare(languages[b]);
  });

  const handleGenerate = async () => {
    if (!description.trim()) {
      setGenStatusColor('text-red-400');
      setGenStatus('Enter a description first.');
      return;
    }
    const langs = Array.from(selectedLangs);
    if (langs.length === 0) {
      setGenStatusColor('text-red-400');
      setGenStatus('Select at least one language.');
      return;
    }

    setGenerating(true);
    setGenStatusColor('text-zinc-500');
    setGenStatus('Generating diverse seeds via Claude Haiku...');

    try {
      const parsed = await api.generateSeeds(id || 'new_intent', description, langs);

      const newSeeds = { ...seedsByLang };
      for (const lang of langs) {
        const langSeeds = parsed.seeds_by_lang[lang] || [];
        const prev = (newSeeds[lang] || '').trim();
        newSeeds[lang] = prev ? prev + '\n' + langSeeds.join('\n') : langSeeds.join('\n');
      }
      setSeedsByLang(newSeeds);

      setGenStatusColor('text-emerald-400');
      setGenStatus(`Generated ${parsed.total} seeds${langs.length > 1 ? ` across ${langs.length} languages` : ''}`);
    } catch (e) {
      setGenStatusColor('text-red-400');
      setGenStatus('Error: ' + (e as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  const handleAdd = async () => {
    const intentId = id.trim();
    if (!intentId) return;

    const finalSeeds: Record<string, string[]> = {};
    let total = 0;
    for (const lang of selectedLangs) {
      const lines = (seedsByLang[lang] || '').split('\n').map(s => s.trim()).filter(Boolean);
      if (lines.length > 0) {
        finalSeeds[lang] = lines;
        total += lines.length;
      }
    }
    if (total === 0) return;

    await api.addIntentMultilingual(intentId, finalSeeds, intentType);
    onDone(intentId);
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">New Intent</h2>
        <button onClick={onCancel} className="text-sm text-zinc-500 hover:text-white transition-colors">
          Cancel
        </button>
      </div>

      <div className="flex gap-4 items-end">
        <div className="flex-1 max-w-sm">
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

      <div>
        <label className="text-xs text-zinc-500 block mb-1">Description (for AI seed generation)</label>
        <textarea
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder="e.g. Customer wants to cancel their subscription or order. They may express frustration, urgency, or simply want the process explained."
          rows={3}
          className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-white text-sm resize-y focus:border-violet-500 focus:outline-none"
        />
      </div>

      {/* Languages */}
      <div>
        <label className="text-xs text-zinc-500 block mb-2">Languages</label>
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 max-h-28 overflow-y-auto bg-zinc-900 border border-zinc-800 rounded-lg p-2">
          {sortedLangs.map(code => (
            <label key={code} className="inline-flex items-center gap-1.5 text-sm text-zinc-300 cursor-pointer py-0.5">
              <input
                type="checkbox"
                checked={selectedLangs.has(code)}
                onChange={() => toggleLang(code)}
                className="accent-violet-500"
              />
              {languages[code]}
            </label>
          ))}
        </div>
      </div>

      {/* Generate */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleGenerate}
          disabled={generating}
          className="px-4 py-2 text-sm border border-violet-500 text-violet-400 rounded hover:bg-violet-500 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {generating ? 'Generating...' : 'Generate Seeds with AI'}
        </button>
        {genStatus && <span className={`text-xs ${genStatusColor}`}>{genStatus}</span>}
      </div>

      {/* Language seed panels */}
      {Array.from(selectedLangs).length > 0 && (
        <div className="grid grid-cols-2 gap-3">
          {Array.from(selectedLangs).map(lang => {
            const count = (seedsByLang[lang] || '').split('\n').filter(s => s.trim()).length;
            return (
              <div key={lang}>
                <div className="text-[11px] font-semibold text-violet-400 uppercase tracking-wide mb-1 bg-zinc-800 rounded px-2 py-1">
                  {languages[lang] || lang} {count > 0 && `(${count})`}
                </div>
                <textarea
                  value={seedsByLang[lang] || ''}
                  onChange={e => setSeedsByLang(prev => ({ ...prev, [lang]: e.target.value }))}
                  placeholder={`One seed per line...`}
                  rows={6}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-2.5 py-2 text-sm text-white font-mono resize-y focus:border-violet-500 focus:outline-none"
                />
              </div>
            );
          })}
        </div>
      )}

      {/* Submit */}
      <div className="flex justify-end pt-2 border-t border-zinc-800">
        <button
          onClick={handleAdd}
          className="px-5 py-2 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded-lg transition-colors font-medium"
        >
          Create Intent
        </button>
      </div>
    </div>
  );
}
