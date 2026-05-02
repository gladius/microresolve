import { useState, useEffect } from 'react';
import { useAppStore } from '@/store';
import { api } from '@/api/client';
import Page from '@/components/Page';

type StopWordStatus = { count: number; source: 'built-in' | 'generated' | 'custom' };

export default function LanguagesPage() {
  const { settings, setLanguages } = useAppStore();
  const ns = settings.selectedNamespaceId;
  const [allLanguages,  setAllLanguages]  = useState<Record<string, string>>({});
  const [pickerOpen,    setPickerOpen]    = useState(false);
  const [search,        setSearch]        = useState('');
  const [stopWords,     setStopWords]     = useState<Record<string, StopWordStatus>>({});
  const [promptLang,    setPromptLang]    = useState<string | null>(null);
  const [generating,    setGenerating]    = useState<string | null>(null);
  const [editLang,      setEditLang]      = useState<string | null>(null);
  const enabledLangs = settings.languages.length > 0 ? settings.languages : ['en'];

  useEffect(() => {
    api.getLanguages().then(setAllLanguages).catch(() => {});
    refreshStopWords();
  }, []);

  const refreshStopWords = () => {
    api.getStopWords().then(setStopWords).catch(() => {});
  };

  const removeLang = (code: string) => {
    if (code === 'en') return;
    setLanguages(enabledLangs.filter(l => l !== code));
  };

  const addLang = (code: string) => {
    if (enabledLangs.includes(code)) return;
    setLanguages([...enabledLangs, code]);
    setSearch('');
    setPickerOpen(false);
    if (code !== 'en' && !stopWords[code]) {
      setPromptLang(code);
    }
  };

  const generateStopWords = async (lang: string) => {
    setGenerating(lang);
    setPromptLang(null);
    try {
      await api.generateStopWords(lang);
      refreshStopWords();
    } catch { /* */ } finally {
      setGenerating(null);
    }
  };

  const commonLangs = ['en', 'es', 'fr', 'de', 'pt', 'it', 'nl', 'ja', 'ko', 'zh', 'ar', 'hi'];
  const sortedLangs = Object.keys(allLanguages).sort((a, b) => {
    const ai = commonLangs.indexOf(a);
    const bi = commonLangs.indexOf(b);
    if (ai >= 0 && bi >= 0) return ai - bi;
    if (ai >= 0) return -1;
    if (bi >= 0) return 1;
    return (allLanguages[a] || '').localeCompare(allLanguages[b] || '');
  });

  const availableLangs = sortedLangs.filter(code => !enabledLangs.includes(code));
  const filteredLangs = search
    ? availableLangs.filter(code =>
        (allLanguages[code] || '').toLowerCase().includes(search.toLowerCase()) ||
        code.toLowerCase().includes(search.toLowerCase()))
    : availableLangs;

  return (
    <Page
      title="Languages"
      subtitle={<>phrase generation languages for <span className="text-emerald-400 font-mono">{ns}</span></>}
      size="sm"
    >
      <div className="space-y-6">

        <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 text-xs text-zinc-500 leading-relaxed space-y-1">
          <div className="text-zinc-300 font-medium text-sm">Per-namespace language config</div>
          <p>
            Select which languages are used when generating training phrases for intents in this namespace.
            Phrases in each selected language are generated on import and via the phrase builder.
          </p>
          <p className="text-zinc-600">
            English is always included. LLM quality varies for non-English phrases — review recommended.
          </p>
        </div>

        {/* Active language chips */}
        <div className="space-y-3">
          <div className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Active languages</div>
          <div className="flex flex-wrap gap-2">
            {enabledLangs.map(code => (
              <span
                key={code}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
                  code === 'en'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                    : 'bg-zinc-800 text-zinc-200 border border-zinc-700'
                }`}
              >
                <span className="text-[9px] uppercase font-bold opacity-60">{code}</span>
                {allLanguages[code] || code}
                {code !== 'en' && (
                  <button
                    onClick={() => removeLang(code)}
                    className="ml-0.5 text-zinc-500 hover:text-red-400 transition-colors leading-none"
                    title={`Remove ${allLanguages[code] || code}`}
                  >
                    ×
                  </button>
                )}
              </span>
            ))}

            <button
              onClick={() => setPickerOpen(v => !v)}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs border border-dashed border-zinc-600 text-zinc-500 hover:text-zinc-100 hover:border-zinc-400 transition-colors"
            >
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add language
            </button>
          </div>

          {pickerOpen && (
            <div className="bg-zinc-900 border border-zinc-700 rounded-lg overflow-hidden">
              <div className="p-2 border-b border-zinc-800">
                <input
                  autoFocus
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Search languages…"
                  className="w-full bg-transparent text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none px-1"
                />
              </div>
              <div className="max-h-48 overflow-y-auto">
                {filteredLangs.slice(0, 40).map(code => (
                  <button
                    key={code}
                    onClick={() => addLang(code)}
                    className="w-full text-left px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 flex items-center gap-2"
                  >
                    <span className="text-[9px] text-zinc-500 uppercase w-6">{code}</span>
                    {allLanguages[code] || code}
                  </button>
                ))}
                {filteredLangs.length === 0 && (
                  <p className="px-3 py-3 text-xs text-zinc-600 text-center">No results</p>
                )}
              </div>
            </div>
          )}

          {/* Inline stop words prompt — fires when a new non-English language is added */}
          {promptLang && (
            <div className="bg-zinc-800/60 border border-zinc-700 rounded-lg px-4 py-3 flex items-center justify-between gap-4">
              <div>
                <div className="text-xs text-zinc-100 font-medium mb-0.5">
                  Generate stop words for {allLanguages[promptLang] || promptLang}?
                </div>
                <div className="text-[11px] text-zinc-500">
                  Common function words will be excluded from phrase matching.
                </div>
              </div>
              <div className="flex gap-2 flex-shrink-0">
                <button onClick={() => setPromptLang(null)}
                  className="text-[11px] text-zinc-500 hover:text-zinc-300 px-2 py-1 transition-colors">
                  Skip
                </button>
                <button onClick={() => generateStopWords(promptLang)}
                  className="text-[11px] bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1 rounded transition-colors">
                  Generate
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Stop words section */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Stop words</div>
            <span className="text-[10px] text-zinc-600">global — shared across all namespaces</span>
          </div>
          <div className="divide-y divide-zinc-800/50 border border-zinc-800 rounded-lg overflow-hidden">
            {enabledLangs.map(code => {
              const sw = stopWords[code];
              const isGenerating = generating === code;
              return (
                <div key={code} className="flex items-center gap-3 px-3 py-2.5 bg-zinc-900/40">
                  <span className="text-[9px] text-zinc-500 uppercase font-bold w-6">{code}</span>
                  {sw ? (
                    <>
                      <span className={`text-[10px] ${sw.source === 'built-in' ? 'text-zinc-500' : 'text-emerald-500'}`}>
                        {sw.source}
                      </span>
                      <span className="text-[10px] text-zinc-600">· {sw.count} words</span>
                      <button onClick={() => setEditLang(code)}
                        className="ml-auto text-[10px] text-zinc-500 hover:text-emerald-400 transition-colors">
                        view / edit
                      </button>
                      {sw.source !== 'built-in' && (
                        <button onClick={() => generateStopWords(code)} disabled={!!generating}
                          className="text-[10px] text-zinc-600 hover:text-emerald-400 disabled:opacity-40 transition-colors">
                          regenerate
                        </button>
                      )}
                    </>
                  ) : isGenerating ? (
                    <div className="flex items-center gap-1.5 text-[10px] text-emerald-400">
                      <div className="w-2.5 h-2.5 border border-emerald-400 border-t-transparent rounded-full animate-spin" />
                      generating…
                    </div>
                  ) : (
                    <>
                      <span className="text-[10px] text-amber-500/70">missing</span>
                      <button onClick={() => generateStopWords(code)} disabled={!!generating}
                        className="ml-auto text-[10px] text-zinc-500 hover:text-emerald-400 disabled:opacity-40 transition-colors">
                        Generate
                      </button>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>

      </div>
      {editLang && (
        <StopWordsModal
          lang={editLang}
          langName={allLanguages[editLang] || editLang}
          onClose={() => { setEditLang(null); refreshStopWords(); }}
        />
      )}
    </Page>
  );
}

function StopWordsModal({ lang, langName, onClose }: {
  lang: string;
  langName: string;
  onClose: () => void;
}) {
  const [words,   setWords]   = useState<string[]>([]);
  const [source,  setSource]  = useState<string>('');
  const [filter,  setFilter]  = useState('');
  const [adding,  setAdding]  = useState('');
  const [loading, setLoading] = useState(true);
  const [err,     setErr]     = useState<string | null>(null);

  const reload = async () => {
    setLoading(true);
    try {
      const d = await api.getStopWordsForLang(lang);
      setWords(d.words.slice().sort());
      setSource(d.source);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { reload(); /* eslint-disable-next-line */ }, [lang]);

  const add = async () => {
    const w = adding.trim().toLowerCase();
    if (!w) return;
    try {
      await api.addStopWord(lang, w);
      setAdding('');
      reload();
    } catch (e) { setErr(String(e)); }
  };

  const remove = async (w: string) => {
    try {
      await api.removeStopWord(lang, w);
      reload();
    } catch (e) { setErr(String(e)); }
  };

  const visible = filter
    ? words.filter(w => w.toLowerCase().includes(filter.toLowerCase()))
    : words;

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-lg max-h-[80vh] flex flex-col" onClick={e => e.stopPropagation()}>
        <div className="px-5 py-3 border-b border-zinc-800 flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-zinc-100">Stop words — {langName}</div>
            <div className="text-[10px] text-zinc-500">
              <span className="font-mono uppercase">{lang}</span> · {source} · {words.length} words
            </div>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 text-lg leading-none">×</button>
        </div>

        <div className="px-5 py-3 border-b border-zinc-800 space-y-2">
          <input
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Filter…"
            className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500"
          />
          <div className="flex gap-2">
            <input
              value={adding}
              onChange={e => setAdding(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && add()}
              placeholder="Add a stop word…"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={add}
              disabled={!adding.trim()}
              className="px-3 py-1 text-xs bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded transition-colors"
            >
              Add
            </button>
          </div>
          {err && <div className="text-[11px] text-red-400">{err}</div>}
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-2">
          {loading ? (
            <div className="text-xs text-zinc-500 text-center py-6">Loading…</div>
          ) : visible.length === 0 ? (
            <div className="text-xs text-zinc-600 text-center py-6">
              {filter ? 'No matches.' : 'No stop words.'}
            </div>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {visible.map(w => (
                <span
                  key={w}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-[11px] text-zinc-200 font-mono group"
                >
                  {w}
                  <button
                    onClick={() => remove(w)}
                    className="text-zinc-600 hover:text-red-400 leading-none transition-colors"
                    title={`Remove "${w}"`}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="px-5 py-2 border-t border-zinc-800 text-[10px] text-zinc-600">
          Edits override the built-in list. Stop words are excluded from phrase matching.
        </div>
      </div>
    </div>
  );
}
