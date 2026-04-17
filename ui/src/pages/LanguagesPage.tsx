import { useState, useEffect } from 'react';
import { useAppStore } from '@/store';
import { api } from '@/api/client';
import Page from '@/components/Page';

export default function LanguagesPage() {
  const { settings, setLanguages } = useAppStore();
  const ns = settings.selectedNamespaceId;
  const [allLanguages, setAllLanguages] = useState<Record<string, string>>({});
  const [pickerOpen, setPickerOpen] = useState(false);
  const [search, setSearch] = useState('');
  const enabledLangs = settings.languages.length > 0 ? settings.languages : ['en'];

  useEffect(() => {
    api.getLanguages().then(setAllLanguages).catch(() => {});
  }, []);

  const removeLang = (code: string) => {
    if (code === 'en') return;
    setLanguages(enabledLangs.filter(l => l !== code));
  };

  const addLang = (code: string) => {
    if (enabledLangs.includes(code)) return;
    setLanguages([...enabledLangs, code]);
    setSearch('');
    setPickerOpen(false);
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
      subtitle={<>phrase generation languages for <span className="text-violet-400 font-mono">{ns}</span></>}
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
                    ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
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
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs border border-dashed border-zinc-600 text-zinc-500 hover:text-white hover:border-zinc-400 transition-colors"
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
                  className="w-full bg-transparent text-sm text-white placeholder-zinc-600 focus:outline-none px-1"
                />
              </div>
              <div className="max-h-48 overflow-y-auto">
                {filteredLangs.slice(0, 40).map(code => (
                  <button
                    key={code}
                    onClick={() => addLang(code)}
                    className="w-full text-left px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center gap-2"
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
        </div>

      </div>
    </Page>
  );
}
