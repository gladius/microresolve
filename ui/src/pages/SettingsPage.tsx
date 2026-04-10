import { useState, useEffect } from 'react';
import { useAppStore } from '@/store';
import { api } from '@/api/client';
import SidebarLayout, { type SidebarItem } from '@/components/SidebarLayout';

export default function SettingsPage() {
  const [section, setSection] = useState('review_mode');

  const items: SidebarItem[] = [
    { id: 'review_mode', label: 'Review Mode' },
    { id: 'llm', label: 'LLM / AI' },
    { id: 'languages', label: 'Languages' },
    { id: 'data', label: 'Data' },
  ];

  return (
    <SidebarLayout
      title="Settings"
      items={items}
      selected={section}
      onSelect={setSection}
    >
      <div className="p-5 max-w-2xl">
        {section === 'review_mode' && <ReviewModeSection />}
        {section === 'llm' && <LLMSection />}
        {section === 'languages' && <LanguagesSection />}
        {section === 'data' && <DataSection />}
      </div>
    </SidebarLayout>
  );
}

function ReviewModeSection() {
  const [mode, setMode] = useState('manual');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getReviewMode().then(d => { setMode(d.mode); setLoading(false); }).catch(() => setLoading(false));
  }, []);

  const handleChange = async (newMode: string) => {
    setMode(newMode);
    await api.setReviewMode(newMode);
  };

  if (loading) return <div className="text-zinc-500 text-sm">Loading...</div>;

  const modes = [
    { id: 'manual', label: 'Manual', desc: 'Review failures manually. You analyze with AI and apply fixes when ready.', color: 'emerald' },
    { id: 'auto', label: 'Auto', desc: 'Every query is reviewed by LLM automatically. Fixes applied immediately. System learns continuously but costs LLM tokens on every query.', color: 'red' },
  ];

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Review Mode</h2>
        <p className="text-xs text-zinc-500 mt-1">Controls how the system learns from queries.</p>
      </div>
      <div className="space-y-2">
        {modes.map(m => (
          <button
            key={m.id}
            onClick={() => handleChange(m.id)}
            className={`w-full text-left p-3 rounded-lg border transition-colors ${
              mode === m.id
                ? 'border-violet-400/50 bg-violet-400/10'
                : 'border-zinc-800 bg-zinc-900 hover:border-zinc-700'
            }`}
          >
            <div className="flex items-center gap-2 mb-0.5">
              <span className={`w-2 h-2 rounded-full ${mode === m.id ? 'bg-violet-400' : 'bg-zinc-600'}`} />
              <span className={`font-semibold text-sm ${mode === m.id ? 'text-violet-400' : 'text-zinc-400'}`}>{m.label}</span>
            </div>
            <p className="text-xs text-zinc-500 pl-4">{m.desc}</p>
          </button>
        ))}
      </div>
      <p className="text-xs text-zinc-600">
        Auto mode requires <code className="text-violet-400">LLM_API_KEY</code> in the server's <code className="text-violet-400">.env</code> file.
      </p>
    </div>
  );
}

function LLMSection() {
  const [status, setStatus] = useState<{ configured: boolean; model: string; url: string } | null>(null);

  useEffect(() => {
    api.getLLMStatus().then(setStatus).catch(() => {});
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">LLM / AI Configuration</h2>
        <p className="text-xs text-zinc-500 mt-1">
          Powers: phrase generation, auto-review, auto-learn, discovery naming.
        </p>
      </div>

      {/* Status */}
      <div className={`rounded-lg border p-4 ${status?.configured ? 'border-emerald-800 bg-emerald-900/10' : 'border-red-800 bg-red-900/10'}`}>
        <div className="flex items-center gap-2 mb-2">
          <span className={`w-2 h-2 rounded-full ${status?.configured ? 'bg-emerald-400' : 'bg-red-400'}`} />
          <span className={`text-sm font-medium ${status?.configured ? 'text-emerald-400' : 'text-red-400'}`}>
            {status?.configured ? 'Connected' : 'Not Configured'}
          </span>
        </div>
        {status?.configured && (
          <div className="space-y-1 text-xs text-zinc-400">
            <div>Model: <span className="text-white font-mono">{status.model}</span></div>
            <div>URL: <span className="text-white font-mono">{status.url}</span></div>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="space-y-3">
        <h3 className="text-xs text-zinc-500 font-semibold uppercase">Configuration</h3>
        <p className="text-xs text-zinc-500">
          LLM configuration is managed via the server's <code className="text-violet-400">.env</code> file.
          Edit the file and restart the server to apply changes.
        </p>
        <div className="bg-zinc-800 rounded-lg p-4 font-mono text-xs space-y-2">
          <div className="text-zinc-500"># Anthropic (default)</div>
          <div><span className="text-cyan-400">LLM_API_URL</span><span className="text-zinc-500">=</span><span className="text-amber-300">https://api.anthropic.com/v1/messages</span></div>
          <div><span className="text-cyan-400">LLM_API_KEY</span><span className="text-zinc-500">=</span><span className="text-amber-300">sk-ant-...</span></div>
          <div><span className="text-cyan-400">LLM_MODEL</span><span className="text-zinc-500">=</span><span className="text-amber-300">claude-haiku-4-5-20251001</span></div>
          <div className="text-zinc-500 mt-3"># OpenAI</div>
          <div><span className="text-zinc-600">LLM_API_URL=https://api.openai.com/v1/chat/completions</span></div>
          <div><span className="text-zinc-600">LLM_API_KEY=sk-...</span></div>
          <div><span className="text-zinc-600">LLM_MODEL=gpt-4o-mini</span></div>
          <div className="text-zinc-500 mt-3"># Local (Ollama)</div>
          <div><span className="text-zinc-600">LLM_API_URL=http://localhost:11434/v1/chat/completions</span></div>
          <div><span className="text-zinc-600">LLM_API_KEY=unused</span></div>
          <div><span className="text-zinc-600">LLM_MODEL=llama3</span></div>
        </div>
      </div>

      {/* What LLM is used for */}
      <div className="space-y-2">
        <h3 className="text-xs text-zinc-500 font-semibold uppercase">Features using LLM</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {[
            { feature: 'Phrase generation', desc: 'Generate diverse training phrases for intents' },
            { feature: 'Auto-review', desc: 'LLM suggests fixes for failed queries' },
            { feature: 'Auto-learn', desc: 'LLM fixes failures automatically' },
            { feature: 'Discovery naming', desc: 'LLM names discovered intent clusters' },
          ].map(f => (
            <div key={f.feature} className="bg-zinc-800 rounded p-2">
              <div className={`font-medium ${status?.configured ? 'text-white' : 'text-zinc-500'}`}>{f.feature}</div>
              <div className="text-zinc-500 mt-0.5">{f.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function LanguagesSection() {
  const { settings, setLanguages } = useAppStore();
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
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Languages</h2>
        <p className="text-xs text-zinc-500 mt-1">
          Languages for AI phrase generation. Phrases in each selected language are generated on import.
        </p>
      </div>

      {/* Active language chips */}
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

        {/* Add language button */}
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

      {/* Picker dropdown */}
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

      <p className="text-xs text-zinc-600">
        English is always included and cannot be removed. LLM quality varies for non-English phrases — review recommended.
      </p>
    </div>
  );
}

function DataSection() {
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">Data</h2>
      <p className="text-xs text-zinc-500">Export, import, or reset router state.</p>
      <div className="flex gap-3">
        <button
          onClick={async () => {
            const data = await api.exportState();
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = 'asv-export.json'; a.click();
            URL.revokeObjectURL(url);
          }}
          className="text-xs text-violet-400 hover:text-violet-300 px-3 py-1.5 border border-violet-400/30 rounded"
        >
          Export State
        </button>
        <label className="text-xs text-violet-400 hover:text-violet-300 px-3 py-1.5 border border-violet-400/30 rounded cursor-pointer">
          Import State
          <input type="file" accept=".json" className="hidden" onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            const text = await file.text();
            try {
              await api.importState(text);
              alert('Imported successfully');
              window.location.reload();
            } catch { alert('Import failed'); }
          }} />
        </label>
        <button
          onClick={async () => {
            if (!confirm('Reset all intents to demo defaults?')) return;
            await api.reset();
            await api.loadDefaults();
            alert('Reset to defaults');
            window.location.reload();
          }}
          className="text-xs text-red-400/70 hover:text-red-400 px-3 py-1.5 border border-red-400/20 rounded"
        >
          Reset to Defaults
        </button>
      </div>
    </div>
  );
}
