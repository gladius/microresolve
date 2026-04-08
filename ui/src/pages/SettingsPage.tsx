import { useState, useEffect } from 'react';
import { useAppStore, type AppMode } from '@/store';
import { api } from '@/api/client';
import SidebarLayout, { type SidebarItem } from '@/components/SidebarLayout';

export default function SettingsPage() {
  const [section, setSection] = useState('mode');

  const items: SidebarItem[] = [
    { id: 'mode', label: 'Router Mode' },
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
        {section === 'mode' && <ModeSection />}
        {section === 'review_mode' && <ReviewModeSection />}
        {section === 'llm' && <LLMSection />}
        {section === 'languages' && <LanguagesSection />}
        {section === 'data' && <DataSection />}
      </div>
    </SidebarLayout>
  );
}

function ModeSection() {
  const { settings, setMode } = useAppStore();

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">Router Mode</h2>
      <div className="flex gap-3">
        {([
          { mode: 'production' as AppMode, label: 'Production', desc: 'Fast routing only.', color: 'emerald' },
          { mode: 'learn' as AppMode, label: 'Learn', desc: 'Every query is reviewed by LLM. Suggestions appear inline.', color: 'amber' },
        ]).map(({ mode, label, desc, color }) => (
          <button
            key={mode}
            onClick={() => setMode(mode)}
            className={`flex-1 p-4 rounded-lg border text-left transition-colors ${
              settings.mode === mode
                ? `border-${color}-400/50 bg-${color}-400/10`
                : 'border-zinc-800 bg-zinc-900 hover:border-zinc-700'
            }`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className={`w-2 h-2 rounded-full ${settings.mode === mode ? `bg-${color}-400` : 'bg-zinc-600'}`} />
              <span className={`font-semibold text-sm ${settings.mode === mode ? `text-${color}-400` : 'text-zinc-400'}`}>
                {label}
              </span>
            </div>
            <p className="text-xs text-zinc-500">{desc}</p>
          </button>
        ))}
      </div>
      <p className="text-xs text-zinc-600">
        LLM features require <code className="text-violet-400">ANTHROPIC_API_KEY</code> in the server's <code className="text-violet-400">.env</code> file.
      </p>
    </div>
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
    { id: 'manual', label: 'Manual', desc: 'Failed queries queued for human review. You decide the correct intent and add seeds.' },
    { id: 'auto_review', label: 'Auto-Review', desc: 'LLM suggests fixes. You approve or reject in the Review tab. Nothing applied without your confirmation.' },
    { id: 'auto_learn', label: 'Auto-Learn', desc: 'LLM detects and fixes failures automatically. Fast but risky — LLM mistakes get applied immediately.' },
  ];

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Review Mode</h2>
        <p className="text-xs text-zinc-500 mt-1">How failed queries are handled when reported from connected libraries.</p>
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
        Auto-Review and Auto-Learn require <code className="text-violet-400">ANTHROPIC_API_KEY</code>.
      </p>
    </div>
  );
}

function LLMSection() {
  const [status, setStatus] = useState<{ configured: boolean; provider: string; model: string; url: string } | null>(null);

  useEffect(() => {
    api.getLLMStatus().then(setStatus).catch(() => {});
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">LLM / AI Configuration</h2>
        <p className="text-xs text-zinc-500 mt-1">
          Powers: seed generation, auto-review, auto-learn, discovery naming.
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
            <div>Provider: <span className="text-white font-mono">{status.provider}</span></div>
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
        <p className="text-xs text-zinc-600">
          After editing <code className="text-violet-400">.env</code>, restart the server for changes to take effect.
          API keys are stored server-side only — never sent to the browser.
        </p>
      </div>

      {/* What LLM is used for */}
      <div className="space-y-2">
        <h3 className="text-xs text-zinc-500 font-semibold uppercase">Features using LLM</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {[
            { feature: 'Seed generation', desc: 'Generate diverse seed phrases for intents' },
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
  const [languages, setLanguages] = useState<Record<string, string>>({});
  const [enabledLangs, setEnabledLangs] = useState<Set<string>>(new Set(['en']));
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    api.getLanguages().then(setLanguages).catch(() => {});
    try {
      const saved = localStorage.getItem('asv_languages');
      if (saved) setEnabledLangs(new Set(JSON.parse(saved)));
    } catch { /* */ }
  }, []);

  const toggleLang = (code: string) => {
    if (code === 'en') return;
    setEnabledLangs(prev => {
      const next = new Set(prev);
      next.has(code) ? next.delete(code) : next.add(code);
      return next;
    });
    setDirty(true);
  };

  const save = () => {
    localStorage.setItem('asv_languages', JSON.stringify(Array.from(enabledLangs)));
    setDirty(false);
  };

  const commonLangs = ['en', 'es', 'fr', 'de', 'pt', 'it', 'nl', 'ja', 'ko', 'zh', 'ar', 'hi'];
  const sortedLangs = Object.keys(languages).sort((a, b) => {
    const ai = commonLangs.indexOf(a);
    const bi = commonLangs.indexOf(b);
    if (ai >= 0 && bi >= 0) return ai - bi;
    if (ai >= 0) return -1;
    if (bi >= 0) return 1;
    return (languages[a] || '').localeCompare(languages[b] || '');
  });

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">Supported Languages ({enabledLangs.size})</h2>
        <p className="text-xs text-zinc-500 mt-1">
          Languages available for AI seed generation. LLM quality varies — human review recommended for non-English seeds.
        </p>
      </div>
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 max-h-64 overflow-y-auto">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {sortedLangs.map(code => (
            <label key={code} className="inline-flex items-center gap-2 text-sm text-zinc-300 cursor-pointer py-0.5 hover:text-white">
              <input
                type="checkbox"
                checked={enabledLangs.has(code)}
                onChange={() => toggleLang(code)}
                disabled={code === 'en'}
                className="accent-violet-500"
              />
              <span className={enabledLangs.has(code) ? 'text-white' : 'text-zinc-500'}>{languages[code]}</span>
              <span className="text-[9px] text-zinc-600 uppercase">{code}</span>
            </label>
          ))}
        </div>
      </div>
      {dirty && (
        <button onClick={save} className="px-4 py-1.5 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded">
          Save Languages
        </button>
      )}
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
