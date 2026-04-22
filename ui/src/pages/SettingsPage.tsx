import { useState, useEffect } from 'react';
import { api } from '@/api/client';
import SidebarLayout, { type SidebarItem } from '@/components/SidebarLayout';
import Page from '@/components/Page';
import { useAppStore, type ThemeMode } from '@/store';

export default function SettingsPage() {
  const [section, setSection] = useState('appearance');

  const items: SidebarItem[] = [
    { id: 'appearance', label: 'Appearance' },
    { id: 'llm', label: 'LLM / AI' },
    { id: 'data', label: 'Data' },
  ];

  return (
    <Page title="Settings" fullscreen>
      <SidebarLayout
        title="Settings"
        items={items}
        selected={section}
        onSelect={setSection}
      >
        <div className="p-5 max-w-2xl">
          {section === 'appearance' && <AppearanceSection />}
          {section === 'llm' && <LLMSection />}
          {section === 'data' && <DataSection />}
        </div>
      </SidebarLayout>
    </Page>
  );
}

function AppearanceSection() {
  const { settings, setTheme } = useAppStore();
  const current = settings.theme;

  const options: { id: ThemeMode; label: string; desc: string; icon: string }[] = [
    { id: 'dark',   label: 'Dark',   desc: 'Always dark',             icon: '◑' },
    { id: 'light',  label: 'Light',  desc: 'Always light',            icon: '○' },
    { id: 'system', label: 'System', desc: 'Follows OS preference',   icon: '◎' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100">Appearance</h2>
        <p className="text-xs text-zinc-500 mt-1">Choose how μResolve looks.</p>
      </div>

      <div>
        <div className="text-xs text-zinc-500 font-semibold uppercase tracking-wide mb-3">Theme</div>
        <div className="flex gap-3">
          {options.map(opt => (
            <button
              key={opt.id}
              onClick={() => setTheme(opt.id)}
              className={`flex-1 flex flex-col items-center gap-2 px-4 py-4 rounded-xl border text-sm transition-colors ${
                current === opt.id
                  ? 'border-violet-500 bg-violet-500/10 text-zinc-100'
                  : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200'
              }`}
            >
              <span className="text-2xl leading-none">{opt.icon}</span>
              <span className="font-medium text-sm">{opt.label}</span>
              <span className="text-[10px] text-zinc-500 text-center leading-tight">{opt.desc}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function LLMSection() {
  const [status, setStatus] = useState<{ configured: boolean; model: string; url: string } | null>(null);

  useEffect(() => {
    api.getLLMStatus().then(setStatus).catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100">LLM / AI Configuration</h2>
        <p className="text-xs text-zinc-500 mt-1">
          Powers phrase generation, auto-review, and auto-learn.
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
            <div>Model: <span className="text-zinc-100 font-mono">{status.model}</span></div>
            <div>URL: <span className="text-zinc-100 font-mono">{status.url}</span></div>
          </div>
        )}
      </div>

      {/* .env instructions */}
      <div className="space-y-3">
        <h3 className="text-xs text-zinc-500 font-semibold uppercase">Configuration</h3>
        <p className="text-xs text-zinc-500">
          Set these in the server's <code className="text-violet-400">.env</code> file and restart the server to apply.
          Any OpenAI-compatible provider works — Anthropic, OpenAI, Gemini, Groq, or local Ollama.
        </p>
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 font-mono text-xs space-y-1.5">
          <div><span className="text-cyan-400">LLM_API_KEY</span><span className="text-zinc-600">=</span><span className="text-amber-300">sk-ant-...</span></div>
          <div><span className="text-cyan-400">LLM_MODEL</span><span className="text-zinc-600">=</span><span className="text-amber-300">claude-haiku-4-5-20251001</span></div>
          <div className="text-zinc-600 text-[10px] pt-1">
            # Optional — defaults to Anthropic if omitted<br />
            # LLM_API_URL=https://api.openai.com/v1/chat/completions<br />
            # LLM_PROVIDER=openai
          </div>
        </div>
      </div>
    </div>
  );
}

function DataSection() {
  const [showClearModal, setShowClearModal] = useState(false);
  const [clearInput, setClearInput] = useState('');
  const [clearing, setClearing] = useState(false);

  const confirmText = 'delete all';

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-zinc-100">Data</h2>
        <p className="text-xs text-zinc-500 mt-1">Export, import, or reset router state.</p>
      </div>

      <div className="flex gap-3 flex-wrap">
        <button
          onClick={async () => {
            const data = await api.exportState();
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = 'microresolve-export.json'; a.click();
            URL.revokeObjectURL(url);
          }}
          className="text-xs text-violet-400 hover:text-violet-300 px-3 py-1.5 border border-violet-400/30 rounded transition-colors"
        >
          Export State
        </button>
        <label className="text-xs text-violet-400 hover:text-violet-300 px-3 py-1.5 border border-violet-400/30 rounded cursor-pointer transition-colors">
          Import State
          <input type="file" accept=".json" className="hidden" onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            const text = await file.text();
            try {
              await api.importState(text);
              alert('Imported successfully');
              window.location.reload();
            } catch { alert('Import failed — select a valid .json export file'); }
          }} />
        </label>
      </div>

      {/* Danger zone */}
      <div className="border border-red-500/20 rounded-xl p-4 space-y-3">
        <div className="text-xs text-red-400/70 font-semibold uppercase tracking-wide">Danger zone</div>
        <div className="flex gap-3 flex-wrap">
          <button
            onClick={() => { setShowClearModal(true); setClearInput(''); }}
            className="text-xs text-red-500/80 hover:text-red-400 px-3 py-1.5 border border-red-500/30 rounded transition-colors"
          >
            Clear All Data
          </button>
        </div>
      </div>

      {showClearModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-red-500/30 rounded-xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-red-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
              </svg>
              <h3 className="text-base font-semibold text-zinc-100">Clear All Data</h3>
            </div>
            <p className="text-sm text-zinc-400">
              This will permanently delete <strong className="text-zinc-100">all workspaces, intents, training data, and query logs</strong>. The server resets to a clean state with only the default workspace.
            </p>
            <p className="text-xs text-zinc-500">This action cannot be undone.</p>
            <div>
              <label className="text-xs text-zinc-400 block mb-1.5">
                Type <span className="font-mono text-red-400">{confirmText}</span> to confirm
              </label>
              <input
                autoFocus
                value={clearInput}
                onChange={e => setClearInput(e.target.value)}
                onKeyDown={async e => {
                  if (e.key === 'Enter' && clearInput === confirmText) {
                    setClearing(true);
                    try { await api.clearAllData(); window.location.reload(); } catch { setClearing(false); }
                  }
                }}
                placeholder="delete all"
                className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-zinc-100 font-mono placeholder-zinc-600 focus:outline-none focus:border-red-500"
              />
            </div>
            <div className="flex gap-2 justify-end pt-1">
              <button onClick={() => setShowClearModal(false)} className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-100">
                Cancel
              </button>
              <button
                onClick={async () => {
                  setClearing(true);
                  try { await api.clearAllData(); window.location.reload(); } catch { setClearing(false); }
                }}
                disabled={clearInput !== confirmText || clearing}
                className="px-5 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-500 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {clearing ? 'Clearing…' : 'Clear All Data'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
