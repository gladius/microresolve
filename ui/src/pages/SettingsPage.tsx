import { useState, useEffect } from 'react';
import { api } from '@/api/client';
import SidebarLayout, { type SidebarItem } from '@/components/SidebarLayout';
import Page from '@/components/Page';
import { useAppStore } from '@/store';

export default function SettingsPage() {
  const [section, setSection] = useState('llm');

  const items: SidebarItem[] = [
    { id: 'llm', label: 'LLM / AI' },
    { id: 'routing', label: 'Routing' },
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
          {section === 'llm' && <LLMSection />}
          {section === 'routing' && <RoutingSection />}
          {section === 'data' && <DataSection />}
        </div>
      </SidebarLayout>
    </Page>
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
          Powers: phrase generation, auto-review, auto-learn.
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

function RoutingSection() {
  const { settings, setReviewSkipThreshold } = useAppStore();
  const val = settings.reviewSkipThreshold ?? 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-white">Routing</h2>
        <p className="text-xs text-zinc-500 mt-1">Controls how routing decisions trigger LLM review.</p>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-white">Auto-review confidence threshold</h3>
            <p className="text-xs text-zinc-500 mt-0.5">
              When routing confidence exceeds this, Turn 1 LLM judge is skipped — routing is trusted as correct.
              Set to 0 to always run the judge (default, safest). Higher values reduce LLM cost but skip review on confident-but-wrong routings.
            </p>
          </div>
          <span className="text-sm font-mono text-violet-400 ml-4 shrink-0">
            {val === 0 ? 'off' : `${Math.round(val * 100)}%`}
          </span>
        </div>
        <input
          type="range"
          min={0} max={1} step={0.05}
          value={val}
          onChange={e => setReviewSkipThreshold(parseFloat(e.target.value))}
          className="w-full accent-violet-500"
        />
        <div className="flex justify-between text-xs text-zinc-600">
          <span>0% — always judge (max LLM cost)</span>
          <span>100% — never judge (zero LLM cost)</span>
        </div>
        {val > 0 && (
          <div className="text-xs text-amber-400/80 bg-amber-400/5 border border-amber-400/20 rounded p-2">
            Queries where top intent scores &ge; {Math.round(val * 100)}% will skip Turn 1 and be treated as correctly routed. Misrouted high-confidence queries will not be auto-corrected.
          </div>
        )}
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
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">Data</h2>
      <p className="text-xs text-zinc-500">Export, import, or reset router state.</p>
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
        <button
          onClick={() => { setShowClearModal(true); setClearInput(''); }}
          className="text-xs text-red-500/70 hover:text-red-400 px-3 py-1.5 border border-red-500/20 rounded"
        >
          Clear All Data
        </button>
      </div>

      {showClearModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-red-500/30 rounded-xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-red-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
              </svg>
              <h3 className="text-base font-semibold text-white">Clear All Data</h3>
            </div>
            <p className="text-sm text-zinc-400">
              This will permanently delete <strong className="text-white">all namespaces, intents, training data, and query logs</strong>. The server will be reset to a clean state with only the default namespace.
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
                className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white font-mono placeholder-zinc-600 focus:outline-none focus:border-red-500"
              />
            </div>
            <div className="flex gap-2 justify-end pt-1">
              <button
                onClick={() => setShowClearModal(false)}
                className="px-4 py-2 text-sm text-zinc-400 hover:text-white"
              >
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

