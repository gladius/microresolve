import { useState, useEffect } from 'react';
import { api, type NamespaceModel } from '@/api/client';
import { useAppStore } from '@/store';
import Page from '@/components/Page';

export default function ModelsPage() {
  const { settings } = useAppStore();
  const ns = settings.selectedNamespaceId;

  const [models,     setModels]     = useState<NamespaceModel[]>([]);
  const [newLabel,   setNewLabel]   = useState('');
  const [newModelId, setNewModelId] = useState('');
  const [saving,     setSaving]     = useState(false);
  const [loading,    setLoading]    = useState(true);

  useEffect(() => {
    api.getNsModels().then(m => { setModels(m); setLoading(false); }).catch(() => setLoading(false));
  }, [ns]);

  const save = async (updated: NamespaceModel[]) => {
    setSaving(true);
    try { await api.setNsModels(updated); setModels(updated); }
    finally { setSaving(false); }
  };

  const add = async () => {
    const label    = newLabel.trim();
    const model_id = newModelId.trim();
    if (!label || !model_id) return;
    await save([...models, { label, model_id }]);
    setNewLabel(''); setNewModelId('');
  };

  const remove = (i: number) => save(models.filter((_, idx) => idx !== i));

  return (
    <Page
      title="Models"
      subtitle={<>routing targets for <span className="text-violet-400 font-mono">{ns}</span></>}
      size="sm"
    >
      <div className="space-y-6">

        {/* Explainer */}
        <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 text-xs text-zinc-500 leading-relaxed space-y-1">
          <div className="text-zinc-300 font-medium text-sm">Per-workspace model registry</div>
          <p>
            Define named models for this workspace. Each intent can then specify which model to route to
            via the <span className="text-violet-400 font-mono">target</span> field in its Details tab.
            Useful when different intents need different speed/quality tradeoffs — e.g. fast model for
            simple queries, a smarter model for complex ones.
          </p>
          <p className="text-zinc-600">
            Models are scoped to the active workspace. Switch workspaces from the sidebar to manage models for others.
          </p>
        </div>

        {/* Model list */}
        {loading ? (
          <div className="text-xs text-zinc-600 py-4 text-center">Loading...</div>
        ) : (
          <div className="space-y-2">
            {models.length === 0 && (
              <div className="text-xs text-zinc-600 py-4 text-center border border-dashed border-zinc-800 rounded-lg">
                No models defined yet. Add one below.
              </div>
            )}
            {models.map((m, i) => (
              <div key={i} className="flex items-center gap-3 bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3">
                <div className="flex-1 min-w-0">
                  <span className="text-sm text-white font-medium">{m.label}</span>
                  <span className="text-zinc-600 mx-2">—</span>
                  <span className="text-sm text-zinc-400 font-mono">{m.model_id}</span>
                </div>
                <button
                  onClick={() => remove(i)}
                  disabled={saving}
                  className="text-zinc-600 hover:text-red-400 transition-colors text-sm px-1 disabled:opacity-40"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Add form */}
        <div className="border border-zinc-800 rounded-xl p-4 space-y-3">
          <div className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">Add model</div>
          <div className="flex gap-2">
            <input
              value={newLabel}
              onChange={e => setNewLabel(e.target.value)}
              placeholder="Label — e.g. Fast"
              className="w-32 bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:border-violet-500 focus:outline-none"
            />
            <input
              value={newModelId}
              onChange={e => setNewModelId(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && add()}
              placeholder="Model ID — e.g. claude-haiku-4-5-20251001"
              className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white font-mono placeholder-zinc-600 focus:border-violet-500 focus:outline-none"
            />
            <button
              onClick={add}
              disabled={!newLabel.trim() || !newModelId.trim() || saving}
              className="px-4 py-2 text-sm bg-violet-600 hover:bg-violet-500 text-white rounded-lg disabled:opacity-40 transition-colors"
            >
              {saving ? '…' : 'Add'}
            </button>
          </div>
          <p className="text-[10px] text-zinc-600">
            Label is shown in the intent Details tab dropdown. Model ID is passed through to your routing consumer.
          </p>
        </div>

      </div>
    </Page>
  );
}
