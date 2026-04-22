import { useState, useEffect, useCallback } from 'react';
import { api } from '@/api/client';
import Page from '@/components/Page';

type BuiltinPattern = {
  label: string;
  display_name: string;
  description: string;
  recommended: boolean;
  regex_patterns: string[];
  context_phrases: string[];
};
type Category = { category: string; patterns: BuiltinPattern[] };

type CustomEntity = {
  label: string;
  display_name: string;
  description: string;
  regex_patterns: string[];
  context_phrases: string[];
  examples: string[];
  source: string;
};

type Distilled = {
  label: string;
  regex_patterns: string[];
  context_phrases: string[];
  examples: string[];
  notes?: string;
};

export default function EntitiesPage() {
  const [categories, setCategories]   = useState<Category[]>([]);
  const [enabled, setEnabled]         = useState(false);
  const [enabledLabels, setEnabledLabels] = useState<Set<string>>(new Set());
  const [custom, setCustom]           = useState<CustomEntity[]>([]);
  const [loading, setLoading]         = useState(true);
  const [saving,  setSaving]          = useState(false);

  // Test pane
  const [testText,    setTestText]    = useState('my email is alice@example.com and ssn 123-45-6789');
  const [testResult,  setTestResult]  = useState<{ labels: string[]; masked: string; latency_us: number } | null>(null);

  // Custom entity creation
  const [showCustomDialog, setShowCustomDialog] = useState(false);
  const [customName,  setCustomName]  = useState('');
  const [customDesc,  setCustomDesc]  = useState('');
  const [customExamples, setCustomExamples] = useState('');
  const [distilling,  setDistilling]  = useState(false);
  const [distilled,   setDistilled]   = useState<Distilled | null>(null);
  const [distillRepairs, setDistillRepairs] = useState<string[]>([]);
  const [distillError, setDistillError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [b, c] = await Promise.all([
        api.listBuiltinEntities(),
        api.getEntityConfig(),
      ]);
      setCategories(b.categories);
      setEnabled(c.enabled);
      setEnabledLabels(new Set(c.config?.enabled_builtins ?? []));
      setCustom(c.config?.custom ?? []);
    } catch (e) {
      console.error('failed to load entities', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const toggleEnabled = async (e: boolean) => {
    setSaving(true);
    try {
      await api.updateEntityConfig({ enabled: e });
      await refresh();
    } catch (err) { alert('Failed: ' + err); }
    finally { setSaving(false); }
  };

  const togglePattern = (label: string) => {
    const next = new Set(enabledLabels);
    if (next.has(label)) next.delete(label); else next.add(label);
    setEnabledLabels(next);
  };

  const saveSelection = async () => {
    setSaving(true);
    try {
      await api.updateEntityConfig({ enabled_builtins: Array.from(enabledLabels) });
      await refresh();
    } catch (err) { alert('Failed: ' + err); }
    finally { setSaving(false); }
  };

  const applyRecommended = async () => {
    setSaving(true);
    try {
      await api.updateEntityConfig({ use_recommended: true });
      await refresh();
    } catch (err) { alert('Failed: ' + err); }
    finally { setSaving(false); }
  };

  const selectAllInCategory = (cat: Category, on: boolean) => {
    const next = new Set(enabledLabels);
    for (const p of cat.patterns) { if (on) next.add(p.label); else next.delete(p.label); }
    setEnabledLabels(next);
  };

  const runTest = async () => {
    if (!testText.trim()) return;
    try {
      const [det, msk] = await Promise.all([
        api.detectEntities(testText),
        api.maskEntities(testText),
      ]);
      setTestResult({ labels: det.labels, masked: msk.masked, latency_us: det.latency_us });
    } catch (err) {
      setTestResult({ labels: [], masked: 'error: ' + err, latency_us: 0 });
    }
  };

  const openCustomDialog = () => {
    setCustomName(''); setCustomDesc(''); setCustomExamples('');
    setDistilled(null); setDistillRepairs([]); setDistillError(null);
    setShowCustomDialog(true);
  };

  const distill = async () => {
    if (!customName.trim() || !customDesc.trim()) {
      setDistillError('name and description required');
      return;
    }
    setDistilling(true); setDistillError(null);
    try {
      const examples = customExamples.split('\n').map(s => s.trim()).filter(Boolean);
      const r = await api.distillEntity({
        name: customName.trim(),
        description: customDesc.trim(),
        examples: examples.length > 0 ? examples : undefined,
      });
      setDistilled(r.proposed);
      setDistillRepairs(r.repairs);
      if (!r.usable) {
        setDistillError('No usable regex survived auto-repair. Try rephrasing the description.');
      }
    } catch (err) {
      setDistillError('Distillation failed: ' + (err instanceof Error ? err.message : err));
    } finally { setDistilling(false); }
  };

  const saveDistilled = async () => {
    if (!distilled) return;
    setSaving(true);
    try {
      await api.saveCustomEntity({
        label: distilled.label.toUpperCase().replace(/[^A-Z0-9_]/g, '_'),
        display_name: customName.trim(),
        description: customDesc.trim() + (distilled.notes ? ` (${distilled.notes})` : ''),
        regex_patterns: distilled.regex_patterns,
        context_phrases: distilled.context_phrases,
        examples: distilled.examples,
        source: 'llm_distillation',
      });
      setShowCustomDialog(false);
      await refresh();
    } catch (err) { alert('Save failed: ' + err); }
    finally { setSaving(false); }
  };

  const deleteCustom = async (label: string) => {
    if (!confirm(`Delete custom entity "${label}"?`)) return;
    try {
      await api.deleteCustomEntity(label);
      await refresh();
    } catch (err) { alert('Delete failed: ' + err); }
  };

  if (loading) {
    return <Page title="Entities" subtitle="PII / credential / identifier detection"><div className="text-xs text-zinc-500 text-center py-12">Loading…</div></Page>;
  }

  return (
    <Page title="Entities" subtitle="PII, credential, and identifier detection — per workspace">
      <div className="max-w-4xl mx-auto p-6 space-y-6">

        {/* Top bar — enable / preset / save */}
        <div className="flex items-center justify-between bg-zinc-800/50 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <button
              type="button"
              role="switch"
              aria-checked={enabled}
              onClick={() => toggleEnabled(!enabled)}
              disabled={saving}
              className={`relative w-12 h-7 rounded-full transition-colors ${enabled ? 'bg-emerald-500' : 'bg-zinc-600'}`}
            >
              <span className={`absolute top-1 left-1 w-5 h-5 rounded-full bg-white shadow transition-transform ${enabled ? 'translate-x-5' : ''}`} />
            </button>
            <div>
              <div className="text-sm text-zinc-100">Entity layer {enabled ? 'enabled' : 'disabled'}</div>
              <div className="text-xs text-zinc-500">
                {enabled ? `${enabledLabels.size} built-in + ${custom.length} custom active` : 'Off — no entity tokens added to routing'}
              </div>
            </div>
          </div>
          {enabled && (
            <div className="flex gap-2">
              <button onClick={applyRecommended} disabled={saving}
                className="text-xs px-3 py-1.5 bg-zinc-700 text-zinc-200 rounded hover:bg-zinc-600 disabled:opacity-50">
                Apply recommended
              </button>
              <button onClick={saveSelection} disabled={saving}
                className="text-xs px-3 py-1.5 bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-50">
                {saving ? 'Saving…' : 'Save selection'}
              </button>
            </div>
          )}
        </div>

        {/* Test pane */}
        {enabled && (
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-2">
            <label className="text-[10px] text-zinc-500 uppercase font-semibold block">Test against text</label>
            <textarea value={testText} onChange={e => setTestText(e.target.value)}
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-violet-500"
              rows={2} />
            <div className="flex justify-end">
              <button onClick={runTest}
                className="text-xs px-3 py-1 bg-violet-600 text-white rounded hover:bg-violet-500">Test</button>
            </div>
            {testResult && (
              <div className="text-xs space-y-1 pt-2 border-t border-zinc-800">
                <div className="text-zinc-400">
                  <span className="text-zinc-500">labels:</span>{' '}
                  {testResult.labels.length > 0
                    ? testResult.labels.map(l => <span key={l} className="inline-block px-1.5 py-0.5 mr-1 bg-violet-500/20 text-violet-300 rounded">{l}</span>)
                    : <span className="text-zinc-600">none</span>}
                  <span className="text-zinc-600 ml-2">{testResult.latency_us}µs</span>
                </div>
                <div className="text-zinc-400 font-mono">
                  <span className="text-zinc-500">masked:</span> {testResult.masked}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Built-in patterns by category */}
        {enabled && categories.map(cat => {
          const enabledInCat = cat.patterns.filter(p => enabledLabels.has(p.label)).length;
          const allOn = enabledInCat === cat.patterns.length;
          return (
            <div key={cat.category} className="border border-zinc-800 rounded-lg overflow-hidden">
              <div className="bg-zinc-800/50 px-4 py-2 flex items-center justify-between">
                <div className="text-sm text-zinc-100 font-medium">{cat.category}</div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-zinc-500">{enabledInCat}/{cat.patterns.length} on</span>
                  <button onClick={() => selectAllInCategory(cat, !allOn)}
                    className="text-[10px] text-zinc-400 hover:text-zinc-200 underline">
                    {allOn ? 'select none' : 'select all'}
                  </button>
                </div>
              </div>
              <div className="divide-y divide-zinc-800/50">
                {cat.patterns.map(p => {
                  const on = enabledLabels.has(p.label);
                  return (
                    <label key={p.label}
                      className={`flex items-start gap-3 px-4 py-2.5 cursor-pointer transition-colors hover:bg-zinc-900/50 ${on ? '' : 'opacity-60'}`}>
                      <input type="checkbox" checked={on} onChange={() => togglePattern(p.label)}
                        className="mt-1 accent-violet-500" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-zinc-200">{p.display_name}</span>
                          <span className="text-[10px] text-zinc-600 font-mono">{p.label}</span>
                          {p.recommended && <span className="text-[9px] uppercase tracking-wide text-emerald-400 bg-emerald-500/10 px-1 py-0.5 rounded">recommended</span>}
                        </div>
                        <div className="text-xs text-zinc-500 truncate">{p.description}</div>
                        <div className="text-[10px] text-zinc-600 font-mono mt-0.5 truncate">
                          {p.regex_patterns.length > 0 ? p.regex_patterns[0] : <span className="text-zinc-700">(context-only — no regex)</span>}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>
          );
        })}

        {/* Custom entities */}
        {enabled && (
          <div className="border border-zinc-800 rounded-lg overflow-hidden">
            <div className="bg-zinc-800/50 px-4 py-2 flex items-center justify-between">
              <div className="text-sm text-zinc-100 font-medium">Custom entities</div>
              <button onClick={openCustomDialog}
                className="text-xs px-3 py-1 bg-violet-600 text-white rounded hover:bg-violet-500">
                + Add custom
              </button>
            </div>
            {custom.length === 0 ? (
              <div className="px-4 py-6 text-xs text-zinc-500 text-center">
                No custom entities yet. Click "Add custom" to define one in plain English — the LLM will generate the patterns.
              </div>
            ) : (
              <div className="divide-y divide-zinc-800/50">
                {custom.map(c => (
                  <div key={c.label} className="px-4 py-3 flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-zinc-200">{c.display_name || c.label}</span>
                        <span className="text-[10px] text-zinc-600 font-mono">{c.label}</span>
                        <span className="text-[9px] uppercase text-zinc-500 bg-zinc-700/50 px-1 py-0.5 rounded">{c.source}</span>
                      </div>
                      <div className="text-xs text-zinc-500">{c.description}</div>
                      <div className="text-[10px] text-zinc-600 font-mono mt-0.5">
                        {c.regex_patterns.map((r, i) => <div key={i} className="truncate">{r}</div>)}
                      </div>
                    </div>
                    <button onClick={() => deleteCustom(c.label)}
                      className="text-xs text-zinc-600 hover:text-red-400">Delete</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Custom entity dialog */}
        {showCustomDialog && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-6"
               onClick={() => !distilling && !saving && setShowCustomDialog(false)}>
            <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-6 max-w-3xl w-full max-h-[90vh] overflow-y-auto"
                 onClick={e => e.stopPropagation()}>
              <h3 className="text-sm text-zinc-100 font-medium mb-1">Add custom entity</h3>
              <p className="text-xs text-zinc-500 mb-4">
                Describe the entity in plain English. The LLM generates regex + context patterns; you review before saving.
              </p>
              <div className="grid grid-cols-2 gap-4">
                {/* Left: input */}
                <div className="space-y-3">
                  <div>
                    <label className="text-[10px] text-zinc-500 uppercase font-semibold block mb-1">Name</label>
                    <input value={customName} onChange={e => setCustomName(e.target.value)}
                      placeholder="e.g. hospital_patient_id"
                      className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-violet-500 font-mono" />
                  </div>
                  <div>
                    <label className="text-[10px] text-zinc-500 uppercase font-semibold block mb-1">Description</label>
                    <textarea value={customDesc} onChange={e => setCustomDesc(e.target.value)}
                      placeholder="Hospital patient identifier — 7 to 10 digits, prefixed with PT- or PID-"
                      className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-violet-500"
                      rows={3} />
                  </div>
                  <div>
                    <label className="text-[10px] text-zinc-500 uppercase font-semibold block mb-1">Examples (optional, one per line)</label>
                    <textarea value={customExamples} onChange={e => setCustomExamples(e.target.value)}
                      placeholder={"PT-1234567\nPID-9876543"}
                      className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm text-zinc-100 focus:outline-none focus:border-violet-500 font-mono"
                      rows={3} />
                  </div>
                  <button onClick={distill} disabled={distilling || !customName || !customDesc}
                    className="w-full text-xs px-3 py-2 bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-50">
                    {distilling ? 'Calling LLM…' : (distilled ? 'Regenerate patterns' : 'Generate patterns')}
                  </button>
                </div>

                {/* Right: review */}
                <div className="bg-zinc-950 border border-zinc-800 rounded p-3 text-xs space-y-2 min-h-[280px]">
                  {distillError && <div className="text-red-400 text-xs">{distillError}</div>}
                  {!distilled && !distillError && (
                    <div className="text-zinc-600 text-center pt-12">
                      LLM-generated patterns will appear here for review.
                    </div>
                  )}
                  {distilled && (
                    <>
                      <div>
                        <div className="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Label</div>
                        <div className="text-zinc-200 font-mono">{distilled.label}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Regex patterns ({distilled.regex_patterns.length})</div>
                        {distilled.regex_patterns.map((p, i) => (
                          <div key={i} className="text-zinc-300 font-mono break-all">{p}</div>
                        ))}
                      </div>
                      <div>
                        <div className="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Context phrases ({distilled.context_phrases.length})</div>
                        <div className="text-zinc-400">{distilled.context_phrases.slice(0, 8).join(', ')}{distilled.context_phrases.length > 8 ? '…' : ''}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Examples ({distilled.examples.length})</div>
                        <div className="text-zinc-400 font-mono">{distilled.examples.slice(0, 4).join(', ')}{distilled.examples.length > 4 ? '…' : ''}</div>
                      </div>
                      {distillRepairs.length > 0 && (
                        <div>
                          <div className="text-[10px] text-amber-500 uppercase font-semibold mb-1">Auto-repaired</div>
                          {distillRepairs.map((r, i) => <div key={i} className="text-amber-400/80 text-[10px]">{r}</div>)}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
              <div className="flex justify-end gap-2 mt-4 pt-4 border-t border-zinc-800">
                <button onClick={() => setShowCustomDialog(false)} disabled={distilling || saving}
                  className="text-xs px-3 py-1.5 text-zinc-400 hover:text-zinc-200">Cancel</button>
                <button onClick={saveDistilled} disabled={!distilled || saving || distilled.regex_patterns.length === 0}
                  className="text-xs px-3 py-1.5 bg-violet-600 text-white rounded hover:bg-violet-500 disabled:opacity-50">
                  {saving ? 'Saving…' : 'Save entity'}
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </Page>
  );
}
