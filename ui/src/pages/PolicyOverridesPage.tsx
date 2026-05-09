import { useEffect, useState } from 'react';
import { api, type PolicyOverrideRow, type IntentInfo, type ResolveOutput } from '@/api/client';
import Page from '@/components/Page';

/// Policy overrides editor — narrow declarative escape hatch (≤10 per pack).
///
/// A policy override fires when ALL listed words appear in the normalised
/// query, adding `bonus` to the target intent. Designed for hard rules pack
/// authors encode for externally-specified policy that the auto-learn loop
/// cannot reasonably teach (Article 5 carve-outs, CSAM detection vs
/// generation, similar). Mechanism is a token conjunction; role is policy
/// override. Every mutation lands in the audit log.
export default function PolicyOverridesPage() {
  const [rows, setRows] = useState<PolicyOverrideRow[]>([]);
  const [intents, setIntents] = useState<IntentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // New rule form
  const [newWords, setNewWords] = useState('');
  const [newIntent, setNewIntent] = useState('');
  const [newBonus, setNewBonus] = useState(2.0);

  // Live preview
  const [previewQuery, setPreviewQuery] = useState('');
  const [previewResult, setPreviewResult] = useState<ResolveOutput | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      const [r, ils] = await Promise.all([api.listPolicyOverrides(), api.listIntents()]);
      setRows(r.policy_overrides || []);
      setIntents(ils);
      if (!newIntent && ils.length > 0) setNewIntent(ils[0].id);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const addRule = async () => {
    const words = newWords
      .split(/[\s,]+/)
      .map(w => w.trim().toLowerCase())
      .filter(w => w.length > 0);
    if (words.length < 2) {
      setErr('Need at least 2 distinct words');
      return;
    }
    if (!newIntent) {
      setErr('Pick an intent');
      return;
    }
    try {
      await api.addPolicyOverride({ words, intent: newIntent, bonus: newBonus });
      setNewWords('');
      setNewBonus(2.0);
      setErr(null);
      refresh();
      if (previewQuery) runPreview(previewQuery);
    } catch (e) {
      setErr(String(e));
    }
  };

  const deleteRule = async (idx: number) => {
    if (!confirm('Remove this policy override?')) return;
    try {
      await api.removePolicyOverride(idx);
      refresh();
      if (previewQuery) runPreview(previewQuery);
    } catch (e) {
      setErr(String(e));
    }
  };

  const runPreview = async (q: string) => {
    setPreviewLoading(true);
    try {
      const r = await api.resolve(q, 0.3, false, true);
      setPreviewResult(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setPreviewLoading(false);
    }
  };

  // Which policy overrides fire on the current preview?
  // The trace's per_intent[].conjunctions_fired contains rule descriptions
  // (the runtime mechanism is still a conjunction) — match against rules.
  const firedRuleSigs = new Set<string>();
  if (previewResult?.trace?.per_intent) {
    for (const pi of previewResult.trace.per_intent) {
      for (const f of pi.conjunctions_fired) {
        // f is like "[word_a + word_b]" — extract sorted words.
        const inner = f.replace(/[\[\]]/g, '');
        const ws = inner.split(' + ').map(w => w.trim()).sort();
        firedRuleSigs.add(`${pi.intent}::${ws.join(' + ')}`);
      }
    }
  }
  const ruleFired = (r: PolicyOverrideRow): boolean => {
    const sig = `${r.intent}::${[...r.words].sort().join(' + ')}`;
    return firedRuleSigs.has(sig);
  };

  return (
    <Page
      title="Policy overrides"
      subtitle="Hard rules — keep ≤10 per pack. Per-intent expansion handles the rest."
    >
      <div className="px-6 py-4 max-w-5xl">
        {err && (
          <div className="mb-3 px-3 py-2 bg-red-500/10 border border-red-500/30 text-red-400 text-xs rounded">
            {err}
          </div>
        )}

        {/* What is this */}
        <div className="mb-4 text-xs text-zinc-500 bg-zinc-900 border border-zinc-800 rounded-lg p-3 leading-relaxed">
          A <span className="text-emerald-400">policy override</span> fires when <span className="text-emerald-400">all</span> listed words appear in a query, adding the bonus to the target intent. Reserved for hard rules where pre-knowledge of policy is more efficient than waiting for the auto-learn loop to discover it (Article 5 carve-outs, CSAM detection vs generation, similar). <span className="text-amber-400">Use sparingly — keep ≤10 per pack.</span> Every add / remove is recorded in the audit log.
        </div>

        {/* New rule form */}
        <div className="mb-4 bg-zinc-900 border border-zinc-800 rounded-lg p-3">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide font-semibold mb-2">Add rule</div>
          <div className="flex gap-2 items-end flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <label className="text-[10px] text-zinc-600 block mb-1">Words (comma or space separated, ≥2)</label>
              <input
                value={newWords}
                onChange={e => setNewWords(e.target.value)}
                placeholder="missing, child"
                className="w-full bg-zinc-950 border border-zinc-700 rounded px-2 py-1.5 text-zinc-100 font-mono text-xs"
              />
            </div>
            <div>
              <label className="text-[10px] text-zinc-600 block mb-1">Boost intent</label>
              <select
                value={newIntent}
                onChange={e => setNewIntent(e.target.value)}
                className="bg-zinc-950 border border-zinc-700 rounded px-2 py-1.5 text-zinc-100 font-mono text-xs"
              >
                {intents.map(i => (
                  <option key={i.id} value={i.id}>{i.id}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-zinc-600 block mb-1">Bonus (0.5-5)</label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="5"
                value={newBonus}
                onChange={e => setNewBonus(parseFloat(e.target.value) || 0)}
                className="w-20 bg-zinc-950 border border-zinc-700 rounded px-2 py-1.5 text-zinc-100 font-mono text-xs"
              />
            </div>
            <button
              onClick={addRule}
              className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded transition-colors"
            >
              Add
            </button>
          </div>
        </div>

        {/* Preview tool */}
        <div className="mb-4 bg-zinc-900 border border-zinc-800 rounded-lg p-3">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide font-semibold mb-2">
            Test query — see which overrides fire
          </div>
          <div className="flex gap-2">
            <input
              value={previewQuery}
              onChange={e => setPreviewQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') runPreview(previewQuery); }}
              placeholder="e.g. live face match for missing child"
              className="flex-1 bg-zinc-950 border border-zinc-700 rounded px-2 py-1.5 text-zinc-100 font-mono text-xs"
            />
            <button
              onClick={() => runPreview(previewQuery)}
              disabled={!previewQuery || previewLoading}
              className="px-4 py-1.5 bg-zinc-700 hover:bg-zinc-600 text-zinc-100 text-xs font-semibold rounded transition-colors disabled:opacity-40"
            >
              {previewLoading ? '…' : 'Test'}
            </button>
          </div>
          {previewResult && (
            <div className="mt-2 text-xs space-y-1">
              <div className="text-zinc-500">
                Top intent:{' '}
                <span className="text-emerald-400 font-mono">
                  {previewResult.intents[0]?.id ?? '(no match)'}
                </span>
                {previewResult.intents[0] && (
                  <span className="text-amber-400 ml-2">
                    {previewResult.intents[0].score.toFixed(2)}
                  </span>
                )}
              </div>
              <div className="text-zinc-600 text-[11px]">
                Fired overrides are highlighted in the rule list below.
              </div>
            </div>
          )}
        </div>

        {/* Rule list */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide font-semibold px-3 py-2 border-b border-zinc-800">
            Active rules ({rows.length})
          </div>
          {loading && (
            <div className="px-3 py-4 text-zinc-600 text-xs">Loading…</div>
          )}
          {!loading && rows.length === 0 && (
            <div className="px-3 py-4 text-zinc-600 text-xs italic">
              No policy overrides yet. Add one above (sparingly).
            </div>
          )}
          {rows.map(r => (
            <div
              key={r.idx}
              className={`flex items-center gap-3 px-3 py-2 border-b border-zinc-800 last:border-b-0 ${
                ruleFired(r) ? 'bg-emerald-500/10' : ''
              }`}
            >
              <span className="text-zinc-600 text-[10px] font-mono w-6">#{r.idx}</span>
              <span className="font-mono text-xs text-cyan-300 flex-1">
                {r.words.map((w, i) => (
                  <span key={i}>
                    {i > 0 && <span className="text-zinc-700"> + </span>}
                    <span>{w}</span>
                  </span>
                ))}
              </span>
              <span className="text-zinc-600 text-xs">→</span>
              <span className="font-mono text-xs text-emerald-400">{r.intent}</span>
              <span className="text-amber-400 font-mono text-xs">+{r.bonus.toFixed(2)}</span>
              {ruleFired(r) && (
                <span className="text-[9px] text-emerald-400 bg-emerald-500/20 px-1.5 py-0.5 rounded uppercase font-bold">
                  fired
                </span>
              )}
              <button
                onClick={() => deleteRule(r.idx)}
                className="text-red-400/60 hover:text-red-400 text-xs px-2 transition-colors"
                title="Remove"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>
    </Page>
  );
}
