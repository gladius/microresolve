import { useState, useEffect, useMemo } from 'react';
import { useAppStore } from '@/store';
import { api } from '@/api/client';
import type { LexicalGroup, LexicalSuggestion } from '@/api/client';
import Page from '@/components/Page';

type Tab = 'morph' | 'abbrev';

export default function LexicalGroupsPage() {
  const { settings } = useAppStore();
  const ns = settings.selectedNamespaceId;
  const enabledLangs = settings.languages.length > 0 ? settings.languages : ['en'];

  const [tab, setTab] = useState<Tab>('morph');
  const [groups, setGroups] = useState<LexicalGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [draftLang, setDraftLang] = useState<string>(enabledLangs[0] || 'en');
  const [draftCanonical, setDraftCanonical] = useState('');
  const [draftVariants, setDraftVariants] = useState('');
  const [adding, setAdding] = useState(false);

  const [suggesting, setSuggesting] = useState(false);
  const [proposals, setProposals] = useState<LexicalSuggestion[] | null>(null);
  const [proposalLang, setProposalLang] = useState<string>('en');

  const reload = async () => {
    setLoading(true);
    try {
      const r = await api.listLexicalGroups();
      setGroups(r.lexical_groups);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { reload(); /* eslint-disable-next-line */ }, [ns]);

  const filtered = useMemo(
    () => groups.filter(g => g.kind === tab),
    [groups, tab],
  );

  const add = async () => {
    const canonical = draftCanonical.trim().toLowerCase();
    const variants = draftVariants
      .split(',')
      .map(v => v.trim().toLowerCase())
      .filter(Boolean);
    if (!canonical || variants.length === 0) {
      setErr('Canonical and at least one variant required.');
      return;
    }
    setAdding(true);
    setErr(null);
    try {
      await api.addLexicalGroup({
        kind: tab,
        lang: draftLang,
        canonical,
        variants,
      });
      setDraftCanonical('');
      setDraftVariants('');
      reload();
    } catch (e) {
      setErr(String(e));
    } finally {
      setAdding(false);
    }
  };

  const remove = async (idx: number) => {
    try {
      await api.removeLexicalGroup(idx);
      reload();
    } catch (e) {
      setErr(String(e));
    }
  };

  const suggest = async () => {
    setSuggesting(true);
    setErr(null);
    setProposals(null);
    try {
      const r = await api.suggestLexicalGroups(tab, proposalLang);
      setProposals(r.proposals);
    } catch (e) {
      setErr(String(e));
    } finally {
      setSuggesting(false);
    }
  };

  const approveProposal = async (p: LexicalSuggestion) => {
    try {
      await api.addLexicalGroup({
        kind: p.kind,
        lang: p.lang,
        canonical: p.canonical,
        variants: p.variants,
      });
      setProposals(prev => prev ? prev.filter(x => x !== p) : null);
      reload();
    } catch (e) {
      setErr(String(e));
    }
  };

  const rejectProposal = (p: LexicalSuggestion) => {
    setProposals(prev => prev ? prev.filter(x => x !== p) : null);
  };

  const heading = tab === 'morph' ? 'Inflection groups' : 'Abbreviations';
  const description = tab === 'morph'
    ? 'Group inflectional variants of a word (child/children, predict/predicts/predicting). Variants get normalized to the canonical at index time and query time.'
    : 'Map short forms to their full phrase (rbi → real-time biometric identification). Abbreviations get expanded to the canonical at index time and query time.';

  return (
    <Page
      title="Lexicon"
      subtitle={<>per-namespace normalization for <span className="text-emerald-400 font-mono">{ns}</span></>}
      size="md"
    >
      <div className="space-y-5">

        <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 text-xs text-zinc-500 leading-relaxed space-y-1">
          <div className="text-zinc-300 font-medium text-sm">Per-namespace lexical normalization</div>
          <p>
            Two distinct kinds of mapping the engine applies during tokenization:
            <span className="text-zinc-400"> morph</span> (inflection variants of one root word) and
            <span className="text-zinc-400"> abbrev</span> (short forms of a longer phrase).
            Both are stored per-namespace, persist in <span className="font-mono text-zinc-400">_ns.json</span>, and rebuild the index on every change.
          </p>
          <p className="text-zinc-600">
            This is NOT synonyms — synonyms cause pollution. Only group items that share the same surface meaning.
          </p>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-zinc-800">
          {(['morph', 'abbrev'] as const).map(t => (
            <button
              key={t}
              onClick={() => { setTab(t); setProposals(null); setErr(null); }}
              className={`px-4 py-2 text-sm border-b-2 transition-colors ${
                tab === t
                  ? 'border-emerald-500 text-emerald-300'
                  : 'border-transparent text-zinc-500 hover:text-zinc-300'
              }`}
            >
              {t === 'morph' ? 'Inflections' : 'Abbreviations'}
              <span className="ml-2 text-[10px] text-zinc-600">
                {groups.filter(g => g.kind === t).length}
              </span>
            </button>
          ))}
        </div>

        {err && (
          <div className="bg-red-500/10 border border-red-500/30 rounded px-3 py-2 text-[11px] text-red-300">
            {err}
          </div>
        )}

        {/* Add form */}
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-4 space-y-3">
          <div className="text-xs text-zinc-300 font-medium">{heading}</div>
          <div className="text-[11px] text-zinc-500">{description}</div>

          <div className="grid grid-cols-12 gap-2">
            <select
              value={draftLang}
              onChange={e => setDraftLang(e.target.value)}
              className="col-span-2 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100 focus:outline-none focus:border-emerald-500"
            >
              {enabledLangs.map(l => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
            <input
              value={draftCanonical}
              onChange={e => setDraftCanonical(e.target.value)}
              placeholder={tab === 'morph' ? 'canonical (e.g. child)' : 'full phrase (e.g. real-time biometric identification)'}
              className="col-span-4 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500"
            />
            <input
              value={draftVariants}
              onChange={e => setDraftVariants(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && add()}
              placeholder={tab === 'morph' ? 'variants comma-separated (child, children)' : 'variants comma-separated (rbi)'}
              className="col-span-5 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={add}
              disabled={adding || !draftCanonical.trim() || !draftVariants.trim()}
              className="col-span-1 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs rounded transition-colors"
            >
              {adding ? '…' : 'Add'}
            </button>
          </div>
        </div>

        {/* LLM Suggest */}
        <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-zinc-300 font-medium">LLM suggester</div>
              <div className="text-[11px] text-zinc-500">
                Operator-triggered. Reads namespace vocabulary, proposes {tab === 'morph' ? 'inflection groups' : 'abbreviation expansions'}. Nothing applies until you approve each one.
              </div>
            </div>
            <div className="flex items-center gap-2">
              <select
                value={proposalLang}
                onChange={e => setProposalLang(e.target.value)}
                disabled={suggesting}
                className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-100 focus:outline-none focus:border-emerald-500"
              >
                {enabledLangs.map(l => (
                  <option key={l} value={l}>{l}</option>
                ))}
              </select>
              <button
                onClick={suggest}
                disabled={suggesting}
                className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 disabled:opacity-40 text-zinc-200 text-xs rounded transition-colors"
              >
                {suggesting ? 'Asking LLM…' : 'Suggest'}
              </button>
            </div>
          </div>

          {proposals && proposals.length === 0 && (
            <div className="text-[11px] text-zinc-600 text-center py-3">
              No proposals returned. Try again or add manually above.
            </div>
          )}

          {proposals && proposals.length > 0 && (
            <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/60">
              {proposals.map((p, i) => (
                <div key={i} className="px-3 py-2 flex items-center gap-3 bg-zinc-900/30">
                  <span className="text-[9px] text-zinc-500 uppercase font-bold w-6">{p.lang}</span>
                  <span className="text-xs text-zinc-200 font-mono flex-shrink-0">{p.canonical}</span>
                  <span className="text-zinc-700">→</span>
                  <span className="text-xs text-zinc-400 font-mono flex-1 truncate">
                    {p.variants.join(', ')}
                  </span>
                  <button
                    onClick={() => approveProposal(p)}
                    className="text-[11px] text-emerald-400 hover:text-emerald-300 px-2 py-0.5 transition-colors"
                  >
                    approve
                  </button>
                  <button
                    onClick={() => rejectProposal(p)}
                    className="text-[11px] text-zinc-600 hover:text-red-400 px-2 py-0.5 transition-colors"
                  >
                    reject
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Existing groups */}
        <div className="space-y-2">
          <div className="text-xs text-zinc-500 font-semibold uppercase tracking-wide">
            Existing — {filtered.length}
          </div>
          {loading ? (
            <div className="text-xs text-zinc-500 text-center py-6">Loading…</div>
          ) : filtered.length === 0 ? (
            <div className="text-[11px] text-zinc-600 text-center py-8 border border-dashed border-zinc-800 rounded-lg">
              No {tab === 'morph' ? 'inflection groups' : 'abbreviations'} yet.
            </div>
          ) : (
            <div className="border border-zinc-800 rounded-lg divide-y divide-zinc-800/60 overflow-hidden">
              {filtered.map(g => (
                <div key={g.idx} className="px-3 py-2 flex items-center gap-3 bg-zinc-900/30 hover:bg-zinc-900/60 transition-colors">
                  <span className="text-[9px] text-zinc-500 uppercase font-bold w-6">{g.lang}</span>
                  <span className="text-xs text-zinc-200 font-mono flex-shrink-0 min-w-[10rem]">{g.canonical}</span>
                  <span className="text-zinc-700">→</span>
                  <span className="text-xs text-zinc-400 font-mono flex-1 truncate">
                    {g.variants.join(', ')}
                  </span>
                  <button
                    onClick={() => g.idx !== undefined && remove(g.idx)}
                    className="text-zinc-600 hover:text-red-400 leading-none px-2 transition-colors"
                    title="Remove"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </Page>
  );
}
