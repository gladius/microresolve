import { useState, useEffect } from 'react';

interface PlanData {
  batch_size: number;
  max_tokens_per_call: number;
  tokens_per_tool: number;
  total_batches: number;
  total_output_tokens: number;
  total_input_tokens: number;
  total_tokens: number;
  phrases_per_tool: number;
}

interface Props {
  numTools: number;
  languages: string[];
  importing?: boolean;
}

export default function GenerationPlan({ numTools, languages, importing }: Props) {
  const [plan, setPlan] = useState<PlanData | null>(null);

  useEffect(() => {
    if (numTools === 0) { setPlan(null); return; }
    fetch(`/api/import/params?num_langs=${languages.length}&num_tools=${numTools}`)
      .then(r => r.json())
      .then(setPlan)
      .catch(() => {});
  }, [numTools, languages.length]);

  if (!plan) return null;

  const isLarge = numTools > 50;
  const isHuge = numTools > 200;
  const batchSizeReduced = plan.batch_size < 10;
  const expectedOutputPerCall = plan.batch_size * plan.tokens_per_tool;
  const bufferPerCall = plan.max_tokens_per_call - expectedOutputPerCall;
  const expectedTotalK = Math.round(plan.total_output_tokens / 1000);

  return (
    <div className={`border rounded-lg px-4 py-3 space-y-3 ${isHuge ? 'border-amber-500/30 bg-amber-900/5' : 'border-zinc-800 bg-zinc-900'}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-300">Generation Plan</span>
        <span className="text-[10px] text-zinc-600">{languages.map(l => l.toUpperCase()).join(' · ')}</span>
      </div>

      {/* Summary sentence */}
      <p className="text-xs text-zinc-400 leading-relaxed">
        {numTools} intent{numTools !== 1 ? 's' : ''} × {plan.phrases_per_tool} phrases
        <span className="text-zinc-600"> ({languages.length} lang{languages.length !== 1 ? 's' : ''} × 10)</span>
        {' '}= <span className="text-white font-medium">{numTools * plan.phrases_per_tool} total phrases</span>,
        sent in <span className="text-violet-400">{plan.total_batches} LLM call{plan.total_batches !== 1 ? 's' : ''}</span>
        {' '}of <span className="text-violet-400">{plan.batch_size} intent{plan.batch_size !== 1 ? 's' : ''} each</span>.
      </p>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        <Stat label="LLM calls" value={plan.total_batches.toString()} />
        <Stat label="Expected output" value={`~${expectedTotalK}K tokens`} />
        <Stat label="Max per call" value={plan.max_tokens_per_call.toLocaleString()} sub="API ceiling" />
      </div>

      {/* Per-call token budget */}
      <div className="border-t border-zinc-800/60 pt-2 space-y-1">
        <div className="flex justify-between text-[10px]">
          <span className="text-zinc-600">{plan.batch_size} intents × {plan.tokens_per_tool} tokens/intent</span>
          <span className="text-zinc-500">= ~{expectedOutputPerCall.toLocaleString()} expected per call</span>
        </div>
        <div className="flex justify-between text-[10px]">
          <span className="text-zinc-600">Max tokens per call</span>
          <span className="text-emerald-400/80">{plan.max_tokens_per_call.toLocaleString()} (+{bufferPerCall.toLocaleString()} buffer)</span>
        </div>
        <div className="text-[10px] text-zinc-700 pt-0.5">
          tokens/intent = {languages.length} lang{languages.length !== 1 ? 's' : ''} × 10 phrases × 10 tokens + 30 JSON overhead
        </div>
        {batchSizeReduced && (
          <div className="text-[10px] text-amber-400/70">
            Batch reduced to {plan.batch_size}/call: {plan.batch_size} × {plan.tokens_per_tool} = {expectedOutputPerCall.toLocaleString()} tokens, leaving headroom under 8192.
          </div>
        )}
      </div>

      {isHuge && (
        <div className="flex items-start gap-2 bg-amber-900/20 border border-amber-500/20 rounded px-3 py-2">
          <span className="text-amber-400 shrink-0 font-bold">!</span>
          <div className="text-[10px] text-amber-300">
            <span className="font-semibold">{numTools} intents</span> = <span className="font-semibold">{plan.total_batches} sequential calls</span>.
            {' '}This will take several minutes. Consider importing in smaller selections.
          </div>
        </div>
      )}
      {isLarge && !isHuge && (
        <div className="text-[10px] text-zinc-600 pt-0.5">
          {plan.total_batches} sequential calls — est. {Math.round(plan.total_batches * 8)}–{Math.round(plan.total_batches * 15)}s.
        </div>
      )}

      {importing && (
        <div className="flex items-center gap-2 text-xs text-violet-400 pt-1 border-t border-zinc-800/60">
          <div className="w-3 h-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin shrink-0" />
          Generating — {plan.total_batches} call{plan.total_batches !== 1 ? 's' : ''} running sequentially...
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="text-center">
      <div className="text-sm font-semibold text-white">{value}</div>
      <div className="text-[10px] text-zinc-600">{label}</div>
      {sub && <div className="text-[9px] text-zinc-700">{sub}</div>}
    </div>
  );
}
