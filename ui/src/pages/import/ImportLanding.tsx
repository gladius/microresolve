import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';
import Page from '@/components/Page';

type ImportCard = {
  label: string;
  description: string;
  cta: string;
  to?: string;
  soon?: boolean;
};

const IMPORT_SOURCES: ImportCard[] = [
  {
    label: 'OpenAPI / Swagger',
    description: 'Import from an OpenAPI 3.x, Swagger 2.0, or Postman Collection spec. Each API operation becomes a routable intent.',
    cta: 'URL or paste spec →',
    to: '/import/openapi',
  },
  {
    label: 'MCP Tools',
    description: "Import from an MCP server's tools/list response. Search the Smithery registry or paste your tools JSON directly.",
    cta: 'Search servers or paste JSON →',
    to: '/import/mcp',
  },
  {
    label: 'OpenAI Function Calling',
    description: 'Paste your OpenAI functions array. Each function definition becomes a routable intent — reuses what you already wrote for the API.',
    cta: 'Paste functions array →',
    to: '/import/openai-functions',
  },
  {
    label: 'LangChain Tools',
    description: 'Import LangChain tool definitions (name + description + args_schema). Drop in your tool list — each becomes a routable intent.',
    cta: 'Paste tools list →',
    to: '/import/langchain',
  },
];

export default function ImportLanding() {
  const navigate = useNavigate();
  const { settings } = useAppStore();
  const [filter, setFilter] = useState('');
  const ns = settings.selectedNamespaceId;
  const domain = settings.selectedDomain;

  const filtered = IMPORT_SOURCES.filter(c =>
    !filter || c.label.toLowerCase().includes(filter.toLowerCase()) || c.description.toLowerCase().includes(filter.toLowerCase())
  );

  const subtitle = (
    <>
      into <span className="text-violet-400 font-mono">{ns}</span>
      {domain && (
        <>
          <span className="text-zinc-600 mx-1">/</span>
          <span className="text-violet-400 font-mono">{domain}</span>
        </>
      )}
      {!domain && <span className="text-zinc-600 ml-1">(no domain prefix)</span>}
    </>
  );

  return (
    <Page title="Import" subtitle={subtitle} size="md">
      <div className="space-y-5">
        {/* Filter */}
        <input
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="Filter import sources..."
          className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-violet-500 transition-colors"
        />

        <div className="grid grid-cols-2 gap-4">
          {filtered.map(card => (
            <div
              key={card.label}
              onClick={() => card.to && navigate(card.to)}
              className={`bg-zinc-800/60 border rounded-lg p-5 transition-colors ${
                card.to
                  ? 'border-zinc-700 cursor-pointer hover:border-violet-500/60 hover:bg-zinc-800'
                  : 'border-zinc-800 cursor-default opacity-50'
              }`}
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="text-sm font-semibold text-zinc-100">{card.label}</div>
                {card.soon && (
                  <span className="text-[9px] text-zinc-500 bg-zinc-700/60 px-1.5 py-0.5 rounded uppercase tracking-wide shrink-0">
                    soon
                  </span>
                )}
              </div>
              <p className="text-xs text-zinc-500 leading-relaxed">{card.description}</p>
              <div className={`mt-3 text-xs ${card.to ? 'text-violet-400' : 'text-zinc-600'}`}>
                {card.cta}
              </div>
            </div>
          ))}

          {filtered.length === 0 && (
            <div className="col-span-2 text-center py-10 text-sm text-zinc-600">
              No import sources match "{filter}"
            </div>
          )}
        </div>
      </div>
    </Page>
  );
}
