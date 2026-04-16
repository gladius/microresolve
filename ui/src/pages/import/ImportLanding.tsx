import { useNavigate } from 'react-router-dom';
import { useAppStore } from '@/store';
import Page from '@/components/Page';

export default function ImportLanding() {
  const navigate = useNavigate();
  const { settings } = useAppStore();
  const ns = settings.selectedNamespaceId;
  const domain = settings.selectedDomain;

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
      <div className="grid grid-cols-2 gap-4">
        <div
          onClick={() => navigate('/import/openapi')}
          className="bg-zinc-800 border border-zinc-700 rounded-lg p-5 cursor-pointer hover:border-violet-500/50 transition-colors"
        >
          <div className="text-sm font-semibold text-white mb-2">OpenAPI / Swagger</div>
          <p className="text-xs text-zinc-500">
            Import from an OpenAPI 3.x, Swagger 2.0, or Postman Collection spec. Each API operation becomes a routable intent.
          </p>
          <div className="mt-3 text-xs text-violet-400">URL or file upload →</div>
        </div>

        <div
          onClick={() => navigate('/import/mcp')}
          className="bg-zinc-800 border border-zinc-700 rounded-lg p-5 cursor-pointer hover:border-violet-500/50 transition-colors"
        >
          <div className="text-sm font-semibold text-white mb-2">MCP Tools</div>
          <p className="text-xs text-zinc-500">
            Import from an MCP server's tool definitions. Pre-curated tools with clear names — typically 20-80 tools instead of 500+ API endpoints.
          </p>
          <div className="mt-3 text-xs text-violet-400">Server URL or paste JSON →</div>
        </div>
      </div>
    </Page>
  );
}
