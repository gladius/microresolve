import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from '@/components/Layout';
import RouterPage from '@/pages/RouterPage';
import IntentsPage from '@/pages/IntentsPage';
import SettingsPage from '@/pages/SettingsPage';
import AutoImprovePage from '@/pages/ScenariosPage';
import ReviewPage from '@/pages/ReviewPage';
import NamespacesPage from '@/pages/NamespacesPage';
import DomainsPage from '@/pages/DomainsPage';
import ImportLanding from '@/pages/import/ImportLanding';
import OpenApiImport from '@/pages/import/OpenApiImport';
import McpImport from '@/pages/import/McpImport';
import InsightsLayout from '@/pages/insights/InsightsLayout';
import Overview from '@/pages/insights/Overview';
import Discovery from '@/pages/insights/Discovery';
import Projections from '@/pages/insights/Projections';
import Workflows from '@/pages/insights/Workflows';
import Temporal from '@/pages/insights/Temporal';
import Escalations from '@/pages/insights/Escalations';
import CoOccurrence from '@/pages/insights/CoOccurrence';
import { AppContext, defaults, type AppSettings } from '@/store';
import { setApiNamespaceId } from '@/api/client';

const BASE = '/api';

async function fetchSettings(): Promise<AppSettings> {
  try {
    const r = await fetch(`${BASE}/settings`);
    if (!r.ok) return defaults;
    const d = await r.json();
    return {
      threshold: d.threshold ?? defaults.threshold,
      selectedNamespaceId: d.selected_namespace_id ?? defaults.selectedNamespaceId,
      selectedDomain: d.selected_domain ?? defaults.selectedDomain,
      languages: d.languages ?? defaults.languages,
    };
  } catch { return defaults; }
}

async function patchSettings(patch: Partial<{
  selected_namespace_id: string;
  selected_domain: string;
  threshold: number;
  languages: string[];
}>) {
  await fetch(`${BASE}/settings`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
}

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(defaults);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    fetchSettings().then(s => {
      setSettings(s);
      setApiNamespaceId(s.selectedNamespaceId);
      setLoaded(true);
    });
  }, []);

  const update = (patch: Partial<AppSettings>) => {
    setSettings(prev => ({ ...prev, ...patch }));
    const serverPatch: Record<string, unknown> = {};
    if (patch.selectedNamespaceId !== undefined) {
      serverPatch.selected_namespace_id = patch.selectedNamespaceId;
      setApiNamespaceId(patch.selectedNamespaceId);
    }
    if (patch.selectedDomain !== undefined) serverPatch.selected_domain = patch.selectedDomain;
    if (patch.threshold !== undefined) serverPatch.threshold = patch.threshold;
    if (patch.languages !== undefined) serverPatch.languages = patch.languages;
    patchSettings(serverPatch);
  };

  const store = {
    settings,
    setThreshold: (threshold: number) => update({ threshold }),
    setSelectedNamespaceId: (selectedNamespaceId: string) => update({ selectedNamespaceId }),
    setSelectedDomain: (selectedDomain: string) => update({ selectedDomain }),
    setLanguages: (languages: string[]) => update({ languages }),
  };

  if (!loaded) return null;

  return (
    <AppContext.Provider value={store}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<RouterPage />} />
            <Route path="/intents" element={<IntentsPage />} />
            <Route path="/review" element={<ReviewPage />} />
            <Route path="/import" element={<ImportLanding />} />
            <Route path="/import/openapi" element={<OpenApiImport />} />
            <Route path="/import/mcp" element={<McpImport />} />
            <Route path="/insights" element={<InsightsLayout />}>
              <Route index element={<Overview />} />
              <Route path="discovery" element={<Discovery />} />
              <Route path="projections" element={<Projections />} />
              <Route path="workflows" element={<Workflows />} />
              <Route path="temporal" element={<Temporal />} />
              <Route path="escalations" element={<Escalations />} />
              <Route path="cooccurrence" element={<CoOccurrence />} />
            </Route>
            <Route path="/auto-improve" element={<AutoImprovePage />} />
            <Route path="/namespaces" element={<NamespacesPage />} />
            <Route path="/namespaces/:nsId" element={<DomainsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppContext.Provider>
  );
}
