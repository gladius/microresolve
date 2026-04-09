import { useState, useCallback, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from '@/components/Layout';
import RouterPage from '@/pages/RouterPage';
import IntentsPage from '@/pages/IntentsPage';
import SettingsPage from '@/pages/SettingsPage';
import AutoImprovePage from '@/pages/ScenariosPage';
import ReviewPage from '@/pages/ReviewPage';
import AppsPage from '@/pages/AppsPage';
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
import { AppContext, loadSettings, saveSettings, type AppSettings } from '@/store';
import { setApiAppId } from '@/api/client';

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(loadSettings);

  const update = useCallback((patch: Partial<AppSettings>) => {
    setSettings(prev => {
      const next = { ...prev, ...patch };
      saveSettings(next);
      return next;
    });
  }, []);

  useEffect(() => {
    setApiAppId(settings.selectedAppId);
  }, [settings.selectedAppId]);

  const store = {
    settings,
    setThreshold: (threshold: number) => update({ threshold }),
    setSelectedAppId: (selectedAppId: string) => update({ selectedAppId }),
  };

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
            <Route path="/apps" element={<AppsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppContext.Provider>
  );
}
