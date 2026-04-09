import { useState, useCallback, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from '@/components/Layout';
import RouterPage from '@/pages/RouterPage';
import IntentsPage from '@/pages/IntentsPage';
import SettingsPage from '@/pages/SettingsPage';
import DashboardPage from '@/pages/DashboardPage';
import ScenariosPage from '@/pages/ScenariosPage';
import DiscoveryPage from '@/pages/DiscoveryPage';
import ReviewPage from '@/pages/ReviewPage';
import ImportPage from '@/pages/ImportPage';
import AppsPage from '@/pages/AppsPage';
import { AppContext, loadSettings, saveSettings, type AppMode, type AppSettings } from '@/store';
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

  // Sync API client with selected app
  useEffect(() => {
    setApiAppId(settings.selectedAppId);
  }, [settings.selectedAppId]);

  const store = {
    settings,
    setMode: (mode: AppMode) => update({ mode }),
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
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/scenarios" element={<ScenariosPage />} />
            <Route path="/discovery" element={<DiscoveryPage />} />
            <Route path="/review" element={<ReviewPage />} />
            <Route path="/apps" element={<AppsPage />} />
            <Route path="/import" element={<ImportPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppContext.Provider>
  );
}
