import { useState, useCallback } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from '@/components/Layout';
import RouterPage from '@/pages/RouterPage';
import IntentsPage from '@/pages/IntentsPage';
import DebugPage from '@/pages/DebugPage';
import SettingsPage from '@/pages/SettingsPage';
import ProjectionsPage from '@/pages/ProjectionsPage';
import ScenariosPage from '@/pages/ScenariosPage';
import { AppContext, loadSettings, saveSettings, type AppMode, type AppSettings } from '@/store';

export default function App() {
  const [settings, setSettings] = useState<AppSettings>(loadSettings);

  const update = useCallback((patch: Partial<AppSettings>) => {
    setSettings(prev => {
      const next = { ...prev, ...patch };
      saveSettings(next);
      return next;
    });
  }, []);

  const store = {
    settings,
    setMode: (mode: AppMode) => update({ mode }),
    setThreshold: (threshold: number) => update({ threshold }),
  };

  return (
    <AppContext.Provider value={store}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<RouterPage />} />
            <Route path="/intents" element={<IntentsPage />} />
            <Route path="/projections" element={<ProjectionsPage />} />
            <Route path="/scenarios" element={<ScenariosPage />} />
            <Route path="/debug" element={<DebugPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppContext.Provider>
  );
}
