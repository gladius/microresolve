import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from '@/components/Layout';
import HomePage from '@/pages/HomePage';
import RouterPage from '@/pages/RouterPage';
import SimulatePage from '@/pages/SimulatePage';
import LayersPage from '@/pages/LayersPage';
import ReviewPage from '@/pages/ReviewPage';
import IntentsPage from '@/pages/IntentsPage';
import SettingsPage from '@/pages/SettingsPage';
import NamespacesPage from '@/pages/NamespacesPage';
import ModelsPage from '@/pages/ModelsPage';
import LanguagesPage from '@/pages/LanguagesPage';
import DomainsPage from '@/pages/DomainsPage';
import ImportLanding from '@/pages/import/ImportLanding';
import OpenApiImport from '@/pages/import/OpenApiImport';
import McpImport from '@/pages/import/McpImport';
import OpenAIFunctionsImport from '@/pages/import/OpenAIFunctionsImport';
import LangChainImport from '@/pages/import/LangChainImport';
import CollisionsPage from '@/pages/CollisionsPage';
import EntitiesPage from '@/pages/EntitiesPage';
import { AppContext, defaults, type AppSettings, type ThemeMode } from '@/store';
import { setApiNamespaceId } from '@/api/client';

const BASE = '/api';

function applyTheme(theme: ThemeMode) {
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const isDark = theme === 'dark' || (theme === 'system' && mq.matches);
  document.documentElement.classList.toggle('light', !isDark);
  localStorage.setItem('theme', theme);
}

// Re-apply when OS preference changes (only relevant in system mode)
if (typeof window !== 'undefined') {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    const theme = (localStorage.getItem('theme') as ThemeMode) || 'dark';
    if (theme === 'system') applyTheme('system');
  });
}

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
      reviewSkipThreshold: d.review_skip_threshold ?? defaults.reviewSkipThreshold,
      theme: (localStorage.getItem('theme') as ThemeMode) || 'dark',
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
      applyTheme(s.theme);
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
    if (patch.reviewSkipThreshold !== undefined) serverPatch.review_skip_threshold = patch.reviewSkipThreshold;
    patchSettings(serverPatch);
  };

  const store = {
    settings,
    setThreshold: (threshold: number) => update({ threshold }),
    setSelectedNamespaceId: (selectedNamespaceId: string) => update({ selectedNamespaceId }),
    setSelectedDomain: (selectedDomain: string) => update({ selectedDomain }),
    setLanguages: (languages: string[]) => update({ languages }),
    setReviewSkipThreshold: (reviewSkipThreshold: number) => update({ reviewSkipThreshold }),
    setTheme: (theme: ThemeMode) => { update({ theme }); applyTheme(theme); },
  };

  if (!loaded) return null;

  return (
    <AppContext.Provider value={store}>
      <BrowserRouter>
        <Routes>
          {/* Standalone — no sidebar */}
          <Route path="/" element={<HomePage />} />
          <Route element={<Layout />}>
            <Route path="/resolve" element={<RouterPage />} />
            <Route path="/simulate" element={<SimulatePage />} />
            <Route path="/layers" element={<LayersPage />} />
            <Route path="/review" element={<ReviewPage />} />
            <Route path="/intents" element={<IntentsPage />} />
            <Route path="/import" element={<ImportLanding />} />
            <Route path="/import/openapi" element={<OpenApiImport />} />
            <Route path="/import/mcp" element={<McpImport />} />
            <Route path="/import/openai-functions" element={<OpenAIFunctionsImport />} />
            <Route path="/import/langchain" element={<LangChainImport />} />
            <Route path="/collisions" element={<CollisionsPage />} />
            <Route path="/entities" element={<EntitiesPage />} />
            <Route path="/namespaces" element={<NamespacesPage />} />
            <Route path="/namespaces/:nsId" element={<DomainsPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/languages" element={<LanguagesPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppContext.Provider>
  );
}
