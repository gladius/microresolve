import { createContext, useContext } from 'react';

export type AppMode = 'production' | 'learn';

export interface AppSettings {
  mode: AppMode;
  threshold: number;
}

export interface AppStore {
  settings: AppSettings;
  setMode: (mode: AppMode) => void;
  setThreshold: (t: number) => void;
}

const defaults: AppSettings = {
  mode: 'production',
  threshold: 0.3,
};

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem('asv_settings');
    if (raw) {
      const parsed = JSON.parse(raw);
      return { ...defaults, ...parsed };
    }
  } catch { /* ignore */ }
  return { ...defaults };
}

export function saveSettings(settings: AppSettings) {
  localStorage.setItem('asv_settings', JSON.stringify(settings));
}

export const AppContext = createContext<AppStore>({
  settings: defaults,
  setMode: () => {},
  setThreshold: () => {},
});

export function useAppStore() {
  return useContext(AppContext);
}
