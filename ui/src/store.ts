import { createContext, useContext } from 'react';

export interface AppSettings {
  threshold: number;
  selectedAppId: string;
}

export interface AppStore {
  settings: AppSettings;
  setThreshold: (t: number) => void;
  setSelectedAppId: (appId: string) => void;
}

const defaults: AppSettings = {
  threshold: 0.3,
  selectedAppId: 'default',
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
  setThreshold: () => {},
  setSelectedAppId: () => {},
});

export function useAppStore() {
  return useContext(AppContext);
}
