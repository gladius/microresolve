import { createContext, useContext } from 'react';

export type ThemeMode = 'dark' | 'light' | 'system';

export interface AppSettings {
  threshold: number;
  selectedNamespaceId: string;
  selectedDomain: string;
  languages: string[];
  reviewSkipThreshold: number;
  theme: ThemeMode;
}

export interface AppStore {
  settings: AppSettings;
  setThreshold: (t: number) => void;
  setSelectedNamespaceId: (namespaceId: string) => void;
  setSelectedDomain: (domain: string) => void;
  setLanguages: (languages: string[]) => void;
  setReviewSkipThreshold: (t: number) => void;
  setTheme: (theme: ThemeMode) => void;
}

export const defaults: AppSettings = {
  threshold: 0.3,
  selectedNamespaceId: 'default',
  selectedDomain: '',
  languages: ['en'],
  reviewSkipThreshold: 0.0,
  theme: 'dark',
};

export const AppContext = createContext<AppStore>({
  settings: defaults,
  setThreshold: () => {},
  setSelectedNamespaceId: () => {},
  setSelectedDomain: () => {},
  setLanguages: () => {},
  setReviewSkipThreshold: () => {},
  setTheme: () => {},
});

export function useAppStore() {
  return useContext(AppContext);
}
