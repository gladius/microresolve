import { createContext, useContext } from 'react';

export interface AppSettings {
  threshold: number;
  selectedNamespaceId: string;
  selectedDomain: string;
  languages: string[];
}

export interface AppStore {
  settings: AppSettings;
  setThreshold: (t: number) => void;
  setSelectedNamespaceId: (namespaceId: string) => void;
  setSelectedDomain: (domain: string) => void;
  setLanguages: (languages: string[]) => void;
}

export const defaults: AppSettings = {
  threshold: 0.3,
  selectedNamespaceId: 'default',
  selectedDomain: '',
  languages: ['en'],
};

export const AppContext = createContext<AppStore>({
  settings: defaults,
  setThreshold: () => {},
  setSelectedNamespaceId: () => {},
  setSelectedDomain: () => {},
  setLanguages: () => {},
});

export function useAppStore() {
  return useContext(AppContext);
}
