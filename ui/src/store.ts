import { createContext, useContext } from 'react';

export interface AppSettings {
  threshold: number;
  selectedNamespaceId: string;
  selectedDomain: string;
  languages: string[];
  reviewSkipThreshold: number;
}

export interface AppStore {
  settings: AppSettings;
  setThreshold: (t: number) => void;
  setSelectedNamespaceId: (namespaceId: string) => void;
  setSelectedDomain: (domain: string) => void;
  setLanguages: (languages: string[]) => void;
  setReviewSkipThreshold: (t: number) => void;
}

export const defaults: AppSettings = {
  threshold: 0.3,
  selectedNamespaceId: 'default',
  selectedDomain: '',
  languages: ['en'],
  reviewSkipThreshold: 0.0,
};

export const AppContext = createContext<AppStore>({
  settings: defaults,
  setThreshold: () => {},
  setSelectedNamespaceId: () => {},
  setSelectedDomain: () => {},
  setLanguages: () => {},
  setReviewSkipThreshold: () => {},
});

export function useAppStore() {
  return useContext(AppContext);
}
