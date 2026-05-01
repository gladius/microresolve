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

/// Reflex-layer toggles for the currently selected namespace.
///
/// Lifted into the global store (rather than each page polling the
/// namespaces endpoint independently) so sidebar status pills update
/// the moment any page toggles a layer — no extra HTTP round trips.
/// Source of truth on the wire is `GET /api/namespaces`; the store
/// is the *cached* active-namespace slice, refreshed on namespace
/// switch and after every PATCH.
export interface LayerStatus {
  l0: boolean;
  l1m: boolean;
  l1s: boolean;
  l1a: boolean;
}

export const defaultLayerStatus: LayerStatus = { l0: true, l1m: true, l1s: true, l1a: true };

export interface AppStore {
  settings: AppSettings;
  setThreshold: (t: number) => void;
  setSelectedNamespaceId: (namespaceId: string) => void;
  setSelectedDomain: (domain: string) => void;
  setLanguages: (languages: string[]) => void;
  setReviewSkipThreshold: (t: number) => void;
  setTheme: (theme: ThemeMode) => void;
  layerStatus: LayerStatus;
  setLayerStatus: (s: LayerStatus) => void;
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
  layerStatus: defaultLayerStatus,
  setLayerStatus: () => {},
});

export function useAppStore() {
  return useContext(AppContext);
}
