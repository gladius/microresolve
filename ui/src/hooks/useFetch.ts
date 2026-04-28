import { useEffect, type DependencyList } from 'react';

/**
 * Deferred data fetch on mount. Wraps useEffect + setTimeout(fn, 0) to avoid
 * state-update loss during React's StrictMode double-invocation in development.
 * In production builds StrictMode does not double-invoke, so the setTimeout is
 * a no-op cost (single tick delay before the first fetch).
 */
export function useFetch(fn: () => void, deps: DependencyList) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { const t = setTimeout(fn, 0); return () => clearTimeout(t); }, deps);
}
