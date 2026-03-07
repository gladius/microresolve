const BASE = '/api';

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const text = await res.text();
  if (!text) return undefined as T;
  return JSON.parse(text);
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

// --- Types ---

export type IntentType = 'action' | 'context';

export interface RouteResult {
  id: string;
  score: number;
}

export interface MultiRouteResult {
  id: string;
  score: number;
  position: number;
  span: [number, number];
  intent_type: IntentType;
}

export interface ProjectedContext {
  id: string;
  co_occurrence: number;
  strength: number;
}

export interface MultiRouteOutput {
  intents: MultiRouteResult[];
  relations: { type: string; [key: string]: unknown }[];
  metadata: Record<string, Record<string, string[]>>;
  projected_context: ProjectedContext[];
}

export interface IntentInfo {
  id: string;
  seeds: string[];
  seeds_by_lang: Record<string, string[]>;
  learned_count: number;
  intent_type: IntentType;
  metadata: Record<string, string[]>;
}

export interface ReviewAnalysis {
  correct: string[];
  false_positives: { id: string; reason: string }[];
  missed: { id: string; reason: string }[];
  suggestions: {
    action: 'learn' | 'correct' | 'add_seed';
    query: string;
    intent_id: string;
    wrong_intent?: string;
    seed?: string;
    reason: string;
  }[];
  confidence: 'high' | 'medium' | 'low';
  summary: string;
}

export interface LogEntry {
  ts: number;
  query: string;
  threshold: number;
  latency_us: number;
  results: { id: string; score: number; intent_type: IntentType; span: [number, number] }[];
}

// --- API ---

export const api = {
  health: () => get<string>('/health'),

  // Routing
  route: (query: string) => post<RouteResult[]>('/route', { query }),
  routeMulti: (query: string, threshold = 0.3) =>
    post<MultiRouteOutput>('/route_multi', { query, threshold }),

  // Intents
  listIntents: () => get<IntentInfo[]>('/intents'),
  addIntent: (id: string, seeds: string[], intent_type?: IntentType, metadata?: Record<string, string[]>) =>
    post<void>('/intents', { id, seeds, intent_type, metadata }),
  addIntentMultilingual: (id: string, seeds_by_lang: Record<string, string[]>, intent_type?: IntentType, metadata?: Record<string, string[]>) =>
    post<void>('/intents/multilingual', { id, seeds_by_lang, intent_type, metadata }),
  addSeed: (intent_id: string, seed: string) =>
    post<void>('/intents/add_seed', { intent_id, seed }),
  deleteIntent: (id: string) => post<void>('/intents/delete', { id }),
  setIntentType: (intent_id: string, intent_type: IntentType) =>
    post<void>('/intents/type', { intent_id, intent_type }),

  // Learning
  learn: (query: string, intent_id: string) =>
    post<void>('/learn', { query, intent_id }),
  correct: (query: string, wrong_intent: string, correct_intent: string) =>
    post<void>('/correct', { query, wrong_intent, correct_intent }),

  // Metadata
  setMetadata: (intent_id: string, key: string, values: string[]) =>
    post<void>('/metadata', { intent_id, key, values }),
  getMetadata: (intent_id: string) =>
    post<Record<string, string[]>>('/metadata/get', { intent_id }),

  // Seed generation
  buildSeedPrompt: (intent_id: string, description: string, languages: string[]) =>
    post<{ prompt: string }>('/seed/prompt', { intent_id, description, languages }),
  parseSeedResponse: (response_text: string, languages: string[]) =>
    post<{ seeds_by_lang: Record<string, string[]>; total: number } | { error: string }>('/seed/parse', { response_text, languages }),

  // State
  loadDefaults: () => post<void>('/intents/load_defaults', {}),
  reset: () => post<void>('/reset', {}),
  exportState: async () => {
    const res = await fetch(`${BASE}/export`);
    return res.text();
  },
  importState: (data: string) =>
    fetch(`${BASE}/import`, { method: 'POST', body: data }).then(r => {
      if (!r.ok) throw new Error('Import failed');
    }),

  // Languages
  getLanguages: () => get<Record<string, string>>('/languages'),

  // Logs
  getLogs: (limit = 100, offset = 0) =>
    get<{ total: number; offset: number; limit: number; entries: LogEntry[] }>(`/logs?limit=${limit}&offset=${offset}`),
  getLogStats: () => get<{ count: number; size_bytes: number; file: string }>('/logs/stats'),
  clearLogs: () => fetch(`${BASE}/logs`, { method: 'DELETE' }).then(r => { if (!r.ok) throw new Error('Clear failed'); }),

  // Co-occurrence & projections
  getCoOccurrence: () => get<{ a: string; b: string; count: number }[]>('/co_occurrence'),
  getProjections: () => get<{
    action: string;
    total_co_occurrences: number;
    projected_context: { id: string; count: number; strength: number }[];
  }[]>('/projections'),

  // Learn mode review (server-side LLM call)
  buildReviewPrompt: (query: string, results: unknown[], threshold: number) =>
    post<{ prompt: string }>('/review/prompt', { query, results, threshold }),
  review: (query: string, results: unknown[], threshold: number) =>
    post<ReviewAnalysis>('/review', { query, results, threshold }),

  // Seed generation (server-side LLM call)
  generateSeeds: (intent_id: string, description: string, languages: string[]) =>
    post<{ seeds_by_lang: Record<string, string[]>; total: number }>('/seed/generate', { intent_id, description, languages }),
};
