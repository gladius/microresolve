const BASE = '/api';

let currentAppId = 'default';

export function setApiAppId(appId: string) {
  currentAppId = appId;
}

function appHeaders(): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json' };
  if (currentAppId && currentAppId !== 'default') {
    h['X-App-ID'] = currentAppId;
  }
  return h;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: appHeaders(),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const text = await res.text();
  if (!text) return undefined as T;
  return JSON.parse(text);
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: appHeaders() });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const data = await res.json();
  // Server returns {"error": "..."} for missing apps — treat as empty
  if (data && typeof data === 'object' && 'error' in data && !Array.isArray(data)) {
    return [] as unknown as T;
  }
  return data;
}

// --- Types ---

export type IntentType = 'action' | 'context';

export interface RouteResult {
  id: string;
  score: number;
}

export type ConfidenceTier = 'high' | 'medium' | 'low';
export type DetectionSource = 'dual' | 'paraphrase' | 'routing';

export interface MultiRouteResult {
  id: string;
  score: number;
  position: number;
  span: [number, number];
  intent_type: IntentType;
  confidence: ConfidenceTier;
  source: DetectionSource;
}

export interface ProjectedContext {
  id: string;
  co_occurrence: number;
  strength: number;
}

export interface MultiRouteOutput {
  confirmed: MultiRouteResult[];
  candidates: MultiRouteResult[];
  relations: { type: string; [key: string]: unknown }[];
  metadata: Record<string, Record<string, string[]>>;
  projected_context: ProjectedContext[];
}

export interface IntentInfo {
  id: string;
  description: string;
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
    action: 'add_seed';
    intent_id: string;
    seed: string;
    reason: string;
    query?: string;
    wrong_intent?: string;
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
  addSeed: (intent_id: string, seed: string, lang = 'en') =>
    post<{
      added: boolean;
      counts: Record<string, number>;
      new_terms: string[];
      conflicts: { term: string; competing_intent: string; severity: number }[];
      redundant: boolean;
      reason: string | null;
    }>('/intents/add_seed', { intent_id, seed, lang }),
  removeSeed: (intent_id: string, seed: string) =>
    post<void>('/intents/remove_seed', { intent_id, seed }),
  deleteIntent: (id: string) => post<void>('/intents/delete', { id }),
  setIntentType: (intent_id: string, intent_type: IntentType) =>
    post<void>('/intents/type', { intent_id, intent_type }),
  setDescription: (intent_id: string, description: string) =>
    post<void>('/intents/description', { intent_id, description }),

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
    const res = await fetch(`${BASE}/export`, { headers: appHeaders() });
    return res.text();
  },
  importState: (data: string) =>
    fetch(`${BASE}/import`, { method: 'POST', headers: appHeaders(), body: data }).then(r => {
      if (!r.ok) throw new Error('Import failed');
    }),

  // Languages
  getLanguages: () => get<Record<string, string>>('/languages'),

  // Logs
  getLogs: (limit = 100, offset = 0) =>
    get<{ total: number; offset: number; limit: number; entries: LogEntry[] }>(`/logs?limit=${limit}&offset=${offset}`),
  getLogStats: () => get<{ count: number; size_bytes: number; file: string }>('/logs/stats'),
  clearLogs: () => fetch(`${BASE}/logs`, { method: 'DELETE' }).then(r => { if (!r.ok) throw new Error('Clear failed'); }),
  checkAccuracy: () => post<AccuracyResult>('/logs/accuracy', {}),

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

  // Training Arena
  trainingGenerate: (config: {
    personality: string;
    sophistication: string;
    verbosity: string;
    turns: number;
    scenario?: string;
  }) => post<{ turns: { customer_message: string; ground_truth: string[]; intent_description: string; agent_response: string }[] }>('/training/generate', config),

  trainingRun: (turns: { message: string; ground_truth: string[] }[]) =>
    post<{
      results: {
        message: string;
        ground_truth: string[];
        confirmed: string[];
        candidates: string[];
        matched: string[];
        missed: string[];
        extra: string[];
        status: 'pass' | 'partial' | 'fail';
        details: { id: string; score: number; confidence: string; source: string }[];
      }[];
      pass_count: number;
      total: number;
      accuracy: number;
    }>('/training/run', { turns }),

  trainingReview: (message: string, detected: { id: string; score: number }[], ground_truth: string[]) =>
    post<{
      analysis: string;
      corrections: { action: string; query?: string; intent?: string; from?: string; phrase?: string }[];
    }>('/training/review', { message, detected, ground_truth }),

  trainingApply: (corrections: { action: string; query?: string; intent?: string; from?: string; phrase?: string }[]) =>
    post<{ applied: number; errors: string[] }>('/training/apply', { corrections }),

  // Simulation
  simulateTurn: (config: {
    personality: string;
    sophistication: string;
    verbosity: string;
    history: { role: string; message: string }[];
    intents: string[];
    mode: string;
  }) => post<{ message: string; ground_truth: string[]; intent_description: string }>('/simulate/turn', config),

  simulateRespond: (config: {
    query: string;
    routed_intents: { id: string; score: number; intent_type: string }[];
    history: { role: string; message: string }[];
  }) => post<{ message: string }>('/simulate/respond', config),

  // Apps
  listApps: () => get<string[]>('/apps'),
  createApp: (app_id: string) => post<{ created: string }>('/apps', { app_id }),
  deleteApp: (app_id: string) =>
    fetch(`${BASE}/apps`, {
      method: 'DELETE',
      headers: appHeaders(),
      body: JSON.stringify({ app_id }),
    }).then(r => { if (!r.ok) throw new Error('Delete failed'); }),

  // LLM
  getLLMStatus: () => get<{ configured: boolean; model: string; url: string }>('/llm/status'),

  // Review
  report: (query: string, detected: string[], flag: string, session_id?: string) =>
    post<{ id: number }>('/report', { query, detected, flag, session_id }),
  getReviewQueue: (status?: string, limit = 50, offset = 0) =>
    get<{ total: number; items: ReviewItem[] }>(`/review/queue?limit=${limit}&offset=${offset}${status ? `&status=${status}` : ''}`),
  reviewApprove: (id: number) => post<{ status: string; intent: string }>('/review/approve', { id }),
  reviewReject: (id: number) => post<{ status: string }>('/review/reject', { id }),
  reviewFix: (id: number, seeds_by_intent: Record<string, { seed: string; lang: string }[]>) =>
    post<ReviewFixResult>('/review/fix', { id, seeds_by_intent }),
  // Review analysis (full 3-turn review in one call)
  reviewAnalyze: (id: number) =>
    post<ReviewAnalyzeResult>('/review/analyze', { id }),
  reviewIntentSeeds: (intent_ids: string[]) =>
    post<Record<string, string[]>>('/review/intent_seeds', { intent_ids }),
  getReviewStats: () => get<ReviewStats>('/review/stats'),
  getReviewMode: () => get<{ mode: string }>('/review/mode'),
  setReviewMode: (mode: string) => post<{ mode: string }>('/review/mode', { mode }),

  // Spec Import
  importSpec: (spec: string) =>
    post<{
      title: string;
      version: string;
      total_operations: number;
      created: number;
      skipped: number;
      intents: { intent_id: string; seeds: number; endpoint: string; method: string; type: string }[];
    }>('/import/spec', { spec }),

  // Discovery
  discover: (queries: string[], expected_intents = 0) =>
    post<DiscoverResult>('/discover', { queries, expected_intents }),
  discoverApply: (clusters: { name: string; representative_queries: string[] }[]) =>
    post<{ created: string[]; count: number }>('/discover/apply', { clusters }),
};

export interface ReviewAnalyzeResult {
  correct_intents: string[];
  wrong_detections: string[];
  languages: string[];
  seeds_to_add: Record<string, string[]>;
  seeds_blocked: { intent: string; seed: string; reason: string }[];
  seeds_to_replace: { intent: string; old_seed: string; new_seed: string; reason: string }[];
  safe_to_apply: boolean;
  summary: string;
}

// Review types
export interface ReviewItem {
  id: number;
  query: string;
  detected: string[];
  flag: string;
  suggested_intent: string | null;
  suggested_seed: string | null;
  status: string;
  timestamp: number;
  session_id: string | null;
}

export interface ReviewStats {
  total: number;
  pending: number;
}

export interface AccuracyResult {
  total: number;
  high: number;
  medium: number;
  low: number;
  miss: number;
  high_pct: number;
  medium_pct: number;
  low_pct: number;
  miss_pct: number;
  pass_pct: number;
  sample_issues: { query: string; detected?: string[] }[];
}

export interface ReviewFixResult {
  status: string;
  added: number;
  blocked: { seed: string; intent: string; reason: string }[];
  resolved_count: number;
}

// Discovery types
export interface DiscoveredCluster {
  suggested_name: string;
  description: string;
  top_terms: string[];
  representative_queries: string[];
  size: number;
  confidence: number;
}

export interface DiscoverResult {
  clusters: DiscoveredCluster[];
  total_clusters: number;
  total_assigned: number;
  total_queries: number;
}
