const BASE = '/api';

let currentNamespaceId = 'default';

export function setApiNamespaceId(namespaceId: string) {
  currentNamespaceId = namespaceId;
}


export function appHeaders(): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json' };
  if (currentNamespaceId && currentNamespaceId !== 'default') {
    h['X-Namespace-ID'] = currentNamespaceId;
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

export interface MultiRouteOutput {
  confirmed: MultiRouteResult[];
  candidates: MultiRouteResult[];
  relations: { type: string; [key: string]: unknown }[];
  routing_us: number;
  ranked?: { id: string; score: number }[];
  debug?: {
    l0_corrected?: string;
    l1_normalized?: string;
    l1_injected?: string[];
    l1_disabled?: boolean;
    l2_tokens?: string[];
    l2_all_scores?: { id: string; score: number }[];
  } | null;
}

export interface NamespaceModel {
  label: string;
  model_id: string;
}

export interface IntentSource {
  type: string;
  label?: string;
  url?: string;
}

export interface IntentTarget {
  type: string;
  url?: string;
  model?: string;
  handler?: string;
}

export interface IntentInfo {
  id: string;
  description: string;
  phrases: string[];
  phrases_by_lang: Record<string, string[]>;
  learned_count: number;
  intent_type: IntentType;
  instructions?: string;
  persona?: string;
  guardrails?: string[];
  source?: IntentSource;
  target?: IntentTarget;
  schema?: Record<string, unknown>;
}

export interface ReviewAnalysis {
  correct: string[];
  false_positives: { id: string; reason: string }[];
  missed: { id: string; reason: string }[];
  suggestions: {
    action: 'add_phrase';
    intent_id: string;
    phrase: string;
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
  routeMulti: (query: string, threshold = 0.3, log = true, debug = false) =>
    post<MultiRouteOutput>('/route_multi', { query, threshold, log, debug }),

  // Intents
  listIntents: () => get<IntentInfo[]>('/intents'),
  addIntent: (id: string, phrases: string[], intent_type?: IntentType) =>
    post<void>('/intents', { id, phrases, intent_type }),
  addIntentMultilingual: (id: string, phrases_by_lang: Record<string, string[]>, intent_type?: IntentType) =>
    post<void>('/intents/multilingual', { id, phrases_by_lang, intent_type }),
  addPhrase: (intent_id: string, phrase: string, lang = 'en') =>
    post<{
      added: boolean;
      counts: Record<string, number>;
      new_terms: string[];
      conflicts: { term: string; competing_intent: string; severity: number }[];
      redundant: boolean;
      reason: string | null;
    }>('/intents/phrase', { intent_id, phrase, lang }),
  removePhrase: (intent_id: string, phrase: string) =>
    post<void>('/intents/phrase/remove', { intent_id, phrase }),
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

  // Intent behavior (instructions, persona, guardrails)
  setInstructions: (intent_id: string, instructions: string) =>
    post<void>('/intents/instructions', { intent_id, instructions }),
  setPersona: (intent_id: string, persona: string) =>
    post<void>('/intents/persona', { intent_id, persona }),
  setGuardrails: (intent_id: string, guardrails: string[]) =>
    post<void>('/intents/guardrails', { intent_id, guardrails }),
  setTarget: (intent_id: string, target: IntentTarget) =>
    post<void>('/intents/target', { intent_id, target }),

  // Namespace model registry
  getNsModels: () => get<NamespaceModel[]>('/ns/models'),
  setNsModels: (models: NamespaceModel[]) => post<void>('/ns/models', models),

  // Phrase generation
  buildPhrasePrompt: (intent_id: string, description: string, languages: string[]) =>
    post<{ prompt: string }>('/phrase/prompt', { intent_id, description, languages }),
  parsePhraseResponse: (response_text: string, languages: string[]) =>
    post<{ phrases_by_lang: Record<string, string[]>; total: number } | { error: string }>('/phrase/parse', { response_text, languages }),

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
  clearAllData: () =>
    fetch(`${BASE}/data/all`, { method: 'DELETE', headers: appHeaders() }).then(r => {
      if (!r.ok) throw new Error('Clear failed');
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

  // Phrase generation (server-side LLM call)
  generatePhrases: (intent_id: string, description: string, languages: string[]) =>
    post<{ phrases_by_lang: Record<string, string[]>; total: number }>('/phrase/generate', { intent_id, description, languages }),

  // Training Arena
  trainingGenerate: (config: {
    personality: string;
    sophistication: string;
    verbosity: string;
    turns: number;
    scenario?: string;
    language?: string;
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

  // Namespaces (isolated routing workspaces)
  listNamespaces: () => get<{ id: string; name: string; description: string; auto_learn: boolean }[]>('/namespaces'),
  createNamespace: (namespace_id: string, name = '', description = '') =>
    post<{ created: string }>('/namespaces', { namespace_id, name, description }),
  deleteNamespace: (namespace_id: string) =>
    fetch(`${BASE}/namespaces`, {
      method: 'DELETE',
      headers: appHeaders(),
      body: JSON.stringify({ namespace_id }),
    }).then(r => { if (!r.ok) throw new Error('Delete failed'); }),
  updateNamespace: (namespace_id: string, patch: { name?: string; description?: string; auto_learn?: boolean }) =>
    fetch(`${BASE}/namespaces`, {
      method: 'PATCH',
      headers: appHeaders(),
      body: JSON.stringify({ namespace_id, ...patch }),
    }).then(r => { if (!r.ok) throw new Error('Update failed'); }),

  // Domain groups within the current namespace (derived from "domain:intent_id" prefixes)
  listDomains: () => get<{ name: string; description: string; intent_count: number }[]>('/domains'),
  listDomainsFor: (namespaceId: string) => {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (namespaceId && namespaceId !== 'default') h['X-Namespace-ID'] = namespaceId;
    return fetch(`${BASE}/domains`, { headers: h })
      .then(r => r.json() as Promise<{ name: string; description: string; intent_count: number }[]>);
  },
  updateDomain: (domain: string, description: string) =>
    fetch(`${BASE}/domains`, {
      method: 'PATCH',
      headers: appHeaders(),
      body: JSON.stringify({ domain, description }),
    }).then(r => { if (!r.ok) throw new Error('Update failed'); }),
  updateDomainFor: (namespaceId: string, domain: string, description: string) => {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (namespaceId && namespaceId !== 'default') h['X-Namespace-ID'] = namespaceId;
    return fetch(`${BASE}/domains`, {
      method: 'PATCH',
      headers: h,
      body: JSON.stringify({ domain, description }),
    }).then(r => { if (!r.ok) throw new Error('Update failed'); });
  },
  createDomainFor: (namespaceId: string, domain: string, description: string) => {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (namespaceId && namespaceId !== 'default') h['X-Namespace-ID'] = namespaceId;
    return fetch(`${BASE}/domains`, {
      method: 'POST',
      headers: h,
      body: JSON.stringify({ domain, description }),
    }).then(r => { if (!r.ok) return r.text().then(t => { throw new Error(t); }); });
  },
  deleteDomainFor: (namespaceId: string, domain: string) => {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (namespaceId && namespaceId !== 'default') h['X-Namespace-ID'] = namespaceId;
    return fetch(`${BASE}/domains`, {
      method: 'DELETE',
      headers: h,
      body: JSON.stringify({ domain }),
    }).then(r => { if (!r.ok) throw new Error('Delete failed'); });
  },

  // LLM
  getLLMStatus: () => get<{ configured: boolean; model: string; url: string }>('/llm/status'),

  // Review
  report: (query: string, detected: string[], flag: string, session_id?: string) =>
    post<{ id: number }>('/report', { query, detected, flag, session_id }),
  getReviewQueue: (status?: string, limit = 50, offset = 0) =>
    get<{ total: number; items: ReviewItem[] }>(`/review/queue?limit=${limit}&offset=${offset}${status ? `&status=${status}` : ''}`),
  reviewReject: (id: number) => post<{ status: string }>('/review/reject', { id }),
  reviewFix: (
    id: number,
    phrases_by_intent: Record<string, { phrase: string; lang: string }[]>,
    correct_intents: string[] = [],
    wrong_detections: string[] = [],
  ) =>
    post<{ status: string; added: number; auto_resolved: number; resolved_count?: number; blocked?: { phrase: string; reason?: string }[] }>(
      '/review/fix', { id, phrases_by_intent, correct_intents, wrong_detections }
    ),
  // Review analysis (full 3-turn review in one call)
  reviewAnalyze: (id: number) =>
    post<ReviewAnalyzeResult>('/review/analyze', { id }),
  reviewIntentPhrases: (intent_ids: string[]) =>
    post<Record<string, string[]>>('/review/intent_phrases', { intent_ids }),
  getReviewStats: () => get<ReviewStats>('/review/stats'),

  // Synchronous learn — bypasses worker queue, returns immediately with SSE events fired.
  // Pass ground_truth when available (simulate tab) to skip Turn 1 LLM.
  learnNow: (query: string, detected_intents: string[] = [], ground_truth?: string[]) =>
    post<{
      correct_intents: string[]; wrong_detections: string[]; missed_intents: string[];
      phrases_added: number; summary: string;
      languages: string[]; version_before: number; version_after: number; model: string;
    }>('/learn/now', { query, detected_intents, ground_truth }),

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
};

/// Single data contract between review and apply — used by all learning flows.
export interface FullReviewResult {
  correct_intents: string[];
  wrong_detections: string[];
  missed_intents: string[];
  languages: string[];
  detection_perfect: boolean;
  phrases_to_add: Record<string, string[]>;
  phrases_blocked: { intent: string; phrase: string; reason: string }[];
  summary: string;
}

// Alias for backwards compat with components that import ReviewAnalyzeResult
export type ReviewAnalyzeResult = FullReviewResult;

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


