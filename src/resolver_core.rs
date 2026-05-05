//! Resolver: constructor, configuration, persistence, accessors.

use crate::*;
use crate::{FxHashMap, FxHashSet};
use std::collections::HashMap;

impl Resolver {
    /// Create a new empty resolver.
    pub fn new() -> Self {
        Self {
            index: crate::scoring::IntentIndex::new(),
            training: HashMap::new(),
            descriptions: HashMap::new(),
            instructions: HashMap::new(),
            persona: HashMap::new(),
            sources: HashMap::new(),
            targets: HashMap::new(),
            schemas: HashMap::new(),
            guardrails: HashMap::new(),
            version: 0,
            namespace_name: String::new(),
            namespace_description: String::new(),
            namespace_default_threshold: None,
            namespace_default_min_voting_tokens: None,
            domain_descriptions: HashMap::new(),
            negative_training_log: Vec::new(),
            oplog: std::collections::VecDeque::new(),
        }
    }

    /// Number of registered intents.
    pub fn intent_count(&self) -> usize {
        self.training.len()
    }

    /// Get all intent IDs. Canonical source is the training map.
    pub fn intent_ids(&self) -> Vec<String> {
        // Union of training keys and descriptions keys to include intents
        // that have a description set but no training phrases yet.
        let mut ids: FxHashSet<String> = self.training.keys().cloned().collect();
        ids.extend(self.descriptions.keys().cloned());
        let mut v: Vec<String> = ids.into_iter().collect();
        v.sort();
        v
    }

    /// Get all training phrases for an intent (flat, all languages combined).
    pub fn training(&self, intent_id: &str) -> Option<Vec<String>> {
        self.training
            .get(intent_id)
            .map(|lang_map| lang_map.values().flat_map(|v| v.clone()).collect())
    }

    /// Get training phrases grouped by language.
    pub fn training_by_lang(&self, intent_id: &str) -> Option<&HashMap<String, Vec<String>>> {
        self.training.get(intent_id)
    }

    /// Get the current version number. Incremented on every mutation.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Force the version counter to a specific value. Used by connected-mode
    /// clients after applying a batch of delta ops, so `Resolver::version()`
    /// matches the server's view (op application via `apply_ops` mutates state
    /// but doesn't bump the counter — the counter is the server's responsibility).
    pub(crate) fn set_version(&mut self, v: u64) {
        self.version = v;
    }

    /// Export resolver state as JSON. Used by the server for namespace
    /// export endpoints and by bindings for in-memory transport. For local
    /// persistence prefer `save_to_dir` / `load_from_dir`.
    pub fn export_json(&self) -> String {
        let state = ResolverState {
            training: self.training.clone(),
            descriptions: self.descriptions.clone(),
            instructions: self.instructions.clone(),
            persona: self.persona.clone(),
            sources: self.sources.clone(),
            targets: self.targets.clone(),
            schemas: self.schemas.clone(),
            guardrails: self.guardrails.clone(),
            version: self.version,
        };
        serde_json::to_string(&state).unwrap_or_default()
    }

    /// Import resolver state from JSON. Companion to `export_json`.
    pub fn import_json(json: &str) -> Result<Self, crate::Error> {
        let state: ResolverState = serde_json::from_str(json)
            .map_err(|e| crate::Error::Parse(format!("invalid JSON: {}", e)))?;

        let mut resolver = Self {
            index: crate::scoring::IntentIndex::new(),
            training: state.training,
            descriptions: state.descriptions,
            instructions: state.instructions,
            persona: state.persona,
            sources: state.sources,
            targets: state.targets,
            schemas: state.schemas,
            guardrails: state.guardrails,
            version: state.version,
            namespace_name: String::new(),
            namespace_description: String::new(),
            namespace_default_threshold: None,
            namespace_default_min_voting_tokens: None,
            domain_descriptions: HashMap::new(),
            negative_training_log: Vec::new(),
            oplog: std::collections::VecDeque::new(),
        };

        // CRITICAL: rebuild L2 from training data so the imported state is
        // actually usable for routing. Without this, training data is restored
        // but the index is empty → routing returns no matches.
        resolver.rebuild_index();

        Ok(resolver)
    }

    // ── Scoring index accessors ───────────────────────────────────────────────

    pub fn index(&self) -> &crate::scoring::IntentIndex {
        &self.index
    }
    pub fn index_mut(&mut self) -> &mut crate::scoring::IntentIndex {
        &mut self.index
    }

    // ── L2b anti-Hebbian v2: token-level negative training ────────────────────

    /// Feed queries as NEGATIVE examples for a set of intents. For each query,
    /// every token's weight in each listed intent is decayed multiplicatively
    /// via the existing `reinforce` primitive (negative delta).
    ///
    /// Design notes:
    /// * Per-(token, intent) — no cross-intent effects. "now" weight in crisis
    ///   drops; "now" weight in scheduling is untouched.
    /// * Asymptotic: `w *= (1 + delta)` with delta ∈ (-1, 0). One call moves
    ///   the weight a few percent, not to zero. Tolerates reviewer noise.
    /// * Bounded: weight can never go below 0 (the multiplicative update
    ///   naturally floors).
    /// * Doesn't hide intents, doesn't break multi-intent, not one-shot
    ///   saturating (the three bugs that killed the old L3 inhibition).
    ///
    /// Typical `alpha`: 0.05 (weak) to 0.3 (aggressive). 0.1 is a reasonable
    /// default for setup-time inoculation from a benign corpus.
    ///
    /// Each call appends to the audit log automatically (see
    /// `negative_training_log`); use `rebuild_index()` to reset both the
    /// weights and the log.
    pub fn decay_for_intents(
        &mut self,
        raw_queries: &[String],
        not_intents: &[String],
        alpha: f32,
    ) {
        if alpha <= 0.0 || alpha >= 1.0 {
            return;
        }
        let delta = -alpha;

        // Collect token lists first (no borrow of self.index yet).
        let token_lists: Vec<Vec<String>> = raw_queries
            .iter()
            .map(|q| {
                crate::tokenizer::tokenize(q)
                    .into_iter()
                    .map(|t| {
                        if let Some(stripped) = t.strip_prefix("not_") {
                            stripped.to_string()
                        } else {
                            t
                        }
                    })
                    .collect()
            })
            .collect();

        // Snapshot weights for all (token, intent) pairs we are about to touch.
        let mut snap: std::collections::HashMap<(String, String), f32> =
            std::collections::HashMap::new();
        for words in &token_lists {
            for word in words {
                for intent_id in not_intents {
                    if let Some(w) = self.index.get_weight(word, intent_id) {
                        snap.entry((word.clone(), intent_id.clone())).or_insert(w);
                    }
                }
            }
        }

        for token_list in &token_lists {
            let word_refs: Vec<&str> = token_list.iter().map(|s| s.as_str()).collect();
            for intent_id in not_intents {
                self.index.reinforce(&word_refs, intent_id, delta);
            }
        }

        let changes = self.diff_weights(&snap);

        // Audit trail — appended automatically.
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.negative_training_log
            .push(crate::NegativeTrainingEntry {
                timestamp,
                corpus_size: raw_queries.len(),
                intents_affected: not_intents.len(),
                alpha,
            });

        if !changes.is_empty() {
            self.bump_with_ops(vec![crate::oplog::Op::WeightUpdates { changes }]);
        }
    }

    /// Lower-level phrase ingestion: tokenizes + indexes the phrase into the scoring index without
    /// the duplicate-check or stop-word filtering that `add_phrase` applies. Use `add_phrase`
    /// for user-driven additions; use `index_phrase` only for trusted, pre-validated phrases
    /// (e.g., from spec import or auto-learn).
    pub fn index_phrase(&mut self, intent_id: &str, phrase: &str) {
        self.index_phrase_no_rebuild(intent_id, phrase);
    }

    pub(crate) fn index_phrase_no_rebuild(&mut self, intent_id: &str, phrase: &str) {
        let words = crate::tokenizer::tokenize(phrase);
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        if !word_refs.is_empty() {
            self.index.learn_phrase(&word_refs, intent_id);
        }
        self.index.index_char_ngrams(phrase, intent_id);
    }

    /// Rebuild the scoring index from scratch using all training phrases currently in this
    /// namespace. Clears the existing index, re-indexes every stored
    /// phrase, and wipes the negative-training audit log.
    pub fn rebuild_index(&mut self) {
        self.index = crate::scoring::IntentIndex::new();
        let all: Vec<(String, String)> = self
            .training
            .iter()
            .flat_map(|(intent_id, lang_map)| {
                lang_map
                    .values()
                    .flat_map(|phrases| phrases.iter().map(|p| (intent_id.clone(), p.clone())))
            })
            .collect();
        for (intent_id, phrase) in &all {
            self.index_phrase_no_rebuild(intent_id, phrase);
        }
        self.index.rebuild_caches();
        // Audit log is now stale — every prior decay_for_intents call has been wiped.
        self.negative_training_log.clear();
    }

    /// Bump version and record one or more ops atomically.
    ///
    /// Every public mutation method that changes state MUST call this instead
    /// of `self.version += 1` directly, so the oplog stays in sync.
    pub(crate) fn bump_with_ops(&mut self, ops: Vec<crate::oplog::Op>) {
        self.version += 1;
        for op in ops {
            self.oplog.push_back((self.version, op));
        }
        while self.oplog.len() > crate::oplog::OPLOG_MAX {
            self.oplog.pop_front();
        }
    }

    /// Diff weights against a snapshot; return non-trivial changes as WeightUpdates triples.
    pub(crate) fn diff_weights(
        &self,
        snapshot: &std::collections::HashMap<(String, String), f32>,
    ) -> Vec<(String, String, f32)> {
        snapshot
            .iter()
            .filter_map(|((t, i), before)| {
                let after = self.index.get_weight(t, i).unwrap_or(0.0);
                if (after - before).abs() > 1e-6 {
                    Some((t.clone(), i.clone(), after))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Collect all (token, intent) pairs currently indexed for a given intent.
    pub(crate) fn intent_weight_pairs(&self, intent_id: &str) -> Vec<(String, String)> {
        self.index
            .word_intent
            .iter()
            .filter_map(|(token, entries)| {
                if entries.iter().any(|(id, _)| id == intent_id) {
                    Some((token.clone(), intent_id.to_string()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Apply the resolver-local part of a review to the index in one shot.
    ///
    /// Performs every mutation a confirmed review should produce:
    ///
    /// 1. **Add new phrases** for `missed_phrases` (per intent → list of new
    ///    training phrases). Each phrase is indexed via the normal pipeline
    ///    so L0/L1/L2 stay in sync.
    /// 2. **Learn span words** — `spans_to_learn` is `(intent_id, span_text)`
    ///    pairs. The span text is tokenized and learned as intent-bearing
    ///    query words (vocabulary growth from the customer's own phrasing).
    /// 3. **Anti-Hebbian shrink** — for each intent in `wrong_detections`,
    ///    decay the weights of every token in `original_query` toward that
    ///    intent (gentle bounded multiplicative decay via `decay_for_intents`).
    /// 4. **Audit log** — append one `NegativeTrainingEntry` summarising the
    ///    shrink, so this review is reversible.
    ///
    /// Does NOT perform: LLM-driven synonym/morphology discovery, persistence
    /// to disk, network calls of any kind. Those orchestration concerns stay
    /// in the server crate; this method is the deterministic local core that
    /// every binding (Python, Node, embedded Rust) can call directly.
    ///
    /// Returns the number of phrases successfully added.
    ///
    /// `negative_alpha` is clamped to (0.0, 0.3]; pass `0.1` for a sensible
    /// default. `0.0` or negative skips anti-Hebbian entirely.
    pub fn apply_review(
        &mut self,
        missed_phrases: &HashMap<String, Vec<String>>,
        spans_to_learn: &[(String, String)],
        wrong_detections: &[String],
        original_query: &str,
        negative_alpha: f32,
    ) -> usize {
        let mut added = 0usize;

        // 1. Index missed phrases.
        for (intent_id, phrases) in missed_phrases {
            for phrase in phrases {
                // Snapshot before indexing.
                let words_pre = crate::tokenizer::tokenize(phrase);
                let snap_pairs: Vec<(String, String)> = words_pre
                    .iter()
                    .map(|w| (w.clone(), intent_id.clone()))
                    .collect();
                let snap: std::collections::HashMap<(String, String), f32> = snap_pairs
                    .iter()
                    .map(|(t, i)| {
                        (
                            (t.clone(), i.clone()),
                            self.index.get_weight(t, i).unwrap_or(0.0),
                        )
                    })
                    .collect();

                self.index_phrase_no_rebuild(intent_id, phrase);
                self.training
                    .entry(intent_id.clone())
                    .or_default()
                    .entry("en".to_string())
                    .or_default()
                    .push(phrase.clone());
                added += 1;

                let changes = self.diff_weights(&snap);
                let mut ops: Vec<crate::oplog::Op> = vec![crate::oplog::Op::PhraseAdded {
                    intent_id: intent_id.clone(),
                    phrase: phrase.clone(),
                    lang: "en".to_string(),
                }];
                if !changes.is_empty() {
                    ops.push(crate::oplog::Op::WeightUpdates { changes });
                }
                self.bump_with_ops(ops);
            }
        }

        // 2. Learn LLM-extracted query spans as intent-bearing words.
        for (intent_id, span_text) in spans_to_learn {
            let span_words: Vec<String> = crate::tokenizer::tokenize(span_text);
            let snap_pairs: Vec<(String, String)> = span_words
                .iter()
                .map(|w| (w.as_str(), intent_id.as_str()))
                .map(|(t, i)| (t.to_string(), i.to_string()))
                .collect();
            let snap: std::collections::HashMap<(String, String), f32> = snap_pairs
                .iter()
                .map(|(t, i)| {
                    (
                        (t.clone(), i.clone()),
                        self.index.get_weight(t, i).unwrap_or(0.0),
                    )
                })
                .collect();
            let span_refs: Vec<&str> = span_words.iter().map(|s| s.as_str()).collect();
            self.index.reinforce_tokens(&span_refs, intent_id);
            let changes = self.diff_weights(&snap);
            if !changes.is_empty() {
                self.bump_with_ops(vec![crate::oplog::Op::WeightUpdates { changes }]);
            }
        }

        // 3. Anti-Hebbian shrink for wrong detections on this query.
        // decay_for_intents now emits its own WeightUpdates op.
        if !wrong_detections.is_empty() && negative_alpha > 0.0 {
            let alpha = negative_alpha.min(0.3);
            self.decay_for_intents(&[original_query.to_string()], wrong_detections, alpha);
        }

        added
    }

    /// Cross-provider disambiguation: when the same action name appears from
    /// multiple providers (e.g. `shopify:list_customers` + `stripe:list_customers`),
    /// pick the provider whose unique query words match best. Only affects
    /// duplicates — different actions are never touched.
    ///
    /// Mutates `scored` in place: removes losing duplicates from groups where
    /// one candidate has more query-unique tokens than the others. If no
    /// candidate has any unique tokens, the group is left intact (genuinely
    /// ambiguous → caller decides).
    pub fn deduplicate_by_provider(&self, scored: &mut Vec<(String, f32)>, query: &str) {
        if scored.len() < 2 {
            return;
        }

        // Group candidate intent indices by action name (part after ':').
        let mut action_groups: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
        for (i, (id, _)) in scored.iter().enumerate() {
            let action = id.split(':').nth(1).unwrap_or(id.as_str());
            action_groups.entry(action).or_default().push(i);
        }
        let duplicate_groups: Vec<Vec<usize>> = action_groups
            .values()
            .filter(|indices| indices.len() > 1)
            .cloned()
            .collect();
        if duplicate_groups.is_empty() {
            return;
        }

        let tokens = crate::tokenizer::tokenize(query);
        let scored_ids: FxHashSet<&str> = scored.iter().map(|(id, _)| id.as_str()).collect();

        // For each token, count it toward an intent only if that intent is the
        // sole candidate it activates (within the current scored set).
        let mut unique_count: FxHashMap<&str, usize> = FxHashMap::default();
        for token in &tokens {
            let base = token.strip_prefix("not_").unwrap_or(token.as_str());
            if let Some(activations) = self.index.word_intent.get(base) {
                let matching: Vec<&str> = activations
                    .iter()
                    .filter(|(id, _)| scored_ids.contains(id.as_str()))
                    .map(|(id, _)| id.as_str())
                    .collect();
                if matching.len() == 1 {
                    *unique_count.entry(matching[0]).or_insert(0) += 1;
                }
            }
        }

        let mut to_remove: FxHashSet<usize> = FxHashSet::default();
        for group in &duplicate_groups {
            let best = group
                .iter()
                .max_by_key(|&&i| unique_count.get(scored[i].0.as_str()).copied().unwrap_or(0));
            if let Some(&best_idx) = best {
                let best_unique = unique_count
                    .get(scored[best_idx].0.as_str())
                    .copied()
                    .unwrap_or(0);
                if best_unique > 0 {
                    for &i in group {
                        if i != best_idx {
                            to_remove.insert(i);
                        }
                    }
                }
            }
        }

        if !to_remove.is_empty() {
            let mut i = 0;
            scored.retain(|_| {
                let keep = !to_remove.contains(&i);
                i += 1;
                keep
            });
        }
    }
}

/// Serializable resolver state for `export_json` / `import_json`.
#[derive(serde::Serialize, serde::Deserialize)]
struct ResolverState {
    training: HashMap<String, HashMap<String, Vec<String>>>,
    #[serde(default)]
    descriptions: HashMap<String, String>,
    #[serde(default)]
    instructions: HashMap<String, String>,
    #[serde(default)]
    persona: HashMap<String, String>,
    #[serde(default)]
    sources: HashMap<String, IntentSource>,
    #[serde(default)]
    targets: HashMap<String, IntentTarget>,
    #[serde(default)]
    schemas: HashMap<String, serde_json::Value>,
    #[serde(default)]
    guardrails: HashMap<String, Vec<String>>,
    #[serde(default)]
    version: u64,
}
