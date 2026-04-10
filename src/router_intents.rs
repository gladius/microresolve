//! Router: intent management and seed guard.

use crate::*;
use crate::tokenizer::*;
use crate::vector::LearnedVector;
use crate::index::InvertedIndex;
use std::collections::{HashMap, HashSet};

impl Router {
    pub fn add_intent(&mut self, id: &str, seed_phrases: &[&str]) -> Vec<SeedCheckResult> {
        self.require_local();
        let mut results = Vec::new();
        let mut accepted: Vec<String> = Vec::new();

        for phrase in seed_phrases {
            let check = self.check_seed(id, phrase);
            // add_intent: always accept seeds (they define the intent).
            // Report conflicts but don't block — caller can review.
            accepted.push(phrase.to_string());
            results.push(SeedCheckResult { added: true, ..check });
        }

        let terms = training_to_terms(&accepted);
        let vector = LearnedVector::from_seed(terms);
        self.vectors.insert(id.to_string(), vector);
        let mut lang_map = HashMap::new();
        lang_map.insert("en".to_string(), accepted.clone());
        self.training.insert(id.to_string(), lang_map);
        let phrase_refs: Vec<&str> = accepted.iter().map(|s| s.as_str()).collect();
        self.add_paraphrases(id, &phrase_refs);
        self.rebuild_index();
        self.version += 1;
        results
    }

    /// Add an intent with seed phrases grouped by language.
    ///
    /// All phrases across all languages are indexed together into one flat vector.
    /// Language grouping is preserved in the datastore for display/export.
    ///
    /// ```
    /// use asv_router::Router;
    /// use std::collections::HashMap;
    ///
    /// let mut router = Router::new();
    /// let mut seeds = HashMap::new();
    /// seeds.insert("en".to_string(), vec!["cancel my order".to_string()]);
    /// seeds.insert("es".to_string(), vec!["cancelar mi pedido".to_string()]);
    /// router.add_intent_multilingual("cancel_order", seeds);
    /// ```
    pub fn add_intent_multilingual(&mut self, id: &str, seeds_by_lang: HashMap<String, Vec<String>>) {
        self.require_local();
        // Enforce per-language seed limit
        let truncated: HashMap<String, Vec<String>> = seeds_by_lang.into_iter()
            .map(|(lang, seeds)| {
                let limited: Vec<String> = seeds.into_iter().take(MAX_SEEDS_PER_LANGUAGE).collect();
                (lang, limited)
            })
            .collect();
        let all_phrases: Vec<String> = truncated.values().flat_map(|v| v.clone()).collect();
        let terms = training_to_terms(&all_phrases);
        let vector = LearnedVector::from_seed(terms);
        self.vectors.insert(id.to_string(), vector);
        // Auto-populate paraphrase index from seeds
        let phrase_refs: Vec<&str> = all_phrases.iter().map(|s| s.as_str()).collect();
        self.add_paraphrases(id, &phrase_refs);
        self.training.insert(id.to_string(), truncated);
        self.rebuild_index();
        self.version += 1;
    }

    /// Add an intent with pre-computed term weights.
    ///
    /// Use this when you have term weights from an external source
    /// (e.g., LLM-generated, imported from another system).
    pub fn add_intent_with_weights(&mut self, id: &str, seed_terms: HashMap<String, f32>) {
        self.require_local();
        let vector = LearnedVector::from_seed(seed_terms);
        self.vectors.insert(id.to_string(), vector);
        self.rebuild_index();
        self.version += 1;
    }

    /// Remove a single seed phrase from an intent.
    /// Recomputes the intent's vector from remaining seeds.
    /// Returns true if the seed was found and removed.
    pub fn remove_seed(&mut self, intent_id: &str, seed: &str) -> bool {
        self.require_local();
        let training = match self.training.get_mut(intent_id) {
            Some(t) => t,
            None => return false,
        };

        // Find and remove the seed from whichever language it belongs to
        let mut found = false;
        for phrases in training.values_mut() {
            if let Some(pos) = phrases.iter().position(|s| s == seed) {
                phrases.remove(pos);
                found = true;
                break;
            }
        }

        if !found { return false; }

        // Remove empty language entries
        training.retain(|_, phrases| !phrases.is_empty());

        // Recompute the vector from remaining seeds
        let all_phrases: Vec<String> = training.values().flat_map(|v| v.clone()).collect();
        if all_phrases.is_empty() {
            // No seeds left — remove the intent entirely
            self.vectors.remove(intent_id);
            self.training.remove(intent_id);
            self.index.remove_intent(intent_id);
        } else {
            let terms = training_to_terms(&all_phrases);
            let vector = LearnedVector::from_seed(terms);
            self.vectors.insert(intent_id.to_string(), vector);
            self.rebuild_index();
        }

        self.version += 1;
        true
    }

    /// Remove an intent.
    pub fn remove_intent(&mut self, id: &str) {
        self.require_local();
        self.vectors.remove(id);
        self.training.remove(id);
        self.index.remove_intent(id);
        self.intent_types.remove(id);
        self.descriptions.remove(id);
        self.metadata.remove(id);

        // Remove paraphrase phrases pointing to this intent
        self.paraphrase_phrases.retain(|_, (intent, _)| intent != id);
        self.rebuild_paraphrase_automaton();
        self.version += 1;
    }

    /// Route a query to matching intents, ranked by score.
    ///
    /// Returns up to `top_k` results (default 10), sorted by score descending.
    /// Empty results means no intent matched any query terms.
    /// Supports both Latin and CJK scripts via dual-path extraction.

    pub fn check_seed(&self, intent_id: &str, seed: &str) -> SeedCheckResult {
        let terms = tokenize(seed);
        // For collision/redundancy, only check unigrams (no spaces).
        // Bigrams are useful for routing but checking them for collisions
        // produces false positives ("refund money" bigram won't be in index).
        let content_terms: Vec<&String> = terms.iter()
            .filter(|t| !t.is_empty() && !t.contains(' '))
            .collect();

        // Empty check
        if content_terms.is_empty() {
            return SeedCheckResult {
                added: false,
                new_terms: vec![],
                conflicts: vec![],
                redundant: false,
                warning: Some("No content terms after tokenization".to_string()),
            };
        }

        let mut new_terms = Vec::new();
        let mut conflicts = Vec::new();
        let mut already_in_intent = 0;

        for term in &content_terms {
            let postings = self.index.postings(term);

            if postings.is_empty() {
                // New term — not in any intent, safe
                new_terms.push(term.to_string());
                continue;
            }

            // Check if term is already in the target intent
            let in_target = postings.iter().any(|(id, _)| id == intent_id);
            if in_target {
                already_in_intent += 1;
            } else {
                new_terms.push(term.to_string());
            }

            // Only check collision if the term is exclusive to one other intent.
            // If already shared across 2+ intents, IDF handles it — no new damage.
            let other_intents: Vec<&(String, f32)> = postings.iter()
                .filter(|(id, _)| id != intent_id)
                .collect();

            if other_intents.len() == 1 {
                // Term is exclusive to one other intent — check severity
                let (other_id, other_weight) = &other_intents[0];
                let total_weight: f32 = postings.iter().map(|(_, w)| w).sum();
                if total_weight > 0.0 {
                    let severity = other_weight / total_weight;
                    // Only flag if the term is truly primary in the other intent:
                    // high severity (>0.7 = mostly in that intent) AND high weight (>0.7 = important seed term)
                    if severity > 0.7 && *other_weight > 0.7 {
                        conflicts.push(TermConflict {
                            term: term.to_string(),
                            competing_intent: other_id.clone(),
                            severity,
                            competing_weight: *other_weight,
                        });
                    }
                }
            }
            // If term is in 2+ other intents, it's already shared — no flag
        }

        // Redundancy: all content unigrams already exist in this intent
        let redundant = content_terms.len() > 0 && already_in_intent == content_terms.len() && new_terms.is_empty();

        // Build warning message
        let warning = if redundant {
            Some("All terms already covered by existing seeds".to_string())
        } else if !conflicts.is_empty() {
            let msgs: Vec<String> = conflicts.iter()
                .map(|c| format!("'{}' is primary in {} ({:.0}%)", c.term, c.competing_intent, c.severity * 100.0))
                .collect();
            Some(format!("Term conflicts: {}", msgs.join("; ")))
        } else {
            None
        };

        SeedCheckResult {
            added: false, // not added yet — this is just a check
            new_terms,
            conflicts,
            redundant,
            warning,
        }
    }

    /// Add a seed phrase with collision checking. Returns full check result.
    /// Blocks addition if: redundant, empty, collision detected, or at language limit.
    /// Only clean seeds with new, non-conflicting vocabulary are accepted.
    pub fn add_seed_checked(&mut self, intent_id: &str, seed: &str, lang: &str) -> SeedCheckResult {
        let mut result = self.check_seed(intent_id, seed);

        // Block: no content terms
        if result.warning.as_deref() == Some("No content terms after tokenization") {
            return result;
        }

        // Block: redundant
        if result.redundant {
            return result;
        }

        // Block: collision detected
        if !result.conflicts.is_empty() {
            return result;
        }

        // Block: at language limit
        if let Some(lang_map) = self.training.get(intent_id) {
            if let Some(seeds) = lang_map.get(lang) {
                if seeds.len() >= MAX_SEEDS_PER_LANGUAGE {
                    result.warning = Some(format!("Language '{}' has reached max {} seeds", lang, MAX_SEEDS_PER_LANGUAGE));
                    return result;
                }
            }
        }

        // Clean — add it
        result.added = self.add_seed(intent_id, seed, lang);
        result
    }

    /// Internal: add a seed without collision checking.
    /// Use `add_seed_checked()` for the public API with guard.
    pub(crate) fn add_seed(&mut self, intent_id: &str, seed: &str, lang: &str) -> bool {
        self.require_local();
        let lang_map = match self.training.get_mut(intent_id) {
            Some(m) => m,
            None => return false,
        };
        let seeds = lang_map.entry(lang.to_string()).or_default();
        if seeds.len() >= MAX_SEEDS_PER_LANGUAGE {
            return false;
        }
        if seeds.iter().any(|s| s == seed) {
            return true; // already exists, not an error
        }
        seeds.push(seed.to_string());

        // Recompute vector from all seeds
        let all_phrases: Vec<String> = lang_map.values().flat_map(|v| v.clone()).collect();
        let terms = training_to_terms(&all_phrases);
        let vector = LearnedVector::from_seed(terms);
        self.vectors.insert(intent_id.to_string(), vector);
        // Add to paraphrase index too
        self.add_paraphrases(intent_id, &[seed]);
        self.rebuild_index();
        self.version += 1;
        true
    }

    // Set the type of an intent (Action or Context).
    //
    // ```
    // use asv_router::{Router, IntentType};
    //
    // let mut router = Router::new();
    // router.add_intent("check_balance", &["check my balance", "account balance"]);
    // router.set_intent_type("check_balance", IntentType::Context);
    // assert_eq!(router.get_intent_type("check_balance"), IntentType::Context);
    // ```

    // --- Namespace utilities ---
    //
    // Namespaced intents use the convention "namespace:intent_id", e.g.
    // "stripe:charge_card", "slack:send_message", "wechat:transfer".
    //
    // In a unified index all apps share one Router. This gives better IDF
    // discrimination (larger N) and a single 30µs routing call covers all apps.
    // Callers use route_ns() / route_multi_ns() to scope results to one namespace,
    // or read all results and filter by the namespace prefix themselves.
    //
    // Situation patterns serve as app fingerprints in unified mode: a pattern
    // "payment" on "stripe:*" fires when the query mentions "payment" even if
    // "payment" is not in any seed — compensating for generic vocabulary shared
    // across namespaces. For CJK apps this is not needed because CJK compound
    // words are inherently app-specific high-IDF vocabulary.

    /// Extract the namespace prefix from a namespaced intent ID.
    ///
    /// Returns the part before the first colon, or `None` if there is no colon.
    ///
    /// ```
    /// use asv_router::Router;
    /// assert_eq!(Router::intent_namespace("stripe:charge_card"), Some("stripe"));
    /// assert_eq!(Router::intent_namespace("cancel_order"), None);
    /// ```
    pub fn intent_namespace(id: &str) -> Option<&str> {
        let colon = id.find(':')?;
        Some(&id[..colon])
    }

    /// All unique namespace prefixes present in this router.
    ///
    /// Intents without a colon (e.g., "cancel_order") are excluded.
    ///
    /// ```
    /// use asv_router::Router;
    /// let mut r = Router::new();
    /// r.add_intent("stripe:charge_card", &["charge my card"]);
    /// r.add_intent("stripe:refund", &["refund payment"]);
    /// r.add_intent("slack:send_message", &["send a message"]);
    /// let mut ns = r.list_namespaces();
    /// ns.sort();
    /// assert_eq!(ns, vec!["slack", "stripe"]);
    /// ```
    pub fn list_namespaces(&self) -> Vec<String> {
        let mut set = std::collections::HashSet::new();
        for id in self.vectors.keys() {
            if let Some(ns) = Self::intent_namespace(id) {
                set.insert(ns.to_string());
            }
        }
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }

    /// All intent IDs belonging to the given namespace.
    ///
    /// Matches intents whose ID starts with `"<ns>:"`.
    ///
    /// ```
    /// use asv_router::Router;
    /// let mut r = Router::new();
    /// r.add_intent("stripe:charge_card", &["charge my card"]);
    /// r.add_intent("stripe:refund", &["refund payment"]);
    /// r.add_intent("slack:send_message", &["send a message"]);
    /// let mut ids = r.intents_in_namespace("stripe");
    /// ids.sort();
    /// assert_eq!(ids, vec!["stripe:charge_card", "stripe:refund"]);
    /// ```
    pub fn intents_in_namespace(&self, ns: &str) -> Vec<String> {
        let prefix = format!("{}:", ns);
        let mut ids: Vec<String> = self.vectors.keys()
            .filter(|id| id.starts_with(&prefix))
            .cloned()
            .collect();
        ids.sort();
        ids
    }
}
