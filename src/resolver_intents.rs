//! Resolver: intent management and phrase storage.

use crate::types::MAX_PHRASES_PER_LANGUAGE;
use crate::*;
use crate::{FxHashMap, FxHashSet};

impl Resolver {
    /// Add an intent with seed phrases.
    ///
    /// `seeds` accepts either `&[&str]` (defaults to English) or a
    /// `HashMap<String, Vec<String>>` keyed by language code. See
    /// [`IntentSeeds`] for more variants.
    ///
    /// Returns the number of phrases indexed (capped per-language at
    /// [`MAX_PHRASES_PER_LANGUAGE`]). The `Result` shape is reserved so
    /// future validation paths can surface errors without breaking callers.
    pub fn add_intent(&mut self, id: &str, seeds: impl Into<IntentSeeds>) -> Result<usize, Error> {
        let seeds_by_lang: FxHashMap<String, Vec<String>> = match seeds.into() {
            IntentSeeds::Mono(phrases) => {
                let mut m = FxHashMap::default();
                m.insert("en".to_string(), phrases);
                m
            }
            IntentSeeds::Multi(m) => m.into_iter().collect(),
        };

        // Truncate per-language to the configured cap.
        let truncated: FxHashMap<String, Vec<String>> = seeds_by_lang
            .into_iter()
            .map(|(lang, seeds)| {
                let limited: Vec<String> =
                    seeds.into_iter().take(MAX_PHRASES_PER_LANGUAGE).collect();
                (lang, limited)
            })
            .collect();

        // Snapshot all existing (token, id) pairs before indexing so we can diff.
        let pre_pairs: Vec<(&str, &str)> = {
            // We'll snapshot empty — new intent has no prior weights.
            vec![]
        };
        let _ = pre_pairs; // intentionally unused; we collect post-weights differently

        let mut total_phrases = 0usize;
        for phrases in truncated.values() {
            for phrase in phrases {
                self.index_phrase_no_rebuild(id, phrase);
                total_phrases += 1;
            }
        }
        let training_std: std::collections::HashMap<String, Vec<String>> =
            truncated.into_iter().collect();
        self.training.insert(id.to_string(), training_std.clone());

        // Collect post-weights for this intent.
        let weight_changes = self.intent_weight_pairs(id);
        let weight_snap: std::collections::HashMap<(String, String), f32> = weight_changes
            .iter()
            .map(|(t, i)| ((t.clone(), i.clone()), 0.0_f32))
            .collect();
        let changes = self.diff_weights(&weight_snap);

        let phrases_by_lang: std::collections::HashMap<String, Vec<String>> =
            self.training.get(id).cloned().unwrap_or_default();

        let mut ops: Vec<crate::oplog::Op> = vec![crate::oplog::Op::IntentAdded {
            id: id.to_string(),
            phrases_by_lang,
            intent_type: None,
            description: None,
            instructions: None,
            persona: None,
        }];
        if !changes.is_empty() {
            ops.push(crate::oplog::Op::WeightUpdates { changes });
        }
        self.bump_with_ops(ops);

        Ok(total_phrases)
    }

    /// Remove a single phrase from an intent.
    /// Returns true if the phrase was found and removed.
    pub fn remove_phrase(&mut self, intent_id: &str, seed: &str) -> bool {
        let training = match self.training.get_mut(intent_id) {
            Some(t) => t,
            None => return false,
        };

        let mut found = false;
        for phrases in training.values_mut() {
            if let Some(pos) = phrases.iter().position(|s| s == seed) {
                phrases.remove(pos);
                found = true;
                break;
            }
        }

        if !found {
            return false;
        }

        // Remove empty language entries
        training.retain(|_, phrases| !phrases.is_empty());

        // Snapshot all tokens before rebuild so we can compute post-values.
        let pairs_before: Vec<(String, String)> = self.intent_weight_pairs(intent_id);
        let snap: std::collections::HashMap<(String, String), f32> = pairs_before
            .iter()
            .map(|(t, i)| {
                (
                    (t.clone(), i.clone()),
                    self.index.get_weight(t, i).unwrap_or(0.0),
                )
            })
            .collect();

        // Rebuild the index from remaining phrases so stale word→intent edges are cleared.
        self.rebuild_index();

        let changes = self.diff_weights(&snap);
        let mut ops: Vec<crate::oplog::Op> = vec![crate::oplog::Op::PhraseRemoved {
            intent_id: intent_id.to_string(),
            phrase: seed.to_string(),
        }];
        if !changes.is_empty() {
            ops.push(crate::oplog::Op::WeightUpdates { changes });
        }
        self.bump_with_ops(ops);
        true
    }

    /// Remove an intent and all its data.
    pub fn remove_intent(&mut self, id: &str) {
        self.training.remove(id);
        self.intent_types.remove(id);
        self.descriptions.remove(id);
        self.instructions.remove(id);
        self.persona.remove(id);
        self.sources.remove(id);
        self.targets.remove(id);
        self.schemas.remove(id);
        self.guardrails.remove(id);
        // Rebuild the index from remaining phrases so stale word→intent edges are cleared.
        self.rebuild_index();
        self.bump_with_ops(vec![crate::oplog::Op::IntentRemoved { id: id.to_string() }]);
    }

    /// Check a phrase before adding it. Returns duplicate/empty info.
    /// Conflict detection (index-based) has been removed; conflicts always empty.
    pub fn check_phrase(&self, intent_id: &str, seed: &str) -> PhraseCheckResult {
        if seed.trim().is_empty() {
            return PhraseCheckResult {
                added: false,
                redundant: false,
                warning: Some("Phrase is empty".to_string()),
            };
        }

        // Check for exact duplicate in any language
        let is_duplicate = self
            .training
            .get(intent_id)
            .map(|m| m.values().any(|phrases| phrases.iter().any(|p| p == seed)))
            .unwrap_or(false);

        PhraseCheckResult {
            added: false,
            redundant: is_duplicate,
            warning: if is_duplicate {
                Some("Phrase already exists in this intent".to_string())
            } else {
                None
            },
        }
    }

    /// Add a phrase with duplicate checking. Blocks redundant/empty phrases.
    pub fn add_phrase_checked(
        &mut self,
        intent_id: &str,
        seed: &str,
        lang: &str,
    ) -> PhraseCheckResult {
        let mut result = self.check_phrase(intent_id, seed);

        if result.warning.as_deref() == Some("Phrase is empty") {
            return result;
        }
        if result.redundant {
            return result;
        }

        // Block: at language limit
        if let Some(lang_map) = self.training.get(intent_id) {
            if let Some(seeds) = lang_map.get(lang) {
                if seeds.len() >= MAX_PHRASES_PER_LANGUAGE {
                    result.warning = Some(format!(
                        "Language '{}' has reached max {} phrases",
                        lang, MAX_PHRASES_PER_LANGUAGE
                    ));
                    return result;
                }
            }
        }

        result.added = self.add_phrase(intent_id, seed, lang);
        result
    }

    pub(crate) fn add_phrase(&mut self, intent_id: &str, seed: &str, lang: &str) -> bool {
        let lang_map = match self.training.get_mut(intent_id) {
            Some(m) => m,
            None => return false,
        };
        let seeds = lang_map.entry(lang.to_string()).or_default();
        if seeds.len() >= MAX_PHRASES_PER_LANGUAGE {
            return false;
        }
        if seeds.iter().any(|s| s == seed) {
            return true;
        }
        seeds.push(seed.to_string());

        // Snapshot before indexing.
        let snap: std::collections::HashMap<(String, String), f32> =
            std::collections::HashMap::new();

        // Index phrase into L2 atomically.
        self.index_phrase(intent_id, seed);

        // Collect weight changes for all tokens in this phrase.
        let words = crate::tokenizer::tokenize(seed);
        let mut changes: Vec<(String, String, f32)> = Vec::new();
        for word in &words {
            if let Some(w) = self.index.get_weight(word, intent_id) {
                // compare against zero — this phrase's tokens didn't exist before (or changed)
                let before = snap
                    .get(&(word.clone(), intent_id.to_string()))
                    .copied()
                    .unwrap_or(0.0);
                if (w - before).abs() > 1e-6 {
                    changes.push((word.clone(), intent_id.to_string(), w));
                }
            }
        }

        let mut ops: Vec<crate::oplog::Op> = vec![crate::oplog::Op::PhraseAdded {
            intent_id: intent_id.to_string(),
            phrase: seed.to_string(),
            lang: lang.to_string(),
        }];
        if !changes.is_empty() {
            ops.push(crate::oplog::Op::WeightUpdates { changes });
        }
        self.bump_with_ops(ops);
        true
    }

    /// Extract the namespace prefix from a namespaced intent ID.
    pub fn intent_namespace(id: &str) -> Option<&str> {
        let colon = id.find(':')?;
        Some(&id[..colon])
    }

    /// All unique namespace prefixes present in this namespace.
    pub fn list_namespaces(&self) -> Vec<String> {
        let mut set = FxHashSet::default();
        for id in self.training.keys() {
            if let Some(ns) = Self::intent_namespace(id) {
                set.insert(ns.to_string());
            }
        }
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    }

    /// All intent IDs belonging to the given namespace.
    pub fn intents_in_namespace(&self, ns: &str) -> Vec<String> {
        let prefix = format!("{}:", ns);
        let mut ids: Vec<String> = self
            .training
            .keys()
            .filter(|id| id.starts_with(&prefix))
            .cloned()
            .collect();
        ids.sort();
        ids
    }
}
