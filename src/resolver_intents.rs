//! Resolver: intent management and phrase storage.

use crate::types::MAX_PHRASES_PER_LANGUAGE;
use crate::*;
use std::collections::HashMap;

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
        let seeds_by_lang: HashMap<String, Vec<String>> = match seeds.into() {
            IntentSeeds::Mono(phrases) => {
                let mut m = HashMap::new();
                m.insert("en".to_string(), phrases);
                m
            }
            IntentSeeds::Multi(m) => m,
        };

        // Truncate per-language to the configured cap.
        let truncated: HashMap<String, Vec<String>> = seeds_by_lang
            .into_iter()
            .map(|(lang, seeds)| {
                let limited: Vec<String> =
                    seeds.into_iter().take(MAX_PHRASES_PER_LANGUAGE).collect();
                (lang, limited)
            })
            .collect();

        // Index all phrases into L2 before storing — rebuild L0 once at end.
        let mut total_phrases = 0usize;
        for phrases in truncated.values() {
            for phrase in phrases {
                self.index_phrase_no_rebuild(id, phrase);
                total_phrases += 1;
            }
        }
        self.training.insert(id.to_string(), truncated);
        self.rebuild_l0();
        self.version += 1;

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

        // Rebuild L2 from remaining phrases so stale word→intent edges are cleared.
        self.rebuild_l2();

        self.version += 1;
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
        // Rebuild L2 from remaining phrases so stale word→intent edges are cleared.
        self.rebuild_l2();
        self.version += 1;
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
        // Index phrase into L2 atomically.
        self.index_phrase(intent_id, seed);
        self.version += 1;
        true
    }

    /// Extract the namespace prefix from a namespaced intent ID.
    pub fn intent_namespace(id: &str) -> Option<&str> {
        let colon = id.find(':')?;
        Some(&id[..colon])
    }

    /// All unique namespace prefixes present in this namespace.
    pub fn list_namespaces(&self) -> Vec<String> {
        let mut set = std::collections::HashSet::new();
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
