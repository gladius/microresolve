//! Router: co-occurrence tracking, temporal flow, workflow discovery.

use crate::*;
use crate::tokenizer::*;
use crate::vector::LearnedVector;
use crate::index::InvertedIndex;
use std::collections::{HashMap, HashSet};

impl Router {
    pub fn record_co_occurrence(&mut self, intent_ids: &[&str]) {
        for i in 0..intent_ids.len() {
            for j in (i + 1)..intent_ids.len() {
                let (a, b) = if intent_ids[i] < intent_ids[j] {
                    (intent_ids[i].to_string(), intent_ids[j].to_string())
                } else {
                    (intent_ids[j].to_string(), intent_ids[i].to_string())
                };
                *self.co_occurrence.entry((a, b)).or_insert(0) += 1;
            }
        }
    }

    /// Get co-occurrence data as a list of (intent_a, intent_b, count) sorted by count desc.
    pub fn get_co_occurrence(&self) -> Vec<(&str, &str, u32)> {
        let mut pairs: Vec<(&str, &str, u32)> = self.co_occurrence
            .iter()
            .map(|((a, b), &count)| (a.as_str(), b.as_str(), count))
            .collect();
        pairs.sort_by(|a, b| b.2.cmp(&a.2));
        pairs
    }

    /// Clear co-occurrence data.
    pub fn clear_co_occurrence(&mut self) {
        self.co_occurrence.clear();
        self.temporal_order.clear();
        self.intent_sequences.clear();
    }

    /// Record a full intent sequence from route_multi (in positional order).
    /// This records co-occurrence, temporal ordering, and the full sequence.
    pub fn record_intent_sequence(&mut self, ordered_intent_ids: &[&str]) {
        if ordered_intent_ids.len() < 2 {
            return;
        }

        // Record pairwise co-occurrence (lexicographic keys)
        for i in 0..ordered_intent_ids.len() {
            for j in (i + 1)..ordered_intent_ids.len() {
                let (a, b) = if ordered_intent_ids[i] < ordered_intent_ids[j] {
                    (ordered_intent_ids[i].to_string(), ordered_intent_ids[j].to_string())
                } else {
                    (ordered_intent_ids[j].to_string(), ordered_intent_ids[i].to_string())
                };
                *self.co_occurrence.entry((a, b)).or_insert(0) += 1;
            }
        }

        // Record temporal ordering (positional order, not lexicographic)
        for i in 0..ordered_intent_ids.len() {
            for j in (i + 1)..ordered_intent_ids.len() {
                let key = (ordered_intent_ids[i].to_string(), ordered_intent_ids[j].to_string());
                *self.temporal_order.entry(key).or_insert(0) += 1;
            }
        }

        // Record full sequence (capped at 1000)
        let seq: Vec<String> = ordered_intent_ids.iter().map(|s| s.to_string()).collect();
        self.intent_sequences.push(seq);
        if self.intent_sequences.len() > 1000 {
            self.intent_sequences.remove(0);
        }
    }

    /// Get temporal ordering: P(B appears after A | A and B co-occur).
    /// Returns (first, second, probability, count) sorted by count desc.
    pub fn get_temporal_order(&self) -> Vec<(&str, &str, f32, u32)> {
        let mut result: Vec<(&str, &str, f32, u32)> = Vec::new();
        // For each co-occurrence pair, check temporal direction
        for ((a, b), &total) in &self.co_occurrence {
            let a_before_b = self.temporal_order.get(&(a.clone(), b.clone())).copied().unwrap_or(0);
            let b_before_a = self.temporal_order.get(&(b.clone(), a.clone())).copied().unwrap_or(0);

            if a_before_b >= b_before_a && a_before_b > 0 {
                let prob = a_before_b as f32 / total as f32;
                result.push((a.as_str(), b.as_str(), prob, a_before_b));
            }
            if b_before_a > a_before_b {
                let prob = b_before_a as f32 / total as f32;
                result.push((b.as_str(), a.as_str(), prob, b_before_a));
            }
        }
        result.sort_by(|a, b| b.3.cmp(&a.3));
        result
    }

    /// Discover intent workflows (clusters) from co-occurrence data.
    ///
    /// Uses connected-component analysis on the co-occurrence graph.
    /// Only includes edges with at least `min_observations` co-occurrences.
    /// Returns clusters sorted by size (largest first), each cluster sorted by
    /// most-connected intent first.
    pub fn discover_workflows(&self, min_observations: u32) -> Vec<Vec<WorkflowIntent>> {
        // Build adjacency list from co-occurrence pairs above threshold
        let mut adj: HashMap<&str, Vec<(&str, u32)>> = HashMap::new();
        for ((a, b), &count) in &self.co_occurrence {
            if count < min_observations {
                continue;
            }
            adj.entry(a.as_str()).or_default().push((b.as_str(), count));
            adj.entry(b.as_str()).or_default().push((a.as_str(), count));
        }

        // Connected components via BFS
        let mut visited: HashSet<&str> = HashSet::new();
        let mut clusters: Vec<Vec<WorkflowIntent>> = Vec::new();

        for &start in adj.keys() {
            if visited.contains(start) {
                continue;
            }
            let mut component: Vec<&str> = Vec::new();
            let mut queue: Vec<&str> = vec![start];

            while let Some(node) = queue.pop() {
                if visited.contains(node) {
                    continue;
                }
                visited.insert(node);
                component.push(node);
                if let Some(neighbors) = adj.get(node) {
                    for &(neighbor, _) in neighbors {
                        if !visited.contains(neighbor) {
                            queue.push(neighbor);
                        }
                    }
                }
            }

            if component.len() >= 2 {
                // Build WorkflowIntent entries with connection strength
                let mut workflow: Vec<WorkflowIntent> = component.iter().map(|&id| {
                    let connections: u32 = adj.get(id)
                        .map(|n| n.iter().filter(|(nid, _)| component.contains(nid)).map(|(_, c)| c).sum())
                        .unwrap_or(0);
                    let neighbors: Vec<String> = adj.get(id)
                        .map(|n| n.iter().filter(|(nid, _)| component.contains(nid)).map(|(nid, _)| nid.to_string()).collect())
                        .unwrap_or_default();
                    WorkflowIntent {
                        id: id.to_string(),
                        connections,
                        neighbors,
                    }
                }).collect();
                workflow.sort_by(|a, b| b.connections.cmp(&a.connections));
                clusters.push(workflow);
            }
        }

        clusters.sort_by(|a, b| b.len().cmp(&a.len()));
        clusters
    }

    /// Detect escalation patterns: sequences where intents progress from
    /// routine to urgent (e.g., track → complaint → contact_human).
    ///
    /// Returns sequences that occur at least `min_occurrences` times,
    /// sorted by frequency.
    pub fn detect_escalation_patterns(&self, min_occurrences: u32) -> Vec<EscalationPattern> {
        // Count subsequences of length 2 and 3 from recorded sequences
        let mut subseq_counts: HashMap<Vec<String>, u32> = HashMap::new();

        for seq in &self.intent_sequences {
            // Length-2 subsequences (pairs in order)
            for i in 0..seq.len() {
                for j in (i + 1)..seq.len() {
                    let sub = vec![seq[i].clone(), seq[j].clone()];
                    *subseq_counts.entry(sub).or_insert(0) += 1;
                }
                // Length-3 subsequences (triples in order)
                for j in (i + 1)..seq.len() {
                    for k in (j + 1)..seq.len() {
                        let sub = vec![seq[i].clone(), seq[j].clone(), seq[k].clone()];
                        *subseq_counts.entry(sub).or_insert(0) += 1;
                    }
                }
            }
        }

        let total_sequences = self.intent_sequences.len() as f32;
        let mut patterns: Vec<EscalationPattern> = subseq_counts.into_iter()
            .filter(|(_, count)| *count >= min_occurrences)
            .map(|(sequence, count)| {
                let frequency = if total_sequences > 0.0 { count as f32 / total_sequences } else { 0.0 };
                EscalationPattern {
                    sequence,
                    occurrences: count,
                    frequency,
                }
            })
            .collect();
        patterns.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));
        patterns
    }

    /// Get total co-occurrence count for a specific intent (how many times it appeared with ANY other intent).
    pub(crate) fn co_occurrence_total(&self, intent_id: &str) -> u32 {
        self.co_occurrence.iter()
            .filter(|((a, b), _)| a == intent_id || b == intent_id)
            .map(|(_, &count)| count)
            .sum()
    }

    /// Get suggested intents based on co-occurrence patterns.
    ///
    /// Given a set of detected intent IDs, returns intents that frequently co-occur
    /// but were NOT detected in this query. Each suggestion includes the conditional
    /// probability P(suggested | detected) and the observation count.
    ///
    /// Only returns suggestions with at least `min_observations` co-occurrences
    /// and conditional probability >= `min_probability`.
    ///
    /// This enables proactive routing: "You asked to cancel. 73% of customers
    /// also want a refund — would you like me to process that too?"
    pub fn suggest_intents(
        &self,
        detected_ids: &[&str],
        min_observations: u32,
        min_probability: f32,
    ) -> Vec<IntentSuggestion> {
        let detected_set: HashSet<&str> = detected_ids.iter().copied().collect();
        let mut suggestions: HashMap<String, (f32, u32, String)> = HashMap::new(); // id -> (max_prob, max_count, because_of)

        for &detected_id in detected_ids {
            let total = self.co_occurrence_total(detected_id);
            if total == 0 {
                continue;
            }

            for ((a, b), &count) in &self.co_occurrence {
                let other = if a == detected_id {
                    b.as_str()
                } else if b == detected_id {
                    a.as_str()
                } else {
                    continue;
                };

                if detected_set.contains(other) || count < min_observations {
                    continue;
                }

                let probability = count as f32 / total as f32;
                if probability < min_probability {
                    continue;
                }

                let entry = suggestions.entry(other.to_string())
                    .or_insert((0.0, 0, String::new()));
                if probability > entry.0 {
                    *entry = (probability, count, detected_id.to_string());
                }
            }
        }

        let mut result: Vec<IntentSuggestion> = suggestions.into_iter()
            .map(|(id, (probability, count, because_of))| IntentSuggestion {
                id,
                probability,
                observations: count,
                because_of,
            })
            .collect();
        result.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Get the current version number. Incremented on every mutation.
    pub fn version(&self) -> u64 {
        self.version
    }

    // Build distributional similarity index from text corpus.
    // Uses Random Indexing (same as neocortex Space 4):
    // - Each word gets a deterministic random vector (128-dim)
    // - For each word, accumulate random vectors of neighboring words
    // - Words in similar contexts get similar accumulated vectors
    // - Cosine similarity between vectors = distributional similarity
    //
    // Call this with accumulated queries/text to improve routing accuracy.
    // The more text, the better the similarity estimates.

}
