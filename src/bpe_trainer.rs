//! BPE (Byte Pair Encoding) Trainer
//!
//! Train BPE tokenizer from scratch on text data.
//! Implements the standard BPE algorithm as described in the original paper.

use hashbrown::HashMap;
use rayon::prelude::*;
use std::cmp::Ordering;

/// BPE Trainer configuration
#[derive(Debug, Clone)]
pub struct BpeTrainerConfig {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum frequency for a pair to be merged
    pub min_frequency: u32,
    /// Special tokens to add to vocabulary
    pub special_tokens: Vec<String>,
    /// Whether to show progress
    pub show_progress: bool,
    /// Initial alphabet (None = derive from data)
    pub initial_alphabet: Option<Vec<char>>,
    /// Limit the number of training samples
    pub limit_alphabet: Option<usize>,
    /// Continuing subword prefix (e.g., "##" for WordPiece style)
    pub continuing_subword_prefix: Option<String>,
    /// End of word suffix (e.g., "</w>" for original BPE)
    pub end_of_word_suffix: Option<String>,
}

impl Default for BpeTrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            min_frequency: 2,
            special_tokens: vec![
                "<unk>".to_string(),
                "<pad>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
            ],
            show_progress: true,
            initial_alphabet: None,
            limit_alphabet: None,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
        }
    }
}

/// Pair with frequency for priority queue
#[derive(Debug, Clone, Eq, PartialEq)]
struct PairFreq {
    pair: (String, String),
    freq: u32,
}

impl Ord for PairFreq {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher frequency = higher priority
        self.freq.cmp(&other.freq)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for PairFreq {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// BPE Trainer
pub struct BpeTrainer {
    config: BpeTrainerConfig,
}

impl BpeTrainer {
    /// Create new BPE trainer with configuration
    pub fn new(config: BpeTrainerConfig) -> Self {
        Self { config }
    }

    /// Create trainer with default configuration
    pub fn with_vocab_size(vocab_size: usize) -> Self {
        Self {
            config: BpeTrainerConfig {
                vocab_size,
                ..Default::default()
            },
        }
    }

    /// Train BPE model on text data
    ///
    /// Returns (vocab, merges) where:
    /// - vocab: HashMap<String, u32> mapping tokens to IDs
    /// - merges: Vec<(String, String)> list of merge operations in order
    pub fn train(&self, texts: &[&str]) -> (HashMap<String, u32>, Vec<(String, String)>) {
        // Step 1: Build initial word frequencies
        let word_freqs = self.build_word_frequencies(texts);

        // Step 2: Build initial character vocabulary
        let mut vocab = self.build_initial_vocab(&word_freqs);

        // Step 3: Split words into characters
        let mut word_splits: HashMap<String, Vec<String>> = word_freqs
            .keys()
            .map(|word| {
                let splits = self.split_word(word);
                (word.clone(), splits)
            })
            .collect();

        // Step 4: Iteratively merge most frequent pairs
        let mut merges: Vec<(String, String)> = Vec::new();

        while vocab.len() < self.config.vocab_size {
            // Count pair frequencies
            let pair_freqs = self.count_pairs(&word_splits, &word_freqs);

            if pair_freqs.is_empty() {
                break;
            }

            // Find best pair
            let best_pair = pair_freqs
                .iter()
                .max_by_key(|(_, freq)| *freq)
                .map(|(pair, _)| pair.clone());

            let best_pair = match best_pair {
                Some(pair) => pair,
                None => break,
            };

            let freq = *pair_freqs.get(&best_pair).unwrap_or(&0);
            if freq < self.config.min_frequency {
                break;
            }

            // Create merged token
            let merged = format!("{}{}", best_pair.0, best_pair.1);

            // Add to vocabulary
            let next_id = vocab.len() as u32;
            vocab.insert(merged.clone(), next_id);

            // Record merge
            merges.push(best_pair.clone());

            // Update word splits
            for (_, splits) in word_splits.iter_mut() {
                *splits = self.merge_pair(splits, &best_pair.0, &best_pair.1, &merged);
            }

            if self.config.show_progress && merges.len() % 1000 == 0 {
                eprintln!(
                    "BPE training: {} merges, vocab size: {}",
                    merges.len(),
                    vocab.len()
                );
            }
        }

        (vocab, merges)
    }

    /// Train on an iterator of texts (for large datasets)
    pub fn train_from_iterator<'a, I>(&self, texts: I) -> (HashMap<String, u32>, Vec<(String, String)>)
    where
        I: Iterator<Item = &'a str>,
    {
        let collected: Vec<&str> = texts.collect();
        self.train(&collected)
    }

    /// Build word frequency map from texts
    fn build_word_frequencies(&self, texts: &[&str]) -> HashMap<String, u32> {
        let word_freqs: HashMap<String, u32> = texts
            .par_iter()
            .flat_map(|text| {
                text.split_whitespace()
                    .map(|word| {
                        // Apply end-of-word suffix if configured
                        if let Some(ref suffix) = self.config.end_of_word_suffix {
                            format!("{}{}", word, suffix)
                        } else {
                            word.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .fold(
                || HashMap::new(),
                |mut acc, word| {
                    *acc.entry(word).or_insert(0) += 1;
                    acc
                },
            )
            .reduce(
                || HashMap::new(),
                |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                },
            );

        word_freqs
    }

    /// Build initial vocabulary from characters
    fn build_initial_vocab(&self, word_freqs: &HashMap<String, u32>) -> HashMap<String, u32> {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut next_id = 0u32;

        // Add special tokens first
        for token in &self.config.special_tokens {
            vocab.insert(token.clone(), next_id);
            next_id += 1;
        }

        // Add initial alphabet if specified
        if let Some(ref alphabet) = self.config.initial_alphabet {
            for c in alphabet {
                let s = c.to_string();
                if !vocab.contains_key(&s) {
                    vocab.insert(s, next_id);
                    next_id += 1;
                }
            }
        }

        // Add all characters from the data
        let mut char_freqs: HashMap<char, u32> = HashMap::new();
        for (word, freq) in word_freqs {
            for c in word.chars() {
                *char_freqs.entry(c).or_insert(0) += freq;
            }
        }

        // Sort by frequency and add to vocab
        let mut char_vec: Vec<_> = char_freqs.into_iter().collect();
        char_vec.sort_by(|a, b| b.1.cmp(&a.1));

        // Limit alphabet if configured
        let limit = self.config.limit_alphabet.unwrap_or(char_vec.len());
        for (c, _) in char_vec.into_iter().take(limit) {
            let s = c.to_string();
            if !vocab.contains_key(&s) {
                vocab.insert(s, next_id);
                next_id += 1;
            }
        }

        vocab
    }

    /// Split word into characters (or subword units)
    fn split_word(&self, word: &str) -> Vec<String> {
        let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // Apply continuing subword prefix if configured
        if let Some(ref prefix) = self.config.continuing_subword_prefix {
            if chars.len() > 1 {
                let mut result = vec![chars[0].clone()];
                for c in chars.into_iter().skip(1) {
                    result.push(format!("{}{}", prefix, c));
                }
                return result;
            }
        }

        chars
    }

    /// Count pair frequencies across all words
    fn count_pairs(
        &self,
        word_splits: &HashMap<String, Vec<String>>,
        word_freqs: &HashMap<String, u32>,
    ) -> HashMap<(String, String), u32> {
        // Convert to Vec for parallel iteration
        let items: Vec<_> = word_splits.iter().collect();

        let pair_freqs: HashMap<(String, String), u32> = items
            .par_iter()
            .filter_map(|(word, splits)| {
                let freq = word_freqs.get(*word).copied().unwrap_or(0);
                if splits.len() < 2 || freq == 0 {
                    return None;
                }

                let mut local_pairs: HashMap<(String, String), u32> = HashMap::new();
                for window in splits.windows(2) {
                    let pair = (window[0].clone(), window[1].clone());
                    *local_pairs.entry(pair).or_insert(0) += freq;
                }
                Some(local_pairs)
            })
            .reduce(
                || HashMap::new(),
                |mut a: HashMap<(String, String), u32>, b: HashMap<(String, String), u32>| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                },
            );

        pair_freqs
    }

    /// Merge a pair in the splits
    fn merge_pair(
        &self,
        splits: &[String],
        first: &str,
        second: &str,
        merged: &str,
    ) -> Vec<String> {
        let mut result = Vec::with_capacity(splits.len());
        let mut i = 0;

        while i < splits.len() {
            if i < splits.len() - 1 && splits[i] == first && splits[i + 1] == second {
                result.push(merged.to_string());
                i += 2;
            } else {
                result.push(splits[i].clone());
                i += 1;
            }
        }

        result
    }

    /// Get configuration reference
    pub fn config(&self) -> &BpeTrainerConfig {
        &self.config
    }
}

/// Builder for BpeTrainerConfig
pub struct BpeTrainerBuilder {
    config: BpeTrainerConfig,
}

impl BpeTrainerBuilder {
    pub fn new() -> Self {
        Self {
            config: BpeTrainerConfig::default(),
        }
    }

    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    pub fn min_frequency(mut self, freq: u32) -> Self {
        self.config.min_frequency = freq;
        self
    }

    pub fn special_tokens(mut self, tokens: Vec<String>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    pub fn initial_alphabet(mut self, alphabet: Vec<char>) -> Self {
        self.config.initial_alphabet = Some(alphabet);
        self
    }

    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    pub fn continuing_subword_prefix(mut self, prefix: &str) -> Self {
        self.config.continuing_subword_prefix = Some(prefix.to_string());
        self
    }

    pub fn end_of_word_suffix(mut self, suffix: &str) -> Self {
        self.config.end_of_word_suffix = Some(suffix.to_string());
        self
    }

    pub fn build(self) -> BpeTrainer {
        BpeTrainer::new(self.config)
    }
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_trainer_basic() {
        let trainer = BpeTrainerBuilder::new()
            .vocab_size(100)
            .min_frequency(1)
            .show_progress(false)
            .build();

        let texts = vec![
            "hello world",
            "hello there",
            "world hello",
            "hello hello hello",
        ];

        let (vocab, merges) = trainer.train(&texts);

        // Should have at least special tokens + characters
        assert!(vocab.len() >= 4);
        // Should have some merges
        assert!(!merges.is_empty() || vocab.len() <= 26);
    }

    #[test]
    fn test_bpe_trainer_with_suffix() {
        let trainer = BpeTrainerBuilder::new()
            .vocab_size(50)
            .min_frequency(1)
            .end_of_word_suffix("</w>")
            .show_progress(false)
            .build();

        let texts = vec!["hello world"];
        let (vocab, _) = trainer.train(&texts);

        // Should have </w> suffix in some tokens
        assert!(vocab.keys().any(|k| k.contains("</w>")));
    }

    #[test]
    fn test_bpe_trainer_config() {
        let config = BpeTrainerConfig {
            vocab_size: 10000,
            min_frequency: 5,
            ..Default::default()
        };

        let trainer = BpeTrainer::new(config);
        assert_eq!(trainer.config().vocab_size, 10000);
        assert_eq!(trainer.config().min_frequency, 5);
    }
}
