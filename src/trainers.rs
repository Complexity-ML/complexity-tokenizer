//! Trainers for different tokenization models
//!
//! - WordPiece trainer (BERT style)
//! - Unigram trainer (SentencePiece style)

use crate::models::{WordPieceModel, UnigramModel};
use crate::normalizers::Normalizer;
use crate::pretokenizers::PreTokenizer;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// =============================================================================
// WordPiece Trainer
// =============================================================================

/// WordPiece trainer configuration
#[derive(Debug, Clone)]
pub struct WordPieceTrainerConfig {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum frequency for a token to be included
    pub min_frequency: u32,
    /// Special tokens to include
    pub special_tokens: Vec<String>,
    /// Continuing subword prefix (usually "##")
    pub continuing_subword_prefix: String,
    /// End of word suffix (optional, for variants)
    pub end_of_word_suffix: Option<String>,
    /// Maximum characters per word
    pub max_input_chars_per_word: usize,
    /// Normalizer to apply
    pub normalizer: Option<Normalizer>,
    /// Pre-tokenizer to apply
    pub pre_tokenizer: Option<PreTokenizer>,
}

impl Default for WordPieceTrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            min_frequency: 2,
            special_tokens: vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[MASK]".to_string(),
            ],
            continuing_subword_prefix: "##".to_string(),
            end_of_word_suffix: None,
            max_input_chars_per_word: 100,
            normalizer: Some(Normalizer::Sequence(vec![
                Normalizer::NFC,
                Normalizer::Lowercase,
            ])),
            pre_tokenizer: Some(PreTokenizer::Whitespace),
        }
    }
}

/// WordPiece trainer
pub struct WordPieceTrainer {
    config: WordPieceTrainerConfig,
    vocab: HashMap<String, u32>,
}

impl WordPieceTrainer {
    /// Create new WordPiece trainer
    pub fn new(config: WordPieceTrainerConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
        }
    }

    /// Pre-tokenize text
    fn pretokenize(&self, text: &str) -> Vec<String> {
        let normalized = match &self.config.normalizer {
            Some(n) => n.normalize(text),
            None => text.to_string(),
        };

        match &self.config.pre_tokenizer {
            Some(pt) => pt.pre_tokenize(&normalized),
            None => normalized.split_whitespace().map(|s| s.to_string()).collect(),
        }
    }

    /// Train from files
    pub fn train<P: AsRef<Path>>(&mut self, files: &[P]) -> Result<WordPieceModel, std::io::Error> {
        // Count word frequencies
        let mut word_freqs: HashMap<String, u32> = HashMap::new();

        for file_path in files {
            let file = File::open(file_path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                let words = self.pretokenize(&line);
                for word in words {
                    *word_freqs.entry(word).or_insert(0) += 1;
                }
            }
        }

        self.train_from_word_freqs(word_freqs)
    }

    /// Train from an iterator of texts
    pub fn train_from_texts<I, S>(&mut self, texts: I) -> Result<WordPieceModel, std::io::Error>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        let mut word_freqs: HashMap<String, u32> = HashMap::new();

        for text in texts {
            let words = self.pretokenize(text.as_ref());
            for word in words {
                *word_freqs.entry(word).or_insert(0) += 1;
            }
        }

        self.train_from_word_freqs(word_freqs)
    }

    /// Train from word frequencies
    fn train_from_word_freqs(&mut self, word_freqs: HashMap<String, u32>) -> Result<WordPieceModel, std::io::Error> {
        // Filter by min frequency
        let word_freqs: HashMap<String, u32> = word_freqs
            .into_iter()
            .filter(|(_, freq)| *freq >= self.config.min_frequency)
            .collect();

        // Initialize vocab with special tokens
        let mut next_id = 0u32;
        for token in &self.config.special_tokens {
            self.vocab.insert(token.clone(), next_id);
            next_id += 1;
        }

        // Collect all unique characters
        let mut chars: HashSet<char> = HashSet::new();
        for word in word_freqs.keys() {
            for c in word.chars() {
                chars.insert(c);
            }
        }

        // Add characters to vocab
        for c in chars {
            let token = c.to_string();
            if !self.vocab.contains_key(&token) {
                self.vocab.insert(token, next_id);
                next_id += 1;
            }
        }

        // Split words into subwords
        let mut subword_freqs: HashMap<String, u32> = HashMap::new();
        for (word, freq) in &word_freqs {
            let chars: Vec<char> = word.chars().collect();
            if chars.is_empty() {
                continue;
            }

            // First character without prefix, rest with prefix
            subword_freqs.entry(chars[0].to_string()).or_insert(0);
            *subword_freqs.entry(chars[0].to_string()).or_insert(0) += freq;

            for c in chars.iter().skip(1) {
                let subword = format!("{}{}", self.config.continuing_subword_prefix, c);
                *subword_freqs.entry(subword).or_insert(0) += freq;
            }
        }

        // Greedy WordPiece training: repeatedly merge most frequent pairs
        while self.vocab.len() < self.config.vocab_size {
            // Count pairs in current tokenizations
            let mut pair_freqs: HashMap<(String, String), u32> = HashMap::new();

            for (word, freq) in &word_freqs {
                let tokens = self.tokenize_for_training(word);
                for i in 0..tokens.len().saturating_sub(1) {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            if pair_freqs.is_empty() {
                break;
            }

            // Find best pair
            let best = pair_freqs
                .iter()
                .max_by_key(|(_, freq)| *freq);

            match best {
                Some(((a, b), _)) => {
                    // Create merged token
                    let merged = if b.starts_with(&self.config.continuing_subword_prefix) {
                        format!("{}{}", a, &b[self.config.continuing_subword_prefix.len()..])
                    } else {
                        format!("{}{}", a, b)
                    };

                    if !self.vocab.contains_key(&merged) {
                        self.vocab.insert(merged, next_id);
                        next_id += 1;
                    }
                }
                None => break,
            }
        }

        Ok(WordPieceModel::new(
            self.vocab.clone(),
            self.config.continuing_subword_prefix.clone(),
            "[UNK]".to_string(),
            self.config.max_input_chars_per_word,
        ))
    }

    /// Tokenize a word for training (greedy longest match)
    fn tokenize_for_training(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        if chars.is_empty() {
            return vec![];
        }

        let mut tokens = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let token = if start > 0 {
                    format!("{}{}", self.config.continuing_subword_prefix, substr)
                } else {
                    substr
                };

                if self.vocab.contains_key(&token) {
                    tokens.push(token);
                    found = true;
                    break;
                }

                end -= 1;
            }

            if !found {
                // Single character fallback
                let token = if start > 0 {
                    format!("{}{}", self.config.continuing_subword_prefix, chars[start])
                } else {
                    chars[start].to_string()
                };
                tokens.push(token);
                start += 1;
            } else {
                start = end;
            }
        }

        tokens
    }

    /// Get vocabulary
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }
}

// =============================================================================
// Unigram Trainer
// =============================================================================

/// Unigram trainer configuration
#[derive(Debug, Clone)]
pub struct UnigramTrainerConfig {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Special tokens
    pub special_tokens: Vec<String>,
    /// Initial vocabulary size before pruning
    pub initial_vocab_size: usize,
    /// Shrinking factor for EM iterations
    pub shrinking_factor: f64,
    /// Number of EM iterations
    pub n_iterations: usize,
    /// Maximum piece length
    pub max_piece_length: usize,
    /// Normalizer
    pub normalizer: Option<Normalizer>,
    /// Pre-tokenizer
    pub pre_tokenizer: Option<PreTokenizer>,
}

impl Default for UnigramTrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8000,
            special_tokens: vec![
                "<unk>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
            ],
            initial_vocab_size: 1000000,
            shrinking_factor: 0.75,
            n_iterations: 16,
            max_piece_length: 16,
            normalizer: Some(Normalizer::NFC),
            pre_tokenizer: Some(PreTokenizer::Metaspace {
                replacement: '▁',
                add_prefix_space: true,
            }),
        }
    }
}

/// Unigram trainer
pub struct UnigramTrainer {
    config: UnigramTrainerConfig,
    vocab: Vec<(String, f64)>,
}

impl UnigramTrainer {
    /// Create new Unigram trainer
    pub fn new(config: UnigramTrainerConfig) -> Self {
        Self {
            config,
            vocab: Vec::new(),
        }
    }

    /// Pre-tokenize text
    fn pretokenize(&self, text: &str) -> Vec<String> {
        let normalized = match &self.config.normalizer {
            Some(n) => n.normalize(text),
            None => text.to_string(),
        };

        match &self.config.pre_tokenizer {
            Some(pt) => pt.pre_tokenize(&normalized),
            None => normalized.split_whitespace().map(|s| s.to_string()).collect(),
        }
    }

    /// Train from files
    pub fn train<P: AsRef<Path>>(&mut self, files: &[P]) -> Result<UnigramModel, std::io::Error> {
        let mut sentences: Vec<String> = Vec::new();

        for file_path in files {
            let file = File::open(file_path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                let words = self.pretokenize(&line);
                sentences.extend(words);
            }
        }

        self.train_from_sentences(&sentences)
    }

    /// Train from an iterator of texts
    pub fn train_from_texts<I, S>(&mut self, texts: I) -> Result<UnigramModel, std::io::Error>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        let mut sentences: Vec<String> = Vec::new();

        for text in texts {
            let words = self.pretokenize(text.as_ref());
            sentences.extend(words);
        }

        self.train_from_sentences(&sentences)
    }

    /// Train from sentences (pre-tokenized words)
    fn train_from_sentences(&mut self, sentences: &[String]) -> Result<UnigramModel, std::io::Error> {
        // Step 1: Build initial vocabulary from substrings
        let mut substr_freqs: HashMap<String, u64> = HashMap::new();

        for sentence in sentences {
            let chars: Vec<char> = sentence.chars().collect();
            let n = chars.len().min(self.config.max_piece_length);

            // Count all substrings up to max_piece_length
            for start in 0..chars.len() {
                for end in (start + 1)..=(start + n).min(chars.len()) {
                    let substr: String = chars[start..end].iter().collect();
                    *substr_freqs.entry(substr).or_insert(0) += 1;
                }
            }
        }

        // Add special tokens
        for token in &self.config.special_tokens {
            substr_freqs.insert(token.clone(), 1);
        }

        // Sort by frequency and take top initial_vocab_size
        let mut items: Vec<(String, u64)> = substr_freqs.into_iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        items.truncate(self.config.initial_vocab_size);

        // Convert to log probabilities
        let total: f64 = items.iter().map(|(_, f)| *f as f64).sum();
        self.vocab = items
            .into_iter()
            .map(|(token, freq)| {
                let log_prob = (freq as f64 / total).ln();
                (token, log_prob)
            })
            .collect();

        // Step 2: EM iterations to prune vocabulary
        for _ in 0..self.config.n_iterations {
            if self.vocab.len() <= self.config.vocab_size {
                break;
            }

            // E-step: compute expected counts using Viterbi
            let mut expected_counts: HashMap<String, f64> = HashMap::new();

            for sentence in sentences {
                let segmentation = self.viterbi_segment(sentence);
                for token in segmentation {
                    *expected_counts.entry(token).or_insert(0.0) += 1.0;
                }
            }

            // M-step: prune low-scoring tokens
            let target_size = ((self.vocab.len() as f64) * self.config.shrinking_factor) as usize;
            let target_size = target_size.max(self.config.vocab_size);

            // Sort by expected count and keep top tokens
            let mut scored: Vec<(String, f64, f64)> = self.vocab
                .iter()
                .map(|(token, log_prob)| {
                    let count = expected_counts.get(token).copied().unwrap_or(0.0);
                    (token.clone(), count, *log_prob)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(target_size);

            // Recalculate log probabilities
            let total_count: f64 = scored.iter().map(|(_, c, _)| c).sum();
            self.vocab = scored
                .into_iter()
                .map(|(token, count, _)| {
                    let log_prob = if total_count > 0.0 {
                        (count / total_count).ln()
                    } else {
                        -100.0
                    };
                    (token, log_prob)
                })
                .collect();
        }

        // Ensure special tokens are present
        for token in &self.config.special_tokens {
            if !self.vocab.iter().any(|(t, _)| t == token) {
                self.vocab.push((token.clone(), -100.0));
            }
        }

        Ok(UnigramModel::new(self.vocab.clone(), "<unk>".to_string()))
    }

    /// Viterbi segmentation for a sentence
    fn viterbi_segment(&self, sentence: &str) -> Vec<String> {
        if sentence.is_empty() {
            return vec![];
        }

        let chars: Vec<char> = sentence.chars().collect();
        let n = chars.len();

        // Build vocab lookup
        let vocab_map: HashMap<&str, f64> = self.vocab
            .iter()
            .map(|(t, s)| (t.as_str(), *s))
            .collect();

        // best_ending[i] = (best_score, prev_pos)
        let mut best_ending: Vec<(f64, i32)> = vec![(f64::NEG_INFINITY, -1); n + 1];
        best_ending[0] = (0.0, -1);

        // best_token[i] = token ending at position i
        let mut best_token: Vec<String> = vec![String::new(); n + 1];

        for end in 1..=n {
            let max_start = end.saturating_sub(self.config.max_piece_length);
            for start in max_start..end {
                let substr: String = chars[start..end].iter().collect();

                if let Some(&score) = vocab_map.get(substr.as_str()) {
                    let new_score = best_ending[start].0 + score;
                    if new_score > best_ending[end].0 {
                        best_ending[end] = (new_score, start as i32);
                        best_token[end] = substr;
                    }
                } else if end - start == 1 {
                    // Single character fallback
                    let score = vocab_map.get("<unk>").copied().unwrap_or(-100.0);
                    let new_score = best_ending[start].0 + score;
                    if new_score > best_ending[end].0 {
                        best_ending[end] = (new_score, start as i32);
                        best_token[end] = substr;
                    }
                }
            }
        }

        // Backtrack
        let mut tokens = Vec::new();
        let mut pos = n;

        while pos > 0 {
            tokens.push(best_token[pos].clone());
            pos = best_ending[pos].1 as usize;
        }

        tokens.reverse();
        tokens
    }

    /// Get vocabulary
    pub fn vocab(&self) -> &[(String, f64)] {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wordpiece_trainer() {
        let config = WordPieceTrainerConfig {
            vocab_size: 100,
            min_frequency: 1,
            ..Default::default()
        };

        let mut trainer = WordPieceTrainer::new(config);
        let texts = vec![
            "hello world",
            "hello there",
            "world peace",
        ];

        let model = trainer.train_from_texts(texts.iter()).unwrap();
        assert!(model.vocab_size() > 0);
    }

    #[test]
    fn test_unigram_trainer() {
        let config = UnigramTrainerConfig {
            vocab_size: 50,
            initial_vocab_size: 100,
            n_iterations: 2,
            ..Default::default()
        };

        let mut trainer = UnigramTrainer::new(config);
        let texts = vec![
            "hello world",
            "hello there",
            "world peace",
        ];

        let model = trainer.train_from_texts(texts.iter()).unwrap();
        assert!(model.vocab_size() > 0);
    }
}
