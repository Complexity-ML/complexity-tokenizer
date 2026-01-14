//! INL-BPE Trainer - OPTIMIZED with heap + ByteLevel pre-tokenization
//!
//! Key optimizations:
//! 1. BinaryHeap for O(log n) best pair selection
//! 2. Configurable normalizer and pre-tokenizer
//! 3. Incremental pair counting
//! 4. Parallel word processing with rayon

use crate::normalizers::Normalizer;
use crate::pretokenizers::PreTokenizer;
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

/// GPT-2 style byte-to-unicode mapping
fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = Vec::new();
    // Printable ASCII range
    bs.extend(b'!'..=b'~');
    bs.extend(0xa1u8..=0xacu8);
    bs.extend(0xaeu8..=0xffu8);

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;

    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    bs.iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (b, char::from_u32(c).unwrap()))
        .collect()
}

/// Pre-tokenize text into ByteLevel tokens (like GPT-2)
fn byte_level_pretokenize(text: &str, byte_encoder: &HashMap<u8, char>) -> Vec<String> {
    // Simple whitespace-based word splitting with ByteLevel encoding
    let mut words = Vec::new();

    for word in text.split_whitespace() {
        // Add space prefix for non-first words (GPT-2 style)
        let word_with_space = format!("Ġ{}", word.replace(' ', "Ġ"));

        // Convert to bytes then to unicode chars
        let encoded: String = word_with_space
            .bytes()
            .filter_map(|b| byte_encoder.get(&b).copied())
            .collect();

        if !encoded.is_empty() {
            words.push(encoded);
        }
    }

    words
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub vocab_size: usize,
    pub min_frequency: u32,
    pub special_tokens: Vec<String>,
    pub min_word_length: usize,
    pub inl_alpha: f32,
    pub inl_beta: f32,
    pub inl_gate: f32,
    pub inl_mu_target: f32,
    pub inl_velocity_max: f32,
    pub inl_beta_max: f32,
    /// Normalizer to apply before pre-tokenization
    pub normalizer: Option<Normalizer>,
    /// Pre-tokenizer to split text into words
    pub pre_tokenizer: Option<PreTokenizer>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            min_frequency: 2,
            special_tokens: vec![
                "</s>".to_string(),
                "<pad>".to_string(),
                "<s>".to_string(),
                "<unk>".to_string(),
            ],
            min_word_length: 1,
            inl_alpha: 0.9,
            inl_beta: 0.3,
            inl_gate: 0.5,
            inl_mu_target: 0.01,
            inl_velocity_max: 10.0,
            inl_beta_max: 2.0,
            normalizer: Some(Normalizer::NFC),
            pre_tokenizer: Some(PreTokenizer::ByteLevel { add_prefix_space: false }),
        }
    }
}

/// Word representation
#[derive(Clone)]
struct Word {
    tokens: Vec<u32>,
    freq: u32,
}

/// Heap entry for best pair selection
#[derive(Clone)]
struct PairScore {
    pair: (u32, u32),
    score: f32,
}

impl PartialEq for PairScore {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for PairScore {}

impl PartialOrd for PairScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PairScore {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max heap: higher score = higher priority
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

/// INL-BPE Trainer - OPTIMIZED
pub struct InlBpeTrainer {
    config: TrainerConfig,
    vocab: HashMap<String, u32>,
    vocab_r: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    token_freqs: HashMap<u32, u64>,
    velocity: HashMap<u32, f32>,
    pair_freqs: HashMap<(u32, u32), i64>,
    word_freqs_accumulator: HashMap<String, u32>,
    byte_encoder: HashMap<u8, char>,
}

impl InlBpeTrainer {
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            merges: Vec::new(),
            token_freqs: HashMap::new(),
            velocity: HashMap::new(),
            pair_freqs: HashMap::new(),
            word_freqs_accumulator: HashMap::new(),
            byte_encoder: bytes_to_unicode(),
        }
    }

    /// Pre-tokenize text using configured normalizer and pre-tokenizer
    fn pretokenize(&self, text: &str) -> Vec<String> {
        // Apply normalizer
        let normalized = match &self.config.normalizer {
            Some(n) => n.normalize(text),
            None => text.to_string(),
        };

        // Apply pre-tokenizer
        match &self.config.pre_tokenizer {
            Some(pt) => pt.pre_tokenize(&normalized),
            None => byte_level_pretokenize(&normalized, &self.byte_encoder),
        }
    }

    pub fn train<P: AsRef<Path>>(&mut self, files: &[P]) -> Result<(), std::io::Error> {
        println!("Step 1: Counting word frequencies...");
        let word_freqs = self.count_words(files)?;
        println!("  Found {} unique words", word_freqs.len());
        self.train_from_word_freqs(word_freqs);
        Ok(())
    }

    pub fn train_from_texts<I, S>(&mut self, texts: I)
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        println!("Step 1: Counting word frequencies (ByteLevel)...");
        let word_freqs = self.count_words_from_iter_bytelevel(texts);
        println!("  Found {} unique words", word_freqs.len());
        self.train_from_word_freqs(word_freqs);
    }

    /// Count words from a batch using configured normalizer/pretokenizer
    pub fn count_batch<I, S>(&mut self, texts: I)
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        for text in texts {
            let words = self.pretokenize(text.as_ref());
            for word in words {
                if word.chars().count() >= self.config.min_word_length {
                    *self.word_freqs_accumulator.entry(word).or_insert(0) += 1;
                }
            }
        }
    }

    /// Finish training after counting all batches
    pub fn finish_training(&mut self) {
        let mut word_freqs = std::mem::take(&mut self.word_freqs_accumulator);
        word_freqs.retain(|_, freq| *freq >= self.config.min_frequency);
        println!("  Found {} unique words", word_freqs.len());
        self.train_from_word_freqs(word_freqs);
    }

    fn train_from_word_freqs(&mut self, word_freqs: HashMap<String, u32>) {
        println!("Step 2: Initializing vocabulary (ByteLevel alphabet)...");
        let mut words = self.init_vocab_bytelevel(&word_freqs);
        println!("  Initial vocab size: {}", self.vocab.len());

        println!("Step 3: Computing initial pair frequencies (parallel)...");
        self.compute_initial_pairs(&words);
        println!("  Found {} unique pairs", self.pair_freqs.len());

        println!("Step 4: Learning merges with INL dynamics (heap-based)...");
        self.learn_merges_heap(&mut words);
        println!("  Final vocab size: {}", self.vocab.len());
        println!("  Total merges: {}", self.merges.len());
    }

    fn count_words_from_iter_bytelevel<I, S>(&self, texts: I) -> HashMap<String, u32>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        let mut word_freqs: HashMap<String, u32> = HashMap::new();
        for text in texts {
            let words = self.pretokenize(text.as_ref());
            for word in words {
                if word.chars().count() >= self.config.min_word_length {
                    *word_freqs.entry(word).or_insert(0) += 1;
                }
            }
        }
        word_freqs.retain(|_, freq| *freq >= self.config.min_frequency);
        word_freqs
    }

    fn count_words<P: AsRef<Path>>(&self, files: &[P]) -> Result<HashMap<String, u32>, std::io::Error> {
        let mut word_freqs: HashMap<String, u32> = HashMap::new();
        for file_path in files {
            let file = File::open(file_path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                let words = self.pretokenize(&line);
                for word in words {
                    if word.chars().count() >= self.config.min_word_length {
                        *word_freqs.entry(word).or_insert(0) += 1;
                    }
                }
            }
        }
        word_freqs.retain(|_, freq| *freq >= self.config.min_frequency);
        Ok(word_freqs)
    }

    /// Initialize vocab with ByteLevel alphabet
    fn init_vocab_bytelevel(&mut self, word_freqs: &HashMap<String, u32>) -> Vec<Word> {
        let mut next_id = 0u32;

        // Add special tokens first
        for token in &self.config.special_tokens {
            self.vocab.insert(token.clone(), next_id);
            self.vocab_r.insert(next_id, token.clone());
            next_id += 1;
        }

        // Collect all unique characters (ByteLevel encoded)
        let mut chars: HashSet<char> = HashSet::new();
        for word in word_freqs.keys() {
            for c in word.chars() {
                chars.insert(c);
            }
        }

        // Add ByteLevel alphabet to vocab
        for c in chars {
            let token = c.to_string();
            if !self.vocab.contains_key(&token) {
                self.vocab.insert(token.clone(), next_id);
                self.vocab_r.insert(next_id, token);
                next_id += 1;
            }
        }

        // Convert words to token ID sequences
        let word_vec: Vec<(&String, &u32)> = word_freqs.iter().collect();
        let words: Vec<Word> = word_vec
            .par_iter()
            .map(|(word, &freq)| {
                let tokens: Vec<u32> = word
                    .chars()
                    .filter_map(|c: char| self.vocab.get(&c.to_string()).copied())
                    .collect();
                Word { tokens, freq }
            })
            .collect();

        // Initialize token frequencies
        for word in &words {
            for &token_id in &word.tokens {
                *self.token_freqs.entry(token_id).or_insert(0) += word.freq as u64;
            }
        }

        // Initialize velocities
        for &id in self.vocab.values() {
            self.velocity.insert(id, 0.0);
        }

        words
    }

    /// Compute initial pair frequencies in parallel
    fn compute_initial_pairs(&mut self, words: &[Word]) {
        let pair_counts: HashMap<(u32, u32), i64> = words
            .par_iter()
            .fold(
                || HashMap::new(),
                |mut acc: HashMap<(u32, u32), i64>, word| {
                    for i in 0..word.tokens.len().saturating_sub(1) {
                        let pair = (word.tokens[i], word.tokens[i + 1]);
                        *acc.entry(pair).or_insert(0) += word.freq as i64;
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::new(),
                |mut a, b| {
                    for (pair, count) in b {
                        *a.entry(pair).or_insert(0) += count;
                    }
                    a
                },
            );

        self.pair_freqs = pair_counts;
    }

    /// Build heap with INL scores
    fn build_heap(&self) -> BinaryHeap<PairScore> {
        let total_freq: u64 = self.token_freqs.values().sum();
        let mu = self.config.inl_mu_target * total_freq as f32;

        self.pair_freqs
            .iter()
            .filter(|(_, &freq)| freq > 0)
            .map(|(&pair, &freq)| {
                let base_score = freq as f32;

                let freq_a = *self.token_freqs.get(&pair.0).unwrap_or(&0) as f32;
                let freq_b = *self.token_freqs.get(&pair.1).unwrap_or(&0) as f32;

                let error_a = freq_a - mu;
                let error_b = freq_b - mu;

                let v_a = *self.velocity.get(&pair.0).unwrap_or(&0.0);
                let v_b = *self.velocity.get(&pair.1).unwrap_or(&0.0);

                let beta_clamped = self.config.inl_beta.min(self.config.inl_beta_max).max(0.0);

                let v_a_new = (self.config.inl_alpha * v_a - beta_clamped * error_a)
                    .max(-self.config.inl_velocity_max)
                    .min(self.config.inl_velocity_max);
                let v_b_new = (self.config.inl_alpha * v_b - beta_clamped * error_b)
                    .max(-self.config.inl_velocity_max)
                    .min(self.config.inl_velocity_max);

                let inl_adjustment = self.config.inl_gate * (v_a_new + v_b_new);
                let score = base_score - inl_adjustment;

                PairScore { pair, score }
            })
            .collect()
    }

    /// Learn merges using heap for O(log n) best pair selection
    fn learn_merges_heap(&mut self, words: &mut Vec<Word>) {
        let target_vocab_size = self.config.vocab_size;
        let initial_vocab_size = self.vocab.len();
        let target_merges = target_vocab_size.saturating_sub(initial_vocab_size);
        let mut iteration = 0;
        let start_time = Instant::now();
        let mut last_progress_update = Instant::now();

        // Rebuild heap periodically (INL dynamics change scores)
        let rebuild_interval = 100;

        while self.vocab.len() < target_vocab_size {
            // Build/rebuild heap
            let mut heap = self.build_heap();

            // Process multiple pairs before rebuilding heap
            for _ in 0..rebuild_interval {
                if self.vocab.len() >= target_vocab_size {
                    break;
                }

                // Get best pair from heap
                let best = loop {
                    match heap.pop() {
                        None => break None,
                        Some(ps) => {
                            // Verify pair still exists with positive frequency
                            if let Some(&freq) = self.pair_freqs.get(&ps.pair) {
                                if freq > 0 {
                                    break Some(ps);
                                }
                            }
                            // Otherwise skip this stale entry
                        }
                    }
                };

                let best = match best {
                    Some(b) => b,
                    None => break,
                };

                let pair = best.pair;

                // Create merged token
                let token_a = self.vocab_r.get(&pair.0).unwrap().clone();
                let token_b = self.vocab_r.get(&pair.1).unwrap().clone();
                let merged_str = format!("{}{}", token_a, token_b);

                // Add to vocab
                let new_id = self.vocab.len() as u32;
                self.vocab.insert(merged_str.clone(), new_id);
                self.vocab_r.insert(new_id, merged_str);

                // Record merge
                self.merges.push((token_a, token_b));

                // Apply merge incrementally
                self.apply_merge_incremental(words, pair, new_id);

                // Update velocities for merged tokens
                let v_a = *self.velocity.get(&pair.0).unwrap_or(&0.0);
                let v_b = *self.velocity.get(&pair.1).unwrap_or(&0.0);
                self.velocity.insert(new_id, (v_a + v_b) / 2.0);

                iteration += 1;

                // Progress bar update (every 100ms or every 100 merges)
                let should_update = last_progress_update.elapsed().as_millis() >= 100
                    || iteration % 100 == 0;

                if should_update && target_merges > 0 {
                    let progress = iteration as f64 / target_merges as f64;
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let eta = if progress > 0.0 {
                        (elapsed / progress) - elapsed
                    } else {
                        0.0
                    };

                    let bar_width = 30;
                    let filled = (progress * bar_width as f64).min(bar_width as f64) as usize;
                    let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);

                    eprint!(
                        "\r  [{bar}] {}/{} ({:.1}%) | ETA: {:.0}s    ",
                        iteration,
                        target_merges,
                        (progress * 100.0).min(100.0),
                        eta
                    );
                    let _ = io::stderr().flush();
                    last_progress_update = Instant::now();
                }
            }

            if self.pair_freqs.iter().filter(|(_, &v)| v > 0).count() == 0 {
                break;
            }
        }

        // Final progress update
        let total_time = start_time.elapsed().as_secs_f64();
        eprintln!(
            "\r  [{}] {}/{} (100.0%) | Done in {:.1}s    ",
            "█".repeat(30),
            iteration,
            target_merges,
            total_time
        );
    }

    /// Apply merge with PARALLEL incremental pair frequency updates
    fn apply_merge_incremental(&mut self, words: &mut Vec<Word>, pair: (u32, u32), new_id: u32) {
        self.pair_freqs.remove(&pair);

        // Parallel: apply merges and collect frequency deltas
        let results: Vec<(HashMap<(u32, u32), i64>, u64)> = words
            .par_iter_mut()
            .map(|word| {
                let mut deltas: HashMap<(u32, u32), i64> = HashMap::new();
                let mut token_freq: u64 = 0;

                let mut i = 0;
                while i < word.tokens.len().saturating_sub(1) {
                    if word.tokens[i] == pair.0 && word.tokens[i + 1] == pair.1 {
                        let freq = word.freq as i64;

                        // Record decrements for old pairs
                        if i > 0 {
                            let left_pair = (word.tokens[i - 1], pair.0);
                            *deltas.entry(left_pair).or_insert(0) -= freq;
                        }
                        if i + 2 < word.tokens.len() {
                            let right_pair = (pair.1, word.tokens[i + 2]);
                            *deltas.entry(right_pair).or_insert(0) -= freq;
                        }

                        // Apply merge
                        word.tokens[i] = new_id;
                        word.tokens.remove(i + 1);

                        // Record increments for new pairs
                        if i > 0 {
                            let new_left = (word.tokens[i - 1], new_id);
                            *deltas.entry(new_left).or_insert(0) += freq;
                        }
                        if i + 1 < word.tokens.len() {
                            let new_right = (new_id, word.tokens[i + 1]);
                            *deltas.entry(new_right).or_insert(0) += freq;
                        }

                        token_freq += word.freq as u64;
                    } else {
                        i += 1;
                    }
                }

                (deltas, token_freq)
            })
            .collect();

        // Aggregate deltas (sequential but fast)
        let mut new_token_freq: u64 = 0;
        for (deltas, freq) in results {
            new_token_freq += freq;
            for (pair_key, delta) in deltas {
                *self.pair_freqs.entry(pair_key).or_insert(0) += delta;
            }
        }

        // Update token frequencies
        if let Some(freq_a) = self.token_freqs.get_mut(&pair.0) {
            *freq_a = freq_a.saturating_sub(new_token_freq);
        }
        if let Some(freq_b) = self.token_freqs.get_mut(&pair.1) {
            *freq_b = freq_b.saturating_sub(new_token_freq);
        }
        self.token_freqs.insert(new_id, new_token_freq);

        // Clean up zero/negative counts
        self.pair_freqs.retain(|_, &mut v| v > 0);
    }

    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        use std::io::Write;

        let merges_str: Vec<String> = self
            .merges
            .iter()
            .map(|(a, b)| format!("{} {}", a, b))
            .collect();

        let added_tokens: Vec<serde_json::Value> = self
            .config
            .special_tokens
            .iter()
            .enumerate()
            .map(|(i, token)| {
                serde_json::json!({
                    "id": i,
                    "content": token,
                    "special": true,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false
                })
            })
            .collect();

        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": self.vocab,
                "merges": merges_str
            },
            "added_tokens": added_tokens,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "use_regex": true
            },
            "decoder": {
                "type": "ByteLevel"
            }
        });

        let json = serde_json::to_string_pretty(&tokenizer_json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_byte_level_encoding() {
        let encoder = bytes_to_unicode();
        assert_eq!(encoder.len(), 256);

        // Test that printable ASCII maps to itself
        assert_eq!(encoder.get(&b'a'), Some(&'a'));
        assert_eq!(encoder.get(&b'Z'), Some(&'Z'));
    }

    #[test]
    fn test_basic_training() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "hello world hello world").unwrap();
        writeln!(file, "hello hello hello").unwrap();

        let config = TrainerConfig {
            vocab_size: 50,
            min_frequency: 1,
            ..Default::default()
        };

        let mut trainer = InlBpeTrainer::new(config);
        trainer.train(&[file.path()]).unwrap();

        assert!(trainer.vocab().len() > 10);
        assert!(!trainer.merges().is_empty());
    }

    #[test]
    fn test_heap_correctness() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "aaa bbb aaa bbb ccc").unwrap();

        let config = TrainerConfig {
            vocab_size: 30,
            min_frequency: 1,
            inl_alpha: 0.0,
            inl_beta: 0.0,
            inl_gate: 0.0,
            ..Default::default()
        };

        let mut trainer = InlBpeTrainer::new(config);
        trainer.train(&[file.path()]).unwrap();

        assert!(!trainer.merges().is_empty());
    }
}
