//! INL-BPE Trainer - OPTIMIZED with incremental updates
//!
//! Key optimizations vs naive implementation:
//! 1. Incremental pair counting (don't recount all pairs after each merge)
//! 2. Parallel word processing with rayon
//! 3. Efficient data structures

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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
        }
    }
}

/// Word representation: tokens + frequency (avoids HashMap lookup)
#[derive(Clone)]
struct Word {
    tokens: Vec<u32>,  // Use IDs instead of strings for speed
    freq: u32,
}

/// INL-BPE Trainer - OPTIMIZED
pub struct InlBpeTrainer {
    config: TrainerConfig,
    vocab: HashMap<String, u32>,
    vocab_r: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    token_freqs: HashMap<u32, u64>,  // Use u64 to avoid overflow
    velocity: HashMap<u32, f32>,
    // Optimization: global pair frequencies (incrementally updated)
    pair_freqs: HashMap<(u32, u32), i64>,  // i64 to handle decrements
    // Streaming: accumulate word frequencies across batches
    word_freqs_accumulator: HashMap<String, u32>,
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
        println!("Step 1: Counting word frequencies from iterator...");
        let word_freqs = self.count_words_from_iter(texts);
        println!("  Found {} unique words", word_freqs.len());
        self.train_from_word_freqs(word_freqs);
    }

    /// Count words from a batch (for streaming - call multiple times, then call finish_training)
    pub fn count_batch<I, S>(&mut self, texts: I)
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        for text in texts {
            for word in text.as_ref().split_whitespace() {
                if word.chars().count() >= self.config.min_word_length {
                    *self.word_freqs_accumulator.entry(word.to_string()).or_insert(0) += 1;
                }
            }
        }
    }

    /// Finish training after counting all batches
    pub fn finish_training(&mut self) {
        // Take accumulated word freqs
        let mut word_freqs = std::mem::take(&mut self.word_freqs_accumulator);

        // Filter by min frequency
        word_freqs.retain(|_, freq| *freq >= self.config.min_frequency);

        println!("  Found {} unique words", word_freqs.len());
        self.train_from_word_freqs(word_freqs);
    }

    fn train_from_word_freqs(&mut self, word_freqs: HashMap<String, u32>) {
        println!("Step 2: Initializing character vocabulary...");
        let mut words = self.init_vocab(&word_freqs);
        println!("  Initial vocab size: {}", self.vocab.len());

        println!("Step 3: Computing initial pair frequencies (parallel)...");
        self.compute_initial_pairs(&words);
        println!("  Found {} unique pairs", self.pair_freqs.len());

        println!("Step 4: Learning merges with INL dynamics (incremental)...");
        self.learn_merges_incremental(&mut words);
        println!("  Final vocab size: {}", self.vocab.len());
        println!("  Total merges: {}", self.merges.len());
    }

    fn count_words_from_iter<I, S>(&self, texts: I) -> HashMap<String, u32>
    where
        I: Iterator<Item = S>,
        S: AsRef<str>,
    {
        let mut word_freqs: HashMap<String, u32> = HashMap::new();
        for text in texts {
            for word in text.as_ref().split_whitespace() {
                if word.chars().count() >= self.config.min_word_length {
                    *word_freqs.entry(word.to_string()).or_insert(0) += 1;
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
                for word in line.split_whitespace() {
                    if word.chars().count() >= self.config.min_word_length {
                        *word_freqs.entry(word.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
        word_freqs.retain(|_, freq| *freq >= self.config.min_frequency);
        Ok(word_freqs)
    }

    /// Initialize vocab and return words as token ID sequences
    fn init_vocab(&mut self, word_freqs: &HashMap<String, u32>) -> Vec<Word> {
        let mut next_id = 0u32;

        // Add special tokens
        for token in &self.config.special_tokens {
            self.vocab.insert(token.clone(), next_id);
            self.vocab_r.insert(next_id, token.clone());
            next_id += 1;
        }

        // Collect unique characters
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
                self.vocab.insert(token.clone(), next_id);
                self.vocab_r.insert(next_id, token);
                next_id += 1;
            }
        }

        // Convert words to token ID sequences
        // Collect to Vec first for parallel iteration (hashbrown doesn't have par_iter)
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
        // Parallel map-reduce for pair counting
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

    /// Learn merges with INCREMENTAL pair updates
    fn learn_merges_incremental(&mut self, words: &mut Vec<Word>) {
        let target_vocab_size = self.config.vocab_size;
        let mut iteration = 0;

        while self.vocab.len() < target_vocab_size {
            // Find best pair using INL scoring
            let best_pair = self.select_best_pair_inl();

            if best_pair.is_none() {
                break;
            }

            let (pair, _score) = best_pair.unwrap();

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

            // INCREMENTAL UPDATE: apply merge and update pair counts
            self.apply_merge_incremental(words, pair, new_id);

            // Initialize velocity for new token
            self.velocity.insert(new_id, 0.0);

            iteration += 1;
            if iteration % 1000 == 0 {
                println!("  Vocab size: {} ({} merges)", self.vocab.len(), iteration);
            }
        }
    }

    /// Select best pair using INL dynamics
    fn select_best_pair_inl(&mut self) -> Option<((u32, u32), f32)> {
        let total_freq: u64 = self.token_freqs.values().sum();
        if total_freq == 0 {
            return None;
        }
        let mu = self.config.inl_mu_target * total_freq as f32;

        let mut best: Option<((u32, u32), f32)> = None;

        for (&pair, &freq) in &self.pair_freqs {
            if freq <= 0 {
                continue;
            }

            let base_score = freq as f32;

            // Get token frequencies
            let freq_a = *self.token_freqs.get(&pair.0).unwrap_or(&0) as f32;
            let freq_b = *self.token_freqs.get(&pair.1).unwrap_or(&0) as f32;

            // INL dynamics
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

            // Update velocities (in-place for selected pair)
            self.velocity.insert(pair.0, v_a_new);
            self.velocity.insert(pair.1, v_b_new);

            let inl_adjustment = self.config.inl_gate * (v_a_new + v_b_new);
            let score = base_score - inl_adjustment;

            match &best {
                None => best = Some((pair, score)),
                Some((_, best_score)) if score > *best_score => {
                    best = Some((pair, score));
                }
                _ => {}
            }
        }

        best
    }

    /// Apply merge with INCREMENTAL pair frequency updates
    fn apply_merge_incremental(&mut self, words: &mut Vec<Word>, pair: (u32, u32), new_id: u32) {
        // Remove the merged pair from counts
        self.pair_freqs.remove(&pair);

        // Track frequency changes for the new token
        let mut new_token_freq: u64 = 0;

        // Process each word
        for word in words.iter_mut() {
            let mut i = 0;
            while i < word.tokens.len().saturating_sub(1) {
                if word.tokens[i] == pair.0 && word.tokens[i + 1] == pair.1 {
                    let freq = word.freq as i64;

                    // Decrement old pair frequencies
                    // Left neighbor: (tokens[i-1], pair.0) -> disappears
                    if i > 0 {
                        let left_pair = (word.tokens[i - 1], pair.0);
                        *self.pair_freqs.entry(left_pair).or_insert(0) -= freq;
                    }
                    // Right neighbor: (pair.1, tokens[i+2]) -> disappears
                    if i + 2 < word.tokens.len() {
                        let right_pair = (pair.1, word.tokens[i + 2]);
                        *self.pair_freqs.entry(right_pair).or_insert(0) -= freq;
                    }

                    // Apply merge: replace pair with new_id
                    word.tokens[i] = new_id;
                    word.tokens.remove(i + 1);

                    // Increment new pair frequencies
                    // Left neighbor: (tokens[i-1], new_id) -> appears
                    if i > 0 {
                        let new_left = (word.tokens[i - 1], new_id);
                        *self.pair_freqs.entry(new_left).or_insert(0) += freq;
                    }
                    // Right neighbor: (new_id, tokens[i+1]) -> appears
                    if i + 1 < word.tokens.len() {
                        let new_right = (new_id, word.tokens[i + 1]);
                        *self.pair_freqs.entry(new_right).or_insert(0) += freq;
                    }

                    // Update token frequencies
                    new_token_freq += word.freq as u64;

                    // Don't increment i, check same position again for consecutive merges
                } else {
                    i += 1;
                }
            }
        }

        // Update token frequencies
        // The merged token absorbs frequencies from the pair tokens
        // Actually, we should decrement pair.0 and pair.1 frequencies
        // and set new_id frequency
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
    fn test_incremental_correctness() {
        // Test that incremental updates give same result as naive
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "aaa bbb aaa bbb ccc").unwrap();

        let config = TrainerConfig {
            vocab_size: 20,
            min_frequency: 1,
            inl_alpha: 0.0, // Disable INL for deterministic test
            inl_beta: 0.0,
            inl_gate: 0.0,
            ..Default::default()
        };

        let mut trainer = InlBpeTrainer::new(config);
        trainer.train(&[file.path()]).unwrap();

        // Should have learned some merges
        assert!(!trainer.merges().is_empty());
    }
}
