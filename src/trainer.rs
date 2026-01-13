//! INL-BPE Trainer - BPE training with INL dynamics for merge selection
//!
//! Unlike standard BPE which only uses frequency, this trainer uses
//! INL dynamics to balance the vocabulary distribution.

use hashbrown::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Target vocabulary size
    pub vocab_size: usize,
    /// Minimum frequency for a token to be included
    pub min_frequency: u32,
    /// Special tokens to add
    pub special_tokens: Vec<String>,
    /// Filter words shorter than this
    pub min_word_length: usize,
    /// INL dynamics: alpha (momentum/inertia)
    pub inl_alpha: f32,
    /// INL dynamics: beta (correction strength) - CLAMPED to [0, 2]
    pub inl_beta: f32,
    /// INL dynamics: gate (amplitude control)
    pub inl_gate: f32,
    /// Target frequency ratio for mu (equilibrium)
    pub inl_mu_target: f32,
    /// Maximum velocity (prevents runaway)
    pub inl_velocity_max: f32,
    /// Maximum beta (stability constraint)
    pub inl_beta_max: f32,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            min_frequency: 2,
            special_tokens: vec![
                "</s>".to_string(),  // EOS (id=0)
                "<pad>".to_string(), // PAD (id=1)
                "<s>".to_string(),   // BOS (id=2)
                "<unk>".to_string(), // UNK (id=3)
            ],
            min_word_length: 1,
            inl_alpha: 0.9,
            inl_beta: 0.3,
            inl_gate: 0.5,
            inl_mu_target: 0.01, // Target: each token ~1% of corpus
            inl_velocity_max: 10.0, // CRITICAL: prevent runaway
            inl_beta_max: 2.0,      // CRITICAL: stability constraint
        }
    }
}

/// INL-BPE Trainer
pub struct InlBpeTrainer {
    config: TrainerConfig,
    /// Current vocabulary: token -> id
    vocab: HashMap<String, u32>,
    /// Reverse vocab: id -> token
    vocab_r: HashMap<u32, String>,
    /// Merge operations in order
    merges: Vec<(String, String)>,
    /// Token frequencies
    token_freqs: HashMap<String, u32>,
    /// INL velocity state per token
    velocity: HashMap<String, f32>,
}

impl InlBpeTrainer {
    /// Create new trainer with config
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            merges: Vec::new(),
            token_freqs: HashMap::new(),
            velocity: HashMap::new(),
        }
    }

    /// Train tokenizer from text files
    pub fn train<P: AsRef<Path>>(&mut self, files: &[P]) -> Result<(), std::io::Error> {
        // Step 1: Count word frequencies
        println!("Step 1: Counting word frequencies...");
        let word_freqs = self.count_words(files)?;
        println!("  Found {} unique words", word_freqs.len());

        // Step 2: Initialize vocab with characters
        println!("Step 2: Initializing character vocabulary...");
        self.init_vocab(&word_freqs);
        println!("  Initial vocab size: {}", self.vocab.len());

        // Step 3: BPE merges with INL dynamics
        println!("Step 3: Learning merges with INL dynamics...");
        self.learn_merges(&word_freqs);
        println!("  Final vocab size: {}", self.vocab.len());
        println!("  Total merges: {}", self.merges.len());

        Ok(())
    }

    /// Count word frequencies from files
    fn count_words<P: AsRef<Path>>(&self, files: &[P]) -> Result<HashMap<String, u32>, std::io::Error> {
        let mut word_freqs: HashMap<String, u32> = HashMap::new();

        for file_path in files {
            let file = File::open(file_path)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                for word in line.split_whitespace() {
                    // Filter by minimum word length
                    if word.chars().count() >= self.config.min_word_length {
                        *word_freqs.entry(word.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }

        // Filter by minimum frequency
        word_freqs.retain(|_, freq| *freq >= self.config.min_frequency);

        Ok(word_freqs)
    }

    /// Initialize vocabulary with special tokens and characters
    fn init_vocab(&mut self, word_freqs: &HashMap<String, u32>) {
        let mut next_id = 0u32;

        // Add special tokens first
        for token in &self.config.special_tokens {
            self.vocab.insert(token.clone(), next_id);
            self.vocab_r.insert(next_id, token.clone());
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
                self.vocab.insert(token.clone(), next_id);
                self.vocab_r.insert(next_id, token);
                next_id += 1;
            }
        }

        // Initialize token frequencies from words
        for (word, freq) in word_freqs {
            for c in word.chars() {
                let token = c.to_string();
                *self.token_freqs.entry(token).or_insert(0) += freq;
            }
        }

        // Initialize INL velocity to zero
        for token in self.vocab.keys() {
            self.velocity.insert(token.clone(), 0.0);
        }
    }

    /// Learn BPE merges using INL dynamics
    fn learn_merges(&mut self, word_freqs: &HashMap<String, u32>) {
        // Represent words as sequences of tokens
        let mut word_tokens: HashMap<String, Vec<String>> = word_freqs
            .keys()
            .map(|word| {
                let tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                (word.clone(), tokens)
            })
            .collect();

        let target_vocab_size = self.config.vocab_size;

        while self.vocab.len() < target_vocab_size {
            // Count pair frequencies
            let pair_freqs = self.count_pairs(&word_tokens, word_freqs);

            if pair_freqs.is_empty() {
                break;
            }

            // Calculate INL-adjusted scores for each pair
            let best_pair = self.select_best_pair_inl(&pair_freqs);

            if let Some((pair, _score)) = best_pair {
                // Create merged token
                let merged = format!("{}{}", pair.0, pair.1);

                // Add to vocab
                let new_id = self.vocab.len() as u32;
                self.vocab.insert(merged.clone(), new_id);
                self.vocab_r.insert(new_id, merged.clone());

                // Record merge
                self.merges.push(pair.clone());

                // Update word representations
                self.apply_merge(&mut word_tokens, &pair, &merged);

                // Update token frequencies
                self.update_frequencies(&pair, &merged, word_freqs, &word_tokens);

                // Update INL velocity for the new token
                self.velocity.insert(merged, 0.0);

                // Progress
                if self.vocab.len() % 1000 == 0 {
                    println!("  Vocab size: {}", self.vocab.len());
                }
            } else {
                break;
            }
        }
    }

    /// Count pair frequencies
    fn count_pairs(
        &self,
        word_tokens: &HashMap<String, Vec<String>>,
        word_freqs: &HashMap<String, u32>,
    ) -> HashMap<(String, String), u32> {
        let mut pair_freqs: HashMap<(String, String), u32> = HashMap::new();

        for (word, tokens) in word_tokens {
            let freq = word_freqs.get(word).copied().unwrap_or(0);
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }

        pair_freqs
    }

    /// Select best pair using INL dynamics
    fn select_best_pair_inl(
        &mut self,
        pair_freqs: &HashMap<(String, String), u32>,
    ) -> Option<((String, String), f32)> {
        let total_freq: u32 = self.token_freqs.values().sum();
        let mu = self.config.inl_mu_target * total_freq as f32;

        let mut best: Option<((String, String), f32)> = None;

        for (pair, &freq) in pair_freqs {
            // Base score = frequency
            let base_score = freq as f32;

            // INL dynamics for each token in the pair
            let _merged = format!("{}{}", pair.0, pair.1);

            // Get current frequencies
            let freq_a = *self.token_freqs.get(&pair.0).unwrap_or(&0) as f32;
            let freq_b = *self.token_freqs.get(&pair.1).unwrap_or(&0) as f32;

            // Error: deviation from equilibrium
            let error_a = freq_a - mu;
            let error_b = freq_b - mu;

            // Get velocities
            let v_a = *self.velocity.get(&pair.0).unwrap_or(&0.0);
            let v_b = *self.velocity.get(&pair.1).unwrap_or(&0.0);

            // CLAMP beta for stability (like in model INL dynamics)
            let beta_clamped = self.config.inl_beta.min(self.config.inl_beta_max).max(0.0);

            // Update velocities (INL dynamics)
            let v_a_new = self.config.inl_alpha * v_a - beta_clamped * error_a;
            let v_b_new = self.config.inl_alpha * v_b - beta_clamped * error_b;

            // CLAMP velocities to prevent runaway (CRITICAL for stability)
            let v_a_clamped = v_a_new.max(-self.config.inl_velocity_max).min(self.config.inl_velocity_max);
            let v_b_clamped = v_b_new.max(-self.config.inl_velocity_max).min(self.config.inl_velocity_max);

            // Store updated velocities (clamped)
            self.velocity.insert(pair.0.clone(), v_a_clamped);
            self.velocity.insert(pair.1.clone(), v_b_clamped);

            // INL adjustment: favor merges that balance the distribution
            // If tokens are too frequent (positive error), reduce their score
            // If tokens are rare (negative error), boost their score
            let inl_adjustment = self.config.inl_gate * (v_a_clamped + v_b_clamped);

            // Final score
            let score = base_score - inl_adjustment;

            match &best {
                None => best = Some((pair.clone(), score)),
                Some((_, best_score)) if score > *best_score => {
                    best = Some((pair.clone(), score));
                }
                _ => {}
            }
        }

        best
    }

    /// Apply merge to all words
    fn apply_merge(
        &self,
        word_tokens: &mut HashMap<String, Vec<String>>,
        pair: &(String, String),
        merged: &str,
    ) {
        for tokens in word_tokens.values_mut() {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    tokens[i] = merged.to_string();
                    tokens.remove(i + 1);
                }
                i += 1;
            }
        }
    }

    /// Update token frequencies after merge
    fn update_frequencies(
        &mut self,
        pair: &(String, String),
        merged: &str,
        word_freqs: &HashMap<String, u32>,
        word_tokens: &HashMap<String, Vec<String>>,
    ) {
        // Recount frequencies
        self.token_freqs.clear();
        for (word, tokens) in word_tokens {
            let freq = word_freqs.get(word).copied().unwrap_or(0);
            for token in tokens {
                *self.token_freqs.entry(token.clone()).or_insert(0) += freq;
            }
        }
    }

    /// Get vocabulary
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    /// Get merges
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Export to HuggingFace tokenizer.json format
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
        // Create test corpus
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
}
