//! Alternative tokenization models
//!
//! Standard tokenization algorithms (not proprietary):
//! - WordPiece (BERT)
//! - Unigram (SentencePiece)
//! - WordLevel (simple lookup)

use hashbrown::HashMap;
use std::cmp::Ordering;

// =============================================================================
// WordPiece Model (BERT style)
// =============================================================================

/// WordPiece tokenizer (BERT style)
#[derive(Debug, Clone)]
pub struct WordPieceModel {
    /// Token string -> ID
    vocab: HashMap<String, u32>,
    /// ID -> Token string
    vocab_r: HashMap<u32, String>,
    /// Subword prefix (usually "##")
    continuing_subword_prefix: String,
    /// Unknown token
    unk_token: String,
    /// Max chars per word
    max_input_chars_per_word: usize,
}

impl WordPieceModel {
    /// Create new WordPiece model
    pub fn new(
        vocab: HashMap<String, u32>,
        continuing_subword_prefix: String,
        unk_token: String,
        max_input_chars_per_word: usize,
    ) -> Self {
        let vocab_r: HashMap<u32, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        Self {
            vocab,
            vocab_r,
            continuing_subword_prefix,
            unk_token,
            max_input_chars_per_word,
        }
    }

    /// Tokenize a single word using WordPiece algorithm
    pub fn tokenize_word(&self, word: &str) -> Vec<u32> {
        let chars: Vec<char> = word.chars().collect();

        if chars.len() > self.max_input_chars_per_word {
            return self.vocab.get(&self.unk_token).copied().into_iter().collect();
        }

        let mut tokens = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let token = if start > 0 {
                    format!("{}{}", self.continuing_subword_prefix, substr)
                } else {
                    substr
                };

                if let Some(&id) = self.vocab.get(&token) {
                    tokens.push(id);
                    found = true;
                    break;
                }

                end -= 1;
            }

            if !found {
                // Unknown token
                if let Some(&unk_id) = self.vocab.get(&self.unk_token) {
                    tokens.push(unk_id);
                }
                start += 1;
            } else {
                start = end;
            }
        }

        tokens
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();

        for word in text.split_whitespace() {
            result.extend(self.tokenize_word(word));
        }

        result
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();

        for &id in ids {
            if let Some(token) = self.vocab_r.get(&id) {
                if token.starts_with(&self.continuing_subword_prefix) {
                    result.push_str(&token[self.continuing_subword_prefix.len()..]);
                } else {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(token);
                }
            }
        }

        result
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Token to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// ID to token
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(&id).map(|s| s.as_str())
    }
}

// =============================================================================
// Unigram Model (SentencePiece style)
// =============================================================================

/// Unigram tokenizer (SentencePiece style)
#[derive(Debug, Clone)]
pub struct UnigramModel {
    /// Token string -> (ID, log probability)
    vocab: HashMap<String, (u32, f64)>,
    /// ID -> Token string
    vocab_r: HashMap<u32, String>,
    /// Unknown token
    unk_token: String,
    /// Unknown token ID
    unk_id: u32,
    /// Minimum score for unknown
    min_score: f64,
}

/// Node for Viterbi decoding
#[derive(Debug, Clone)]
struct ViterbiNode {
    /// Token ID at this position
    token_id: u32,
    /// Score (log probability)
    score: f64,
    /// Start position in text
    start: usize,
    /// End position in text
    end: usize,
    /// Previous node index (-1 for start)
    prev: i32,
}

impl Eq for ViterbiNode {}

impl PartialEq for ViterbiNode {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
    }
}

impl Ord for ViterbiNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score is better (less negative log prob)
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ViterbiNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl UnigramModel {
    /// Create new Unigram model
    pub fn new(vocab: Vec<(String, f64)>, unk_token: String) -> Self {
        let mut vocab_map = HashMap::new();
        let mut vocab_r = HashMap::new();
        let mut min_score = 0.0f64;

        for (id, (token, score)) in vocab.iter().enumerate() {
            let id = id as u32;
            vocab_map.insert(token.clone(), (id, *score));
            vocab_r.insert(id, token.clone());
            min_score = min_score.min(*score);
        }

        let unk_id = vocab_map.get(&unk_token).map(|(id, _)| *id).unwrap_or(0);

        Self {
            vocab: vocab_map,
            vocab_r,
            unk_token,
            unk_id,
            min_score: min_score - 10.0, // Penalty for unknown
        }
    }

    /// Tokenize using Viterbi algorithm
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        // best_ending[i] = best score ending at position i
        let mut best_ending: Vec<(f64, i32, u32)> = vec![(f64::NEG_INFINITY, -1, 0); n + 1];
        best_ending[0] = (0.0, -1, 0);

        for end in 1..=n {
            for start in 0..end {
                let substr: String = chars[start..end].iter().collect();

                let (token_id, score) = if let Some((id, s)) = self.vocab.get(&substr) {
                    (*id, *s)
                } else if end - start == 1 {
                    // Single character fallback
                    (self.unk_id, self.min_score)
                } else {
                    continue;
                };

                let new_score = best_ending[start].0 + score;
                if new_score > best_ending[end].0 {
                    best_ending[end] = (new_score, start as i32, token_id);
                }
            }
        }

        // Backtrack to get tokens
        let mut tokens = Vec::new();
        let mut pos = n as i32;

        while pos > 0 {
            let (_, prev, token_id) = best_ending[pos as usize];
            tokens.push(token_id);
            pos = prev;
        }

        tokens.reverse();
        tokens
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenize(text)
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab_r.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Token to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).map(|(id, _)| *id)
    }

    /// ID to token
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(&id).map(|s| s.as_str())
    }
}

// =============================================================================
// WordLevel Model (Simple lookup)
// =============================================================================

/// Word-level tokenizer (simple vocabulary lookup)
#[derive(Debug, Clone)]
pub struct WordLevelModel {
    /// Token string -> ID
    vocab: HashMap<String, u32>,
    /// ID -> Token string
    vocab_r: HashMap<u32, String>,
    /// Unknown token
    unk_token: String,
}

impl WordLevelModel {
    /// Create new WordLevel model
    pub fn new(vocab: HashMap<String, u32>, unk_token: String) -> Self {
        let vocab_r: HashMap<u32, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        Self {
            vocab,
            vocab_r,
            unk_token,
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let unk_id = self.vocab.get(&self.unk_token).copied().unwrap_or(0);

        text.split_whitespace()
            .map(|word| self.vocab.get(word).copied().unwrap_or(unk_id))
            .collect()
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab_r.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Token to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// ID to token
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(&id).map(|s| s.as_str())
    }
}

// =============================================================================
// CharBPE Model (character-level BPE with word boundary markers)
// =============================================================================

/// CharBPE tokenizer (original GPT-style with </w> suffix)
#[derive(Debug, Clone)]
pub struct CharBpeModel {
    /// Token string -> ID
    vocab: HashMap<String, u32>,
    /// ID -> Token string
    vocab_r: HashMap<u32, String>,
    /// Merge operations in order: (pair_a, pair_b) -> merged
    merges: Vec<((String, String), String)>,
    /// Merge priority: (pair_a, pair_b) -> rank
    merge_ranks: HashMap<(String, String), usize>,
    /// Word-end suffix (usually "</w>")
    end_of_word_suffix: String,
    /// Unknown token
    unk_token: String,
}

impl CharBpeModel {
    /// Create new CharBPE model
    pub fn new(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        end_of_word_suffix: String,
        unk_token: String,
    ) -> Self {
        let vocab_r: HashMap<u32, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        let mut merge_ops = Vec::new();
        let mut merge_ranks = HashMap::new();

        for (rank, (a, b)) in merges.iter().enumerate() {
            let merged = format!("{}{}", a, b);
            merge_ops.push(((a.clone(), b.clone()), merged));
            merge_ranks.insert((a.clone(), b.clone()), rank);
        }

        Self {
            vocab,
            vocab_r,
            merges: merge_ops,
            merge_ranks,
            end_of_word_suffix,
            unk_token,
        }
    }

    /// Tokenize a single word
    fn tokenize_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }

        // Split into characters, add </w> suffix to last
        let chars: Vec<char> = word.chars().collect();
        let mut tokens: Vec<String> = chars[..chars.len()-1]
            .iter()
            .map(|c| c.to_string())
            .collect();

        // Last character gets the end-of-word suffix
        if let Some(&last) = chars.last() {
            tokens.push(format!("{}{}", last, self.end_of_word_suffix));
        }

        // Apply BPE merges
        loop {
            let mut best_merge: Option<(usize, usize, String)> = None;

            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    match &best_merge {
                        None => {
                            let merged = format!("{}{}", pair.0, pair.1);
                            best_merge = Some((i, rank, merged));
                        }
                        Some((_, best_rank, _)) if rank < *best_rank => {
                            let merged = format!("{}{}", pair.0, pair.1);
                            best_merge = Some((i, rank, merged));
                        }
                        _ => {}
                    }
                }
            }

            match best_merge {
                Some((idx, _, merged)) => {
                    tokens[idx] = merged;
                    tokens.remove(idx + 1);
                }
                None => break,
            }
        }

        tokens
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let unk_id = self.vocab.get(&self.unk_token).copied().unwrap_or(0);

        text.split_whitespace()
            .flat_map(|word| {
                self.tokenize_word(word)
                    .into_iter()
                    .map(|t| self.vocab.get(&t).copied().unwrap_or(unk_id))
            })
            .collect()
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();

        for &id in ids {
            if let Some(token) = self.vocab_r.get(&id) {
                if token.ends_with(&self.end_of_word_suffix) {
                    // Remove suffix and add space after
                    let word_part = &token[..token.len() - self.end_of_word_suffix.len()];
                    result.push_str(word_part);
                    result.push(' ');
                } else {
                    result.push_str(token);
                }
            }
        }

        result.trim_end().to_string()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Token to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// ID to token
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(&id).map(|s| s.as_str())
    }
}

// =============================================================================
// Model Enum (unified interface)
// =============================================================================

/// Unified model interface
#[derive(Debug, Clone)]
pub enum Model {
    /// BPE model (existing)
    BPE,
    /// CharBPE model (original GPT-style)
    CharBPE(CharBpeModel),
    /// WordPiece model (BERT)
    WordPiece(WordPieceModel),
    /// Unigram model (SentencePiece)
    Unigram(UnigramModel),
    /// Word-level model
    WordLevel(WordLevelModel),
}

impl Model {
    /// Create CharBPE model
    pub fn char_bpe(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        suffix: &str,
        unk_token: &str,
    ) -> Self {
        Model::CharBPE(CharBpeModel::new(
            vocab,
            merges,
            suffix.to_string(),
            unk_token.to_string(),
        ))
    }

    /// Create WordPiece model
    pub fn wordpiece(
        vocab: HashMap<String, u32>,
        prefix: &str,
        unk_token: &str,
    ) -> Self {
        Model::WordPiece(WordPieceModel::new(
            vocab,
            prefix.to_string(),
            unk_token.to_string(),
            100,
        ))
    }

    /// Create Unigram model
    pub fn unigram(vocab: Vec<(String, f64)>, unk_token: &str) -> Self {
        Model::Unigram(UnigramModel::new(vocab, unk_token.to_string()))
    }

    /// Create WordLevel model
    pub fn word_level(vocab: HashMap<String, u32>, unk_token: &str) -> Self {
        Model::WordLevel(WordLevelModel::new(vocab, unk_token.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wordpiece() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);
        vocab.insert("##ing".to_string(), 3);
        vocab.insert("play".to_string(), 4);
        vocab.insert("##ed".to_string(), 5);

        let model = WordPieceModel::new(vocab, "##".to_string(), "[UNK]".to_string(), 100);

        let tokens = model.encode("hello world");
        assert_eq!(tokens, vec![1, 2]);

        let decoded = model.decode(&tokens);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_wordpiece_subwords() {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("play".to_string(), 1);
        vocab.insert("##ing".to_string(), 2);
        vocab.insert("##ed".to_string(), 3);

        let model = WordPieceModel::new(vocab, "##".to_string(), "[UNK]".to_string(), 100);

        let tokens = model.encode("playing");
        assert_eq!(tokens, vec![1, 2]); // play + ##ing

        let decoded = model.decode(&tokens);
        assert_eq!(decoded, "playing");
    }

    #[test]
    fn test_unigram() {
        let vocab = vec![
            ("<unk>".to_string(), -10.0),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
            ("c".to_string(), -1.0),
            ("ab".to_string(), -0.5),
            ("bc".to_string(), -0.5),
            ("abc".to_string(), -0.2),
        ];

        let model = UnigramModel::new(vocab, "<unk>".to_string());

        // Should prefer "abc" over "a"+"b"+"c" due to better score
        let tokens = model.encode("abc");
        assert_eq!(tokens.len(), 1);
        assert_eq!(model.id_to_token(tokens[0]), Some("abc"));
    }

    #[test]
    fn test_word_level() {
        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        let model = WordLevelModel::new(vocab, "<unk>".to_string());

        let tokens = model.encode("hello world");
        assert_eq!(tokens, vec![1, 2]);

        let decoded = model.decode(&tokens);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_word_level_unknown() {
        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("hello".to_string(), 1);

        let model = WordLevelModel::new(vocab, "<unk>".to_string());

        let tokens = model.encode("hello unknown");
        assert_eq!(tokens, vec![1, 0]); // "unknown" -> <unk>
    }

    #[test]
    fn test_char_bpe() {
        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("h".to_string(), 1);
        vocab.insert("i</w>".to_string(), 2);
        vocab.insert("hi</w>".to_string(), 3);

        let merges = vec![
            ("h".to_string(), "i</w>".to_string()),
        ];

        let model = CharBpeModel::new(vocab, merges, "</w>".to_string(), "<unk>".to_string());

        let tokens = model.encode("hi");
        assert_eq!(tokens, vec![3]); // "hi" merged

        let decoded = model.decode(&tokens);
        assert_eq!(decoded, "hi");
    }
}
