//! BPE (Byte Pair Encoding) core algorithm

use hashbrown::HashMap;
use rayon::prelude::*;

/// Merge operation (pair -> new token)
#[derive(Debug, Clone)]
pub struct Merge {
    pub pair: (u32, u32),
    pub new_id: u32,
}

/// BPE Tokenizer core
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token string -> ID
    vocab: HashMap<String, u32>,
    /// ID -> Token string
    vocab_r: HashMap<u32, String>,
    /// Merge operations in order
    merges: Vec<Merge>,
    /// Pair -> merge rank (lower = higher priority)
    merge_ranks: HashMap<(u32, u32), usize>,
}

impl BpeTokenizer {
    /// Create new BPE tokenizer from vocab and merges
    pub fn new(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let vocab_r: HashMap<u32, String> = vocab.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        let mut merge_ranks = HashMap::new();
        let mut merge_ops = Vec::with_capacity(merges.len());

        for (rank, (a, b)) in merges.iter().enumerate() {
            if let (Some(&id_a), Some(&id_b)) = (vocab.get(a), vocab.get(b)) {
                let merged = format!("{}{}", a, b);
                if let Some(&new_id) = vocab.get(&merged) {
                    merge_ranks.insert((id_a, id_b), rank);
                    merge_ops.push(Merge {
                        pair: (id_a, id_b),
                        new_id,
                    });
                }
            }
        }

        Self {
            vocab,
            vocab_r,
            merges: merge_ops,
            merge_ranks,
        }
    }

    /// Encode text to token IDs using BPE
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Start with character-level tokens
        let mut tokens: Vec<u32> = text
            .chars()
            .filter_map(|c| self.vocab.get(&c.to_string()).copied())
            .collect();

        if tokens.is_empty() {
            return vec![];
        }

        // Apply merges greedily
        loop {
            // Find best merge (lowest rank)
            let best_merge = self.find_best_merge(&tokens);

            match best_merge {
                Some((idx, new_id)) => {
                    // Apply merge: replace pair at idx with new_id
                    tokens[idx] = new_id;
                    tokens.remove(idx + 1);
                }
                None => break,
            }
        }

        tokens
    }

    /// Find the best merge to apply (lowest rank)
    fn find_best_merge(&self, tokens: &[u32]) -> Option<(usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None; // (idx, rank, new_id)

        for i in 0..tokens.len().saturating_sub(1) {
            let pair = (tokens[i], tokens[i + 1]);
            if let Some(&rank) = self.merge_ranks.get(&pair) {
                let new_id = self.merges[rank].new_id;
                match &best {
                    None => best = Some((i, rank, new_id)),
                    Some((_, best_rank, _)) if rank < *best_rank => {
                        best = Some((i, rank, new_id));
                    }
                    _ => {}
                }
            }
        }

        best.map(|(idx, _, new_id)| (idx, new_id))
    }

    /// Encode batch of texts in parallel
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.par_iter()
            .map(|text| self.encode(text))
            .collect()
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|id| self.vocab_r.get(id))
            .cloned()
            .collect()
    }

    /// Decode batch in parallel
    pub fn decode_batch(&self, batch: &[Vec<u32>]) -> Vec<String> {
        batch.par_iter()
            .map(|ids| self.decode(ids))
            .collect()
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Token to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// ID to token
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    /// Get vocab reference
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encode_decode() {
        let mut vocab = HashMap::new();
        vocab.insert("h".to_string(), 0);
        vocab.insert("e".to_string(), 1);
        vocab.insert("l".to_string(), 2);
        vocab.insert("o".to_string(), 3);
        vocab.insert("he".to_string(), 4);
        vocab.insert("ll".to_string(), 5);
        vocab.insert("hel".to_string(), 6);
        vocab.insert("hell".to_string(), 7);
        vocab.insert("hello".to_string(), 8);

        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("he".to_string(), "l".to_string()),
            ("hel".to_string(), "l".to_string()),
            ("hell".to_string(), "o".to_string()),
        ];

        let tokenizer = BpeTokenizer::new(vocab, merges);
        let encoded = tokenizer.encode("hello");

        assert_eq!(encoded, vec![8]); // "hello" -> single token
    }
}
