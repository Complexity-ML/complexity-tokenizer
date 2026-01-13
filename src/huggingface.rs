//! HuggingFace tokenizer.json compatibility

use crate::bpe::BpeTokenizer;
use crate::vocab::{SpecialTokens, Vocab};
use hashbrown::HashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, BufReader};
use std::path::Path;

/// HuggingFace tokenizer.json format
#[derive(Debug, Deserialize, Serialize)]
pub struct TokenizerJson {
    pub version: Option<String>,
    pub model: ModelJson,
    #[serde(default)]
    pub added_tokens: Vec<AddedToken>,
    pub pre_tokenizer: Option<serde_json::Value>,
    pub post_processor: Option<serde_json::Value>,
    pub decoder: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelJson {
    #[serde(rename = "type")]
    pub model_type: Option<String>,
    pub vocab: HashMap<String, u32>,
    #[serde(default)]
    pub merges: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
    #[serde(default)]
    pub single_word: bool,
    #[serde(default)]
    pub lstrip: bool,
    #[serde(default)]
    pub rstrip: bool,
    #[serde(default)]
    pub normalized: bool,
}

/// HuggingFace compatible tokenizer
#[derive(Debug, Clone)]
pub struct HuggingFaceTokenizer {
    bpe: BpeTokenizer,
    vocab: Vocab,
    special_tokens: HashMap<String, u32>,
    added_tokens: HashMap<String, u32>,
}

impl HuggingFaceTokenizer {
    /// Load from tokenizer.json file
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let tokenizer_json: TokenizerJson = serde_json::from_reader(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Self::from_tokenizer_json(tokenizer_json)
    }

    /// Load from HuggingFace Hub
    pub fn from_pretrained(repo_id: &str) -> io::Result<Self> {
        // Build URL to tokenizer.json
        let url = format!(
            "https://huggingface.co/{}/resolve/main/tokenizer.json",
            repo_id
        );

        // Download (blocking for simplicity - could be async)
        let response = ureq::get(&url)
            .call()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        let tokenizer_json: TokenizerJson = response
            .into_json()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Self::from_tokenizer_json(tokenizer_json)
    }

    /// Build from parsed tokenizer.json
    fn from_tokenizer_json(json: TokenizerJson) -> io::Result<Self> {
        // Parse merges: "token1 token2" -> (token1, token2)
        let merges: Vec<(String, String)> = json
            .model
            .merges
            .iter()
            .filter_map(|m| {
                let parts: Vec<&str> = m.split(' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        // Create BPE tokenizer
        let bpe = BpeTokenizer::new(json.model.vocab.clone(), merges);

        // Detect special tokens from added_tokens
        let mut special_tokens = SpecialTokens::default();
        let mut special_token_map = HashMap::new();
        let mut added_tokens_map = HashMap::new();

        for token in &json.added_tokens {
            added_tokens_map.insert(token.content.clone(), token.id);

            if token.special {
                special_token_map.insert(token.content.clone(), token.id);

                // Try to identify special token types by common names
                let content_lower = token.content.to_lowercase();
                if content_lower.contains("unk") {
                    special_tokens.unk_token = Some(token.content.clone());
                } else if content_lower == "<s>" || content_lower.contains("bos") {
                    special_tokens.bos_token = Some(token.content.clone());
                } else if content_lower == "</s>" || content_lower.contains("eos") {
                    special_tokens.eos_token = Some(token.content.clone());
                } else if content_lower.contains("pad") {
                    special_tokens.pad_token = Some(token.content.clone());
                } else if content_lower.contains("sep") {
                    special_tokens.sep_token = Some(token.content.clone());
                } else if content_lower.contains("cls") {
                    special_tokens.cls_token = Some(token.content.clone());
                } else if content_lower.contains("mask") {
                    special_tokens.mask_token = Some(token.content.clone());
                }
            }
        }

        // Create vocab
        let vocab = Vocab::new(json.model.vocab, special_tokens);

        Ok(Self {
            bpe,
            vocab,
            special_tokens: special_token_map,
            added_tokens: added_tokens_map,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Handle added tokens first (special tokens)
        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Check for special/added tokens at start
            let mut found_special = false;
            for (token, &id) in &self.added_tokens {
                if remaining.starts_with(token) {
                    result.push(id);
                    remaining = &remaining[token.len()..];
                    found_special = true;
                    break;
                }
            }

            if !found_special {
                // Find next special token or end
                let next_special_pos = self
                    .added_tokens
                    .keys()
                    .filter_map(|t| remaining.find(t))
                    .min()
                    .unwrap_or(remaining.len());

                if next_special_pos > 0 {
                    // Encode regular text with BPE
                    let chunk = &remaining[..next_special_pos];
                    result.extend(self.bpe.encode(chunk));
                    remaining = &remaining[next_special_pos..];
                }
            }
        }

        result
    }

    /// Encode batch (parallel)
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.par_iter().map(|t| self.encode(t)).collect()
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        self.bpe.decode(ids)
    }

    /// Decode batch (parallel)
    pub fn decode_batch(&self, batch: &[Vec<u32>]) -> Vec<String> {
        batch.par_iter().map(|ids| self.decode(ids)).collect()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Token to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }

    /// ID to token
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id).map(|s| s.to_string())
    }

    /// Get special tokens map
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }

    /// Save tokenizer to file (HuggingFace format)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let vocab = self.bpe.vocab().clone();

        // Reconstruct merges (we don't store them, so this is limited)
        let merges: Vec<String> = Vec::new();

        let added_tokens: Vec<AddedToken> = self
            .added_tokens
            .iter()
            .map(|(content, &id)| AddedToken {
                id,
                content: content.clone(),
                special: self.special_tokens.contains_key(content),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
            })
            .collect();

        let tokenizer_json = TokenizerJson {
            version: Some("1.0".to_string()),
            model: ModelJson {
                model_type: Some("BPE".to_string()),
                vocab,
                merges,
            },
            added_tokens,
            pre_tokenizer: None,
            post_processor: None,
            decoder: None,
        };

        let json = serde_json::to_string_pretty(&tokenizer_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        fs::write(path, json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_tokenizer_json() {
        // Create minimal tokenizer.json
        let json = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {
                    "h": 0,
                    "e": 1,
                    "l": 2,
                    "o": 3,
                    " ": 4,
                    "w": 5,
                    "r": 6,
                    "d": 7
                },
                "merges": []
            },
            "added_tokens": []
        }"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(json.as_bytes()).unwrap();

        let tokenizer = HuggingFaceTokenizer::from_file(file.path()).unwrap();
        assert_eq!(tokenizer.vocab_size(), 8);
    }
}
