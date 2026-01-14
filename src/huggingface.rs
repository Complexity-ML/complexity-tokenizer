//! HuggingFace tokenizer.json compatibility

use crate::bpe::BpeTokenizer;
use crate::decoders::Decoder;
use crate::encoding::Encoding;
use crate::normalizers::Normalizer;
use crate::postprocessors::PostProcessor;
use crate::pretokenizers::PreTokenizer;
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
    /// Normalizer pipeline
    normalizer: Option<Normalizer>,
    /// Pre-tokenizer pipeline
    pre_tokenizer: Option<PreTokenizer>,
    /// Post-processor pipeline
    post_processor: Option<PostProcessor>,
    /// Decoder pipeline
    decoder: Option<Decoder>,
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

        // Parse pipeline components from JSON
        let normalizer = parse_normalizer(&json.pre_tokenizer);
        let pre_tokenizer = parse_pre_tokenizer(&json.pre_tokenizer);
        let post_processor = parse_post_processor(&json.post_processor, &special_token_map);
        let decoder = parse_decoder(&json.decoder);

        Ok(Self {
            bpe,
            vocab,
            special_tokens: special_token_map,
            added_tokens: added_tokens_map,
            normalizer,
            pre_tokenizer,
            post_processor,
            decoder,
        })
    }

    /// Encode text to full Encoding (with attention mask, type ids, etc.)
    pub fn encode_to_encoding(&self, text: &str) -> Encoding {
        // Normalize
        let normalized = match &self.normalizer {
            Some(n) => n.normalize(text),
            None => text.to_string(),
        };

        // Pre-tokenize
        let words = match &self.pre_tokenizer {
            Some(pt) => pt.pre_tokenize(&normalized),
            None => vec![normalized],
        };

        // BPE encode each word
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut word_ids = Vec::new();

        let mut char_offset = 0usize;
        for (word_idx, word) in words.iter().enumerate() {
            let word_ids_part = self.bpe.encode(word);
            for &id in &word_ids_part {
                ids.push(id);
                let token_str = self.vocab.get_token(id).unwrap_or("").to_string();
                let token_len = token_str.len();
                offsets.push((char_offset, char_offset + token_len));
                char_offset += token_len;
                tokens.push(token_str);
                word_ids.push(Some(word_idx));
            }
        }

        // Post-process (add special tokens)
        let processed_ids = match &self.post_processor {
            Some(pp) => pp.process(ids.clone(), None),
            None => ids.clone(),
        };

        // Build full encoding
        let len = processed_ids.len();
        let mut encoding = Encoding {
            ids: processed_ids,
            type_ids: vec![0; len],
            tokens: tokens.clone(),
            attention_mask: vec![1; len],
            special_tokens_mask: vec![0; len],
            offsets,
            word_ids,
            overflowing: Vec::new(),
        };

        // Mark special tokens
        let special_ids: Vec<u32> = self.special_tokens.values().copied().collect();
        encoding.mark_special_tokens(&special_ids);

        encoding
    }

    /// Add a token dynamically
    pub fn add_token(&mut self, content: &str, id: u32, special: bool) {
        self.added_tokens.insert(content.to_string(), id);
        if special {
            self.special_tokens.insert(content.to_string(), id);
        }
    }

    /// Add multiple tokens dynamically
    pub fn add_tokens(&mut self, tokens: Vec<(String, u32, bool)>) {
        for (content, id, special) in tokens {
            self.add_token(&content, id, special);
        }
    }

    /// Set normalizer
    pub fn set_normalizer(&mut self, normalizer: Normalizer) {
        self.normalizer = Some(normalizer);
    }

    /// Set pre-tokenizer
    pub fn set_pre_tokenizer(&mut self, pre_tokenizer: PreTokenizer) {
        self.pre_tokenizer = Some(pre_tokenizer);
    }

    /// Set post-processor
    pub fn set_post_processor(&mut self, post_processor: PostProcessor) {
        self.post_processor = Some(post_processor);
    }

    /// Set decoder
    pub fn set_decoder(&mut self, decoder: Decoder) {
        self.decoder = Some(decoder);
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
        // Get tokens from IDs
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.vocab.get_token(id).map(|s| s.to_string()))
            .collect();

        // Apply decoder if set
        match &self.decoder {
            Some(d) => d.decode(&tokens),
            None => self.bpe.decode(ids),
        }
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

// =============================================================================
// JSON Parsing Helpers
// =============================================================================

/// Parse normalizer from JSON
fn parse_normalizer(_json: &Option<serde_json::Value>) -> Option<Normalizer> {
    // For now, default to NFC - can be extended to parse from JSON
    Some(Normalizer::NFC)
}

/// Parse pre-tokenizer from JSON
fn parse_pre_tokenizer(json: &Option<serde_json::Value>) -> Option<PreTokenizer> {
    if let Some(value) = json {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type") {
                let type_str = type_val.as_str().unwrap_or("");
                return match type_str {
                    "ByteLevel" => {
                        let add_prefix = obj
                            .get("add_prefix_space")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        Some(PreTokenizer::ByteLevel {
                            add_prefix_space: add_prefix,
                        })
                    }
                    "Metaspace" => {
                        let replacement = obj
                            .get("replacement")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.chars().next())
                            .unwrap_or('▁');
                        let add_prefix = obj
                            .get("add_prefix_space")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        Some(PreTokenizer::Metaspace {
                            replacement,
                            add_prefix_space: add_prefix,
                        })
                    }
                    "Whitespace" => Some(PreTokenizer::Whitespace),
                    "Punctuation" => Some(PreTokenizer::Punctuation),
                    _ => None,
                };
            }
        }
    }
    // Default to ByteLevel
    Some(PreTokenizer::ByteLevel {
        add_prefix_space: false,
    })
}

/// Parse post-processor from JSON
fn parse_post_processor(
    json: &Option<serde_json::Value>,
    special_tokens: &HashMap<String, u32>,
) -> Option<PostProcessor> {
    if let Some(value) = json {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type") {
                let type_str = type_val.as_str().unwrap_or("");
                return match type_str {
                    "TemplateProcessing" => {
                        let single = obj
                            .get("single")
                            .and_then(|v| v.as_array())
                            .map(|arr| template_from_array(arr))
                            .unwrap_or_else(|| "<s> $A </s>".to_string());
                        let pair = obj
                            .get("pair")
                            .and_then(|v| v.as_array())
                            .map(|arr| template_from_array(arr));
                        let tokens: Vec<(String, u32)> = special_tokens
                            .iter()
                            .map(|(k, v)| (k.clone(), *v))
                            .collect();
                        Some(PostProcessor::TemplateProcessing {
                            single,
                            pair,
                            special_tokens: tokens,
                        })
                    }
                    "RobertaProcessing" => {
                        let bos = special_tokens.get("<s>").copied().unwrap_or(0);
                        let eos = special_tokens.get("</s>").copied().unwrap_or(2);
                        Some(PostProcessor::RobertaProcessing {
                            bos: ("<s>".to_string(), bos),
                            eos: ("</s>".to_string(), eos),
                            add_prefix_space: false,
                        })
                    }
                    "BertProcessing" => {
                        let cls = special_tokens.get("[CLS]").copied().unwrap_or(101);
                        let sep = special_tokens.get("[SEP]").copied().unwrap_or(102);
                        Some(PostProcessor::BertProcessing {
                            cls: ("[CLS]".to_string(), cls),
                            sep: ("[SEP]".to_string(), sep),
                        })
                    }
                    _ => None,
                };
            }
        }
    }
    None
}

/// Convert template array to string
fn template_from_array(arr: &[serde_json::Value]) -> String {
    arr.iter()
        .filter_map(|item| {
            if let Some(obj) = item.as_object() {
                if let Some(special) = obj.get("SpecialToken") {
                    return special
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                }
                if let Some(seq) = obj.get("Sequence") {
                    return seq
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| format!("${}", s));
                }
            }
            None
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse decoder from JSON
fn parse_decoder(json: &Option<serde_json::Value>) -> Option<Decoder> {
    if let Some(value) = json {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type") {
                let type_str = type_val.as_str().unwrap_or("");
                return match type_str {
                    "ByteLevel" => Some(Decoder::ByteLevel),
                    "Metaspace" => {
                        let replacement = obj
                            .get("replacement")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.chars().next())
                            .unwrap_or('▁');
                        let add_prefix = obj
                            .get("add_prefix_space")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        Some(Decoder::Metaspace {
                            replacement,
                            add_prefix_space: add_prefix,
                        })
                    }
                    "WordPiece" => {
                        let prefix = obj
                            .get("prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("##")
                            .to_string();
                        let cleanup = obj
                            .get("cleanup")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        Some(Decoder::WordPiece { prefix, cleanup })
                    }
                    "BPE" => {
                        let suffix = obj
                            .get("suffix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("</w>")
                            .to_string();
                        Some(Decoder::BPE { suffix })
                    }
                    _ => None,
                };
            }
        }
    }
    // Default to ByteLevel
    Some(Decoder::ByteLevel)
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
