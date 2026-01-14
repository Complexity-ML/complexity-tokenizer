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
    pub normalizer: Option<serde_json::Value>,
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
        let normalizer = parse_normalizer(&json.normalizer);
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
        self.encode_to_encoding_impl(text, None, None, None)
    }

    /// Encode text pair to full Encoding (for NLI, QA, etc.)
    pub fn encode_pair_to_encoding(&self, text: &str, text_pair: &str) -> Encoding {
        self.encode_to_encoding_impl(text, Some(text_pair), None, None)
    }

    /// Encode with truncation and stride for long texts
    pub fn encode_to_encoding_with_truncation(
        &self,
        text: &str,
        text_pair: Option<&str>,
        max_length: usize,
        stride: usize,
    ) -> Encoding {
        self.encode_to_encoding_impl(text, text_pair, Some(max_length), Some(stride))
    }

    /// Internal encoding implementation
    fn encode_to_encoding_impl(
        &self,
        text: &str,
        text_pair: Option<&str>,
        max_length: Option<usize>,
        stride: Option<usize>,
    ) -> Encoding {
        // Encode first sequence
        let mut encoding = self.encode_single_to_encoding(text, 0);

        // Encode second sequence if provided
        if let Some(pair) = text_pair {
            let pair_encoding = self.encode_single_to_encoding(pair, 1);
            encoding.merge(pair_encoding, 1);
        }

        // Post-process (add special tokens)
        let processed_ids = match &self.post_processor {
            Some(pp) => pp.process(encoding.ids.clone(), None),
            None => encoding.ids.clone(),
        };

        // Update encoding with processed IDs
        let added_count = processed_ids.len() - encoding.ids.len();
        encoding.ids = processed_ids;

        // Extend masks for added special tokens
        encoding.attention_mask.extend(std::iter::repeat(1).take(added_count));
        encoding.special_tokens_mask.extend(std::iter::repeat(1).take(added_count));
        encoding.type_ids.extend(std::iter::repeat(0).take(added_count));

        // Mark special tokens
        let special_ids: Vec<u32> = self.special_tokens.values().copied().collect();
        encoding.mark_special_tokens(&special_ids);

        // Handle truncation with stride (for long texts)
        if let Some(max_len) = max_length {
            if encoding.len() > max_len {
                let stride = stride.unwrap_or(0);
                encoding.truncate_with_stride(max_len, stride);
            }
        }

        encoding
    }

    /// Encode a single text to Encoding
    fn encode_single_to_encoding(&self, text: &str, type_id: u32) -> Encoding {
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

        let len = ids.len();
        Encoding {
            ids,
            type_ids: vec![type_id; len],
            tokens,
            attention_mask: vec![1; len],
            special_tokens_mask: vec![0; len],
            offsets,
            word_ids,
            sequence_ids: vec![Some(type_id as usize); len],
            overflowing: Vec::new(),
        }
    }

    /// Encode batch of texts to full Encodings (parallel)
    pub fn encode_batch_to_encoding(&self, texts: &[&str]) -> Vec<Encoding> {
        texts.par_iter().map(|t| self.encode_to_encoding(t)).collect()
    }

    /// Encode batch of text pairs to full Encodings (parallel)
    pub fn encode_batch_pairs_to_encoding(&self, pairs: &[(&str, &str)]) -> Vec<Encoding> {
        pairs.par_iter()
            .map(|(a, b)| self.encode_pair_to_encoding(a, b))
            .collect()
    }

    /// Encode batch with automatic padding to longest sequence
    pub fn encode_batch_with_padding(
        &self,
        texts: &[&str],
        pad_to_max: Option<usize>,
        pad_left: bool,
    ) -> Vec<Encoding> {
        let mut encodings: Vec<Encoding> = texts.par_iter()
            .map(|t| self.encode_to_encoding(t))
            .collect();

        // Find max length
        let max_len = pad_to_max.unwrap_or_else(|| {
            encodings.iter().map(|e| e.len()).max().unwrap_or(0)
        });

        // Get pad token ID
        let pad_id = self.special_tokens.get("[PAD]")
            .or_else(|| self.special_tokens.get("<pad>"))
            .copied()
            .unwrap_or(0);

        let pad_token = self.vocab.get_token(pad_id).unwrap_or("<pad>");

        // Pad all sequences
        for enc in &mut encodings {
            enc.pad(max_len, pad_id, pad_token, pad_left);
        }

        encodings
    }

    /// Encode batch of pairs with automatic padding
    pub fn encode_batch_pairs_with_padding(
        &self,
        pairs: &[(&str, &str)],
        pad_to_max: Option<usize>,
        pad_left: bool,
    ) -> Vec<Encoding> {
        let mut encodings: Vec<Encoding> = pairs.par_iter()
            .map(|(a, b)| self.encode_pair_to_encoding(a, b))
            .collect();

        let max_len = pad_to_max.unwrap_or_else(|| {
            encodings.iter().map(|e| e.len()).max().unwrap_or(0)
        });

        let pad_id = self.special_tokens.get("[PAD]")
            .or_else(|| self.special_tokens.get("<pad>"))
            .copied()
            .unwrap_or(0);

        let pad_token = self.vocab.get_token(pad_id).unwrap_or("<pad>");

        for enc in &mut encodings {
            enc.pad(max_len, pad_id, pad_token, pad_left);
        }

        encodings
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

        // Reconstruct merges from BPE
        let merges: Vec<String> = self.bpe.merges()
            .iter()
            .map(|m| {
                let a = self.bpe.vocab_r().get(&m.pair.0).cloned().unwrap_or_default();
                let b = self.bpe.vocab_r().get(&m.pair.1).cloned().unwrap_or_default();
                format!("{} {}", a, b)
            })
            .collect();

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
            normalizer: self.normalizer.as_ref().map(serialize_normalizer),
            pre_tokenizer: self.pre_tokenizer.as_ref().map(serialize_pre_tokenizer),
            post_processor: self.post_processor.as_ref().map(|pp| serialize_post_processor(pp, &self.special_tokens)),
            decoder: self.decoder.as_ref().map(serialize_decoder),
        };

        let json = serde_json::to_string_pretty(&tokenizer_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        fs::write(path, json)
    }

    /// Save tokenizer to a directory (HuggingFace pretrained format)
    pub fn save_pretrained<P: AsRef<Path>>(&self, dir: P) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        // Save tokenizer.json
        self.save(dir.join("tokenizer.json"))?;

        // Save tokenizer_config.json
        let config = serde_json::json!({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_type": "bpe",
            "bos_token": self.vocab.special_tokens().bos_token,
            "eos_token": self.vocab.special_tokens().eos_token,
            "unk_token": self.vocab.special_tokens().unk_token,
            "pad_token": self.vocab.special_tokens().pad_token,
            "sep_token": self.vocab.special_tokens().sep_token,
            "cls_token": self.vocab.special_tokens().cls_token,
            "mask_token": self.vocab.special_tokens().mask_token,
        });

        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        fs::write(dir.join("tokenizer_config.json"), config_json)?;

        // Save special_tokens_map.json
        let special_map = serde_json::json!({
            "bos_token": self.vocab.special_tokens().bos_token,
            "eos_token": self.vocab.special_tokens().eos_token,
            "unk_token": self.vocab.special_tokens().unk_token,
            "pad_token": self.vocab.special_tokens().pad_token,
            "sep_token": self.vocab.special_tokens().sep_token,
            "cls_token": self.vocab.special_tokens().cls_token,
            "mask_token": self.vocab.special_tokens().mask_token,
        });

        let special_json = serde_json::to_string_pretty(&special_map)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        fs::write(dir.join("special_tokens_map.json"), special_json)?;

        Ok(())
    }
}

// =============================================================================
// JSON Parsing Helpers
// =============================================================================

/// Parse normalizer from JSON
fn parse_normalizer(json: &Option<serde_json::Value>) -> Option<Normalizer> {
    if let Some(value) = json {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type") {
                let type_str = type_val.as_str().unwrap_or("");
                return match type_str {
                    "NFC" => Some(Normalizer::NFC),
                    "NFD" => Some(Normalizer::NFD),
                    "NFKC" => Some(Normalizer::NFKC),
                    "NFKD" => Some(Normalizer::NFKD),
                    "Lowercase" => Some(Normalizer::Lowercase),
                    "Strip" => Some(Normalizer::Strip),
                    "StripAccents" => Some(Normalizer::StripAccents),
                    "Replace" => {
                        let pattern = obj
                            .get("pattern")
                            .and_then(|v| v.get("String"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let replacement = obj
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        Some(Normalizer::Replace { pattern, replacement })
                    }
                    "Prepend" => {
                        let prepend = obj
                            .get("prepend")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        Some(Normalizer::Prepend(prepend))
                    }
                    "Sequence" => {
                        if let Some(normalizers) = obj.get("normalizers").and_then(|v| v.as_array()) {
                            let parsed: Vec<Normalizer> = normalizers
                                .iter()
                                .filter_map(|n| parse_normalizer(&Some(n.clone())))
                                .collect();
                            if !parsed.is_empty() {
                                Some(Normalizer::Sequence(parsed))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    "BertNormalizer" => {
                        // BERT normalizer: NFC + Lowercase (optionally) + StripAccents (optionally)
                        let lowercase = obj.get("lowercase").and_then(|v| v.as_bool()).unwrap_or(true);
                        let strip_accents = obj.get("strip_accents").and_then(|v| v.as_bool()).unwrap_or(false);
                        let clean_text = obj.get("clean_text").and_then(|v| v.as_bool()).unwrap_or(true);

                        let mut normalizers = vec![Normalizer::NFC];
                        if clean_text {
                            normalizers.push(Normalizer::Strip);
                        }
                        if lowercase {
                            normalizers.push(Normalizer::Lowercase);
                        }
                        if strip_accents {
                            normalizers.push(Normalizer::StripAccents);
                        }
                        Some(Normalizer::Sequence(normalizers))
                    }
                    _ => None,
                };
            }
        }
    }
    // Default to NFC if no normalizer specified
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

// =============================================================================
// Serialization Helpers
// =============================================================================

/// Serialize normalizer to JSON
fn serialize_normalizer(normalizer: &Normalizer) -> serde_json::Value {
    match normalizer {
        Normalizer::NFC => serde_json::json!({"type": "NFC"}),
        Normalizer::NFD => serde_json::json!({"type": "NFD"}),
        Normalizer::NFKC => serde_json::json!({"type": "NFKC"}),
        Normalizer::NFKD => serde_json::json!({"type": "NFKD"}),
        Normalizer::Lowercase => serde_json::json!({"type": "Lowercase"}),
        Normalizer::Strip => serde_json::json!({"type": "Strip"}),
        Normalizer::StripAccents => serde_json::json!({"type": "StripAccents"}),
        Normalizer::Replace { pattern, replacement } => serde_json::json!({
            "type": "Replace",
            "pattern": {"String": pattern},
            "content": replacement
        }),
        Normalizer::Prepend(prepend) => serde_json::json!({
            "type": "Prepend",
            "prepend": prepend
        }),
        Normalizer::Append(append) => serde_json::json!({
            "type": "Append",
            "append": append
        }),
        Normalizer::Sequence(normalizers) => serde_json::json!({
            "type": "Sequence",
            "normalizers": normalizers.iter().map(serialize_normalizer).collect::<Vec<_>>()
        }),
    }
}

/// Serialize pre-tokenizer to JSON
fn serialize_pre_tokenizer(pre_tokenizer: &PreTokenizer) -> serde_json::Value {
    match pre_tokenizer {
        PreTokenizer::ByteLevel { add_prefix_space } => serde_json::json!({
            "type": "ByteLevel",
            "add_prefix_space": add_prefix_space,
            "trim_offsets": true,
            "use_regex": true
        }),
        PreTokenizer::Metaspace { replacement, add_prefix_space } => serde_json::json!({
            "type": "Metaspace",
            "replacement": replacement.to_string(),
            "add_prefix_space": add_prefix_space
        }),
        PreTokenizer::Whitespace => serde_json::json!({"type": "Whitespace"}),
        PreTokenizer::WhitespaceSplit => serde_json::json!({"type": "WhitespaceSplit"}),
        PreTokenizer::Punctuation => serde_json::json!({"type": "Punctuation"}),
        PreTokenizer::Digits { individual_digits } => serde_json::json!({
            "type": "Digits",
            "individual_digits": individual_digits
        }),
        PreTokenizer::Split { pattern, invert } => serde_json::json!({
            "type": "Split",
            "pattern": {"Regex": pattern},
            "behavior": "Removed",
            "invert": invert
        }),
        PreTokenizer::GPT2 => serde_json::json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        }),
        PreTokenizer::Sequence(pretokenizers) => serde_json::json!({
            "type": "Sequence",
            "pretokenizers": pretokenizers.iter().map(serialize_pre_tokenizer).collect::<Vec<_>>()
        }),
    }
}

/// Serialize post-processor to JSON
fn serialize_post_processor(post_processor: &PostProcessor, special_tokens: &HashMap<String, u32>) -> serde_json::Value {
    match post_processor {
        PostProcessor::TemplateProcessing { single, pair, special_tokens: tokens } => {
            let special_tokens_json: Vec<serde_json::Value> = tokens.iter()
                .map(|(token, id)| serde_json::json!({
                    "id": token,
                    "ids": [id],
                    "tokens": [token]
                }))
                .collect();

            serde_json::json!({
                "type": "TemplateProcessing",
                "single": parse_template_to_json(single),
                "pair": pair.as_ref().map(|p| parse_template_to_json(p)),
                "special_tokens": special_tokens_json
            })
        }
        PostProcessor::RobertaProcessing { bos, eos, add_prefix_space } => serde_json::json!({
            "type": "RobertaProcessing",
            "sep": [eos.0.clone(), eos.1],
            "cls": [bos.0.clone(), bos.1],
            "trim_offsets": true,
            "add_prefix_space": add_prefix_space
        }),
        PostProcessor::BertProcessing { cls, sep } => serde_json::json!({
            "type": "BertProcessing",
            "sep": [sep.0.clone(), sep.1],
            "cls": [cls.0.clone(), cls.1]
        }),
        _ => serde_json::json!(null),
    }
}

/// Parse template string to JSON array
fn parse_template_to_json(template: &str) -> Vec<serde_json::Value> {
    template.split_whitespace()
        .map(|part| {
            if part.starts_with('$') {
                serde_json::json!({"Sequence": {"id": &part[1..], "type_id": 0}})
            } else {
                serde_json::json!({"SpecialToken": {"id": part, "type_id": 0}})
            }
        })
        .collect()
}

/// Serialize decoder to JSON
fn serialize_decoder(decoder: &Decoder) -> serde_json::Value {
    match decoder {
        Decoder::ByteLevel => serde_json::json!({"type": "ByteLevel"}),
        Decoder::Metaspace { replacement, add_prefix_space } => serde_json::json!({
            "type": "Metaspace",
            "replacement": replacement.to_string(),
            "add_prefix_space": add_prefix_space
        }),
        Decoder::WordPiece { prefix, cleanup } => serde_json::json!({
            "type": "WordPiece",
            "prefix": prefix,
            "cleanup": cleanup
        }),
        Decoder::BPE { suffix } => serde_json::json!({
            "type": "BPE",
            "suffix": suffix
        }),
        Decoder::Replace { pattern, replacement } => serde_json::json!({
            "type": "Replace",
            "pattern": pattern,
            "content": replacement
        }),
        Decoder::Sequence(decoders) => serde_json::json!({
            "type": "Sequence",
            "decoders": decoders.iter().map(serialize_decoder).collect::<Vec<_>>()
        }),
    }
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
