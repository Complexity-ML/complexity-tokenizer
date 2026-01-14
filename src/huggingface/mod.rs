//! HuggingFace tokenizer.json compatibility
//!
//! This module provides full compatibility with HuggingFace's tokenizer format.

mod config;
mod parsing;
mod serialization;
mod chat;

pub use config::{PaddingConfig, TruncationConfig};
pub use chat::ChatTemplateResult;

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

// =============================================================================
// JSON Schema Types
// =============================================================================

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

// =============================================================================
// Main Tokenizer Struct
// =============================================================================

/// Internal representation of an added token with its behavior flags
#[derive(Debug, Clone)]
struct AddedTokenInternal {
    id: u32,
    special: bool,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
}

/// HuggingFace compatible tokenizer
#[derive(Debug, Clone)]
pub struct HuggingFaceTokenizer {
    bpe: BpeTokenizer,
    vocab: Vocab,
    special_tokens: HashMap<String, u32>,
    added_tokens: HashMap<String, u32>,
    added_tokens_config: HashMap<String, AddedTokenInternal>,
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    post_processor: Option<PostProcessor>,
    decoder: Option<Decoder>,
    model_max_length: usize,
    padding_side: String,
    truncation_side: String,
    chat_template: Option<String>,
    padding_config: PaddingConfig,
    truncation_config: TruncationConfig,
}

impl HuggingFaceTokenizer {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Load from tokenizer.json file
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let tokenizer_json: TokenizerJson = serde_json::from_reader(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Self::from_tokenizer_json(tokenizer_json)
    }

    /// Load from JSON string
    pub fn from_str(json: &str) -> io::Result<Self> {
        let tokenizer_json: TokenizerJson = serde_json::from_str(json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Self::from_tokenizer_json(tokenizer_json)
    }

    /// Load from bytes buffer
    pub fn from_buffer(buffer: &[u8]) -> io::Result<Self> {
        let tokenizer_json: TokenizerJson = serde_json::from_slice(buffer)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Self::from_tokenizer_json(tokenizer_json)
    }

    /// Load from HuggingFace Hub
    pub fn from_pretrained(repo_id: &str) -> io::Result<Self> {
        Self::from_pretrained_with_options(repo_id, None, false)
    }

    /// Load from HuggingFace Hub with options
    pub fn from_pretrained_with_options(
        repo_id: &str,
        revision: Option<&str>,
        local_files_only: bool,
    ) -> io::Result<Self> {
        let rev = revision.unwrap_or("main");

        if local_files_only {
            let config = crate::hub::HubConfig::default();
            let repo_cache = config.cache_dir.join(repo_id.replace('/', "--"));
            let cached_file = repo_cache.join("tokenizer.json");
            if cached_file.exists() {
                return Self::from_file(cached_file);
            }
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Model '{}' not found in cache and local_files_only=true", repo_id),
            ));
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/{}/tokenizer.json",
            repo_id, rev
        );

        let response = ureq::get(&url)
            .call()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        let tokenizer_json: TokenizerJson = response
            .into_json()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let config_url = format!(
            "https://huggingface.co/{}/resolve/{}/tokenizer_config.json",
            repo_id, rev
        );

        let mut model_max_length = 512;
        let mut chat_template = None;

        if let Ok(config_response) = ureq::get(&config_url).call() {
            if let Ok(config_json) = config_response.into_json::<serde_json::Value>() {
                if let Some(max_len) = config_json.get("model_max_length").and_then(|v| v.as_u64()) {
                    model_max_length = max_len as usize;
                }
                if let Some(template) = config_json.get("chat_template").and_then(|v| v.as_str()) {
                    chat_template = Some(template.to_string());
                }
            }
        }

        Self::from_tokenizer_json_with_config(tokenizer_json, model_max_length, chat_template)
    }

    fn from_tokenizer_json(json: TokenizerJson) -> io::Result<Self> {
        Self::from_tokenizer_json_with_config(json, 512, None)
    }

    fn from_tokenizer_json_with_config(
        json: TokenizerJson,
        model_max_length: usize,
        chat_template: Option<String>,
    ) -> io::Result<Self> {
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

        let bpe = BpeTokenizer::new(json.model.vocab.clone(), merges);

        let mut special_tokens = SpecialTokens::default();
        let mut special_token_map = HashMap::new();
        let mut added_tokens_map = HashMap::new();

        let mut added_tokens_config = HashMap::new();

        for token in &json.added_tokens {
            added_tokens_map.insert(token.content.clone(), token.id);
            added_tokens_config.insert(token.content.clone(), AddedTokenInternal {
                id: token.id,
                special: token.special,
                single_word: token.single_word,
                lstrip: token.lstrip,
                rstrip: token.rstrip,
                normalized: token.normalized,
            });

            if token.special {
                special_token_map.insert(token.content.clone(), token.id);

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

        let vocab = Vocab::new(json.model.vocab, special_tokens);

        let normalizer = parsing::parse_normalizer(&json.normalizer);
        let pre_tokenizer = parsing::parse_pre_tokenizer(&json.pre_tokenizer);
        let post_processor = parsing::parse_post_processor(&json.post_processor, &special_token_map);
        let decoder = parsing::parse_decoder(&json.decoder);

        Ok(Self {
            bpe,
            vocab,
            special_tokens: special_token_map,
            added_tokens: added_tokens_map,
            added_tokens_config,
            normalizer,
            pre_tokenizer,
            post_processor,
            decoder,
            model_max_length,
            padding_side: "right".to_string(),
            truncation_side: "right".to_string(),
            chat_template,
            padding_config: PaddingConfig::default(),
            truncation_config: TruncationConfig {
                max_length: model_max_length,
                ..Default::default()
            },
        })
    }

    // =========================================================================
    // Encoding Methods
    // =========================================================================

    pub fn encode_to_encoding(&self, text: &str) -> Encoding {
        self.encode_to_encoding_impl(text, None, None, None)
    }

    pub fn encode_pair_to_encoding(&self, text: &str, text_pair: &str) -> Encoding {
        self.encode_to_encoding_impl(text, Some(text_pair), None, None)
    }

    pub fn encode_to_encoding_with_truncation(
        &self,
        text: &str,
        text_pair: Option<&str>,
        max_length: usize,
        stride: usize,
    ) -> Encoding {
        self.encode_to_encoding_impl(text, text_pair, Some(max_length), Some(stride))
    }

    fn encode_to_encoding_impl(
        &self,
        text: &str,
        text_pair: Option<&str>,
        max_length: Option<usize>,
        stride: Option<usize>,
    ) -> Encoding {
        let mut encoding = self.encode_single_to_encoding(text, 0);

        if let Some(pair) = text_pair {
            let pair_encoding = self.encode_single_to_encoding(pair, 1);
            encoding.merge(pair_encoding, 1);
        }

        let processed_ids = match &self.post_processor {
            Some(pp) => pp.process(encoding.ids.clone(), None),
            None => encoding.ids.clone(),
        };

        let added_count = processed_ids.len() - encoding.ids.len();
        encoding.ids = processed_ids;

        encoding.attention_mask.extend(std::iter::repeat(1).take(added_count));
        encoding.special_tokens_mask.extend(std::iter::repeat(1).take(added_count));
        encoding.type_ids.extend(std::iter::repeat(0).take(added_count));

        let special_ids: Vec<u32> = self.special_tokens.values().copied().collect();
        encoding.mark_special_tokens(&special_ids);

        if let Some(max_len) = max_length {
            if encoding.len() > max_len {
                let stride = stride.unwrap_or(0);
                encoding.truncate_with_stride(max_len, stride);
            }
        }

        encoding
    }

    fn encode_single_to_encoding(&self, text: &str, type_id: u32) -> Encoding {
        // Track original text positions
        let original_text = text;

        let normalized = match &self.normalizer {
            Some(n) => n.normalize(text),
            None => text.to_string(),
        };

        // Pre-tokenize and track word boundaries in original text
        let words_with_offsets = self.pre_tokenize_with_offsets(&normalized, original_text);

        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut word_ids = Vec::new();

        for (word_idx, (word, word_start, word_end)) in words_with_offsets.iter().enumerate() {
            let word_ids_part = self.bpe.encode(word);

            // Calculate per-token offsets within the word
            let mut token_char_offset = *word_start;
            for &id in &word_ids_part {
                ids.push(id);
                let token_str = self.vocab.get_token(id).unwrap_or("").to_string();

                // Calculate token length in original text (approximate)
                let token_byte_len = token_str.len();
                let token_end = (token_char_offset + token_byte_len).min(*word_end);

                offsets.push((token_char_offset, token_end));
                token_char_offset = token_end;
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

    /// Pre-tokenize and return words with their character offsets in the original text
    fn pre_tokenize_with_offsets(&self, normalized: &str, original: &str) -> Vec<(String, usize, usize)> {
        let words = match &self.pre_tokenizer {
            Some(pt) => pt.pre_tokenize(normalized),
            None => vec![normalized.to_string()],
        };

        // Try to map words back to original text positions
        let mut result = Vec::new();
        let mut search_start = 0;

        for word in words {
            // Find word in original text (starting from where we left off)
            let word_trimmed = word.trim_start_matches(|c: char| c == 'Ġ' || c == '▁');
            let word_to_find = if word_trimmed.is_empty() { &word } else { word_trimmed };

            if let Some(pos) = original[search_start..].find(word_to_find) {
                let start = search_start + pos;
                let end = start + word_to_find.len();
                result.push((word.clone(), start, end));
                search_start = end;
            } else {
                // Fallback: use position based on accumulated length
                let start = search_start;
                let end = (start + word.len()).min(original.len());
                result.push((word.clone(), start, end));
                search_start = end;
            }
        }

        result
    }

    pub fn encode_batch_to_encoding(&self, texts: &[&str]) -> Vec<Encoding> {
        texts.par_iter().map(|t| self.encode_to_encoding(t)).collect()
    }

    pub fn encode_batch_pairs_to_encoding(&self, pairs: &[(&str, &str)]) -> Vec<Encoding> {
        pairs.par_iter()
            .map(|(a, b)| self.encode_pair_to_encoding(a, b))
            .collect()
    }

    pub fn encode_batch_with_padding(
        &self,
        texts: &[&str],
        pad_to_max: Option<usize>,
        pad_left: bool,
    ) -> Vec<Encoding> {
        let mut encodings: Vec<Encoding> = texts.par_iter()
            .map(|t| self.encode_to_encoding(t))
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

    // =========================================================================
    // Basic Encode/Decode
    // =========================================================================

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut found_special = false;
            let mut best_match: Option<(&str, u32, usize)> = None;

            for (token, &id) in &self.added_tokens {
                if let Some(config) = self.added_tokens_config.get(token) {
                    // Check for match with proper lstrip/rstrip/single_word handling
                    if let Some(pos) = self.find_added_token(remaining, token, config) {
                        if pos == 0 {
                            // Direct match at start
                            if best_match.is_none() || token.len() > best_match.unwrap().0.len() {
                                best_match = Some((token, id, token.len()));
                            }
                        }
                    }
                } else if remaining.starts_with(token) {
                    if best_match.is_none() || token.len() > best_match.unwrap().0.len() {
                        best_match = Some((token, id, token.len()));
                    }
                }
            }

            if let Some((_, id, len)) = best_match {
                result.push(id);
                remaining = &remaining[len..];
                found_special = true;
            }

            if !found_special {
                let next_special_pos = self.find_next_added_token_pos(remaining);

                if next_special_pos > 0 {
                    let chunk = &remaining[..next_special_pos];
                    result.extend(self.bpe.encode(chunk));
                    remaining = &remaining[next_special_pos..];
                } else if next_special_pos == 0 {
                    // Shouldn't happen, but safety net
                    break;
                }
            }
        }

        result
    }

    /// Find an added token considering lstrip/rstrip/single_word flags
    fn find_added_token(&self, text: &str, token: &str, config: &AddedTokenInternal) -> Option<usize> {
        let pos = text.find(token)?;

        // Check single_word constraint
        if config.single_word {
            // Token must be surrounded by word boundaries
            let before_ok = pos == 0 || {
                let before_char = text[..pos].chars().last();
                before_char.map_or(true, |c| !c.is_alphanumeric())
            };

            let after_ok = pos + token.len() >= text.len() || {
                let after_char = text[pos + token.len()..].chars().next();
                after_char.map_or(true, |c| !c.is_alphanumeric())
            };

            if !before_ok || !after_ok {
                return None;
            }
        }

        // Check lstrip constraint (token should be preceded by whitespace or start)
        if config.lstrip && pos > 0 {
            let before_char = text[..pos].chars().last();
            if !before_char.map_or(true, |c| c.is_whitespace()) {
                return None;
            }
        }

        // Check rstrip constraint (token should be followed by whitespace or end)
        if config.rstrip && pos + token.len() < text.len() {
            let after_char = text[pos + token.len()..].chars().next();
            if !after_char.map_or(true, |c| c.is_whitespace()) {
                return None;
            }
        }

        Some(pos)
    }

    /// Find the position of the next added token in text
    fn find_next_added_token_pos(&self, text: &str) -> usize {
        let mut min_pos = text.len();

        for (token, _) in &self.added_tokens {
            if let Some(config) = self.added_tokens_config.get(token) {
                if let Some(pos) = self.find_added_token(text, token, config) {
                    min_pos = min_pos.min(pos);
                }
            } else if let Some(pos) = text.find(token) {
                min_pos = min_pos.min(pos);
            }
        }

        min_pos
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.par_iter().map(|t| self.encode(t)).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_impl(ids, false, true)
    }

    pub fn decode_with_options(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> String {
        self.decode_impl(ids, skip_special_tokens, clean_up_tokenization_spaces)
    }

    fn decode_impl(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> String {
        let ids_to_decode: Vec<u32> = if skip_special_tokens {
            ids.iter()
                .copied()
                .filter(|&id| {
                    if let Some(token) = self.vocab.get_token(id) {
                        !self.special_tokens.contains_key(token)
                    } else {
                        true
                    }
                })
                .collect()
        } else {
            ids.to_vec()
        };

        let tokens: Vec<String> = ids_to_decode
            .iter()
            .filter_map(|&id| self.vocab.get_token(id).map(|s| s.to_string()))
            .collect();

        let text = match &self.decoder {
            Some(d) => d.decode(&tokens),
            None => self.bpe.decode(&ids_to_decode),
        };

        if clean_up_tokenization_spaces {
            self.clean_up_tokenization_spaces(&text)
        } else {
            text
        }
    }

    pub fn clean_up_tokenization_spaces(&self, text: &str) -> String {
        text
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" :", ":")
            .replace(" ;", ";")
            .replace("\" ", "\"")
            .replace(" \"", "\"")
            .replace("' ", "'")
            .replace(" '", "'")
            .replace("( ", "(")
            .replace(" )", ")")
            .replace("[ ", "[")
            .replace(" ]", "]")
            .replace(" - ", "-")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn decode_batch(&self, batch: &[Vec<u32>]) -> Vec<String> {
        batch.par_iter().map(|ids| self.decode(ids)).collect()
    }

    pub fn decode_batch_with_options(
        &self,
        batch: &[Vec<u32>],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> Vec<String> {
        batch
            .par_iter()
            .map(|ids| self.decode_with_options(ids, skip_special_tokens, clean_up_tokenization_spaces))
            .collect()
    }

    // =========================================================================
    // Token Management
    // =========================================================================

    pub fn add_token(&mut self, content: &str, id: u32, special: bool) {
        self.added_tokens.insert(content.to_string(), id);
        self.added_tokens_config.insert(content.to_string(), AddedTokenInternal {
            id,
            special,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: !special,
        });
        if special {
            self.special_tokens.insert(content.to_string(), id);
        }
    }

    /// Add a token with full configuration
    pub fn add_token_with_config(
        &mut self,
        content: &str,
        id: u32,
        special: bool,
        single_word: bool,
        lstrip: bool,
        rstrip: bool,
    ) {
        self.added_tokens.insert(content.to_string(), id);
        self.added_tokens_config.insert(content.to_string(), AddedTokenInternal {
            id,
            special,
            single_word,
            lstrip,
            rstrip,
            normalized: !special,
        });
        if special {
            self.special_tokens.insert(content.to_string(), id);
        }
    }

    pub fn add_tokens(&mut self, tokens: Vec<(String, u32, bool)>) {
        for (content, id, special) in tokens {
            self.add_token(&content, id, special);
        }
    }

    pub fn set_normalizer(&mut self, normalizer: Normalizer) {
        self.normalizer = Some(normalizer);
    }

    pub fn set_pre_tokenizer(&mut self, pre_tokenizer: PreTokenizer) {
        self.pre_tokenizer = Some(pre_tokenizer);
    }

    pub fn set_post_processor(&mut self, post_processor: PostProcessor) {
        self.post_processor = Some(post_processor);
    }

    pub fn set_decoder(&mut self, decoder: Decoder) {
        self.decoder = Some(decoder);
    }

    // =========================================================================
    // Vocabulary Access
    // =========================================================================

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id).map(|s| s.to_string())
    }

    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.bpe.vocab().clone()
    }

    pub fn convert_ids_to_tokens(&self, ids: &[u32], skip_special_tokens: bool) -> Vec<Option<String>> {
        ids.iter()
            .map(|&id| {
                if let Some(token) = self.vocab.get_token(id) {
                    if skip_special_tokens && self.special_tokens.contains_key(token) {
                        None
                    } else {
                        Some(token.to_string())
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn convert_tokens_to_string(&self, tokens: &[String]) -> String {
        match &self.decoder {
            Some(d) => d.decode(tokens),
            None => tokens.join(""),
        }
    }

    pub fn get_special_tokens_mask(&self, ids: &[u32], already_has_special_tokens: bool) -> Vec<u32> {
        if already_has_special_tokens {
            ids.iter()
                .map(|&id| {
                    if let Some(token) = self.vocab.get_token(id) {
                        if self.special_tokens.contains_key(token) { 1 } else { 0 }
                    } else {
                        0
                    }
                })
                .collect()
        } else {
            vec![0; ids.len()]
        }
    }

    pub fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        match &self.post_processor {
            Some(PostProcessor::BertProcessing { .. }) => {
                if is_pair { 3 } else { 2 }
            }
            Some(PostProcessor::RobertaProcessing { .. }) => {
                if is_pair { 4 } else { 2 }
            }
            Some(PostProcessor::TemplateProcessing { single, pair, .. }) => {
                let template = if is_pair { pair.as_ref().unwrap_or(single) } else { single };
                template
                    .split_whitespace()
                    .filter(|part| !part.starts_with('$'))
                    .count()
            }
            _ => 0,
        }
    }

    pub fn is_fast(&self) -> bool {
        true
    }

    // =========================================================================
    // Properties
    // =========================================================================

    pub fn model_max_length(&self) -> usize {
        self.model_max_length
    }

    pub fn set_model_max_length(&mut self, max_length: usize) {
        self.model_max_length = max_length;
    }

    pub fn padding_side(&self) -> &str {
        &self.padding_side
    }

    pub fn set_padding_side(&mut self, side: &str) {
        self.padding_side = side.to_string();
    }

    pub fn truncation_side(&self) -> &str {
        &self.truncation_side
    }

    pub fn set_truncation_side(&mut self, side: &str) {
        self.truncation_side = side.to_string();
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn set_chat_template(&mut self, template: Option<String>) {
        self.chat_template = template;
    }

    // =========================================================================
    // Special Token Properties
    // =========================================================================

    pub fn bos_token(&self) -> Option<&str> {
        self.vocab.special_tokens().bos_token.as_deref()
    }

    pub fn set_bos_token(&mut self, token: Option<String>) {
        if let Some(ref tok) = token {
            if let Some(id) = self.vocab.get_id(tok) {
                self.special_tokens.insert(tok.clone(), id);
            }
        }
    }

    pub fn eos_token(&self) -> Option<&str> {
        self.vocab.special_tokens().eos_token.as_deref()
    }

    pub fn pad_token(&self) -> Option<&str> {
        self.vocab.special_tokens().pad_token.as_deref()
    }

    pub fn unk_token(&self) -> Option<&str> {
        self.vocab.special_tokens().unk_token.as_deref()
    }

    pub fn sep_token(&self) -> Option<&str> {
        self.vocab.special_tokens().sep_token.as_deref()
    }

    pub fn cls_token(&self) -> Option<&str> {
        self.vocab.special_tokens().cls_token.as_deref()
    }

    pub fn mask_token(&self) -> Option<&str> {
        self.vocab.special_tokens().mask_token.as_deref()
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.vocab.bos_id()
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.vocab.eos_id()
    }

    pub fn pad_token_id(&self) -> Option<u32> {
        self.vocab.pad_id()
    }

    pub fn unk_token_id(&self) -> Option<u32> {
        self.vocab.unk_id()
    }

    pub fn sep_token_id(&self) -> Option<u32> {
        self.sep_token().and_then(|tok| self.vocab.get_id(tok))
    }

    pub fn cls_token_id(&self) -> Option<u32> {
        self.cls_token().and_then(|tok| self.vocab.get_id(tok))
    }

    pub fn mask_token_id(&self) -> Option<u32> {
        self.mask_token().and_then(|tok| self.vocab.get_id(tok))
    }

    pub fn all_special_tokens(&self) -> Vec<String> {
        let mut tokens = Vec::new();
        if let Some(tok) = self.bos_token() { tokens.push(tok.to_string()); }
        if let Some(tok) = self.eos_token() { tokens.push(tok.to_string()); }
        if let Some(tok) = self.pad_token() { tokens.push(tok.to_string()); }
        if let Some(tok) = self.unk_token() { tokens.push(tok.to_string()); }
        if let Some(tok) = self.sep_token() { tokens.push(tok.to_string()); }
        if let Some(tok) = self.cls_token() { tokens.push(tok.to_string()); }
        if let Some(tok) = self.mask_token() { tokens.push(tok.to_string()); }
        for tok in self.special_tokens.keys() {
            if !tokens.contains(tok) {
                tokens.push(tok.clone());
            }
        }
        tokens
    }

    pub fn all_special_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Some(id) = self.bos_token_id() { ids.push(id); }
        if let Some(id) = self.eos_token_id() { ids.push(id); }
        if let Some(id) = self.pad_token_id() { ids.push(id); }
        if let Some(id) = self.unk_token_id() { ids.push(id); }
        if let Some(id) = self.sep_token_id() { ids.push(id); }
        if let Some(id) = self.cls_token_id() { ids.push(id); }
        if let Some(id) = self.mask_token_id() { ids.push(id); }
        for &id in self.special_tokens.values() {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }

    // =========================================================================
    // Tokenization
    // =========================================================================

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let normalized = match &self.normalizer {
            Some(n) => n.normalize(text),
            None => text.to_string(),
        };

        let words = match &self.pre_tokenizer {
            Some(pt) => pt.pre_tokenize(&normalized),
            None => vec![normalized],
        };

        let mut tokens = Vec::new();
        for word in &words {
            let word_ids = self.bpe.encode(word);
            for &id in &word_ids {
                if let Some(token) = self.vocab.get_token(id) {
                    tokens.push(token.to_string());
                }
            }
        }
        tokens
    }

    pub fn convert_tokens_to_ids(&self, tokens: &[String]) -> Vec<Option<u32>> {
        tokens.iter()
            .map(|token| self.vocab.get_id(token))
            .collect()
    }

    pub fn convert_token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }

    // =========================================================================
    // Padding/Truncation Configuration
    // =========================================================================

    pub fn enable_padding(
        &mut self,
        direction: Option<&str>,
        pad_to_multiple_of: Option<usize>,
        pad_id: Option<u32>,
        pad_token: Option<&str>,
        length: Option<usize>,
    ) {
        self.padding_config.enabled = true;
        self.padding_config.direction = direction.unwrap_or("right").to_string();
        self.padding_config.pad_to_multiple_of = pad_to_multiple_of;
        if let Some(dir) = direction {
            self.padding_side = dir.to_string();
        }
        if length.is_some() {
            self.padding_config.strategy = "max_length".to_string();
        } else {
            self.padding_config.strategy = "longest".to_string();
        }
        if let (Some(tok), Some(id)) = (pad_token, pad_id) {
            self.add_token(tok, id, true);
        }
    }

    pub fn no_padding(&mut self) {
        self.padding_config.enabled = false;
    }

    pub fn enable_truncation(
        &mut self,
        max_length: usize,
        stride: Option<usize>,
        strategy: Option<&str>,
        direction: Option<&str>,
    ) {
        self.truncation_config.enabled = true;
        self.truncation_config.max_length = max_length;
        self.truncation_config.stride = stride.unwrap_or(0);
        self.truncation_config.strategy = strategy.unwrap_or("longest_first").to_string();
        self.truncation_config.direction = direction.unwrap_or("right").to_string();
        if let Some(dir) = direction {
            self.truncation_side = dir.to_string();
        }
    }

    pub fn no_truncation(&mut self) {
        self.truncation_config.enabled = false;
    }

    pub fn padding(&self) -> Option<&PaddingConfig> {
        if self.padding_config.enabled {
            Some(&self.padding_config)
        } else {
            None
        }
    }

    pub fn truncation(&self) -> Option<&TruncationConfig> {
        if self.truncation_config.enabled {
            Some(&self.truncation_config)
        } else {
            None
        }
    }

    // =========================================================================
    // Add Special Tokens
    // =========================================================================

    pub fn add_special_tokens_dict(&mut self, special_tokens_dict: &std::collections::HashMap<String, String>) -> usize {
        let mut num_added = 0;

        for (_key, value) in special_tokens_dict {
            let already_exists = self.vocab.get_id(value).is_some();

            let id = self.vocab.get_id(value).unwrap_or_else(|| {
                let new_id = self.vocab_size() as u32;
                num_added += 1;
                new_id
            });

            self.special_tokens.insert(value.clone(), id);
            self.added_tokens.insert(value.clone(), id);

            if !already_exists {
                // Token will be handled via added_tokens during encoding
            }
        }

        num_added
    }

    pub fn add_special_tokens_list(&mut self, tokens: &[String]) -> usize {
        let mut num_added = 0;

        for token in tokens {
            if self.vocab.get_id(token).is_none() && !self.added_tokens.contains_key(token) {
                let new_id = (self.vocab_size() + self.added_tokens.len()) as u32;
                self.special_tokens.insert(token.clone(), new_id);
                self.added_tokens.insert(token.clone(), new_id);
                num_added += 1;
            }
        }

        num_added
    }

    // =========================================================================
    // Training
    // =========================================================================

    /// Train a new tokenizer from texts using the same configuration
    /// This keeps normalizer, pre_tokenizer, post_processor, decoder, and special tokens
    /// but trains a new vocabulary from the provided texts
    pub fn train_new_from_iterator<'a, I>(&self, texts: I, vocab_size: usize) -> io::Result<Self>
    where
        I: Iterator<Item = &'a str>,
    {
        use crate::bpe_trainer::{BpeTrainer, BpeTrainerConfig};

        // Collect special tokens
        let special_tokens: Vec<String> = self.all_special_tokens();

        // Configure trainer
        let config = BpeTrainerConfig {
            vocab_size,
            min_frequency: 2,
            special_tokens: special_tokens.clone(),
            show_progress: true,
            end_of_word_suffix: None,
            continuing_subword_prefix: None,
            ..Default::default()
        };

        let trainer = BpeTrainer::new(config);

        // Collect texts
        let texts_vec: Vec<&str> = texts.collect();

        // Pre-tokenize texts if we have a pre-tokenizer
        let processed_texts: Vec<String> = if self.pre_tokenizer.is_some() {
            texts_vec.iter()
                .flat_map(|text| {
                    let normalized = match &self.normalizer {
                        Some(n) => n.normalize(text),
                        None => text.to_string(),
                    };
                    match &self.pre_tokenizer {
                        Some(pt) => pt.pre_tokenize(&normalized),
                        None => vec![normalized],
                    }
                })
                .collect()
        } else {
            texts_vec.iter().map(|s| s.to_string()).collect()
        };

        let refs: Vec<&str> = processed_texts.iter().map(|s| s.as_str()).collect();
        let (vocab, merges) = trainer.train(&refs);

        // Create new tokenizer with trained vocab
        let bpe = crate::bpe::BpeTokenizer::new(vocab.clone(), merges);

        // Rebuild special tokens map
        let mut special_token_map = HashMap::new();
        let mut added_tokens_map = HashMap::new();
        let mut added_tokens_config = HashMap::new();

        for token in &special_tokens {
            if let Some(&id) = vocab.get(token) {
                special_token_map.insert(token.clone(), id);
                added_tokens_map.insert(token.clone(), id);
                added_tokens_config.insert(token.clone(), AddedTokenInternal {
                    id,
                    special: true,
                    single_word: false,
                    lstrip: false,
                    rstrip: false,
                    normalized: false,
                });
            }
        }

        // Copy special tokens config from original
        let vocab_obj = crate::vocab::Vocab::new(vocab, self.vocab.special_tokens().clone());

        Ok(Self {
            bpe,
            vocab: vocab_obj,
            special_tokens: special_token_map,
            added_tokens: added_tokens_map,
            added_tokens_config,
            normalizer: self.normalizer.clone(),
            pre_tokenizer: self.pre_tokenizer.clone(),
            post_processor: self.post_processor.clone(),
            decoder: self.decoder.clone(),
            model_max_length: self.model_max_length,
            padding_side: self.padding_side.clone(),
            truncation_side: self.truncation_side.clone(),
            chat_template: self.chat_template.clone(),
            padding_config: self.padding_config.clone(),
            truncation_config: self.truncation_config.clone(),
        })
    }

    // =========================================================================
    // Post-Processing
    // =========================================================================

    /// Apply post-processing to an encoding
    /// This is a standalone method for applying post-processing separately
    pub fn post_process(&self, encoding: Encoding, pair_encoding: Option<Encoding>) -> Encoding {
        let mut result = encoding;

        // Merge pair encoding if provided
        if let Some(pair) = pair_encoding {
            result.merge(pair, 1);
        }

        // Apply post-processor
        if let Some(ref pp) = self.post_processor {
            let original_len = result.ids.len();
            let processed_ids = pp.process(result.ids.clone(), None);
            let added_count = processed_ids.len() - original_len;

            result.ids = processed_ids;
            result.attention_mask.extend(std::iter::repeat(1).take(added_count));
            result.special_tokens_mask.extend(std::iter::repeat(1).take(added_count));
            result.type_ids.extend(std::iter::repeat(0).take(added_count));
            result.offsets.extend(std::iter::repeat((0, 0)).take(added_count));
            result.word_ids.extend(std::iter::repeat(None).take(added_count));
            result.sequence_ids.extend(std::iter::repeat(None).take(added_count));

            // Mark special tokens
            let special_ids: Vec<u32> = self.special_tokens.values().copied().collect();
            result.mark_special_tokens(&special_ids);
        }

        result
    }

    // =========================================================================
    // Chat Template
    // =========================================================================

    pub fn apply_chat_template(
        &self,
        messages: &[std::collections::HashMap<String, String>],
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> Result<ChatTemplateResult, String> {
        let template = self.chat_template.as_ref()
            .ok_or_else(|| "No chat template set for this tokenizer".to_string())?;

        let bos_token = self.vocab.special_tokens().bos_token.as_deref().unwrap_or("<s>");
        let eos_token = self.vocab.special_tokens().eos_token.as_deref().unwrap_or("</s>");

        let result = chat::apply_chat_template(template, messages, add_generation_prompt, bos_token, eos_token);

        if tokenize {
            let ids = self.encode(&result);
            Ok(ChatTemplateResult::Tokenized(ids))
        } else {
            Ok(ChatTemplateResult::Text(result))
        }
    }

    // =========================================================================
    // Prepare for Model
    // =========================================================================

    pub fn prepare_for_model(
        &self,
        ids: Vec<u32>,
        pair_ids: Option<Vec<u32>>,
        add_special_tokens: bool,
        padding: Option<&str>,
        truncation: bool,
        max_length: Option<usize>,
        stride: usize,
        return_attention_mask: bool,
    ) -> Encoding {
        let mut encoding = if let Some(pair) = pair_ids {
            let text_encoding = Encoding::from_ids(
                ids.clone(),
                ids.iter().filter_map(|&id| self.vocab.get_token(id).map(|s| s.to_string())).collect(),
            );
            let pair_encoding = Encoding::from_ids(
                pair.clone(),
                pair.iter().filter_map(|&id| self.vocab.get_token(id).map(|s| s.to_string())).collect(),
            );
            let mut merged = text_encoding;
            merged.merge(pair_encoding, 1);
            merged
        } else {
            Encoding::from_ids(
                ids.clone(),
                ids.iter().filter_map(|&id| self.vocab.get_token(id).map(|s| s.to_string())).collect(),
            )
        };

        if add_special_tokens {
            if let Some(ref pp) = self.post_processor {
                let processed_ids = pp.process(encoding.ids.clone(), None);
                let added_count = processed_ids.len() - encoding.ids.len();
                encoding.ids = processed_ids;
                encoding.attention_mask.extend(std::iter::repeat(1).take(added_count));
                encoding.special_tokens_mask.extend(std::iter::repeat(1).take(added_count));
                encoding.type_ids.extend(std::iter::repeat(0).take(added_count));
            }
        }

        let max_len = max_length.unwrap_or(self.model_max_length);
        if truncation && encoding.len() > max_len {
            if stride > 0 {
                encoding.truncate_with_stride(max_len, stride);
            } else {
                encoding.truncate(max_len);
            }
        }

        if let Some(padding_strategy) = padding {
            let pad_id = self.special_tokens.get("[PAD]")
                .or_else(|| self.special_tokens.get("<pad>"))
                .copied()
                .unwrap_or(0);
            let pad_token = self.vocab.get_token(pad_id).unwrap_or("<pad>");
            let pad_left = padding_strategy == "left" || self.padding_side == "left";

            match padding_strategy {
                "max_length" => {
                    encoding.pad(max_len, pad_id, pad_token, pad_left);
                }
                "longest" | "left" | "right" => {
                    encoding.pad(max_len, pad_id, pad_token, pad_left);
                }
                _ => {}
            }
        }

        if return_attention_mask {
            // Already set
        }

        encoding
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let vocab = self.bpe.vocab().clone();

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
            .map(|(content, &id)| {
                // Get the config if available
                let config = self.added_tokens_config.get(content);
                AddedToken {
                    id,
                    content: content.clone(),
                    special: config.map_or(self.special_tokens.contains_key(content), |c| c.special),
                    single_word: config.map_or(false, |c| c.single_word),
                    lstrip: config.map_or(false, |c| c.lstrip),
                    rstrip: config.map_or(false, |c| c.rstrip),
                    normalized: config.map_or(false, |c| c.normalized),
                }
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
            normalizer: self.normalizer.as_ref().map(serialization::serialize_normalizer),
            pre_tokenizer: self.pre_tokenizer.as_ref().map(serialization::serialize_pre_tokenizer),
            post_processor: self.post_processor.as_ref().map(|pp| serialization::serialize_post_processor(pp, &self.special_tokens)),
            decoder: self.decoder.as_ref().map(serialization::serialize_decoder),
        };

        let json = serde_json::to_string_pretty(&tokenizer_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        fs::write(path, json)
    }

    pub fn save_pretrained<P: AsRef<Path>>(&self, dir: P) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        self.save(dir.join("tokenizer.json"))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_tokenizer_json() {
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
