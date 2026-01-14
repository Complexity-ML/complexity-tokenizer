//! Complexity Tokenizer - Fast BPE tokenizer with HuggingFace compatibility
//!
//! A high-performance Byte Pair Encoding (BPE) tokenizer written in Rust
//! with Python bindings via PyO3.
//!
//! Features:
//! - Fast inference (encode/decode)
//! - INL-BPE training with dynamics-based merge selection
//! - HuggingFace tokenizer.json compatibility
//! - Full encoding with attention masks, token type IDs, offsets
//! - Normalizers, pre-tokenizers, post-processors, decoders
//! - Multiple model types: BPE, WordPiece, Unigram, WordLevel

mod bpe;
mod vocab;
mod huggingface;
mod trainer;
mod trainers;
mod normalizers;
mod pretokenizers;
mod postprocessors;
mod decoders;
mod encoding;
mod models;
mod hub;
mod bpe_trainer;

pub use bpe::BpeTokenizer;
pub use vocab::Vocab;
pub use huggingface::HuggingFaceTokenizer;
pub use trainer::{InlBpeTrainer, TrainerConfig};
pub use trainers::{WordPieceTrainer, WordPieceTrainerConfig, UnigramTrainer, UnigramTrainerConfig};
pub use normalizers::Normalizer;
pub use pretokenizers::PreTokenizer;
pub use postprocessors::{PostProcessor, TruncationStrategy, PaddingStrategy};
pub use decoders::Decoder;
pub use encoding::{Encoding, AddedToken};
pub use models::{WordPieceModel, UnigramModel, WordLevelModel, CharBpeModel, ByteLevelBpeModel, Model};
pub use hub::{HubConfig, download_tokenizer, resolve_model_path};
pub use bpe_trainer::{BpeTrainer, BpeTrainerConfig, BpeTrainerBuilder};

use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashMap as StdHashMap;

/// Python module
#[pymodule]
fn complexity_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyTrainer>()?;
    m.add_class::<PyWordPieceTrainer>()?;
    m.add_class::<PyUnigramTrainer>()?;
    m.add_class::<PyBpeTrainer>()?;
    m.add_class::<PyEncoding>()?;
    m.add_class::<PyNormalizer>()?;
    m.add_class::<PyPreTokenizer>()?;
    m.add_class::<PyPostProcessor>()?;
    m.add_class::<PyDecoder>()?;
    m.add_class::<PyWordPieceModel>()?;
    m.add_class::<PyUnigramModel>()?;
    m.add_class::<PyWordLevelModel>()?;
    m.add_class::<PyCharBpeModel>()?;
    m.add_class::<PyByteLevelBpeModel>()?;
    m.add("__version__", "0.2.8")?;
    Ok(())
}

/// Python-exposed Tokenizer class
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: HuggingFaceTokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load tokenizer from HuggingFace tokenizer.json file
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = HuggingFaceTokenizer::from_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Load tokenizer from HuggingFace Hub
    #[staticmethod]
    fn from_pretrained(repo_id: &str) -> PyResult<Self> {
        let inner = HuggingFaceTokenizer::from_pretrained(repo_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        Ok(self.inner.encode(text))
    }

    /// Encode batch of texts (parallel)
    fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        Ok(self.inner.encode_batch(&refs))
    }

    /// Decode token IDs to text
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        Ok(self.inner.decode(&ids))
    }

    /// Decode with options (skip_special_tokens, clean_up_tokenization_spaces)
    #[pyo3(signature = (ids, skip_special_tokens = false, clean_up_tokenization_spaces = true))]
    fn decode_with_options(
        &self,
        ids: Vec<u32>,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> PyResult<String> {
        Ok(self.inner.decode_with_options(&ids, skip_special_tokens, clean_up_tokenization_spaces))
    }

    /// Decode batch of token IDs (parallel)
    fn decode_batch(&self, batch: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        Ok(self.inner.decode_batch(&batch))
    }

    /// Decode batch with options (parallel)
    #[pyo3(signature = (batch, skip_special_tokens = false, clean_up_tokenization_spaces = true))]
    fn decode_batch_with_options(
        &self,
        batch: Vec<Vec<u32>>,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> PyResult<Vec<String>> {
        Ok(self.inner.decode_batch_with_options(&batch, skip_special_tokens, clean_up_tokenization_spaces))
    }

    /// Convert tokens to string
    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        self.inner.convert_tokens_to_string(&tokens)
    }

    /// Get special tokens mask
    #[pyo3(signature = (ids, already_has_special_tokens = true))]
    fn get_special_tokens_mask(&self, ids: Vec<u32>, already_has_special_tokens: bool) -> Vec<u32> {
        self.inner.get_special_tokens_mask(&ids, already_has_special_tokens)
    }

    /// Get number of special tokens that would be added
    #[pyo3(signature = (is_pair = false))]
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        self.inner.num_special_tokens_to_add(is_pair)
    }

    /// Check if this is a fast tokenizer (always true)
    #[getter]
    fn is_fast(&self) -> bool {
        self.inner.is_fast()
    }

    /// Alias for encode_to_encoding (HuggingFace compatibility)
    fn encode_plus(&self, text: &str) -> PyEncoding {
        self.encode_to_encoding(text)
    }

    /// Batch encode_plus (HuggingFace compatibility)
    fn batch_encode_plus(&self, texts: Vec<String>) -> Vec<PyEncoding> {
        self.encode_batch_to_encoding(texts)
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Get token ID for a token string
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Get token string for a token ID
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Get special tokens
    #[getter]
    fn special_tokens(&self) -> PyResult<StdHashMap<String, u32>> {
        // Convert hashbrown::HashMap to std::collections::HashMap
        Ok(self.inner.special_tokens().iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect())
    }

    /// Save tokenizer to file (HuggingFace format)
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    /// Save tokenizer to directory (HuggingFace pretrained format)
    fn save_pretrained(&self, dir: &str) -> PyResult<()> {
        self.inner.save_pretrained(dir)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    /// Encode text to full Encoding (with attention mask, type ids, etc.)
    fn encode_to_encoding(&self, text: &str) -> PyEncoding {
        PyEncoding {
            inner: self.inner.encode_to_encoding(text),
        }
    }

    /// Encode text pair to full Encoding (for NLI, QA, etc.)
    fn encode_pair_to_encoding(&self, text: &str, text_pair: &str) -> PyEncoding {
        PyEncoding {
            inner: self.inner.encode_pair_to_encoding(text, text_pair),
        }
    }

    /// Encode with truncation and stride for long texts
    #[pyo3(signature = (text, text_pair = None, max_length = 512, stride = 0))]
    fn encode_with_truncation(
        &self,
        text: &str,
        text_pair: Option<&str>,
        max_length: usize,
        stride: usize,
    ) -> PyEncoding {
        PyEncoding {
            inner: self.inner.encode_to_encoding_with_truncation(text, text_pair, max_length, stride),
        }
    }

    /// Encode batch of texts to full Encodings (parallel)
    fn encode_batch_to_encoding(&self, texts: Vec<String>) -> Vec<PyEncoding> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner.encode_batch_to_encoding(&refs)
            .into_iter()
            .map(|e| PyEncoding { inner: e })
            .collect()
    }

    /// Encode batch of text pairs to full Encodings (parallel)
    fn encode_batch_pairs_to_encoding(&self, pairs: Vec<(String, String)>) -> Vec<PyEncoding> {
        let refs: Vec<(&str, &str)> = pairs.iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        self.inner.encode_batch_pairs_to_encoding(&refs)
            .into_iter()
            .map(|e| PyEncoding { inner: e })
            .collect()
    }

    /// Encode batch with automatic padding to longest sequence
    #[pyo3(signature = (texts, max_length = None, pad_left = false))]
    fn encode_batch_with_padding(
        &self,
        texts: Vec<String>,
        max_length: Option<usize>,
        pad_left: bool,
    ) -> Vec<PyEncoding> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner.encode_batch_with_padding(&refs, max_length, pad_left)
            .into_iter()
            .map(|e| PyEncoding { inner: e })
            .collect()
    }

    /// Encode batch of pairs with automatic padding
    #[pyo3(signature = (pairs, max_length = None, pad_left = false))]
    fn encode_batch_pairs_with_padding(
        &self,
        pairs: Vec<(String, String)>,
        max_length: Option<usize>,
        pad_left: bool,
    ) -> Vec<PyEncoding> {
        let refs: Vec<(&str, &str)> = pairs.iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        self.inner.encode_batch_pairs_with_padding(&refs, max_length, pad_left)
            .into_iter()
            .map(|e| PyEncoding { inner: e })
            .collect()
    }

    /// Add a token dynamically
    fn add_token(&mut self, content: &str, id: u32, special: bool) {
        self.inner.add_token(content, id, special);
    }

    /// Add multiple tokens dynamically
    fn add_tokens(&mut self, tokens: Vec<(String, u32, bool)>) {
        self.inner.add_tokens(tokens);
    }

    /// Set normalizer
    fn set_normalizer(&mut self, normalizer: &PyNormalizer) {
        self.inner.set_normalizer(normalizer.inner.clone());
    }

    /// Set pre-tokenizer
    fn set_pre_tokenizer(&mut self, pre_tokenizer: &PyPreTokenizer) {
        self.inner.set_pre_tokenizer(pre_tokenizer.inner.clone());
    }

    /// Set post-processor
    fn set_post_processor(&mut self, post_processor: &PyPostProcessor) {
        self.inner.set_post_processor(post_processor.inner.clone());
    }

    /// Set decoder
    fn set_decoder(&mut self, decoder: &PyDecoder) {
        self.inner.set_decoder(decoder.inner.clone());
    }
}

/// Python-exposed Trainer class for INL-BPE training
#[pyclass(name = "Trainer")]
pub struct PyTrainer {
    inner: InlBpeTrainer,
}

#[pymethods]
impl PyTrainer {
    /// Create a new trainer with configuration
    #[new]
    #[pyo3(signature = (
        vocab_size = 32000,
        min_frequency = 2,
        special_tokens = None,
        min_word_length = 1,
        inl_alpha = 0.9,
        inl_beta = 0.3,
        inl_gate = 0.5
    ))]
    fn new(
        vocab_size: usize,
        min_frequency: u32,
        special_tokens: Option<Vec<String>>,
        min_word_length: usize,
        inl_alpha: f32,
        inl_beta: f32,
        inl_gate: f32,
    ) -> Self {
        let config = TrainerConfig {
            vocab_size,
            min_frequency,
            special_tokens: special_tokens.unwrap_or_else(|| vec![
                "</s>".to_string(),
                "<pad>".to_string(),
                "<s>".to_string(),
                "<unk>".to_string(),
            ]),
            min_word_length,
            inl_alpha,
            inl_beta,
            inl_gate,
            ..Default::default()
        };
        Self {
            inner: InlBpeTrainer::new(config),
        }
    }

    /// Train tokenizer from text files
    fn train(&mut self, files: Vec<String>) -> PyResult<()> {
        let paths: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
        self.inner.train(&paths)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    /// Train tokenizer from an iterator of text strings (for streaming datasets)
    fn train_from_iterator(&mut self, texts: Vec<String>) -> PyResult<()> {
        self.inner.train_from_texts(texts.iter().map(|s| s.as_str()));
        Ok(())
    }

    /// Count words from a batch (streaming - low memory)
    /// Call this multiple times with batches, then call finish_training()
    fn count_batch(&mut self, texts: Vec<String>) -> PyResult<()> {
        self.inner.count_batch(texts.iter().map(|s| s.as_str()));
        Ok(())
    }

    /// Finish training after counting all batches
    fn finish_training(&mut self) -> PyResult<()> {
        self.inner.finish_training();
        Ok(())
    }

    /// Save trained tokenizer to file (HuggingFace format)
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab().len()
    }

    /// Get number of merges
    #[getter]
    fn num_merges(&self) -> usize {
        self.inner.merges().len()
    }
}

// =============================================================================
// Encoding Python Bindings
// =============================================================================

/// Python-exposed Encoding class
#[pyclass(name = "Encoding")]
pub struct PyEncoding {
    inner: Encoding,
}

#[pymethods]
impl PyEncoding {
    /// Create encoding from token IDs
    #[staticmethod]
    fn from_ids(ids: Vec<u32>, tokens: Vec<String>) -> Self {
        Self {
            inner: Encoding::from_ids(ids, tokens),
        }
    }

    /// Get token IDs
    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.inner.ids.clone()
    }

    /// Get tokens as strings
    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens.clone()
    }

    /// Get attention mask
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        self.inner.attention_mask.clone()
    }

    /// Get token type IDs
    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        self.inner.type_ids.clone()
    }

    /// Get special tokens mask
    #[getter]
    fn special_tokens_mask(&self) -> Vec<u32> {
        self.inner.special_tokens_mask.clone()
    }

    /// Get character offsets
    #[getter]
    fn offsets(&self) -> Vec<(usize, usize)> {
        self.inner.offsets.clone()
    }

    /// Get length
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get word IDs (which word each token belongs to)
    #[getter]
    fn word_ids(&self) -> Vec<Option<usize>> {
        self.inner.word_ids.clone()
    }

    /// Get number of overflowing encodings
    #[getter]
    fn n_overflowing(&self) -> usize {
        self.inner.n_overflowing()
    }

    /// Get overflowing encodings
    #[getter]
    fn overflowing(&self) -> Vec<PyEncoding> {
        self.inner.overflowing()
            .iter()
            .map(|e| PyEncoding { inner: e.clone() })
            .collect()
    }

    /// Pad encoding
    fn pad(&mut self, target_length: usize, pad_id: u32, pad_token: &str, pad_left: bool) {
        self.inner.pad(target_length, pad_id, pad_token, pad_left);
    }

    /// Truncate encoding
    fn truncate(&mut self, max_length: usize) {
        self.inner.truncate(max_length);
    }

    /// Truncate with stride (for long documents with overlap)
    fn truncate_with_stride(&mut self, max_length: usize, stride: usize) {
        self.inner.truncate_with_stride(max_length, stride);
    }

    /// Get sequence IDs (None for special tokens, 0 for first seq, 1 for second)
    #[getter]
    fn sequence_ids(&self) -> Vec<Option<usize>> {
        self.inner.sequence_ids.clone()
    }

    /// Get the token index for a character position
    fn char_to_token(&self, char_pos: usize) -> Option<usize> {
        self.inner.char_to_token(char_pos)
    }

    /// Get the token index for a character position within a specific sequence
    fn char_to_token_with_sequence(&self, char_pos: usize, sequence_id: usize) -> Option<usize> {
        self.inner.char_to_token_with_sequence(char_pos, sequence_id)
    }

    /// Get the character span for a token index (start, end)
    fn token_to_chars(&self, token_idx: usize) -> Option<(usize, usize)> {
        self.inner.token_to_chars(token_idx)
    }

    /// Get the word index for a token
    fn token_to_word(&self, token_idx: usize) -> Option<usize> {
        self.inner.token_to_word(token_idx)
    }

    /// Get the sequence ID for a token
    fn token_to_sequence(&self, token_idx: usize) -> Option<usize> {
        self.inner.token_to_sequence(token_idx)
    }

    /// Get token IDs as numpy array
    fn ids_as_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u32>> {
        numpy::PyArray1::from_vec(py, self.inner.ids.clone())
    }

    /// Get attention mask as numpy array
    fn attention_mask_as_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u32>> {
        numpy::PyArray1::from_vec(py, self.inner.attention_mask.clone())
    }

    /// Get token type IDs as numpy array
    fn type_ids_as_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u32>> {
        numpy::PyArray1::from_vec(py, self.inner.type_ids.clone())
    }

    /// Get special tokens mask as numpy array
    fn special_tokens_mask_as_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u32>> {
        numpy::PyArray1::from_vec(py, self.inner.special_tokens_mask.clone())
    }
}

// =============================================================================
// Normalizer Python Bindings
// =============================================================================

/// Python-exposed Normalizer class
#[pyclass(name = "Normalizer")]
pub struct PyNormalizer {
    inner: Normalizer,
}

#[pymethods]
impl PyNormalizer {
    /// Create NFC normalizer
    #[staticmethod]
    fn nfc() -> Self {
        Self { inner: Normalizer::NFC }
    }

    /// Create NFD normalizer
    #[staticmethod]
    fn nfd() -> Self {
        Self { inner: Normalizer::NFD }
    }

    /// Create NFKC normalizer
    #[staticmethod]
    fn nfkc() -> Self {
        Self { inner: Normalizer::NFKC }
    }

    /// Create NFKD normalizer
    #[staticmethod]
    fn nfkd() -> Self {
        Self { inner: Normalizer::NFKD }
    }

    /// Create lowercase normalizer
    #[staticmethod]
    fn lowercase() -> Self {
        Self { inner: Normalizer::Lowercase }
    }

    /// Create strip normalizer
    #[staticmethod]
    fn strip() -> Self {
        Self { inner: Normalizer::Strip }
    }

    /// Create strip accents normalizer
    #[staticmethod]
    fn strip_accents() -> Self {
        Self { inner: Normalizer::StripAccents }
    }

    /// Create replace normalizer
    #[staticmethod]
    fn replace(pattern: String, replacement: String) -> Self {
        Self {
            inner: Normalizer::Replace { pattern, replacement },
        }
    }

    /// Normalize text
    fn normalize(&self, text: &str) -> String {
        self.inner.normalize(text)
    }
}

// =============================================================================
// PreTokenizer Python Bindings
// =============================================================================

/// Python-exposed PreTokenizer class
#[pyclass(name = "PreTokenizer")]
pub struct PyPreTokenizer {
    inner: PreTokenizer,
}

#[pymethods]
impl PyPreTokenizer {
    /// Create whitespace pre-tokenizer
    #[staticmethod]
    fn whitespace() -> Self {
        Self { inner: PreTokenizer::Whitespace }
    }

    /// Create ByteLevel pre-tokenizer (GPT-2 style)
    #[staticmethod]
    #[pyo3(signature = (add_prefix_space = false))]
    fn byte_level(add_prefix_space: bool) -> Self {
        Self {
            inner: PreTokenizer::ByteLevel { add_prefix_space },
        }
    }

    /// Create Metaspace pre-tokenizer (SentencePiece style)
    #[staticmethod]
    #[pyo3(signature = (replacement = '▁', add_prefix_space = true))]
    fn metaspace(replacement: char, add_prefix_space: bool) -> Self {
        Self {
            inner: PreTokenizer::Metaspace { replacement, add_prefix_space },
        }
    }

    /// Create punctuation pre-tokenizer
    #[staticmethod]
    fn punctuation() -> Self {
        Self { inner: PreTokenizer::Punctuation }
    }

    /// Create digits pre-tokenizer
    #[staticmethod]
    #[pyo3(signature = (individual_digits = false))]
    fn digits(individual_digits: bool) -> Self {
        Self {
            inner: PreTokenizer::Digits { individual_digits },
        }
    }

    /// Create GPT-2 pre-tokenizer
    #[staticmethod]
    fn gpt2() -> Self {
        Self { inner: PreTokenizer::GPT2 }
    }

    /// Pre-tokenize text
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        self.inner.pre_tokenize(text)
    }
}

// =============================================================================
// PostProcessor Python Bindings
// =============================================================================

/// Python-exposed PostProcessor class
#[pyclass(name = "PostProcessor")]
pub struct PyPostProcessor {
    inner: PostProcessor,
}

#[pymethods]
impl PyPostProcessor {
    /// Create BERT post-processor
    #[staticmethod]
    fn bert(cls_token: String, cls_id: u32, sep_token: String, sep_id: u32) -> Self {
        Self {
            inner: PostProcessor::BertProcessing {
                cls: (cls_token, cls_id),
                sep: (sep_token, sep_id),
            },
        }
    }

    /// Create RoBERTa post-processor
    #[staticmethod]
    #[pyo3(signature = (bos_token, bos_id, eos_token, eos_id, add_prefix_space = false))]
    fn roberta(bos_token: String, bos_id: u32, eos_token: String, eos_id: u32, add_prefix_space: bool) -> Self {
        Self {
            inner: PostProcessor::RobertaProcessing {
                bos: (bos_token, bos_id),
                eos: (eos_token, eos_id),
                add_prefix_space,
            },
        }
    }

    /// Create template post-processor
    #[staticmethod]
    #[pyo3(signature = (single, pair = None, special_tokens = vec![]))]
    fn template(single: String, pair: Option<String>, special_tokens: Vec<(String, u32)>) -> Self {
        Self {
            inner: PostProcessor::TemplateProcessing {
                single,
                pair,
                special_tokens,
            },
        }
    }

    /// Process token IDs
    #[pyo3(signature = (ids, pair_ids = None))]
    fn process(&self, ids: Vec<u32>, pair_ids: Option<Vec<u32>>) -> Vec<u32> {
        self.inner.process(ids, pair_ids)
    }

    /// Get number of added tokens for single sequence
    fn added_tokens_single(&self) -> usize {
        self.inner.added_tokens_single()
    }

    /// Get number of added tokens for pair sequence
    fn added_tokens_pair(&self) -> usize {
        self.inner.added_tokens_pair()
    }
}

// =============================================================================
// Decoder Python Bindings
// =============================================================================

/// Python-exposed Decoder class
#[pyclass(name = "Decoder")]
pub struct PyDecoder {
    inner: Decoder,
}

#[pymethods]
impl PyDecoder {
    /// Create ByteLevel decoder (GPT-2 style)
    #[staticmethod]
    fn byte_level() -> Self {
        Self { inner: Decoder::ByteLevel }
    }

    /// Create Metaspace decoder (SentencePiece style)
    #[staticmethod]
    #[pyo3(signature = (replacement = '▁', add_prefix_space = true))]
    fn metaspace(replacement: char, add_prefix_space: bool) -> Self {
        Self {
            inner: Decoder::Metaspace { replacement, add_prefix_space },
        }
    }

    /// Create WordPiece decoder (BERT style)
    #[staticmethod]
    #[pyo3(signature = (prefix = "##".to_string(), cleanup = true))]
    fn wordpiece(prefix: String, cleanup: bool) -> Self {
        Self {
            inner: Decoder::WordPiece { prefix, cleanup },
        }
    }

    /// Create BPE decoder
    #[staticmethod]
    #[pyo3(signature = (suffix = "</w>".to_string()))]
    fn bpe(suffix: String) -> Self {
        Self {
            inner: Decoder::BPE { suffix },
        }
    }

    /// Decode tokens to text
    fn decode(&self, tokens: Vec<String>) -> String {
        self.inner.decode(&tokens)
    }
}

// =============================================================================
// WordPiece Model Python Bindings
// =============================================================================

/// Python-exposed WordPiece model class
#[pyclass(name = "WordPieceModel")]
pub struct PyWordPieceModel {
    inner: WordPieceModel,
}

#[pymethods]
impl PyWordPieceModel {
    /// Create new WordPiece model
    #[new]
    #[pyo3(signature = (vocab, prefix = "##".to_string(), unk_token = "[UNK]".to_string(), max_input_chars_per_word = 100))]
    fn new(
        vocab: StdHashMap<String, u32>,
        prefix: String,
        unk_token: String,
        max_input_chars_per_word: usize,
    ) -> Self {
        let vocab_hash: hashbrown::HashMap<String, u32> = vocab.into_iter().collect();
        Self {
            inner: WordPieceModel::new(vocab_hash, prefix, unk_token, max_input_chars_per_word),
        }
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Decode token IDs to text
    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Token to ID
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// ID to token
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}

// =============================================================================
// Unigram Model Python Bindings
// =============================================================================

/// Python-exposed Unigram model class
#[pyclass(name = "UnigramModel")]
pub struct PyUnigramModel {
    inner: UnigramModel,
}

#[pymethods]
impl PyUnigramModel {
    /// Create new Unigram model
    #[new]
    #[pyo3(signature = (vocab, unk_token = "<unk>".to_string()))]
    fn new(vocab: Vec<(String, f64)>, unk_token: String) -> Self {
        Self {
            inner: UnigramModel::new(vocab, unk_token),
        }
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Decode token IDs to text
    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Token to ID
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// ID to token
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}

// =============================================================================
// WordLevel Model Python Bindings
// =============================================================================

/// Python-exposed WordLevel model class
#[pyclass(name = "WordLevelModel")]
pub struct PyWordLevelModel {
    inner: WordLevelModel,
}

#[pymethods]
impl PyWordLevelModel {
    /// Create new WordLevel model
    #[new]
    #[pyo3(signature = (vocab, unk_token = "<unk>".to_string()))]
    fn new(vocab: StdHashMap<String, u32>, unk_token: String) -> Self {
        let vocab_hash: hashbrown::HashMap<String, u32> = vocab.into_iter().collect();
        Self {
            inner: WordLevelModel::new(vocab_hash, unk_token),
        }
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Decode token IDs to text
    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Token to ID
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// ID to token
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}

// =============================================================================
// CharBPE Model Python Bindings
// =============================================================================

/// Python-exposed CharBPE model class
#[pyclass(name = "CharBpeModel")]
pub struct PyCharBpeModel {
    inner: CharBpeModel,
}

#[pymethods]
impl PyCharBpeModel {
    /// Create new CharBPE model
    #[new]
    #[pyo3(signature = (vocab, merges, end_of_word_suffix = "</w>".to_string(), unk_token = "<unk>".to_string()))]
    fn new(
        vocab: StdHashMap<String, u32>,
        merges: Vec<(String, String)>,
        end_of_word_suffix: String,
        unk_token: String,
    ) -> Self {
        let vocab_hash: hashbrown::HashMap<String, u32> = vocab.into_iter().collect();
        Self {
            inner: CharBpeModel::new(vocab_hash, merges, end_of_word_suffix, unk_token),
        }
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Decode token IDs to text
    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Token to ID
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// ID to token
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}

// =============================================================================
// ByteLevel BPE Model Python Bindings
// =============================================================================

/// Python-exposed ByteLevel BPE model class (GPT-2/GPT-3/LLaMA style)
#[pyclass(name = "ByteLevelBpeModel")]
pub struct PyByteLevelBpeModel {
    inner: ByteLevelBpeModel,
}

#[pymethods]
impl PyByteLevelBpeModel {
    /// Create new ByteLevel BPE model
    #[new]
    #[pyo3(signature = (vocab, merges, unk_token = "<unk>".to_string(), add_prefix_space = true))]
    fn new(
        vocab: StdHashMap<String, u32>,
        merges: Vec<(String, String)>,
        unk_token: String,
        add_prefix_space: bool,
    ) -> Self {
        let vocab_hash: hashbrown::HashMap<String, u32> = vocab.into_iter().collect();
        Self {
            inner: ByteLevelBpeModel::new(vocab_hash, merges, unk_token, add_prefix_space),
        }
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Decode token IDs to text
    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Token to ID
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// ID to token
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}

// =============================================================================
// WordPiece Trainer Python Bindings
// =============================================================================

/// Python-exposed WordPiece trainer class
#[pyclass(name = "WordPieceTrainer")]
pub struct PyWordPieceTrainer {
    inner: WordPieceTrainer,
}

#[pymethods]
impl PyWordPieceTrainer {
    /// Create new WordPiece trainer
    #[new]
    #[pyo3(signature = (
        vocab_size = 30000,
        min_frequency = 2,
        special_tokens = None,
        continuing_subword_prefix = "##".to_string(),
        max_input_chars_per_word = 100
    ))]
    fn new(
        vocab_size: usize,
        min_frequency: u32,
        special_tokens: Option<Vec<String>>,
        continuing_subword_prefix: String,
        max_input_chars_per_word: usize,
    ) -> Self {
        let config = WordPieceTrainerConfig {
            vocab_size,
            min_frequency,
            special_tokens: special_tokens.unwrap_or_else(|| vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[MASK]".to_string(),
            ]),
            continuing_subword_prefix,
            max_input_chars_per_word,
            ..Default::default()
        };
        Self {
            inner: WordPieceTrainer::new(config),
        }
    }

    /// Train tokenizer from text files
    fn train(&mut self, files: Vec<String>) -> PyResult<PyWordPieceModel> {
        let paths: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
        let model = self.inner.train(&paths)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyWordPieceModel { inner: model })
    }

    /// Train tokenizer from an iterator of text strings
    fn train_from_iterator(&mut self, texts: Vec<String>) -> PyResult<PyWordPieceModel> {
        let model = self.inner.train_from_texts(texts.iter().map(|s| s.as_str()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyWordPieceModel { inner: model })
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab().len()
    }
}

// =============================================================================
// Unigram Trainer Python Bindings
// =============================================================================

/// Python-exposed Unigram trainer class
#[pyclass(name = "UnigramTrainer")]
pub struct PyUnigramTrainer {
    inner: UnigramTrainer,
}

#[pymethods]
impl PyUnigramTrainer {
    /// Create new Unigram trainer
    #[new]
    #[pyo3(signature = (
        vocab_size = 8000,
        special_tokens = None,
        initial_vocab_size = 1000000,
        shrinking_factor = 0.75,
        n_iterations = 16,
        max_piece_length = 16
    ))]
    fn new(
        vocab_size: usize,
        special_tokens: Option<Vec<String>>,
        initial_vocab_size: usize,
        shrinking_factor: f64,
        n_iterations: usize,
        max_piece_length: usize,
    ) -> Self {
        let config = UnigramTrainerConfig {
            vocab_size,
            special_tokens: special_tokens.unwrap_or_else(|| vec![
                "<unk>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
            ]),
            initial_vocab_size,
            shrinking_factor,
            n_iterations,
            max_piece_length,
            ..Default::default()
        };
        Self {
            inner: UnigramTrainer::new(config),
        }
    }

    /// Train tokenizer from text files
    fn train(&mut self, files: Vec<String>) -> PyResult<PyUnigramModel> {
        let paths: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
        let model = self.inner.train(&paths)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyUnigramModel { inner: model })
    }

    /// Train tokenizer from an iterator of text strings
    fn train_from_iterator(&mut self, texts: Vec<String>) -> PyResult<PyUnigramModel> {
        let model = self.inner.train_from_texts(texts.iter().map(|s| s.as_str()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyUnigramModel { inner: model })
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab().len()
    }
}

// =============================================================================
// BPE Trainer Python Bindings
// =============================================================================

/// Python-exposed BPE trainer class
#[pyclass(name = "BpeTrainer")]
pub struct PyBpeTrainer {
    inner: BpeTrainer,
}

#[pymethods]
impl PyBpeTrainer {
    /// Create new BPE trainer
    #[new]
    #[pyo3(signature = (
        vocab_size = 30000,
        min_frequency = 2,
        special_tokens = None,
        show_progress = true,
        end_of_word_suffix = None,
        continuing_subword_prefix = None
    ))]
    fn new(
        vocab_size: usize,
        min_frequency: u32,
        special_tokens: Option<Vec<String>>,
        show_progress: bool,
        end_of_word_suffix: Option<String>,
        continuing_subword_prefix: Option<String>,
    ) -> Self {
        let config = BpeTrainerConfig {
            vocab_size,
            min_frequency,
            special_tokens: special_tokens.unwrap_or_else(|| vec![
                "<unk>".to_string(),
                "<pad>".to_string(),
                "<s>".to_string(),
                "</s>".to_string(),
            ]),
            show_progress,
            end_of_word_suffix,
            continuing_subword_prefix,
            ..Default::default()
        };
        Self {
            inner: BpeTrainer::new(config),
        }
    }

    /// Train BPE tokenizer from text strings
    /// Returns (vocab, merges) tuple
    fn train(&self, texts: Vec<String>) -> PyResult<(StdHashMap<String, u32>, Vec<(String, String)>)> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let (vocab, merges) = self.inner.train(&refs);

        // Convert hashbrown HashMap to std HashMap
        let std_vocab: StdHashMap<String, u32> = vocab.into_iter().collect();

        Ok((std_vocab, merges))
    }

    /// Get target vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.config().vocab_size
    }

    /// Get minimum frequency
    #[getter]
    fn min_frequency(&self) -> u32 {
        self.inner.config().min_frequency
    }
}
