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
mod normalizers;
mod pretokenizers;
mod postprocessors;
mod decoders;
mod encoding;
mod models;

pub use bpe::BpeTokenizer;
pub use vocab::Vocab;
pub use huggingface::HuggingFaceTokenizer;
pub use trainer::{InlBpeTrainer, TrainerConfig};
pub use normalizers::Normalizer;
pub use pretokenizers::PreTokenizer;
pub use postprocessors::{PostProcessor, TruncationStrategy, PaddingStrategy};
pub use decoders::Decoder;
pub use encoding::{Encoding, AddedToken};
pub use models::{WordPieceModel, UnigramModel, WordLevelModel, Model};

use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashMap as StdHashMap;

/// Python module
#[pymodule]
fn complexity_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyTrainer>()?;
    m.add_class::<PyEncoding>()?;
    m.add_class::<PyNormalizer>()?;
    m.add_class::<PyPreTokenizer>()?;
    m.add_class::<PyPostProcessor>()?;
    m.add_class::<PyDecoder>()?;
    m.add_class::<PyWordPieceModel>()?;
    m.add_class::<PyUnigramModel>()?;
    m.add_class::<PyWordLevelModel>()?;
    m.add("__version__", "0.2.4")?;
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

    /// Decode batch of token IDs (parallel)
    fn decode_batch(&self, batch: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        Ok(self.inner.decode_batch(&batch))
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

    /// Encode text to full Encoding (with attention mask, type ids, etc.)
    fn encode_to_encoding(&self, text: &str) -> PyEncoding {
        PyEncoding {
            inner: self.inner.encode_to_encoding(text),
        }
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

    /// Pad encoding
    fn pad(&mut self, target_length: usize, pad_id: u32, pad_token: &str, pad_left: bool) {
        self.inner.pad(target_length, pad_id, pad_token, pad_left);
    }

    /// Truncate encoding
    fn truncate(&mut self, max_length: usize) {
        self.inner.truncate(max_length);
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
