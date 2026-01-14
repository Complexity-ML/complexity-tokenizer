//! Complexity Tokenizer - Fast BPE tokenizer with HuggingFace compatibility
//!
//! A high-performance Byte Pair Encoding (BPE) tokenizer written in Rust
//! with Python bindings via PyO3.
//!
//! Features:
//! - Fast inference (encode/decode)
//! - INL-BPE training with dynamics-based merge selection
//! - HuggingFace tokenizer.json compatibility

mod bpe;
mod vocab;
mod huggingface;
mod trainer;

pub use bpe::BpeTokenizer;
pub use vocab::Vocab;
pub use huggingface::HuggingFaceTokenizer;
pub use trainer::{InlBpeTrainer, TrainerConfig};

use pyo3::prelude::*;
use std::collections::HashMap as StdHashMap;

/// Python module
#[pymodule]
fn complexity_tokenizer(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyTrainer>()?;
    m.add("__version__", "0.1.9")?;
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
    fn encode_batch(&self, texts: Vec<&str>) -> PyResult<Vec<Vec<u32>>> {
        Ok(self.inner.encode_batch(&texts))
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
