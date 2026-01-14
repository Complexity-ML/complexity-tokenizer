//! Python bindings for Trainer classes

use crate::trainer::{InlBpeTrainer, TrainerConfig};
use crate::trainers::{WordPieceTrainer, WordPieceTrainerConfig, UnigramTrainer, UnigramTrainerConfig};
use crate::bpe_trainer::{BpeTrainer, BpeTrainerConfig};
use super::models::{PyWordPieceModel, PyUnigramModel};
use pyo3::prelude::*;
use std::collections::HashMap as StdHashMap;

/// Python-exposed Trainer class for INL-BPE training
#[pyclass(name = "Trainer")]
pub struct PyTrainer {
    inner: InlBpeTrainer,
}

#[pymethods]
impl PyTrainer {
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

    fn train(&mut self, files: Vec<String>) -> PyResult<()> {
        let paths: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
        self.inner.train(&paths)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    fn train_from_iterator(&mut self, texts: Vec<String>) -> PyResult<()> {
        self.inner.train_from_texts(texts.iter().map(|s| s.as_str()));
        Ok(())
    }

    fn count_batch(&mut self, texts: Vec<String>) -> PyResult<()> {
        self.inner.count_batch(texts.iter().map(|s| s.as_str()));
        Ok(())
    }

    fn finish_training(&mut self) -> PyResult<()> {
        self.inner.finish_training();
        Ok(())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab().len()
    }

    #[getter]
    fn num_merges(&self) -> usize {
        self.inner.merges().len()
    }
}

/// Python-exposed WordPiece trainer class
#[pyclass(name = "WordPieceTrainer")]
pub struct PyWordPieceTrainer {
    inner: WordPieceTrainer,
}

#[pymethods]
impl PyWordPieceTrainer {
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

    fn train(&mut self, files: Vec<String>) -> PyResult<PyWordPieceModel> {
        let paths: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
        let model = self.inner.train(&paths)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyWordPieceModel { inner: model })
    }

    fn train_from_iterator(&mut self, texts: Vec<String>) -> PyResult<PyWordPieceModel> {
        let model = self.inner.train_from_texts(texts.iter().map(|s| s.as_str()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyWordPieceModel { inner: model })
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab().len()
    }
}

/// Python-exposed Unigram trainer class
#[pyclass(name = "UnigramTrainer")]
pub struct PyUnigramTrainer {
    inner: UnigramTrainer,
}

#[pymethods]
impl PyUnigramTrainer {
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

    fn train(&mut self, files: Vec<String>) -> PyResult<PyUnigramModel> {
        let paths: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
        let model = self.inner.train(&paths)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyUnigramModel { inner: model })
    }

    fn train_from_iterator(&mut self, texts: Vec<String>) -> PyResult<PyUnigramModel> {
        let model = self.inner.train_from_texts(texts.iter().map(|s| s.as_str()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(PyUnigramModel { inner: model })
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab().len()
    }
}

/// Python-exposed BPE trainer class
#[pyclass(name = "BpeTrainer")]
pub struct PyBpeTrainer {
    inner: BpeTrainer,
}

#[pymethods]
impl PyBpeTrainer {
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

    fn train(&self, texts: Vec<String>) -> PyResult<(StdHashMap<String, u32>, Vec<(String, String)>)> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let (vocab, merges) = self.inner.train(&refs);

        let std_vocab: StdHashMap<String, u32> = vocab.into_iter().collect();

        Ok((std_vocab, merges))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.config().vocab_size
    }

    #[getter]
    fn min_frequency(&self) -> u32 {
        self.inner.config().min_frequency
    }
}
