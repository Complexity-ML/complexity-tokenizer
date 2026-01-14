//! Python bindings for Model classes

use crate::models::{WordPieceModel, UnigramModel, WordLevelModel, CharBpeModel, ByteLevelBpeModel};
use pyo3::prelude::*;
use std::collections::HashMap as StdHashMap;

/// Python-exposed WordPiece model class
#[pyclass(name = "WordPieceModel")]
pub struct PyWordPieceModel {
    pub(crate) inner: WordPieceModel,
}

#[pymethods]
impl PyWordPieceModel {
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

    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}

/// Python-exposed Unigram model class
#[pyclass(name = "UnigramModel")]
pub struct PyUnigramModel {
    pub(crate) inner: UnigramModel,
}

#[pymethods]
impl PyUnigramModel {
    #[new]
    #[pyo3(signature = (vocab, unk_token = "<unk>".to_string()))]
    fn new(vocab: Vec<(String, f64)>, unk_token: String) -> Self {
        Self {
            inner: UnigramModel::new(vocab, unk_token),
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}

/// Python-exposed WordLevel model class
#[pyclass(name = "WordLevelModel")]
pub struct PyWordLevelModel {
    inner: WordLevelModel,
}

#[pymethods]
impl PyWordLevelModel {
    #[new]
    #[pyo3(signature = (vocab, unk_token = "<unk>".to_string()))]
    fn new(vocab: StdHashMap<String, u32>, unk_token: String) -> Self {
        let vocab_hash: hashbrown::HashMap<String, u32> = vocab.into_iter().collect();
        Self {
            inner: WordLevelModel::new(vocab_hash, unk_token),
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}

/// Python-exposed CharBPE model class
#[pyclass(name = "CharBpeModel")]
pub struct PyCharBpeModel {
    inner: CharBpeModel,
}

#[pymethods]
impl PyCharBpeModel {
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

    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}

/// Python-exposed ByteLevel BPE model class (GPT-2/GPT-3/LLaMA style)
#[pyclass(name = "ByteLevelBpeModel")]
pub struct PyByteLevelBpeModel {
    inner: ByteLevelBpeModel,
}

#[pymethods]
impl PyByteLevelBpeModel {
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

    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(|s| s.to_string())
    }
}
