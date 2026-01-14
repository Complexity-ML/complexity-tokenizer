//! Python bindings for Encoding classes

use crate::encoding::Encoding;
use pyo3::prelude::*;

/// Python-exposed Encoding class
#[pyclass(name = "Encoding")]
pub struct PyEncoding {
    pub(crate) inner: Encoding,
}

#[pymethods]
impl PyEncoding {
    #[staticmethod]
    fn from_ids(ids: Vec<u32>, tokens: Vec<String>) -> Self {
        Self {
            inner: Encoding::from_ids(ids, tokens),
        }
    }

    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.inner.ids.clone()
    }

    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens.clone()
    }

    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        self.inner.attention_mask.clone()
    }

    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        self.inner.type_ids.clone()
    }

    #[getter]
    fn special_tokens_mask(&self) -> Vec<u32> {
        self.inner.special_tokens_mask.clone()
    }

    #[getter]
    fn offsets(&self) -> Vec<(usize, usize)> {
        self.inner.offsets.clone()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn word_ids(&self) -> Vec<Option<usize>> {
        self.inner.word_ids.clone()
    }

    #[getter]
    fn n_overflowing(&self) -> usize {
        self.inner.n_overflowing()
    }

    #[getter]
    fn overflowing(&self) -> Vec<PyEncoding> {
        self.inner.overflowing()
            .iter()
            .map(|e| PyEncoding { inner: e.clone() })
            .collect()
    }

    fn pad(&mut self, target_length: usize, pad_id: u32, pad_token: &str, pad_left: bool) {
        self.inner.pad(target_length, pad_id, pad_token, pad_left);
    }

    fn truncate(&mut self, max_length: usize) {
        self.inner.truncate(max_length);
    }

    fn truncate_with_stride(&mut self, max_length: usize, stride: usize) {
        self.inner.truncate_with_stride(max_length, stride);
    }

    #[getter]
    fn sequence_ids(&self) -> Vec<Option<usize>> {
        self.inner.sequence_ids.clone()
    }

    fn char_to_token(&self, char_pos: usize) -> Option<usize> {
        self.inner.char_to_token(char_pos)
    }

    fn char_to_token_with_sequence(&self, char_pos: usize, sequence_id: usize) -> Option<usize> {
        self.inner.char_to_token_with_sequence(char_pos, sequence_id)
    }

    fn token_to_chars(&self, token_idx: usize) -> Option<(usize, usize)> {
        self.inner.token_to_chars(token_idx)
    }

    fn token_to_word(&self, token_idx: usize) -> Option<usize> {
        self.inner.token_to_word(token_idx)
    }

    fn token_to_sequence(&self, token_idx: usize) -> Option<usize> {
        self.inner.token_to_sequence(token_idx)
    }

    /// Get the token indices for a word
    /// Returns (start, end) indices where start is inclusive and end is exclusive
    fn word_to_tokens(&self, word_idx: usize) -> Option<(usize, usize)> {
        self.inner.word_to_tokens(word_idx)
    }

    /// Get the token indices for a word in a specific sequence
    #[pyo3(signature = (word_idx, sequence_id = 0))]
    fn word_to_tokens_with_sequence(&self, word_idx: usize, sequence_id: usize) -> Option<(usize, usize)> {
        self.inner.word_to_tokens_with_sequence(word_idx, sequence_id)
    }

    /// Get the character span for a word
    fn word_to_chars(&self, word_idx: usize) -> Option<(usize, usize)> {
        self.inner.word_to_chars(word_idx)
    }

    /// Get the character span for a word in a specific sequence
    #[pyo3(signature = (word_idx, sequence_id = 0))]
    fn word_to_chars_with_sequence(&self, word_idx: usize, sequence_id: usize) -> Option<(usize, usize)> {
        self.inner.word_to_chars_with_sequence(word_idx, sequence_id)
    }

    /// Get all token indices for a word
    fn word_token_indices(&self, word_idx: usize) -> Vec<usize> {
        self.inner.word_token_indices(word_idx)
    }

    /// Get the number of words
    #[getter]
    fn n_words(&self) -> usize {
        self.inner.n_words()
    }

    fn ids_as_numpy<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<u32>> {
        numpy::PyArray1::from_vec(py, self.inner.ids.clone())
    }

    fn attention_mask_as_numpy<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<u32>> {
        numpy::PyArray1::from_vec(py, self.inner.attention_mask.clone())
    }

    fn type_ids_as_numpy<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<u32>> {
        numpy::PyArray1::from_vec(py, self.inner.type_ids.clone())
    }

    fn special_tokens_mask_as_numpy<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<u32>> {
        numpy::PyArray1::from_vec(py, self.inner.special_tokens_mask.clone())
    }
}

/// Python-exposed BatchEncoding class (result of tokenizer("text"))
#[pyclass(name = "BatchEncoding")]
pub struct PyBatchEncoding {
    pub(crate) encodings: Vec<Encoding>,
    pub(crate) return_attention_mask: bool,
    pub(crate) return_token_type_ids: bool,
    pub(crate) return_offsets_mapping: bool,
    pub(crate) return_special_tokens_mask: bool,
}

impl PyBatchEncoding {
    pub fn new(
        encodings: Vec<Encoding>,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        return_offsets_mapping: bool,
        return_special_tokens_mask: bool,
    ) -> Self {
        Self {
            encodings,
            return_attention_mask,
            return_token_type_ids,
            return_offsets_mapping,
            return_special_tokens_mask,
        }
    }
}

#[pymethods]
impl PyBatchEncoding {
    #[getter]
    fn input_ids(&self) -> Vec<Vec<u32>> {
        self.encodings.iter().map(|e| e.ids.clone()).collect()
    }

    #[getter]
    fn attention_mask(&self) -> Vec<Vec<u32>> {
        if self.return_attention_mask {
            self.encodings.iter().map(|e| e.attention_mask.clone()).collect()
        } else {
            vec![]
        }
    }

    #[getter]
    fn token_type_ids(&self) -> Vec<Vec<u32>> {
        if self.return_token_type_ids {
            self.encodings.iter().map(|e| e.type_ids.clone()).collect()
        } else {
            vec![]
        }
    }

    #[getter]
    fn special_tokens_mask(&self) -> Vec<Vec<u32>> {
        if self.return_special_tokens_mask {
            self.encodings.iter().map(|e| e.special_tokens_mask.clone()).collect()
        } else {
            vec![]
        }
    }

    #[getter]
    fn offset_mapping(&self) -> Vec<Vec<(usize, usize)>> {
        if self.return_offsets_mapping {
            self.encodings.iter().map(|e| e.offsets.clone()).collect()
        } else {
            vec![]
        }
    }

    fn encodings(&self) -> Vec<PyEncoding> {
        self.encodings.iter()
            .map(|e| PyEncoding { inner: e.clone() })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.encodings.len()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<PyEncoding> {
        self.encodings.get(idx)
            .map(|e| PyEncoding { inner: e.clone() })
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of range"))
    }

    fn keys(&self) -> Vec<&'static str> {
        let mut keys = vec!["input_ids"];
        if self.return_attention_mask {
            keys.push("attention_mask");
        }
        if self.return_token_type_ids {
            keys.push("token_type_ids");
        }
        if self.return_special_tokens_mask {
            keys.push("special_tokens_mask");
        }
        if self.return_offsets_mapping {
            keys.push("offset_mapping");
        }
        keys
    }

    fn input_ids_as_numpy<'py>(&self, py: Python<'py>) -> Vec<pyo3::Bound<'py, numpy::PyArray1<u32>>> {
        self.encodings.iter()
            .map(|e| numpy::PyArray1::from_vec(py, e.ids.clone()))
            .collect()
    }

    fn attention_mask_as_numpy<'py>(&self, py: Python<'py>) -> Vec<pyo3::Bound<'py, numpy::PyArray1<u32>>> {
        self.encodings.iter()
            .map(|e| numpy::PyArray1::from_vec(py, e.attention_mask.clone()))
            .collect()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        use pyo3::types::PyDict;

        let dict = PyDict::new(py);
        dict.set_item("input_ids", self.input_ids())?;
        if self.return_attention_mask {
            dict.set_item("attention_mask", self.attention_mask())?;
        }
        if self.return_token_type_ids {
            dict.set_item("token_type_ids", self.token_type_ids())?;
        }
        if self.return_special_tokens_mask {
            dict.set_item("special_tokens_mask", self.special_tokens_mask())?;
        }
        if self.return_offsets_mapping {
            dict.set_item("offset_mapping", self.offset_mapping())?;
        }
        Ok(dict.into())
    }
}
