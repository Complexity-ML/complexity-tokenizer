//! Python bindings for Tokenizer components

use crate::normalizers::Normalizer;
use crate::pretokenizers::PreTokenizer;
use crate::postprocessors::PostProcessor;
use crate::decoders::Decoder;
use pyo3::prelude::*;

/// Python-exposed Normalizer class
#[pyclass(name = "Normalizer")]
pub struct PyNormalizer {
    pub(crate) inner: Normalizer,
}

#[pymethods]
impl PyNormalizer {
    #[staticmethod]
    fn nfc() -> Self {
        Self { inner: Normalizer::NFC }
    }

    #[staticmethod]
    fn nfd() -> Self {
        Self { inner: Normalizer::NFD }
    }

    #[staticmethod]
    fn nfkc() -> Self {
        Self { inner: Normalizer::NFKC }
    }

    #[staticmethod]
    fn nfkd() -> Self {
        Self { inner: Normalizer::NFKD }
    }

    #[staticmethod]
    fn lowercase() -> Self {
        Self { inner: Normalizer::Lowercase }
    }

    #[staticmethod]
    fn strip() -> Self {
        Self { inner: Normalizer::Strip }
    }

    #[staticmethod]
    fn strip_accents() -> Self {
        Self { inner: Normalizer::StripAccents }
    }

    #[staticmethod]
    fn replace(pattern: String, replacement: String) -> Self {
        Self {
            inner: Normalizer::Replace { pattern, replacement },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (clean_text = true, handle_chinese_chars = true, strip_accents = None, lowercase = true))]
    fn bert(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> Self {
        Self {
            inner: Normalizer::BertNormalizer {
                clean_text,
                handle_chinese_chars,
                strip_accents,
                lowercase,
            },
        }
    }

    #[staticmethod]
    fn precompiled(charsmap: Vec<(String, String)>) -> Self {
        Self {
            inner: Normalizer::Precompiled { charsmap },
        }
    }

    fn normalize(&self, text: &str) -> String {
        self.inner.normalize(text)
    }
}

/// Python-exposed PreTokenizer class
#[pyclass(name = "PreTokenizer")]
pub struct PyPreTokenizer {
    pub(crate) inner: PreTokenizer,
}

#[pymethods]
impl PyPreTokenizer {
    #[staticmethod]
    fn whitespace() -> Self {
        Self { inner: PreTokenizer::Whitespace }
    }

    #[staticmethod]
    #[pyo3(signature = (add_prefix_space = false))]
    fn byte_level(add_prefix_space: bool) -> Self {
        Self {
            inner: PreTokenizer::ByteLevel { add_prefix_space },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (replacement = '▁', add_prefix_space = true))]
    fn metaspace(replacement: char, add_prefix_space: bool) -> Self {
        Self {
            inner: PreTokenizer::Metaspace { replacement, add_prefix_space },
        }
    }

    #[staticmethod]
    fn punctuation() -> Self {
        Self { inner: PreTokenizer::Punctuation }
    }

    #[staticmethod]
    #[pyo3(signature = (individual_digits = false))]
    fn digits(individual_digits: bool) -> Self {
        Self {
            inner: PreTokenizer::Digits { individual_digits },
        }
    }

    #[staticmethod]
    fn gpt2() -> Self {
        Self { inner: PreTokenizer::GPT2 }
    }

    #[staticmethod]
    fn bert() -> Self {
        Self { inner: PreTokenizer::BertPreTokenizer }
    }

    #[staticmethod]
    fn char_delimiter_split(delimiter: char) -> Self {
        Self { inner: PreTokenizer::CharDelimiterSplit { delimiter } }
    }

    #[staticmethod]
    fn unicode_scripts() -> Self {
        Self { inner: PreTokenizer::UnicodeScripts }
    }

    #[staticmethod]
    #[pyo3(signature = (pattern, behavior = "Removed", invert = false))]
    fn split(pattern: String, behavior: &str, invert: bool) -> Self {
        let split_behavior = match behavior {
            "Isolated" => crate::pretokenizers::SplitBehavior::Isolated,
            "MergedWithPrevious" => crate::pretokenizers::SplitBehavior::MergedWithPrevious,
            "MergedWithNext" => crate::pretokenizers::SplitBehavior::MergedWithNext,
            "Contiguous" => crate::pretokenizers::SplitBehavior::Contiguous,
            _ => crate::pretokenizers::SplitBehavior::Removed,
        };
        Self {
            inner: PreTokenizer::SplitWithBehavior {
                pattern,
                behavior: split_behavior,
                invert,
            },
        }
    }

    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        self.inner.pre_tokenize(text)
    }
}

/// Python-exposed PostProcessor class
#[pyclass(name = "PostProcessor")]
pub struct PyPostProcessor {
    pub(crate) inner: PostProcessor,
}

#[pymethods]
impl PyPostProcessor {
    #[staticmethod]
    fn bert(cls_token: String, cls_id: u32, sep_token: String, sep_id: u32) -> Self {
        Self {
            inner: PostProcessor::BertProcessing {
                cls: (cls_token, cls_id),
                sep: (sep_token, sep_id),
            },
        }
    }

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

    #[pyo3(signature = (ids, pair_ids = None))]
    fn process(&self, ids: Vec<u32>, pair_ids: Option<Vec<u32>>) -> Vec<u32> {
        self.inner.process(ids, pair_ids)
    }

    fn added_tokens_single(&self) -> usize {
        self.inner.added_tokens_single()
    }

    fn added_tokens_pair(&self) -> usize {
        self.inner.added_tokens_pair()
    }
}

/// Python-exposed Decoder class
#[pyclass(name = "Decoder")]
pub struct PyDecoder {
    pub(crate) inner: Decoder,
}

#[pymethods]
impl PyDecoder {
    #[staticmethod]
    fn byte_level() -> Self {
        Self { inner: Decoder::ByteLevel }
    }

    #[staticmethod]
    #[pyo3(signature = (replacement = '▁', add_prefix_space = true))]
    fn metaspace(replacement: char, add_prefix_space: bool) -> Self {
        Self {
            inner: Decoder::Metaspace { replacement, add_prefix_space },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (prefix = "##".to_string(), cleanup = true))]
    fn wordpiece(prefix: String, cleanup: bool) -> Self {
        Self {
            inner: Decoder::WordPiece { prefix, cleanup },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (suffix = "</w>".to_string()))]
    fn bpe(suffix: String) -> Self {
        Self {
            inner: Decoder::BPE { suffix },
        }
    }

    #[staticmethod]
    #[pyo3(signature = (pad_token = "<pad>".to_string(), word_delimiter_token = None))]
    fn ctc(pad_token: String, word_delimiter_token: Option<String>) -> Self {
        Self {
            inner: Decoder::CTC { pad_token, word_delimiter_token },
        }
    }

    #[staticmethod]
    fn fuse() -> Self {
        Self { inner: Decoder::Fuse }
    }

    #[staticmethod]
    #[pyo3(signature = (content = ' ', start = 0, stop = 0))]
    fn strip(content: char, start: usize, stop: usize) -> Self {
        Self {
            inner: Decoder::Strip { content, start, stop },
        }
    }

    fn decode(&self, tokens: Vec<String>) -> String {
        self.inner.decode(&tokens)
    }
}
