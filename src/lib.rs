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
mod bindings;

pub use bpe::BpeTokenizer;
pub use vocab::Vocab;
pub use huggingface::{HuggingFaceTokenizer, ChatTemplateResult, PaddingConfig, TruncationConfig};
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

/// Python module
#[pymodule]
fn complexity_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bindings::PyTokenizer>()?;
    m.add_class::<bindings::PyTrainer>()?;
    m.add_class::<bindings::PyWordPieceTrainer>()?;
    m.add_class::<bindings::PyUnigramTrainer>()?;
    m.add_class::<bindings::PyBpeTrainer>()?;
    m.add_class::<bindings::PyEncoding>()?;
    m.add_class::<bindings::PyBatchEncoding>()?;
    m.add_class::<bindings::PyNormalizer>()?;
    m.add_class::<bindings::PyPreTokenizer>()?;
    m.add_class::<bindings::PyPostProcessor>()?;
    m.add_class::<bindings::PyDecoder>()?;
    m.add_class::<bindings::PyWordPieceModel>()?;
    m.add_class::<bindings::PyUnigramModel>()?;
    m.add_class::<bindings::PyWordLevelModel>()?;
    m.add_class::<bindings::PyCharBpeModel>()?;
    m.add_class::<bindings::PyByteLevelBpeModel>()?;
    m.add("__version__", "0.3.1")?;
    Ok(())
}
