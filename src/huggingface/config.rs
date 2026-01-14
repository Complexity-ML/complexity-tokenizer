//! Configuration structs for HuggingFace tokenizer

/// Padding configuration
#[derive(Debug, Clone, Default)]
pub struct PaddingConfig {
    pub enabled: bool,
    pub strategy: String, // "longest", "max_length"
    pub pad_to_multiple_of: Option<usize>,
    pub direction: String, // "right" or "left"
}

/// Truncation configuration
#[derive(Debug, Clone, Default)]
pub struct TruncationConfig {
    pub enabled: bool,
    pub max_length: usize,
    pub strategy: String, // "longest_first", "only_first", "only_second"
    pub stride: usize,
    pub direction: String, // "right" or "left"
}
