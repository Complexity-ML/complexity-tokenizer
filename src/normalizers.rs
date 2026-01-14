//! Text normalizers - Unicode normalization, lowercase, strip, etc.
//!
//! Standard text normalization algorithms (not proprietary).

use unicode_normalization::UnicodeNormalization;

/// Normalizer types
#[derive(Debug, Clone)]
pub enum Normalizer {
    /// Unicode NFC normalization (canonical decomposition + canonical composition)
    NFC,
    /// Unicode NFD normalization (canonical decomposition)
    NFD,
    /// Unicode NFKC normalization (compatibility decomposition + canonical composition)
    NFKC,
    /// Unicode NFKD normalization (compatibility decomposition)
    NFKD,
    /// Convert to lowercase
    Lowercase,
    /// Strip leading/trailing whitespace
    Strip,
    /// Strip accents (decompose + remove combining marks)
    StripAccents,
    /// Replace pattern with replacement
    Replace { pattern: String, replacement: String },
    /// Prepend a string
    Prepend(String),
    /// Append a string
    Append(String),
    /// Sequence of normalizers
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    /// Apply normalization to text
    pub fn normalize(&self, text: &str) -> String {
        match self {
            Normalizer::NFC => text.nfc().collect(),
            Normalizer::NFD => text.nfd().collect(),
            Normalizer::NFKC => text.nfkc().collect(),
            Normalizer::NFKD => text.nfkd().collect(),
            Normalizer::Lowercase => text.to_lowercase(),
            Normalizer::Strip => text.trim().to_string(),
            Normalizer::StripAccents => strip_accents(text),
            Normalizer::Replace { pattern, replacement } => {
                text.replace(pattern, replacement)
            }
            Normalizer::Prepend(s) => format!("{}{}", s, text),
            Normalizer::Append(s) => format!("{}{}", text, s),
            Normalizer::Sequence(normalizers) => {
                let mut result = text.to_string();
                for normalizer in normalizers {
                    result = normalizer.normalize(&result);
                }
                result
            }
        }
    }
}

/// Strip accents by decomposing and removing combining characters
fn strip_accents(text: &str) -> String {
    text.nfd()
        .filter(|c| !is_combining_mark(*c))
        .collect()
}

/// Check if character is a combining mark (Unicode category M)
fn is_combining_mark(c: char) -> bool {
    let code = c as u32;
    // Combining Diacritical Marks: 0x0300 - 0x036F
    // Combining Diacritical Marks Extended: 0x1AB0 - 0x1AFF
    // Combining Diacritical Marks Supplement: 0x1DC0 - 0x1DFF
    // Combining Diacritical Marks for Symbols: 0x20D0 - 0x20FF
    // Combining Half Marks: 0xFE20 - 0xFE2F
    matches!(code,
        0x0300..=0x036F |
        0x1AB0..=0x1AFF |
        0x1DC0..=0x1DFF |
        0x20D0..=0x20FF |
        0xFE20..=0xFE2F
    )
}

/// Default normalizer for BPE training (NFC)
pub fn default_normalizer() -> Normalizer {
    Normalizer::NFC
}

/// Create a BERT-style normalizer (NFC + lowercase + strip accents)
pub fn bert_normalizer() -> Normalizer {
    Normalizer::Sequence(vec![
        Normalizer::NFC,
        Normalizer::Lowercase,
        Normalizer::StripAccents,
        Normalizer::Strip,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfc() {
        let normalizer = Normalizer::NFC;
        // é as e + combining accent should become single é
        let text = "e\u{0301}"; // e + combining acute accent
        let normalized = normalizer.normalize(text);
        assert_eq!(normalized, "é");
    }

    #[test]
    fn test_lowercase() {
        let normalizer = Normalizer::Lowercase;
        assert_eq!(normalizer.normalize("HELLO World"), "hello world");
    }

    #[test]
    fn test_strip_accents() {
        let normalizer = Normalizer::StripAccents;
        assert_eq!(normalizer.normalize("café"), "cafe");
        assert_eq!(normalizer.normalize("naïve"), "naive");
    }

    #[test]
    fn test_sequence() {
        let normalizer = bert_normalizer();
        assert_eq!(normalizer.normalize("  CAFÉ  "), "cafe");
    }
}
