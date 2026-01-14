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
    /// BERT-style normalizer (clean text + handle chinese chars + strip accents + lowercase)
    BertNormalizer {
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    },
    /// Precompiled normalizer (uses precompiled charsmap for fast normalization)
    Precompiled { charsmap: Vec<(String, String)> },
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
            Normalizer::BertNormalizer {
                clean_text,
                handle_chinese_chars,
                strip_accents,
                lowercase,
            } => {
                let mut result = text.to_string();

                // Clean text: remove control characters, replace whitespace
                if *clean_text {
                    result = bert_clean_text(&result);
                }

                // Handle Chinese characters: add spaces around them
                if *handle_chinese_chars {
                    result = bert_handle_chinese_chars(&result);
                }

                // Apply NFC normalization
                result = result.nfc().collect();

                // Strip accents (if enabled, or if lowercase is enabled)
                let should_strip = strip_accents.unwrap_or(*lowercase);
                if should_strip {
                    result = strip_accents_impl(&result);
                }

                // Lowercase
                if *lowercase {
                    result = result.to_lowercase();
                }

                result
            }
            Normalizer::Precompiled { charsmap } => {
                precompiled_normalize(text, charsmap)
            }
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
    strip_accents_impl(text)
}

/// Strip accents implementation (used by both StripAccents and BertNormalizer)
fn strip_accents_impl(text: &str) -> String {
    text.nfd()
        .filter(|c| !is_combining_mark(*c))
        .collect()
}

/// BERT clean text: remove control characters and replace whitespace
fn bert_clean_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for c in text.chars() {
        // Skip control characters (except tab, newline, carriage return)
        if is_control_char(c) {
            continue;
        }
        // Replace all whitespace with regular space
        if c.is_whitespace() {
            result.push(' ');
        } else {
            result.push(c);
        }
    }
    result
}

/// Check if character is a control character (not tab/newline/cr)
fn is_control_char(c: char) -> bool {
    let code = c as u32;
    // Exclude tab (0x09), newline (0x0A), carriage return (0x0D)
    if c == '\t' || c == '\n' || c == '\r' {
        return false;
    }
    // Control characters: 0x00-0x1F and 0x7F-0x9F
    matches!(code, 0x0000..=0x001F | 0x007F..=0x009F)
}

/// BERT handle Chinese characters: add spaces around Chinese chars
fn bert_handle_chinese_chars(text: &str) -> String {
    let mut result = String::with_capacity(text.len() * 2);
    for c in text.chars() {
        if is_chinese_char_bert(c) {
            result.push(' ');
            result.push(c);
            result.push(' ');
        } else {
            result.push(c);
        }
    }
    result
}

/// Check if character is Chinese (for BERT)
fn is_chinese_char_bert(c: char) -> bool {
    let code = c as u32;
    matches!(code,
        0x4E00..=0x9FFF |    // CJK Unified Ideographs
        0x3400..=0x4DBF |    // CJK Extension A
        0x20000..=0x2A6DF | // CJK Extension B
        0x2A700..=0x2B73F | // CJK Extension C
        0x2B740..=0x2B81F | // CJK Extension D
        0x2B820..=0x2CEAF | // CJK Extension E
        0xF900..=0xFAFF |   // CJK Compatibility Ideographs
        0x2F800..=0x2FA1F   // CJK Compatibility Supplement
    )
}

/// Precompiled normalize using a character map
fn precompiled_normalize(text: &str, charsmap: &[(String, String)]) -> String {
    let mut result = text.to_string();
    for (from, to) in charsmap {
        result = result.replace(from, to);
    }
    result
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

    #[test]
    fn test_bert_normalizer() {
        let normalizer = Normalizer::BertNormalizer {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: Some(true),
            lowercase: true,
        };
        assert_eq!(normalizer.normalize("HELLO"), "hello");
        assert_eq!(normalizer.normalize("Café"), "cafe");
    }

    #[test]
    fn test_bert_normalizer_chinese() {
        let normalizer = Normalizer::BertNormalizer {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
        };
        let result = normalizer.normalize("Hello世界");
        assert!(result.contains(" 世 "));
    }

    #[test]
    fn test_precompiled() {
        let charsmap = vec![
            ("ﬁ".to_string(), "fi".to_string()),
            ("ﬂ".to_string(), "fl".to_string()),
        ];
        let normalizer = Normalizer::Precompiled { charsmap };
        assert_eq!(normalizer.normalize("ﬁle"), "file");
    }
}
