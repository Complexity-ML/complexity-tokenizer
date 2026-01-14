//! Pre-tokenizers - Split text into words before BPE
//!
//! Standard text splitting algorithms (not proprietary).

use regex::Regex;
use std::sync::LazyLock;

/// GPT-2/GPT-4 style regex pattern for splitting
static GPT2_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .unwrap()
});

/// Pre-tokenizer types
#[derive(Debug, Clone)]
pub enum PreTokenizer {
    /// Split on whitespace
    Whitespace,
    /// Split on whitespace and punctuation
    WhitespaceSplit,
    /// ByteLevel (GPT-2 style) - handles all bytes
    ByteLevel { add_prefix_space: bool },
    /// Metaspace (SentencePiece style) - uses ▁ for spaces
    Metaspace { replacement: char, add_prefix_space: bool },
    /// Split on punctuation
    Punctuation,
    /// Split digits into individual tokens
    Digits { individual_digits: bool },
    /// Split using regex pattern
    Split { pattern: String, invert: bool },
    /// GPT-2 style regex splitting
    GPT2,
    /// Sequence of pre-tokenizers
    Sequence(Vec<PreTokenizer>),
}

impl PreTokenizer {
    /// Pre-tokenize text into words
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        match self {
            PreTokenizer::Whitespace => {
                text.split_whitespace()
                    .map(|s| s.to_string())
                    .collect()
            }
            PreTokenizer::WhitespaceSplit => {
                text.split(|c: char| c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect()
            }
            PreTokenizer::ByteLevel { add_prefix_space } => {
                byte_level_pretokenize(text, *add_prefix_space)
            }
            PreTokenizer::Metaspace { replacement, add_prefix_space } => {
                metaspace_pretokenize(text, *replacement, *add_prefix_space)
            }
            PreTokenizer::Punctuation => {
                punctuation_split(text)
            }
            PreTokenizer::Digits { individual_digits } => {
                digits_split(text, *individual_digits)
            }
            PreTokenizer::Split { pattern, invert } => {
                regex_split(text, pattern, *invert)
            }
            PreTokenizer::GPT2 => {
                gpt2_pretokenize(text)
            }
            PreTokenizer::Sequence(pretokenizers) => {
                let mut words = vec![text.to_string()];
                for pt in pretokenizers {
                    let mut new_words = Vec::new();
                    for word in words {
                        new_words.extend(pt.pre_tokenize(&word));
                    }
                    words = new_words;
                }
                words
            }
        }
    }
}

/// GPT-2 byte-to-unicode mapping
pub fn bytes_to_unicode() -> std::collections::HashMap<u8, char> {
    use std::collections::HashMap;

    let mut bs: Vec<u8> = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(0xa1u8..=0xacu8);
    bs.extend(0xaeu8..=0xffu8);

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;

    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    bs.iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (b, char::from_u32(c).unwrap()))
        .collect::<HashMap<_, _>>()
}

/// ByteLevel pre-tokenization (GPT-2 style)
fn byte_level_pretokenize(text: &str, add_prefix_space: bool) -> Vec<String> {
    let byte_encoder = bytes_to_unicode();
    let mut words = Vec::new();

    let text = if add_prefix_space && !text.starts_with(' ') {
        format!(" {}", text)
    } else {
        text.to_string()
    };

    for word in text.split_whitespace() {
        // Prefix with Ġ (encoded space)
        let word_with_space = format!("Ġ{}", word);

        let encoded: String = word_with_space
            .bytes()
            .filter_map(|b| byte_encoder.get(&b).copied())
            .collect();

        if !encoded.is_empty() {
            words.push(encoded);
        }
    }

    words
}

/// Metaspace pre-tokenization (SentencePiece style)
fn metaspace_pretokenize(text: &str, replacement: char, add_prefix_space: bool) -> Vec<String> {
    let text = if add_prefix_space {
        format!("{}{}", replacement, text)
    } else {
        text.to_string()
    };

    text.replace(' ', &replacement.to_string())
        .split(|c: char| c.is_whitespace() && c != replacement)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Split on punctuation
fn punctuation_split(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c.is_ascii_punctuation() || is_unicode_punctuation(c) {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            words.push(c.to_string());
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}

/// Check if character is Unicode punctuation
fn is_unicode_punctuation(c: char) -> bool {
    let code = c as u32;
    // General punctuation ranges
    matches!(code,
        0x0021..=0x002F | // !"#$%&'()*+,-./
        0x003A..=0x0040 | // :;<=>?@
        0x005B..=0x0060 | // [\]^_`
        0x007B..=0x007E | // {|}~
        0x00A1..=0x00BF | // Latin-1 punctuation
        0x2000..=0x206F | // General punctuation
        0x2E00..=0x2E7F | // Supplemental punctuation
        0x3000..=0x303F   // CJK punctuation
    )
}

/// Split digits
fn digits_split(text: &str, individual_digits: bool) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut in_digits = false;

    for c in text.chars() {
        let is_digit = c.is_ascii_digit();

        if is_digit != in_digits {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            in_digits = is_digit;
        }

        if is_digit && individual_digits {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            words.push(c.to_string());
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}

/// Split using regex pattern
fn regex_split(text: &str, pattern: &str, invert: bool) -> Vec<String> {
    if let Ok(re) = Regex::new(pattern) {
        if invert {
            // Return parts that DON'T match
            re.split(text)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        } else {
            // Return parts that DO match
            re.find_iter(text)
                .map(|m| m.as_str().to_string())
                .collect()
        }
    } else {
        vec![text.to_string()]
    }
}

/// GPT-2 style pre-tokenization with regex
fn gpt2_pretokenize(text: &str) -> Vec<String> {
    GPT2_PATTERN
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Default pre-tokenizer (ByteLevel)
pub fn default_pretokenizer() -> PreTokenizer {
    PreTokenizer::ByteLevel { add_prefix_space: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace() {
        let pt = PreTokenizer::Whitespace;
        assert_eq!(pt.pre_tokenize("hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn test_punctuation() {
        let pt = PreTokenizer::Punctuation;
        let result = pt.pre_tokenize("hello, world!");
        assert_eq!(result, vec!["hello", ",", " world", "!"]);
    }

    #[test]
    fn test_digits() {
        let pt = PreTokenizer::Digits { individual_digits: true };
        let result = pt.pre_tokenize("hello123world");
        assert_eq!(result, vec!["hello", "1", "2", "3", "world"]);
    }

    #[test]
    fn test_gpt2() {
        let pt = PreTokenizer::GPT2;
        let result = pt.pre_tokenize("Hello, world!");
        assert!(result.len() > 1);
    }

    #[test]
    fn test_metaspace() {
        let pt = PreTokenizer::Metaspace {
            replacement: '▁',
            add_prefix_space: true
        };
        let result = pt.pre_tokenize("hello world");
        assert!(result[0].starts_with('▁'));
    }
}
