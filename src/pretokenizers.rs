//! Pre-tokenizers - Split text into words before BPE
//!
//! Standard text splitting algorithms (not proprietary).

use regex::Regex;
use std::sync::LazyLock;

/// GPT-2/GPT-4 style regex pattern for splitting
/// Note: Original pattern uses look-ahead which is not supported by rust regex crate
/// This simplified pattern handles most cases
static GPT2_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    // Simplified pattern without look-ahead: contractions, words, numbers, punctuation, whitespace
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
        .unwrap()
});

/// Split behavior for regex patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitBehavior {
    /// Remove the matched pattern from output
    Removed,
    /// Keep the pattern as its own isolated token
    Isolated,
    /// Merge matched pattern with the previous token
    MergedWithPrevious,
    /// Merge matched pattern with the next token
    MergedWithNext,
    /// Consecutive matches are merged together
    Contiguous,
}

impl Default for SplitBehavior {
    fn default() -> Self {
        SplitBehavior::Removed
    }
}

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
    /// Split using regex pattern (legacy, uses Removed behavior)
    Split { pattern: String, invert: bool },
    /// Split using regex pattern with behavior control
    SplitWithBehavior { pattern: String, behavior: SplitBehavior, invert: bool },
    /// GPT-2 style regex splitting
    GPT2,
    /// BERT-style pre-tokenizer (whitespace + punctuation + chinese chars)
    BertPreTokenizer,
    /// Split on a specific character delimiter
    CharDelimiterSplit { delimiter: char },
    /// Split on Unicode script changes (Latin, Han, Hiragana, etc.)
    UnicodeScripts,
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
            PreTokenizer::SplitWithBehavior { pattern, behavior, invert } => {
                regex_split_with_behavior(text, pattern, *behavior, *invert)
            }
            PreTokenizer::GPT2 => {
                gpt2_pretokenize(text)
            }
            PreTokenizer::BertPreTokenizer => {
                bert_pretokenize(text)
            }
            PreTokenizer::CharDelimiterSplit { delimiter } => {
                char_delimiter_split(text, *delimiter)
            }
            PreTokenizer::UnicodeScripts => {
                unicode_scripts_split(text)
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
/// This uses the GPT2 regex pattern to split text, preserving spaces at the start of words,
/// then encodes each byte using the GPT2 byte-to-unicode mapping.
fn byte_level_pretokenize(text: &str, add_prefix_space: bool) -> Vec<String> {
    let byte_encoder = bytes_to_unicode();
    let mut words = Vec::new();

    // Optionally add prefix space
    let text = if add_prefix_space && !text.is_empty() && !text.starts_with(' ') {
        format!(" {}", text)
    } else {
        text.to_string()
    };

    // Use GPT2 regex to split - this preserves spaces at the beginning of words
    for mat in GPT2_PATTERN.find_iter(&text) {
        let word = mat.as_str();

        // Encode each byte using GPT2 byte-to-unicode mapping
        let encoded: String = word
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

/// Split using regex pattern with behavior control
fn regex_split_with_behavior(text: &str, pattern: &str, behavior: SplitBehavior, invert: bool) -> Vec<String> {
    let re = match Regex::new(pattern) {
        Ok(r) => r,
        Err(_) => return vec![text.to_string()],
    };

    // Collect all matches and their positions
    let matches: Vec<_> = re.find_iter(text).collect();
    if matches.is_empty() {
        return vec![text.to_string()];
    }

    // Build result based on behavior
    let mut result = Vec::new();
    let mut last_end = 0;

    match behavior {
        SplitBehavior::Removed => {
            // Same as invert=true in legacy Split
            for m in &matches {
                if invert {
                    // Keep non-matching parts only
                    if m.start() > last_end {
                        result.push(text[last_end..m.start()].to_string());
                    }
                } else {
                    // Keep matching parts only
                    result.push(m.as_str().to_string());
                }
                last_end = m.end();
            }
            if invert && last_end < text.len() {
                result.push(text[last_end..].to_string());
            }
        }
        SplitBehavior::Isolated => {
            // Keep both matches and non-matches as separate tokens
            for m in &matches {
                if m.start() > last_end {
                    let before = &text[last_end..m.start()];
                    if !before.is_empty() {
                        result.push(before.to_string());
                    }
                }
                result.push(m.as_str().to_string());
                last_end = m.end();
            }
            if last_end < text.len() {
                result.push(text[last_end..].to_string());
            }
        }
        SplitBehavior::MergedWithPrevious => {
            // Merge matches with the token before them
            for m in &matches {
                if m.start() > last_end {
                    let before = &text[last_end..m.start()];
                    if !before.is_empty() {
                        // Merge this part with the match
                        result.push(format!("{}{}", before, m.as_str()));
                    } else if !result.is_empty() {
                        // Append match to previous token
                        let prev = result.pop().unwrap();
                        result.push(format!("{}{}", prev, m.as_str()));
                    } else {
                        result.push(m.as_str().to_string());
                    }
                } else if !result.is_empty() {
                    let prev = result.pop().unwrap();
                    result.push(format!("{}{}", prev, m.as_str()));
                } else {
                    result.push(m.as_str().to_string());
                }
                last_end = m.end();
            }
            if last_end < text.len() {
                result.push(text[last_end..].to_string());
            }
        }
        SplitBehavior::MergedWithNext => {
            // Merge matches with the token after them
            let mut pending_match: Option<&str> = None;
            for m in &matches {
                if m.start() > last_end {
                    let before = &text[last_end..m.start()];
                    if let Some(pm) = pending_match {
                        result.push(format!("{}{}", pm, before));
                    } else if !before.is_empty() {
                        result.push(before.to_string());
                    }
                } else if let Some(pm) = pending_match {
                    // No text between matches, output pending
                    result.push(pm.to_string());
                }
                pending_match = Some(m.as_str());
                last_end = m.end();
            }
            // Handle remaining text
            if last_end < text.len() {
                let remaining = &text[last_end..];
                if let Some(pm) = pending_match {
                    result.push(format!("{}{}", pm, remaining));
                } else {
                    result.push(remaining.to_string());
                }
            } else if let Some(pm) = pending_match {
                result.push(pm.to_string());
            }
        }
        SplitBehavior::Contiguous => {
            // Merge consecutive matches together
            let mut current_match = String::new();
            for m in &matches {
                if m.start() > last_end {
                    let before = &text[last_end..m.start()];
                    if !current_match.is_empty() {
                        result.push(current_match.clone());
                        current_match.clear();
                    }
                    if !before.is_empty() {
                        result.push(before.to_string());
                    }
                }
                current_match.push_str(m.as_str());
                last_end = m.end();
            }
            if !current_match.is_empty() {
                result.push(current_match);
            }
            if last_end < text.len() {
                result.push(text[last_end..].to_string());
            }
        }
    }

    result.into_iter().filter(|s| !s.is_empty()).collect()
}

/// GPT-2 style pre-tokenization with regex
fn gpt2_pretokenize(text: &str) -> Vec<String> {
    GPT2_PATTERN
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// BERT-style pre-tokenization
/// Splits on whitespace, punctuation, and isolates Chinese characters
fn bert_pretokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c.is_whitespace() {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
        } else if is_chinese_char(c) {
            // Chinese characters are always isolated
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            words.push(c.to_string());
        } else if c.is_ascii_punctuation() || is_unicode_punctuation(c) {
            // Punctuation is isolated
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

/// Check if character is a Chinese character
fn is_chinese_char(c: char) -> bool {
    let code = c as u32;
    matches!(code,
        0x4E00..=0x9FFF |    // CJK Unified Ideographs
        0x3400..=0x4DBF |    // CJK Unified Ideographs Extension A
        0x20000..=0x2A6DF |  // CJK Unified Ideographs Extension B
        0x2A700..=0x2B73F |  // CJK Unified Ideographs Extension C
        0x2B740..=0x2B81F |  // CJK Unified Ideographs Extension D
        0x2B820..=0x2CEAF |  // CJK Unified Ideographs Extension E
        0x2CEB0..=0x2EBEF |  // CJK Unified Ideographs Extension F
        0x30000..=0x3134F |  // CJK Unified Ideographs Extension G
        0xF900..=0xFAFF |    // CJK Compatibility Ideographs
        0x2F800..=0x2FA1F    // CJK Compatibility Ideographs Supplement
    )
}

/// Split on a specific character delimiter
fn char_delimiter_split(text: &str, delimiter: char) -> Vec<String> {
    text.split(delimiter)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Split on Unicode script changes
/// Groups consecutive characters from the same Unicode script together
fn unicode_scripts_split(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut current_script: Option<UnicodeScript> = None;

    for c in text.chars() {
        if c.is_whitespace() {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
                current_script = None;
            }
            continue;
        }

        let script = get_unicode_script(c);

        if current_script.is_none() || current_script == Some(script) || script == UnicodeScript::Common {
            current.push(c);
            if current_script.is_none() && script != UnicodeScript::Common {
                current_script = Some(script);
            }
        } else {
            // Script change - start new word
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            current.push(c);
            current_script = Some(script);
        }
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}

/// Unicode script categories (simplified)
#[derive(Debug, Clone, Copy, PartialEq)]
enum UnicodeScript {
    Latin,
    Greek,
    Cyrillic,
    Arabic,
    Hebrew,
    Han,
    Hiragana,
    Katakana,
    Hangul,
    Thai,
    Common,  // Punctuation, numbers, etc.
    Unknown,
}

/// Get the Unicode script for a character
fn get_unicode_script(c: char) -> UnicodeScript {
    let code = c as u32;
    match code {
        // Latin
        0x0041..=0x007A | 0x00C0..=0x024F | 0x1E00..=0x1EFF => UnicodeScript::Latin,
        // Greek
        0x0370..=0x03FF | 0x1F00..=0x1FFF => UnicodeScript::Greek,
        // Cyrillic
        0x0400..=0x04FF | 0x0500..=0x052F => UnicodeScript::Cyrillic,
        // Arabic
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF => UnicodeScript::Arabic,
        // Hebrew
        0x0590..=0x05FF => UnicodeScript::Hebrew,
        // CJK/Han
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF => UnicodeScript::Han,
        // Hiragana
        0x3040..=0x309F => UnicodeScript::Hiragana,
        // Katakana
        0x30A0..=0x30FF | 0x31F0..=0x31FF => UnicodeScript::Katakana,
        // Hangul
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => UnicodeScript::Hangul,
        // Thai
        0x0E00..=0x0E7F => UnicodeScript::Thai,
        // Common (punctuation, numbers, spaces)
        0x0000..=0x0040 | 0x005B..=0x0060 | 0x007B..=0x00BF |
        0x2000..=0x206F | 0x3000..=0x303F => UnicodeScript::Common,
        _ => UnicodeScript::Unknown,
    }
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

    #[test]
    fn test_bert_pretokenizer() {
        let pt = PreTokenizer::BertPreTokenizer;
        let result = pt.pre_tokenize("Hello, world!");
        assert_eq!(result, vec!["Hello", ",", "world", "!"]);
    }

    #[test]
    fn test_bert_pretokenizer_chinese() {
        let pt = PreTokenizer::BertPreTokenizer;
        let result = pt.pre_tokenize("Hello世界");
        assert_eq!(result, vec!["Hello", "世", "界"]);
    }

    #[test]
    fn test_char_delimiter_split() {
        let pt = PreTokenizer::CharDelimiterSplit { delimiter: '_' };
        let result = pt.pre_tokenize("hello_world_test");
        assert_eq!(result, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_unicode_scripts() {
        let pt = PreTokenizer::UnicodeScripts;
        let result = pt.pre_tokenize("Helloこんにちは");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "Hello");
        assert_eq!(result[1], "こんにちは");
    }

    #[test]
    fn test_split_isolated() {
        let pt = PreTokenizer::SplitWithBehavior {
            pattern: r"\s".to_string(),
            behavior: SplitBehavior::Isolated,
            invert: false,
        };
        let result = pt.pre_tokenize("hello world test");
        // Should keep spaces as isolated tokens
        assert_eq!(result, vec!["hello", " ", "world", " ", "test"]);
    }

    #[test]
    fn test_split_merged_with_previous() {
        let pt = PreTokenizer::SplitWithBehavior {
            pattern: r"!".to_string(),
            behavior: SplitBehavior::MergedWithPrevious,
            invert: false,
        };
        let result = pt.pre_tokenize("hello! world!");
        // Exclamation marks should merge with previous word
        assert_eq!(result, vec!["hello!", " world!"]);
    }

    #[test]
    fn test_split_merged_with_next() {
        let pt = PreTokenizer::SplitWithBehavior {
            pattern: r"\$".to_string(),
            behavior: SplitBehavior::MergedWithNext,
            invert: false,
        };
        let result = pt.pre_tokenize("price $100 and $50");
        // Dollar signs should merge with next token
        assert_eq!(result, vec!["price ", "$100 and ", "$50"]);
    }

    #[test]
    fn test_split_contiguous() {
        let pt = PreTokenizer::SplitWithBehavior {
            pattern: r"\d".to_string(),
            behavior: SplitBehavior::Contiguous,
            invert: false,
        };
        let result = pt.pre_tokenize("abc123def456");
        // Consecutive digits should be merged
        assert_eq!(result, vec!["abc", "123", "def", "456"]);
    }
}
