//! Decoders - Convert token IDs back to text
//!
//! Standard decoding algorithms (not proprietary).

use std::collections::HashMap;

/// Decoder types
#[derive(Debug, Clone)]
pub enum Decoder {
    /// ByteLevel decoder (GPT-2 style)
    ByteLevel,
    /// Metaspace decoder (SentencePiece style)
    Metaspace { replacement: char, add_prefix_space: bool },
    /// WordPiece decoder (BERT style)
    WordPiece { prefix: String, cleanup: bool },
    /// BPE decoder
    BPE { suffix: String },
    /// Replace patterns
    Replace { pattern: String, replacement: String },
    /// CTC decoder - removes blank tokens and merges consecutive duplicates
    CTC { pad_token: String, word_delimiter_token: Option<String> },
    /// Fuse decoder - merges all tokens into a single string
    Fuse,
    /// Strip decoder - strips specific characters from the decoded string
    Strip { content: char, start: usize, stop: usize },
    /// Sequence of decoders
    Sequence(Vec<Decoder>),
}

impl Decoder {
    /// Decode tokens to text
    pub fn decode(&self, tokens: &[String]) -> String {
        match self {
            Decoder::ByteLevel => byte_level_decode(tokens),
            Decoder::Metaspace { replacement, add_prefix_space } => {
                metaspace_decode(tokens, *replacement, *add_prefix_space)
            }
            Decoder::WordPiece { prefix, cleanup } => {
                wordpiece_decode(tokens, prefix, *cleanup)
            }
            Decoder::BPE { suffix } => {
                bpe_decode(tokens, suffix)
            }
            Decoder::Replace { pattern, replacement } => {
                let text = tokens.join("");
                text.replace(pattern, replacement)
            }
            Decoder::CTC { pad_token, word_delimiter_token } => {
                ctc_decode(tokens, pad_token, word_delimiter_token.as_deref())
            }
            Decoder::Fuse => {
                fuse_decode(tokens)
            }
            Decoder::Strip { content, start, stop } => {
                strip_decode(tokens, *content, *start, *stop)
            }
            Decoder::Sequence(decoders) => {
                let mut result = tokens.to_vec();
                for decoder in decoders {
                    let text = decoder.decode(&result);
                    result = vec![text];
                }
                result.join("")
            }
        }
    }
}

/// Unicode to byte mapping (reverse of bytes_to_unicode)
fn unicode_to_bytes() -> HashMap<char, u8> {
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

    cs.iter()
        .zip(bs.iter())
        .filter_map(|(&c, &b)| char::from_u32(c).map(|ch| (ch, b)))
        .collect()
}

/// ByteLevel decoding (GPT-2 style)
fn byte_level_decode(tokens: &[String]) -> String {
    let decoder_map = unicode_to_bytes();

    let joined = tokens.join("");

    // Convert unicode chars back to bytes
    let bytes: Vec<u8> = joined
        .chars()
        .filter_map(|c| {
            if c == 'Ġ' {
                Some(b' ')
            } else {
                decoder_map.get(&c).copied().or_else(|| {
                    // For regular ASCII chars, just use them
                    if c.is_ascii() {
                        Some(c as u8)
                    } else {
                        None
                    }
                })
            }
        })
        .collect();

    String::from_utf8_lossy(&bytes).to_string()
}

/// Metaspace decoding (SentencePiece style)
fn metaspace_decode(tokens: &[String], replacement: char, add_prefix_space: bool) -> String {
    let mut result = tokens.join("");
    result = result.replace(replacement, " ");

    if add_prefix_space && result.starts_with(' ') {
        result = result[1..].to_string();
    }

    result
}

/// WordPiece decoding (BERT style)
fn wordpiece_decode(tokens: &[String], prefix: &str, cleanup: bool) -> String {
    let mut result = String::new();

    for token in tokens {
        if token.starts_with(prefix) {
            result.push_str(&token[prefix.len()..]);
        } else {
            if !result.is_empty() {
                result.push(' ');
            }
            result.push_str(token);
        }
    }

    if cleanup {
        // Remove spaces before punctuation
        result = result
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" :", ":")
            .replace(" ;", ";")
            .replace(" '", "'")
            .replace("' ", "'");
    }

    result
}

/// BPE decoding
fn bpe_decode(tokens: &[String], suffix: &str) -> String {
    let mut result = String::new();

    for token in tokens {
        if token.ends_with(suffix) {
            result.push_str(&token[..token.len() - suffix.len()]);
            result.push(' ');
        } else {
            result.push_str(token);
        }
    }

    result.trim_end().to_string()
}

/// CTC decoding - removes blank tokens and merges consecutive duplicates
/// Used for speech recognition models (Wav2Vec2, etc.)
fn ctc_decode(tokens: &[String], pad_token: &str, word_delimiter_token: Option<&str>) -> String {
    let mut result = Vec::new();
    let mut prev_token: Option<&str> = None;

    for token in tokens {
        // Skip pad/blank tokens
        if token == pad_token {
            prev_token = None;
            continue;
        }

        // Handle word delimiter
        if let Some(delim) = word_delimiter_token {
            if token == delim {
                result.push(" ".to_string());
                prev_token = None;
                continue;
            }
        }

        // Merge consecutive duplicates (CTC collapse)
        if prev_token != Some(token.as_str()) {
            result.push(token.clone());
        }
        prev_token = Some(token.as_str());
    }

    result.join("")
}

/// Fuse decoding - simply concatenates all tokens
fn fuse_decode(tokens: &[String]) -> String {
    tokens.join("")
}

/// Strip decoding - strips specific characters from start/end of result
fn strip_decode(tokens: &[String], content: char, start: usize, stop: usize) -> String {
    let text = tokens.join("");

    let mut result = text.clone();

    // Strip from start
    for _ in 0..start {
        if result.starts_with(content) {
            result = result[content.len_utf8()..].to_string();
        } else {
            break;
        }
    }

    // Strip from end
    for _ in 0..stop {
        if result.ends_with(content) {
            let new_len = result.len() - content.len_utf8();
            result.truncate(new_len);
        } else {
            break;
        }
    }

    result
}

/// Default decoder (ByteLevel)
pub fn default_decoder() -> Decoder {
    Decoder::ByteLevel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metaspace_decode() {
        let decoder = Decoder::Metaspace {
            replacement: '▁',
            add_prefix_space: true,
        };
        let tokens = vec!["▁Hello".to_string(), "▁world".to_string()];
        assert_eq!(decoder.decode(&tokens), "Hello world");
    }

    #[test]
    fn test_wordpiece_decode() {
        let decoder = Decoder::WordPiece {
            prefix: "##".to_string(),
            cleanup: true,
        };
        let tokens = vec!["Hello".to_string(), "##world".to_string()];
        assert_eq!(decoder.decode(&tokens), "Helloworld");
    }

    #[test]
    fn test_byte_level_decode() {
        let decoder = Decoder::ByteLevel;
        let tokens = vec!["ĠHello".to_string(), "Ġworld".to_string()];
        let result = decoder.decode(&tokens);
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_ctc_decode() {
        let decoder = Decoder::CTC {
            pad_token: "<pad>".to_string(),
            word_delimiter_token: Some("|".to_string()),
        };
        // CTC output often has duplicates and blanks
        let tokens = vec![
            "H".to_string(), "H".to_string(), "E".to_string(),
            "<pad>".to_string(), "L".to_string(), "L".to_string(),
            "O".to_string(), "|".to_string(), "W".to_string(),
        ];
        let result = decoder.decode(&tokens);
        assert_eq!(result, "HELO W");
    }

    #[test]
    fn test_fuse_decode() {
        let decoder = Decoder::Fuse;
        let tokens = vec!["Hello".to_string(), " ".to_string(), "World".to_string()];
        assert_eq!(decoder.decode(&tokens), "Hello World");
    }

    #[test]
    fn test_strip_decode() {
        let decoder = Decoder::Strip {
            content: '_',
            start: 1,
            stop: 1,
        };
        let tokens = vec!["_Hello_".to_string()];
        assert_eq!(decoder.decode(&tokens), "Hello");
    }
}
