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
}
