//! JSON parsing helpers for HuggingFace tokenizer format

use crate::decoders::Decoder;
use crate::normalizers::Normalizer;
use crate::postprocessors::PostProcessor;
use crate::pretokenizers::PreTokenizer;
use hashbrown::HashMap;

/// Parse normalizer from JSON
pub fn parse_normalizer(json: &Option<serde_json::Value>) -> Option<Normalizer> {
    if let Some(value) = json {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type") {
                let type_str = type_val.as_str().unwrap_or("");
                return match type_str {
                    "NFC" => Some(Normalizer::NFC),
                    "NFD" => Some(Normalizer::NFD),
                    "NFKC" => Some(Normalizer::NFKC),
                    "NFKD" => Some(Normalizer::NFKD),
                    "Lowercase" => Some(Normalizer::Lowercase),
                    "Strip" => Some(Normalizer::Strip),
                    "StripAccents" => Some(Normalizer::StripAccents),
                    "Replace" => {
                        let pattern = obj
                            .get("pattern")
                            .and_then(|v| v.get("String"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let replacement = obj
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        Some(Normalizer::Replace { pattern, replacement })
                    }
                    "Prepend" => {
                        let prepend = obj
                            .get("prepend")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        Some(Normalizer::Prepend(prepend))
                    }
                    "Sequence" => {
                        if let Some(normalizers) = obj.get("normalizers").and_then(|v| v.as_array()) {
                            let parsed: Vec<Normalizer> = normalizers
                                .iter()
                                .filter_map(|n| parse_normalizer(&Some(n.clone())))
                                .collect();
                            if !parsed.is_empty() {
                                Some(Normalizer::Sequence(parsed))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    "BertNormalizer" => {
                        let lowercase = obj.get("lowercase").and_then(|v| v.as_bool()).unwrap_or(true);
                        let strip_accents = obj.get("strip_accents").and_then(|v| v.as_bool());
                        let clean_text = obj.get("clean_text").and_then(|v| v.as_bool()).unwrap_or(true);
                        let handle_chinese_chars = obj.get("handle_chinese_chars").and_then(|v| v.as_bool()).unwrap_or(true);

                        Some(Normalizer::BertNormalizer {
                            clean_text,
                            handle_chinese_chars,
                            strip_accents,
                            lowercase,
                        })
                    }
                    "Precompiled" => {
                        // Parse precompiled charsmap
                        let charsmap = obj.get("precompiled_charsmap")
                            .and_then(|v| v.as_str())
                            .map(|s| {
                                // Simple parsing - in practice this would need more sophisticated handling
                                vec![(s.to_string(), s.to_string())]
                            })
                            .unwrap_or_default();
                        Some(Normalizer::Precompiled { charsmap })
                    }
                    _ => None,
                };
            }
        }
    }
    Some(Normalizer::NFC)
}

/// Parse pre-tokenizer from JSON
pub fn parse_pre_tokenizer(json: &Option<serde_json::Value>) -> Option<PreTokenizer> {
    if let Some(value) = json {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type") {
                let type_str = type_val.as_str().unwrap_or("");
                return match type_str {
                    "ByteLevel" => {
                        let add_prefix = obj
                            .get("add_prefix_space")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        Some(PreTokenizer::ByteLevel {
                            add_prefix_space: add_prefix,
                        })
                    }
                    "Metaspace" => {
                        let replacement = obj
                            .get("replacement")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.chars().next())
                            .unwrap_or('▁');
                        let add_prefix = obj
                            .get("add_prefix_space")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        Some(PreTokenizer::Metaspace {
                            replacement,
                            add_prefix_space: add_prefix,
                        })
                    }
                    "Whitespace" => Some(PreTokenizer::Whitespace),
                    "WhitespaceSplit" => Some(PreTokenizer::WhitespaceSplit),
                    "Punctuation" => Some(PreTokenizer::Punctuation),
                    "BertPreTokenizer" => Some(PreTokenizer::BertPreTokenizer),
                    "CharDelimiterSplit" => {
                        let delimiter = obj
                            .get("delimiter")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.chars().next())
                            .unwrap_or(' ');
                        Some(PreTokenizer::CharDelimiterSplit { delimiter })
                    }
                    "UnicodeScripts" => Some(PreTokenizer::UnicodeScripts),
                    "Digits" => {
                        let individual = obj
                            .get("individual_digits")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        Some(PreTokenizer::Digits { individual_digits: individual })
                    }
                    "Split" => {
                        let pattern = obj
                            .get("pattern")
                            .and_then(|v| v.get("Regex"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let invert = obj
                            .get("invert")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        let behavior_str = obj
                            .get("behavior")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Removed");
                        let behavior = match behavior_str {
                            "Isolated" => crate::pretokenizers::SplitBehavior::Isolated,
                            "MergedWithPrevious" => crate::pretokenizers::SplitBehavior::MergedWithPrevious,
                            "MergedWithNext" => crate::pretokenizers::SplitBehavior::MergedWithNext,
                            "Contiguous" => crate::pretokenizers::SplitBehavior::Contiguous,
                            _ => crate::pretokenizers::SplitBehavior::Removed,
                        };
                        Some(PreTokenizer::SplitWithBehavior { pattern, behavior, invert })
                    }
                    "Sequence" => {
                        if let Some(pretokenizers) = obj.get("pretokenizers").and_then(|v| v.as_array()) {
                            let parsed: Vec<PreTokenizer> = pretokenizers
                                .iter()
                                .filter_map(|p| parse_pre_tokenizer(&Some(p.clone())))
                                .collect();
                            if !parsed.is_empty() {
                                Some(PreTokenizer::Sequence(parsed))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
            }
        }
    }
    Some(PreTokenizer::ByteLevel {
        add_prefix_space: false,
    })
}

/// Parse post-processor from JSON
pub fn parse_post_processor(
    json: &Option<serde_json::Value>,
    special_tokens: &HashMap<String, u32>,
) -> Option<PostProcessor> {
    if let Some(value) = json {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type") {
                let type_str = type_val.as_str().unwrap_or("");
                return match type_str {
                    "TemplateProcessing" => {
                        let single = obj
                            .get("single")
                            .and_then(|v| v.as_array())
                            .map(|arr| template_from_array(arr))
                            .unwrap_or_else(|| "<s> $A </s>".to_string());
                        let pair = obj
                            .get("pair")
                            .and_then(|v| v.as_array())
                            .map(|arr| template_from_array(arr));
                        let tokens: Vec<(String, u32)> = special_tokens
                            .iter()
                            .map(|(k, v)| (k.clone(), *v))
                            .collect();
                        Some(PostProcessor::TemplateProcessing {
                            single,
                            pair,
                            special_tokens: tokens,
                        })
                    }
                    "RobertaProcessing" => {
                        let bos = special_tokens.get("<s>").copied().unwrap_or(0);
                        let eos = special_tokens.get("</s>").copied().unwrap_or(2);
                        Some(PostProcessor::RobertaProcessing {
                            bos: ("<s>".to_string(), bos),
                            eos: ("</s>".to_string(), eos),
                            add_prefix_space: false,
                        })
                    }
                    "BertProcessing" => {
                        let cls = special_tokens.get("[CLS]").copied().unwrap_or(101);
                        let sep = special_tokens.get("[SEP]").copied().unwrap_or(102);
                        Some(PostProcessor::BertProcessing {
                            cls: ("[CLS]".to_string(), cls),
                            sep: ("[SEP]".to_string(), sep),
                        })
                    }
                    _ => None,
                };
            }
        }
    }
    None
}

/// Convert template array to string
fn template_from_array(arr: &[serde_json::Value]) -> String {
    arr.iter()
        .filter_map(|item| {
            if let Some(obj) = item.as_object() {
                if let Some(special) = obj.get("SpecialToken") {
                    return special
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                }
                if let Some(seq) = obj.get("Sequence") {
                    return seq
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| format!("${}", s));
                }
            }
            None
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse decoder from JSON
pub fn parse_decoder(json: &Option<serde_json::Value>) -> Option<Decoder> {
    if let Some(value) = json {
        if let Some(obj) = value.as_object() {
            if let Some(type_val) = obj.get("type") {
                let type_str = type_val.as_str().unwrap_or("");
                return match type_str {
                    "ByteLevel" => Some(Decoder::ByteLevel),
                    "Metaspace" => {
                        let replacement = obj
                            .get("replacement")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.chars().next())
                            .unwrap_or('▁');
                        let add_prefix = obj
                            .get("add_prefix_space")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        Some(Decoder::Metaspace {
                            replacement,
                            add_prefix_space: add_prefix,
                        })
                    }
                    "WordPiece" => {
                        let prefix = obj
                            .get("prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("##")
                            .to_string();
                        let cleanup = obj
                            .get("cleanup")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        Some(Decoder::WordPiece { prefix, cleanup })
                    }
                    "BPE" => {
                        let suffix = obj
                            .get("suffix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("</w>")
                            .to_string();
                        Some(Decoder::BPE { suffix })
                    }
                    "CTC" => {
                        let pad_token = obj
                            .get("pad_token")
                            .and_then(|v| v.as_str())
                            .unwrap_or("<pad>")
                            .to_string();
                        let word_delimiter_token = obj
                            .get("word_delimiter_token")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        Some(Decoder::CTC { pad_token, word_delimiter_token })
                    }
                    "Fuse" => Some(Decoder::Fuse),
                    "Strip" => {
                        let content = obj
                            .get("content")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.chars().next())
                            .unwrap_or(' ');
                        let start = obj
                            .get("start")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let stop = obj
                            .get("stop")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        Some(Decoder::Strip { content, start, stop })
                    }
                    "Sequence" => {
                        if let Some(decoders) = obj.get("decoders").and_then(|v| v.as_array()) {
                            let parsed: Vec<Decoder> = decoders
                                .iter()
                                .filter_map(|d| parse_decoder(&Some(d.clone())))
                                .collect();
                            if !parsed.is_empty() {
                                Some(Decoder::Sequence(parsed))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
            }
        }
    }
    Some(Decoder::ByteLevel)
}
