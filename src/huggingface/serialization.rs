//! JSON serialization helpers for HuggingFace tokenizer format

use crate::decoders::Decoder;
use crate::normalizers::Normalizer;
use crate::postprocessors::PostProcessor;
use crate::pretokenizers::PreTokenizer;
use hashbrown::HashMap;

/// Serialize normalizer to JSON
pub fn serialize_normalizer(normalizer: &Normalizer) -> serde_json::Value {
    match normalizer {
        Normalizer::NFC => serde_json::json!({"type": "NFC"}),
        Normalizer::NFD => serde_json::json!({"type": "NFD"}),
        Normalizer::NFKC => serde_json::json!({"type": "NFKC"}),
        Normalizer::NFKD => serde_json::json!({"type": "NFKD"}),
        Normalizer::Lowercase => serde_json::json!({"type": "Lowercase"}),
        Normalizer::Strip => serde_json::json!({"type": "Strip"}),
        Normalizer::StripAccents => serde_json::json!({"type": "StripAccents"}),
        Normalizer::Replace { pattern, replacement } => serde_json::json!({
            "type": "Replace",
            "pattern": {"String": pattern},
            "content": replacement
        }),
        Normalizer::Prepend(prepend) => serde_json::json!({
            "type": "Prepend",
            "prepend": prepend
        }),
        Normalizer::Append(append) => serde_json::json!({
            "type": "Append",
            "append": append
        }),
        Normalizer::BertNormalizer { clean_text, handle_chinese_chars, strip_accents, lowercase } => serde_json::json!({
            "type": "BertNormalizer",
            "clean_text": clean_text,
            "handle_chinese_chars": handle_chinese_chars,
            "strip_accents": strip_accents,
            "lowercase": lowercase
        }),
        Normalizer::Precompiled { charsmap } => serde_json::json!({
            "type": "Precompiled",
            "precompiled_charsmap": charsmap.iter()
                .map(|(from, to)| format!("{}:{}", from, to))
                .collect::<Vec<_>>()
                .join(",")
        }),
        Normalizer::Sequence(normalizers) => serde_json::json!({
            "type": "Sequence",
            "normalizers": normalizers.iter().map(serialize_normalizer).collect::<Vec<_>>()
        }),
    }
}

/// Serialize pre-tokenizer to JSON
pub fn serialize_pre_tokenizer(pre_tokenizer: &PreTokenizer) -> serde_json::Value {
    match pre_tokenizer {
        PreTokenizer::ByteLevel { add_prefix_space } => serde_json::json!({
            "type": "ByteLevel",
            "add_prefix_space": add_prefix_space,
            "trim_offsets": true,
            "use_regex": true
        }),
        PreTokenizer::Metaspace { replacement, add_prefix_space } => serde_json::json!({
            "type": "Metaspace",
            "replacement": replacement.to_string(),
            "add_prefix_space": add_prefix_space
        }),
        PreTokenizer::Whitespace => serde_json::json!({"type": "Whitespace"}),
        PreTokenizer::WhitespaceSplit => serde_json::json!({"type": "WhitespaceSplit"}),
        PreTokenizer::Punctuation => serde_json::json!({"type": "Punctuation"}),
        PreTokenizer::Digits { individual_digits } => serde_json::json!({
            "type": "Digits",
            "individual_digits": individual_digits
        }),
        PreTokenizer::Split { pattern, invert } => serde_json::json!({
            "type": "Split",
            "pattern": {"Regex": pattern},
            "behavior": "Removed",
            "invert": invert
        }),
        PreTokenizer::GPT2 => serde_json::json!({
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        }),
        PreTokenizer::BertPreTokenizer => serde_json::json!({"type": "BertPreTokenizer"}),
        PreTokenizer::CharDelimiterSplit { delimiter } => serde_json::json!({
            "type": "CharDelimiterSplit",
            "delimiter": delimiter.to_string()
        }),
        PreTokenizer::UnicodeScripts => serde_json::json!({"type": "UnicodeScripts"}),
        PreTokenizer::Sequence(pretokenizers) => serde_json::json!({
            "type": "Sequence",
            "pretokenizers": pretokenizers.iter().map(serialize_pre_tokenizer).collect::<Vec<_>>()
        }),
    }
}

/// Serialize post-processor to JSON
pub fn serialize_post_processor(post_processor: &PostProcessor, _special_tokens: &HashMap<String, u32>) -> serde_json::Value {
    match post_processor {
        PostProcessor::TemplateProcessing { single, pair, special_tokens: tokens } => {
            let special_tokens_json: Vec<serde_json::Value> = tokens.iter()
                .map(|(token, id)| serde_json::json!({
                    "id": token,
                    "ids": [id],
                    "tokens": [token]
                }))
                .collect();

            serde_json::json!({
                "type": "TemplateProcessing",
                "single": parse_template_to_json(single),
                "pair": pair.as_ref().map(|p| parse_template_to_json(p)),
                "special_tokens": special_tokens_json
            })
        }
        PostProcessor::RobertaProcessing { bos, eos, add_prefix_space } => serde_json::json!({
            "type": "RobertaProcessing",
            "sep": [eos.0.clone(), eos.1],
            "cls": [bos.0.clone(), bos.1],
            "trim_offsets": true,
            "add_prefix_space": add_prefix_space
        }),
        PostProcessor::BertProcessing { cls, sep } => serde_json::json!({
            "type": "BertProcessing",
            "sep": [sep.0.clone(), sep.1],
            "cls": [cls.0.clone(), cls.1]
        }),
        _ => serde_json::json!(null),
    }
}

/// Parse template string to JSON array
fn parse_template_to_json(template: &str) -> Vec<serde_json::Value> {
    template.split_whitespace()
        .map(|part| {
            if part.starts_with('$') {
                serde_json::json!({"Sequence": {"id": &part[1..], "type_id": 0}})
            } else {
                serde_json::json!({"SpecialToken": {"id": part, "type_id": 0}})
            }
        })
        .collect()
}

/// Serialize decoder to JSON
pub fn serialize_decoder(decoder: &Decoder) -> serde_json::Value {
    match decoder {
        Decoder::ByteLevel => serde_json::json!({"type": "ByteLevel"}),
        Decoder::Metaspace { replacement, add_prefix_space } => serde_json::json!({
            "type": "Metaspace",
            "replacement": replacement.to_string(),
            "add_prefix_space": add_prefix_space
        }),
        Decoder::WordPiece { prefix, cleanup } => serde_json::json!({
            "type": "WordPiece",
            "prefix": prefix,
            "cleanup": cleanup
        }),
        Decoder::BPE { suffix } => serde_json::json!({
            "type": "BPE",
            "suffix": suffix
        }),
        Decoder::Replace { pattern, replacement } => serde_json::json!({
            "type": "Replace",
            "pattern": pattern,
            "content": replacement
        }),
        Decoder::Sequence(decoders) => serde_json::json!({
            "type": "Sequence",
            "decoders": decoders.iter().map(serialize_decoder).collect::<Vec<_>>()
        }),
    }
}
