//! Chat template processing

use std::collections::HashMap;

/// Result of applying a chat template
#[derive(Debug, Clone)]
pub enum ChatTemplateResult {
    /// Raw text (when tokenize=false)
    Text(String),
    /// Token IDs (when tokenize=true)
    Tokenized(Vec<u32>),
}

/// Apply chat template to messages
/// Messages format: Vec<HashMap<"role" -> "user"|"assistant"|"system", "content" -> "...">>
pub fn apply_chat_template(
    template: &str,
    messages: &[HashMap<String, String>],
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
) -> String {
    let mut result = String::new();

    // Process each message according to template patterns
    // Common patterns: ChatML, Llama, Mistral, etc.
    if template.contains("<|im_start|>") {
        // ChatML format
        for msg in messages {
            let role = msg.get("role").map(|s| s.as_str()).unwrap_or("user");
            let content = msg.get("content").map(|s| s.as_str()).unwrap_or("");
            result.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }
        if add_generation_prompt {
            result.push_str("<|im_start|>assistant\n");
        }
    } else if template.contains("[INST]") {
        // Llama/Mistral format
        result.push_str(bos_token);
        for msg in messages {
            let role = msg.get("role").map(|s| s.as_str()).unwrap_or("user");
            let content = msg.get("content").map(|s| s.as_str()).unwrap_or("");
            match role {
                "system" => {
                    result.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", content));
                }
                "user" => {
                    result.push_str(&format!("[INST] {} [/INST]", content));
                }
                "assistant" => {
                    result.push_str(&format!(" {}{}", content, eos_token));
                    result.push_str(bos_token);
                }
                _ => {}
            }
        }
    } else if template.contains("### ") {
        // Alpaca-like format
        for msg in messages {
            let role = msg.get("role").map(|s| s.as_str()).unwrap_or("user");
            let content = msg.get("content").map(|s| s.as_str()).unwrap_or("");
            match role {
                "system" => {
                    result.push_str(&format!("### System:\n{}\n\n", content));
                }
                "user" => {
                    result.push_str(&format!("### Human:\n{}\n\n", content));
                }
                "assistant" => {
                    result.push_str(&format!("### Assistant:\n{}\n\n", content));
                }
                _ => {}
            }
        }
        if add_generation_prompt {
            result.push_str("### Assistant:\n");
        }
    } else {
        // Default simple format
        for msg in messages {
            let role = msg.get("role").map(|s| s.as_str()).unwrap_or("user");
            let content = msg.get("content").map(|s| s.as_str()).unwrap_or("");
            result.push_str(&format!("{}: {}\n", role, content));
        }
        if add_generation_prompt {
            result.push_str("assistant: ");
        }
    }

    result
}
