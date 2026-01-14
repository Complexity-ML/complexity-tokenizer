//! Post-processors - Add special tokens, truncate, pad
//!
//! Standard post-processing algorithms (not proprietary).

/// Post-processor types
#[derive(Debug, Clone)]
pub enum PostProcessor {
    /// Template-based processing (add BOS/EOS tokens)
    TemplateProcessing {
        /// Single sequence template, e.g., "<s> $A </s>"
        single: String,
        /// Pair sequence template, e.g., "<s> $A </s> $B </s>"
        pair: Option<String>,
        /// Special tokens with their IDs
        special_tokens: Vec<(String, u32)>,
    },
    /// BERT-style processing ([CLS] ... [SEP])
    BertProcessing {
        cls: (String, u32),
        sep: (String, u32),
    },
    /// RoBERTa-style processing (<s> ... </s>)
    RobertaProcessing {
        bos: (String, u32),
        eos: (String, u32),
        add_prefix_space: bool,
    },
    /// Sequence of post-processors
    Sequence(Vec<PostProcessor>),
}

impl PostProcessor {
    /// Apply post-processing to token IDs
    pub fn process(&self, ids: Vec<u32>, pair_ids: Option<Vec<u32>>) -> Vec<u32> {
        match self {
            PostProcessor::TemplateProcessing { single, pair, special_tokens } => {
                template_process(ids, pair_ids, single, pair.as_deref(), special_tokens)
            }
            PostProcessor::BertProcessing { cls, sep } => {
                bert_process(ids, pair_ids, cls, sep)
            }
            PostProcessor::RobertaProcessing { bos, eos, .. } => {
                roberta_process(ids, pair_ids, bos, eos)
            }
            PostProcessor::Sequence(processors) => {
                let mut result = ids;
                let mut pair_result = pair_ids;
                for processor in processors {
                    result = processor.process(result, pair_result.take());
                }
                result
            }
        }
    }

    /// Get number of special tokens added for single sequence
    pub fn added_tokens_single(&self) -> usize {
        match self {
            PostProcessor::TemplateProcessing { single, special_tokens, .. } => {
                count_special_tokens(single, special_tokens)
            }
            PostProcessor::BertProcessing { .. } => 2, // [CLS] + [SEP]
            PostProcessor::RobertaProcessing { .. } => 2, // <s> + </s>
            PostProcessor::Sequence(processors) => {
                processors.iter().map(|p| p.added_tokens_single()).sum()
            }
        }
    }

    /// Get number of special tokens added for pair sequence
    pub fn added_tokens_pair(&self) -> usize {
        match self {
            PostProcessor::TemplateProcessing { pair, special_tokens, .. } => {
                pair.as_ref()
                    .map(|p| count_special_tokens(p, special_tokens))
                    .unwrap_or(0)
            }
            PostProcessor::BertProcessing { .. } => 3, // [CLS] + [SEP] + [SEP]
            PostProcessor::RobertaProcessing { .. } => 4, // <s> + </s> + </s> + </s>
            PostProcessor::Sequence(processors) => {
                processors.iter().map(|p| p.added_tokens_pair()).sum()
            }
        }
    }
}

/// Template-based post-processing
fn template_process(
    ids: Vec<u32>,
    pair_ids: Option<Vec<u32>>,
    single_template: &str,
    pair_template: Option<&str>,
    special_tokens: &[(String, u32)],
) -> Vec<u32> {
    let template = if pair_ids.is_some() {
        pair_template.unwrap_or(single_template)
    } else {
        single_template
    };

    let mut result = Vec::new();

    let mut i = 0;
    let chars: Vec<char> = template.chars().collect();

    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() {
            match chars[i + 1] {
                'A' => {
                    result.extend(&ids);
                    i += 2;
                }
                'B' => {
                    if let Some(ref pair) = pair_ids {
                        result.extend(pair);
                    }
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        } else if chars[i] == '<' || chars[i] == '[' {
            // Find end of token
            let end_char = if chars[i] == '<' { '>' } else { ']' };
            let start = i;
            while i < chars.len() && chars[i] != end_char {
                i += 1;
            }
            if i < chars.len() {
                i += 1; // include end char
            }
            let token: String = chars[start..i].iter().collect();
            let token = token.trim();

            // Find token ID
            if let Some((_, id)) = special_tokens.iter().find(|(t, _)| t == token) {
                result.push(*id);
            }
        } else if !chars[i].is_whitespace() {
            i += 1;
        } else {
            i += 1;
        }
    }

    result
}

/// BERT-style post-processing
fn bert_process(
    ids: Vec<u32>,
    pair_ids: Option<Vec<u32>>,
    cls: &(String, u32),
    sep: &(String, u32),
) -> Vec<u32> {
    let mut result = vec![cls.1];
    result.extend(&ids);
    result.push(sep.1);

    if let Some(pair) = pair_ids {
        result.extend(&pair);
        result.push(sep.1);
    }

    result
}

/// RoBERTa-style post-processing
fn roberta_process(
    ids: Vec<u32>,
    pair_ids: Option<Vec<u32>>,
    bos: &(String, u32),
    eos: &(String, u32),
) -> Vec<u32> {
    let mut result = vec![bos.1];
    result.extend(&ids);
    result.push(eos.1);

    if let Some(pair) = pair_ids {
        result.push(eos.1);
        result.extend(&pair);
        result.push(eos.1);
    }

    result
}

/// Count special tokens in template
fn count_special_tokens(template: &str, special_tokens: &[(String, u32)]) -> usize {
    special_tokens
        .iter()
        .filter(|(token, _)| template.contains(token.as_str()))
        .count()
}

/// Truncation strategy
#[derive(Debug, Clone, Copy)]
pub enum TruncationStrategy {
    /// Truncate only the first sequence
    OnlyFirst,
    /// Truncate only the second sequence
    OnlySecond,
    /// Truncate the longest sequence
    LongestFirst,
}

/// Truncate sequences to max length
pub fn truncate(
    ids: &mut Vec<u32>,
    mut pair_ids: Option<&mut Vec<u32>>,
    max_length: usize,
    strategy: TruncationStrategy,
) {
    let total_len = ids.len() + pair_ids.as_ref().map(|p| p.len()).unwrap_or(0);

    if total_len <= max_length {
        return;
    }

    let to_remove = total_len - max_length;

    match strategy {
        TruncationStrategy::OnlyFirst => {
            let remove = to_remove.min(ids.len());
            ids.truncate(ids.len() - remove);
        }
        TruncationStrategy::OnlySecond => {
            if let Some(pair) = pair_ids {
                let remove = to_remove.min(pair.len());
                pair.truncate(pair.len() - remove);
            }
        }
        TruncationStrategy::LongestFirst => {
            let mut remaining = to_remove;
            while remaining > 0 {
                let ids_len = ids.len();
                let pair_len = pair_ids.as_ref().map(|p| p.len()).unwrap_or(0);

                if ids_len >= pair_len && ids_len > 0 {
                    ids.pop();
                    remaining -= 1;
                } else if let Some(ref mut pair) = pair_ids {
                    if !pair.is_empty() {
                        pair.pop();
                        remaining -= 1;
                    }
                } else {
                    break;
                }
            }
        }
    }
}

/// Padding strategy
#[derive(Debug, Clone, Copy)]
pub enum PaddingStrategy {
    /// Pad to the longest sequence in batch
    BatchLongest,
    /// Pad to a fixed length
    Fixed(usize),
}

/// Pad sequence to target length
pub fn pad(ids: &mut Vec<u32>, target_length: usize, pad_token_id: u32, pad_left: bool) {
    if ids.len() >= target_length {
        return;
    }

    let pad_count = target_length - ids.len();

    if pad_left {
        let mut padded = vec![pad_token_id; pad_count];
        padded.extend(ids.iter());
        *ids = padded;
    } else {
        ids.extend(std::iter::repeat(pad_token_id).take(pad_count));
    }
}

/// Create default post-processor for LLM (BOS + EOS)
pub fn default_postprocessor() -> PostProcessor {
    PostProcessor::TemplateProcessing {
        single: "<s> $A </s>".to_string(),
        pair: Some("<s> $A </s> $B </s>".to_string()),
        special_tokens: vec![
            ("<s>".to_string(), 2),
            ("</s>".to_string(), 0),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_processing() {
        let processor = PostProcessor::BertProcessing {
            cls: ("[CLS]".to_string(), 101),
            sep: ("[SEP]".to_string(), 102),
        };

        let ids = vec![1, 2, 3];
        let result = processor.process(ids, None);

        assert_eq!(result, vec![101, 1, 2, 3, 102]);
    }

    #[test]
    fn test_roberta_processing() {
        let processor = PostProcessor::RobertaProcessing {
            bos: ("<s>".to_string(), 0),
            eos: ("</s>".to_string(), 2),
            add_prefix_space: false,
        };

        let ids = vec![1, 2, 3];
        let result = processor.process(ids, None);

        assert_eq!(result, vec![0, 1, 2, 3, 2]);
    }

    #[test]
    fn test_truncation() {
        let mut ids = vec![1, 2, 3, 4, 5];
        truncate(&mut ids, None, 3, TruncationStrategy::OnlyFirst);
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_padding() {
        let mut ids = vec![1, 2, 3];
        pad(&mut ids, 5, 0, false);
        assert_eq!(ids, vec![1, 2, 3, 0, 0]);
    }
}
