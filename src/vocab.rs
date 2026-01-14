//! Vocabulary management with special tokens

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// Special tokens configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub unk_token: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub pad_token: Option<String>,
    pub sep_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            unk_token: Some("<unk>".to_string()),
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            pad_token: Some("<pad>".to_string()),
            sep_token: None,
            cls_token: None,
            mask_token: None,
        }
    }
}

/// Vocabulary with special tokens support
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Token -> ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID -> Token mapping
    id_to_token: HashMap<u32, String>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// Special token IDs
    special_token_ids: HashMap<String, u32>,
}

impl Vocab {
    /// Create vocabulary from token->id mapping
    pub fn new(token_to_id: HashMap<String, u32>, special_tokens: SpecialTokens) -> Self {
        let id_to_token: HashMap<u32, String> = token_to_id
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        let mut special_token_ids = HashMap::new();

        // Map special tokens to their IDs
        if let Some(ref tok) = special_tokens.unk_token {
            if let Some(&id) = token_to_id.get(tok) {
                special_token_ids.insert("unk".to_string(), id);
            }
        }
        if let Some(ref tok) = special_tokens.bos_token {
            if let Some(&id) = token_to_id.get(tok) {
                special_token_ids.insert("bos".to_string(), id);
            }
        }
        if let Some(ref tok) = special_tokens.eos_token {
            if let Some(&id) = token_to_id.get(tok) {
                special_token_ids.insert("eos".to_string(), id);
            }
        }
        if let Some(ref tok) = special_tokens.pad_token {
            if let Some(&id) = token_to_id.get(tok) {
                special_token_ids.insert("pad".to_string(), id);
            }
        }

        Self {
            token_to_id,
            id_to_token,
            special_tokens,
            special_token_ids,
        }
    }

    /// Get token ID
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token string
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Get UNK token ID
    pub fn unk_id(&self) -> Option<u32> {
        self.special_token_ids.get("unk").copied()
    }

    /// Get BOS token ID
    pub fn bos_id(&self) -> Option<u32> {
        self.special_token_ids.get("bos").copied()
    }

    /// Get EOS token ID
    pub fn eos_id(&self) -> Option<u32> {
        self.special_token_ids.get("eos").copied()
    }

    /// Get PAD token ID
    pub fn pad_id(&self) -> Option<u32> {
        self.special_token_ids.get("pad").copied()
    }

    /// Vocabulary size
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// Is vocabulary empty
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// Get all special token IDs
    pub fn special_token_ids(&self) -> &HashMap<String, u32> {
        &self.special_token_ids
    }

    /// Get token->id mapping reference
    pub fn token_to_id(&self) -> &HashMap<String, u32> {
        &self.token_to_id
    }

    /// Get special tokens configuration
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Check if token is special
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.unk_token.as_deref() == Some(token)
            || self.special_tokens.bos_token.as_deref() == Some(token)
            || self.special_tokens.eos_token.as_deref() == Some(token)
            || self.special_tokens.pad_token.as_deref() == Some(token)
            || self.special_tokens.sep_token.as_deref() == Some(token)
            || self.special_tokens.cls_token.as_deref() == Some(token)
            || self.special_tokens.mask_token.as_deref() == Some(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_special_tokens() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("<unk>".to_string(), 0);
        token_to_id.insert("<s>".to_string(), 1);
        token_to_id.insert("</s>".to_string(), 2);
        token_to_id.insert("<pad>".to_string(), 3);
        token_to_id.insert("hello".to_string(), 4);

        let vocab = Vocab::new(token_to_id, SpecialTokens::default());

        assert_eq!(vocab.unk_id(), Some(0));
        assert_eq!(vocab.bos_id(), Some(1));
        assert_eq!(vocab.eos_id(), Some(2));
        assert_eq!(vocab.pad_id(), Some(3));
        assert_eq!(vocab.len(), 5);
    }
}
