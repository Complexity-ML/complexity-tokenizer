//! Encoding - Full encoding output with attention masks, offsets, etc.
//!
//! Standard encoding features (not proprietary).

/// Full encoding output
#[derive(Debug, Clone)]
pub struct Encoding {
    /// Token IDs
    pub ids: Vec<u32>,
    /// Token type IDs (0 for first sequence, 1 for second)
    pub type_ids: Vec<u32>,
    /// Tokens as strings
    pub tokens: Vec<String>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u32>,
    /// Special tokens mask (1 for special, 0 for normal)
    pub special_tokens_mask: Vec<u32>,
    /// Character offsets for each token (start, end)
    pub offsets: Vec<(usize, usize)>,
    /// Word IDs (which word each token belongs to)
    pub word_ids: Vec<Option<usize>>,
    /// Sequence IDs (None for special tokens, 0 for first seq, 1 for second)
    pub sequence_ids: Vec<Option<usize>>,
    /// Overflowing tokens (if truncated)
    pub overflowing: Vec<Encoding>,
}

impl Encoding {
    /// Create a new empty encoding
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            type_ids: Vec::new(),
            tokens: Vec::new(),
            attention_mask: Vec::new(),
            special_tokens_mask: Vec::new(),
            offsets: Vec::new(),
            word_ids: Vec::new(),
            sequence_ids: Vec::new(),
            overflowing: Vec::new(),
        }
    }

    /// Create encoding from token IDs
    pub fn from_ids(ids: Vec<u32>, tokens: Vec<String>) -> Self {
        let len = ids.len();
        Self {
            ids,
            type_ids: vec![0; len],
            tokens,
            attention_mask: vec![1; len],
            special_tokens_mask: vec![0; len],
            offsets: Vec::new(),
            word_ids: Vec::new(),
            sequence_ids: vec![Some(0); len],
            overflowing: Vec::new(),
        }
    }

    /// Get the number of tokens
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Set token type ID for a range
    pub fn set_type_ids(&mut self, type_id: u32, start: usize, end: usize) {
        for i in start..end.min(self.type_ids.len()) {
            self.type_ids[i] = type_id;
        }
    }

    /// Mark special tokens
    pub fn mark_special_tokens(&mut self, special_ids: &[u32]) {
        for (i, id) in self.ids.iter().enumerate() {
            if special_ids.contains(id) {
                self.special_tokens_mask[i] = 1;
            }
        }
    }

    /// Pad encoding to target length
    pub fn pad(&mut self, target_length: usize, pad_id: u32, pad_token: &str, pad_left: bool) {
        if self.len() >= target_length {
            return;
        }

        let pad_count = target_length - self.len();

        if pad_left {
            // Prepend padding
            let mut new_ids = vec![pad_id; pad_count];
            new_ids.extend(&self.ids);
            self.ids = new_ids;

            let mut new_type_ids = vec![0; pad_count];
            new_type_ids.extend(&self.type_ids);
            self.type_ids = new_type_ids;

            let mut new_tokens: Vec<String> = vec![pad_token.to_string(); pad_count];
            new_tokens.extend(self.tokens.clone());
            self.tokens = new_tokens;

            let mut new_attention = vec![0; pad_count];
            new_attention.extend(&self.attention_mask);
            self.attention_mask = new_attention;

            let mut new_special = vec![1; pad_count];  // Padding is special
            new_special.extend(&self.special_tokens_mask);
            self.special_tokens_mask = new_special;

            let mut new_seq_ids: Vec<Option<usize>> = vec![None; pad_count];
            new_seq_ids.extend(&self.sequence_ids);
            self.sequence_ids = new_seq_ids;
        } else {
            // Append padding
            self.ids.extend(std::iter::repeat(pad_id).take(pad_count));
            self.type_ids.extend(std::iter::repeat(0).take(pad_count));
            self.tokens.extend(std::iter::repeat(pad_token.to_string()).take(pad_count));
            self.attention_mask.extend(std::iter::repeat(0).take(pad_count));
            self.special_tokens_mask.extend(std::iter::repeat(1).take(pad_count));
            self.sequence_ids.extend(std::iter::repeat(None).take(pad_count));
        }
    }

    /// Truncate encoding to max length
    pub fn truncate(&mut self, max_length: usize) {
        if self.len() <= max_length {
            return;
        }

        // Store overflowing tokens
        let overflow = Encoding {
            ids: self.ids[max_length..].to_vec(),
            type_ids: self.type_ids[max_length..].to_vec(),
            tokens: self.tokens[max_length..].to_vec(),
            attention_mask: self.attention_mask[max_length..].to_vec(),
            special_tokens_mask: self.special_tokens_mask[max_length..].to_vec(),
            offsets: if self.offsets.len() > max_length {
                self.offsets[max_length..].to_vec()
            } else {
                Vec::new()
            },
            word_ids: if self.word_ids.len() > max_length {
                self.word_ids[max_length..].to_vec()
            } else {
                Vec::new()
            },
            sequence_ids: if self.sequence_ids.len() > max_length {
                self.sequence_ids[max_length..].to_vec()
            } else {
                Vec::new()
            },
            overflowing: Vec::new(),
        };

        self.overflowing.push(overflow);

        // Truncate
        self.ids.truncate(max_length);
        self.type_ids.truncate(max_length);
        self.tokens.truncate(max_length);
        self.attention_mask.truncate(max_length);
        self.special_tokens_mask.truncate(max_length);
        self.offsets.truncate(max_length);
        self.word_ids.truncate(max_length);
        self.sequence_ids.truncate(max_length);
    }

    /// Truncate with stride - creates overlapping windows for long documents
    pub fn truncate_with_stride(&mut self, max_length: usize, stride: usize) {
        if self.len() <= max_length {
            return;
        }

        // Create overlapping windows
        let mut pos = max_length;
        while pos < self.ids.len() {
            let start = pos.saturating_sub(stride);
            let end = (start + max_length).min(self.ids.len());

            let overflow = Encoding {
                ids: self.ids[start..end].to_vec(),
                type_ids: self.type_ids[start..end].to_vec(),
                tokens: self.tokens[start..end].to_vec(),
                attention_mask: self.attention_mask[start..end].to_vec(),
                special_tokens_mask: self.special_tokens_mask[start..end].to_vec(),
                offsets: if self.offsets.len() > start {
                    self.offsets[start..end.min(self.offsets.len())].to_vec()
                } else {
                    Vec::new()
                },
                word_ids: if self.word_ids.len() > start {
                    self.word_ids[start..end.min(self.word_ids.len())].to_vec()
                } else {
                    Vec::new()
                },
                sequence_ids: if self.sequence_ids.len() > start {
                    self.sequence_ids[start..end.min(self.sequence_ids.len())].to_vec()
                } else {
                    Vec::new()
                },
                overflowing: Vec::new(),
            };

            self.overflowing.push(overflow);
            pos = end;
        }

        // Truncate main encoding
        self.ids.truncate(max_length);
        self.type_ids.truncate(max_length);
        self.tokens.truncate(max_length);
        self.attention_mask.truncate(max_length);
        self.special_tokens_mask.truncate(max_length);
        self.offsets.truncate(max_length);
        self.word_ids.truncate(max_length);
        self.sequence_ids.truncate(max_length);
    }

    /// Get word IDs
    pub fn word_ids(&self) -> &[Option<usize>] {
        &self.word_ids
    }

    /// Get overflowing encodings
    pub fn overflowing(&self) -> &[Encoding] {
        &self.overflowing
    }

    /// Get the number of overflowing encodings
    pub fn n_overflowing(&self) -> usize {
        self.overflowing.len()
    }

    /// Merge two encodings (for sequence pairs)
    pub fn merge(&mut self, other: Encoding, type_id: u32) {
        let other_len = other.ids.len();

        self.ids.extend(other.ids);
        self.tokens.extend(other.tokens);
        self.attention_mask.extend(other.attention_mask);
        self.special_tokens_mask.extend(other.special_tokens_mask);
        self.offsets.extend(other.offsets);
        self.word_ids.extend(other.word_ids);

        // Set type IDs for second sequence
        self.type_ids.extend(std::iter::repeat(type_id).take(other_len));

        // Set sequence IDs for second sequence
        self.sequence_ids.extend(std::iter::repeat(Some(type_id as usize)).take(other_len));
    }

    /// Get sequence IDs
    pub fn sequence_ids(&self) -> &[Option<usize>] {
        &self.sequence_ids
    }

    /// Get the token index for a character position
    /// Returns None if the character is not part of any token
    pub fn char_to_token(&self, char_pos: usize) -> Option<usize> {
        for (i, &(start, end)) in self.offsets.iter().enumerate() {
            if char_pos >= start && char_pos < end {
                return Some(i);
            }
        }
        None
    }

    /// Get the token index for a character position within a specific sequence
    pub fn char_to_token_with_sequence(&self, char_pos: usize, sequence_id: usize) -> Option<usize> {
        for (i, &(start, end)) in self.offsets.iter().enumerate() {
            if let Some(seq_id) = self.sequence_ids.get(i).and_then(|s| *s) {
                if seq_id == sequence_id && char_pos >= start && char_pos < end {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Get the character span for a token index
    /// Returns (start, end) character positions
    pub fn token_to_chars(&self, token_idx: usize) -> Option<(usize, usize)> {
        self.offsets.get(token_idx).copied()
    }

    /// Get the word index for a token
    pub fn token_to_word(&self, token_idx: usize) -> Option<usize> {
        self.word_ids.get(token_idx).and_then(|w| *w)
    }

    /// Get the sequence ID for a token
    pub fn token_to_sequence(&self, token_idx: usize) -> Option<usize> {
        self.sequence_ids.get(token_idx).and_then(|s| *s)
    }
}

impl Default for Encoding {
    fn default() -> Self {
        Self::new()
    }
}

/// Added token - a special token that can be added dynamically
#[derive(Debug, Clone)]
pub struct AddedToken {
    /// The token string
    pub content: String,
    /// Whether it's a special token
    pub special: bool,
    /// Whether it should match single word only
    pub single_word: bool,
    /// Left strip
    pub lstrip: bool,
    /// Right strip
    pub rstrip: bool,
    /// Whether to normalize
    pub normalized: bool,
}

impl AddedToken {
    /// Create a new added token
    pub fn new(content: &str, special: bool) -> Self {
        Self {
            content: content.to_string(),
            special,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: !special,  // Special tokens are not normalized
        }
    }

    /// Create a special token
    pub fn special(content: &str) -> Self {
        Self::new(content, true)
    }

    /// Create a normal added token
    pub fn normal(content: &str) -> Self {
        Self::new(content, false)
    }

    /// Set single word matching
    pub fn single_word(mut self, single_word: bool) -> Self {
        self.single_word = single_word;
        self
    }

    /// Set left strip
    pub fn lstrip(mut self, lstrip: bool) -> Self {
        self.lstrip = lstrip;
        self
    }

    /// Set right strip
    pub fn rstrip(mut self, rstrip: bool) -> Self {
        self.rstrip = rstrip;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_from_ids() {
        let enc = Encoding::from_ids(
            vec![1, 2, 3],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );
        assert_eq!(enc.len(), 3);
        assert_eq!(enc.attention_mask, vec![1, 1, 1]);
        assert_eq!(enc.type_ids, vec![0, 0, 0]);
        assert_eq!(enc.sequence_ids, vec![Some(0), Some(0), Some(0)]);
    }

    #[test]
    fn test_padding() {
        let mut enc = Encoding::from_ids(
            vec![1, 2],
            vec!["a".to_string(), "b".to_string()],
        );
        enc.pad(5, 0, "<pad>", false);
        assert_eq!(enc.len(), 5);
        assert_eq!(enc.attention_mask, vec![1, 1, 0, 0, 0]);
        assert_eq!(enc.sequence_ids, vec![Some(0), Some(0), None, None, None]);
    }

    #[test]
    fn test_truncation() {
        let mut enc = Encoding::from_ids(
            vec![1, 2, 3, 4, 5],
            vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string(), "e".to_string()],
        );
        enc.truncate(3);
        assert_eq!(enc.len(), 3);
        assert_eq!(enc.overflowing.len(), 1);
        assert_eq!(enc.overflowing[0].len(), 2);
    }

    #[test]
    fn test_added_token() {
        let token = AddedToken::special("<eos>").lstrip(true);
        assert!(token.special);
        assert!(token.lstrip);
    }

    #[test]
    fn test_char_to_token() {
        let mut enc = Encoding::from_ids(
            vec![1, 2, 3],
            vec!["hello".to_string(), " ".to_string(), "world".to_string()],
        );
        enc.offsets = vec![(0, 5), (5, 6), (6, 11)];

        assert_eq!(enc.char_to_token(0), Some(0)); // 'h'
        assert_eq!(enc.char_to_token(4), Some(0)); // 'o' in hello
        assert_eq!(enc.char_to_token(5), Some(1)); // space
        assert_eq!(enc.char_to_token(6), Some(2)); // 'w'
        assert_eq!(enc.char_to_token(11), None);   // past end
    }

    #[test]
    fn test_token_to_chars() {
        let mut enc = Encoding::from_ids(
            vec![1, 2],
            vec!["hello".to_string(), "world".to_string()],
        );
        enc.offsets = vec![(0, 5), (5, 10)];

        assert_eq!(enc.token_to_chars(0), Some((0, 5)));
        assert_eq!(enc.token_to_chars(1), Some((5, 10)));
        assert_eq!(enc.token_to_chars(2), None);
    }
}
