"""
Complexity Tokenizer - Fast BPE tokenizer with HuggingFace compatibility.

A high-performance Byte Pair Encoding (BPE) tokenizer written in Rust
with Python bindings.

Example:
    >>> from complexity_tokenizer import Tokenizer
    >>> tok = Tokenizer.from_file("tokenizer.json")
    >>> ids = tok.encode("Hello world!")
    >>> print(tok.decode(ids))
    Hello world!
"""

from typing import Dict, List, Optional, Iterator, Iterable

# Import from Rust extension
from .complexity_tokenizer import Tokenizer as _Tokenizer
from .complexity_tokenizer import Trainer as _Trainer
from .complexity_tokenizer import __version__

__all__ = ["Tokenizer", "Trainer", "__version__"]


class Tokenizer(_Tokenizer):
    """
    Fast BPE tokenizer with HuggingFace compatibility.

    This tokenizer is implemented in Rust for maximum performance
    and is fully compatible with HuggingFace's tokenizer.json format.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        special_tokens: Dictionary of special token names to IDs.

    Example:
        >>> tok = Tokenizer.from_file("tokenizer.json")
        >>> tok.encode("Hello")
        [123, 456]
        >>> tok.decode([123, 456])
        'Hello'
    """

    @staticmethod
    def from_file(path: str) -> "Tokenizer":
        """
        Load tokenizer from a HuggingFace tokenizer.json file.

        Args:
            path: Path to the tokenizer.json file.

        Returns:
            Loaded Tokenizer instance.

        Raises:
            IOError: If the file cannot be read or parsed.
        """
        return _Tokenizer.from_file(path)

    @staticmethod
    def from_pretrained(repo_id: str) -> "Tokenizer":
        """
        Load tokenizer from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID (e.g., "Pacific-Prime/pacific-prime").

        Returns:
            Loaded Tokenizer instance.

        Raises:
            IOError: If the tokenizer cannot be downloaded.
        """
        return _Tokenizer.from_pretrained(repo_id)

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token IDs.
        """
        return super().encode(text)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts in parallel.

        Args:
            texts: List of input texts to tokenize.

        Returns:
            List of token ID lists.
        """
        return super().encode_batch(texts)

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text.
        """
        return super().decode(ids)

    def decode_batch(self, batch: List[List[int]]) -> List[str]:
        """
        Decode multiple token ID lists in parallel.

        Args:
            batch: List of token ID lists.

        Returns:
            List of decoded texts.
        """
        return super().decode_batch(batch)

    def token_to_id(self, token: str) -> Optional[int]:
        """
        Get the ID for a token.

        Args:
            token: Token string.

        Returns:
            Token ID if found, None otherwise.
        """
        return super().token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """
        Get the token string for an ID.

        Args:
            id: Token ID.

        Returns:
            Token string if found, None otherwise.
        """
        return super().id_to_token(id)

    def save(self, path: str) -> None:
        """
        Save tokenizer to file in HuggingFace format.

        Args:
            path: Output path for tokenizer.json.
        """
        super().save(path)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return super().vocab_size

    @property
    def special_tokens(self) -> Dict[str, int]:
        """Get special tokens mapping."""
        return super().special_tokens


class Trainer(_Trainer):
    """
    INL-BPE Trainer with dynamics-based merge selection.

    Unlike standard BPE which only uses frequency, this trainer uses
    INL dynamics to balance the vocabulary distribution.

    Example:
        >>> from complexity_tokenizer import Trainer
        >>> trainer = Trainer(vocab_size=32000)
        >>> trainer.train_from_iterator(texts)
        >>> trainer.save("tokenizer.json")
    """

    def train(self, files: List[str]) -> None:
        """
        Train tokenizer from text files.

        Args:
            files: List of paths to text files.
        """
        super().train(files)

    def train_from_iterator(self, texts: Iterable[str]) -> None:
        """
        Train tokenizer from an iterator of text strings.

        This is the recommended method for streaming datasets
        (e.g., HuggingFace datasets with streaming=True).

        Args:
            texts: Iterable of text strings.

        Example:
            >>> from datasets import load_dataset
            >>> ds = load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)
            >>> trainer.train_from_iterator(row["text"] for row in ds["train"])
        """
        # Convert iterator to list (required by Rust binding)
        super().train_from_iterator(list(texts))

    def save(self, path: str) -> None:
        """
        Save trained tokenizer to file in HuggingFace format.

        Args:
            path: Output path for tokenizer.json.
        """
        super().save(path)

    @property
    def vocab_size(self) -> int:
        """Get current vocabulary size."""
        return super().vocab_size

    @property
    def num_merges(self) -> int:
        """Get number of learned merges."""
        return super().num_merges
