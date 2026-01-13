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

# Import from Rust extension - re-export directly
from .complexity_tokenizer import Tokenizer, Trainer, __version__

__all__ = ["Tokenizer", "Trainer", "__version__"]
