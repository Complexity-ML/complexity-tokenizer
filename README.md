# Complexity Tokenizer

**Fast BPE tokenizer in Rust with HuggingFace compatibility.**

## Features

- **Fast**: Written in Rust, 10-20x faster than pure Python
- **Parallel**: Batch encoding/decoding uses all CPU cores (rayon)
- **HuggingFace Compatible**: Loads `tokenizer.json` directly
- **Hub Support**: `from_pretrained("repo/model")` works out of the box

## Installation

```bash
pip install complexity-tokenizer
```

## Usage

### Load from file

```python
from complexity_tokenizer import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")

# Encode
ids = tok.encode("Hello, world!")
print(ids)  # [123, 456, 789, ...]

# Decode
text = tok.decode(ids)
print(text)  # "Hello, world!"
```

### Load from HuggingFace Hub

```python
from complexity_tokenizer import Tokenizer

tok = Tokenizer.from_pretrained("Pacific-Prime/pacific-prime")
ids = tok.encode("Bonjour le monde!")
```

### Batch processing (parallel)

```python
texts = ["Hello", "World", "Foo", "Bar"] * 1000

# Uses all CPU cores
ids_batch = tok.encode_batch(texts)
texts_back = tok.decode_batch(ids_batch)
```

### Special tokens

```python
print(tok.vocab_size)        # 100000
print(tok.special_tokens)    # {'eos': 0, 'bos': 2, 'pad': 1, ...}

# Token <-> ID conversion
tok.token_to_id("<s>")       # 2
tok.id_to_token(2)           # "<s>"
```

## Performance

| Operation | complexity-tokenizer (Rust) | tokenizers (HF) | Python |
|-----------|---------------------|-----------------|--------|
| Encode 1K texts | ~5ms | ~8ms | ~100ms |
| Decode 1K texts | ~3ms | ~5ms | ~80ms |
| Batch encode 10K | ~20ms | ~40ms | ~1s |

## Build from source

Requires Rust 1.70+ and maturin:

```bash
# Install maturin
pip install maturin

# Build and install
maturin develop --release

# Or build wheel
maturin build --release
```

## License

Apache-2.0
