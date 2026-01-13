"""
Train a 32K tokenizer on FineWeb-Edu dataset using INL-BPE dynamics.

Usage:
    pip install datasets
    python train_eduweb.py
"""

from datasets import load_dataset
from complexity_tokenizer import Trainer
import itertools

# Config
VOCAB_SIZE = 32000
NUM_SAMPLES = 100_000  # Number of samples to use (streaming)
OUTPUT_PATH = "tokenizer_32k.json"

# INL dynamics parameters
INL_ALPHA = 0.9   # momentum
INL_BETA = 0.3    # correction strength
INL_GATE = 0.5    # amplitude control

def main():
    print(f"Training {VOCAB_SIZE} vocab tokenizer on FineWeb-Edu...")
    print(f"Using {NUM_SAMPLES} samples")

    # Load dataset with streaming
    print("Loading dataset (streaming)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # Use sample subset
        split="train",
        streaming=True
    )

    # Create trainer with INL dynamics
    trainer = Trainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        min_word_length=1,
        inl_alpha=INL_ALPHA,
        inl_beta=INL_BETA,
        inl_gate=INL_GATE,
    )

    # Extract texts (limited to NUM_SAMPLES)
    print(f"Extracting {NUM_SAMPLES} text samples...")
    texts = [row["text"] for row in itertools.islice(ds, NUM_SAMPLES)]
    print(f"  Got {len(texts)} texts")

    # Train
    print("Training tokenizer with INL-BPE dynamics...")
    trainer.train_from_iterator(texts)

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    trainer.save(OUTPUT_PATH)

    print(f"Done! Vocab size: {trainer.vocab_size}, Merges: {trainer.num_merges}")

if __name__ == "__main__":
    main()
