#!/usr/bin/env python3
# dump_vocab.py
"""
Script to dump the vocabulary and metadata from the o200k_base tokenizer.
This will be used as a reference for building a custom tokenizer.
"""
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import pandas as pd
import numpy as np

# Load the raw token→rank mapping
mergeable_ranks = load_tiktoken_bpe(
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    expected_hash="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
)

# Flip it to rank → token for vocab dump
id_to_token = {rank: token for token, rank in mergeable_ranks.items()}

# Initialize the tokenizer to get special tokens
enc = tiktoken.get_encoding("o200k_base")

# Create a comprehensive vocabulary dictionary
vocab_data = []
for idx, token in id_to_token.items():
    # Convert bytes to readable string
    try:
        decoded_str = token.decode("utf-8")
    except UnicodeDecodeError:
        decoded_str = token.decode("utf-8", errors="replace")
    
    # Check if it's a special token
    is_special = False
    for special_token in enc._special_tokens.values():
        if token == special_token:
            is_special = True
            break
    
    vocab_data.append({
        "token_id": idx,
        "token_bytes": token.hex(),  # Store raw bytes as hex string
        "token_string": decoded_str,
        "token_length": len(token),
        "is_special": is_special,
        "is_printable": decoded_str.isprintable(),
        "is_ascii": decoded_str.isascii(),
    })

# Convert to DataFrame
df = pd.DataFrame(vocab_data)

# Sort by token_id to maintain order
df = df.sort_values("token_id")

# Save to CSV with detailed metadata
df.to_csv("o200k_vocab_detailed.csv", index=False)

# Print some statistics
print("\nVocabulary Statistics:")
print(f"Total tokens: {len(df)}")
print(f"Special tokens: {df['is_special'].sum()}")
print(f"Average token length: {df['token_length'].mean():.2f} bytes")
print(f"Printable tokens: {df['is_printable'].sum()}")
print(f"ASCII tokens: {df['is_ascii'].sum()}")

# Example: print first 20 tokens with their metadata
print("\nFirst 20 tokens:")
for _, row in df.head(20).iterrows():
    print(f"ID: {row['token_id']:4d} | Length: {row['token_length']:2d} | Special: {row['is_special']} | String: {row['token_string']}")




"""
🧠 How BPE Tokenization Works (Simplified)
Text is first converted to bytes, often using UTF-8.

Each byte or short string becomes an initial token candidate.

The BPE algorithm then:

Merges frequent adjacent byte sequences based on training

Builds longer-and-longer merges like:

b'g' + b'i' → b'gi'

b'gi' + b'r' → b'gir'

...

Eventually it stops merging when the next candidate is not in the vocabulary.

So for "giraffe", depending on the tokenizer's learned merges:

It might find b'gir' as a frequent subword

Then b'aff'

Then b'e' as leftovers

Each of those gets mapped to a separate token ID.


"""

enc = tiktoken.get_encoding("o200k_base")

word = "giraffe"

tokens = enc.encode(word, disallowed_special=())

print(f"{word} as tokens: {tokens}")
