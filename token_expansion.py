import tiktoken
from tiktoken.load import load_tiktoken_bpe
import pandas as pd
import numpy as np
from typing import List

def load_vocab(path: str) -> pd.DataFrame:
    """Load vocab CSV with proper dtypes."""
    dtypes = {
        'token_id': np.int64,
        'token_bytes': str,
        'token_string': str,
        'token_length': np.int64,
        'is_special': bool,
        'is_printable': bool,
        'is_ascii': bool,
        'tokenizer_name': str
    }
    df_o200_vocab = pd.read_csv(path, dtype=dtypes)
    return df_o200_vocab


df_o200_vocab = load_vocab("o200k_vocab_detailed.csv")
enc = tiktoken.get_encoding("o200k_base")

def analyze_words(words: List[str], df_o200_vocab: pd.DataFrame=df_o200_vocab, enc=enc) -> None:
    """Tokenize each word and show ID, string, count, avg byte-len, and token/char ratios."""
    print("\n=== Per-Word Tokenization & Ratios ===")
    for word in words:
        token_ids = enc.encode(word)
        sub = df_o200_vocab.set_index('token_id').loc[token_ids]
        token_count = len(token_ids)
        char_count = len(word)
        avg_len = sub['token_length'].mean()
        tokens_per_char = token_count / char_count if char_count else float('nan')
        chars_per_token = char_count / token_count if token_count else float('nan')
        print(f"\n>> '{word}': {token_count} tokens, {char_count} chars, "
              f"avg {avg_len:.2f} bytes/token, "
              f"{tokens_per_char:.2f} tokens/char, {chars_per_token:.2f} chars/token")
        for tid, txt in zip(token_ids, sub['token_string']):
            print(f"   • ID {tid:5d}: {txt!r}")






phrase = "I am going to see her next week."
words = phrase.split()
tokens = enc.encode(phrase)



print(tokens)



