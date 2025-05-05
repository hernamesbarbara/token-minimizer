#! /usr/bin/env python3
"""token2chars_ratio_analysis

Usage:
  token2chars_ratio_analysis CSV [-W WORDS | --words=WORDS]
  token2chars_ratio_analysis -h | --help
  token2chars_ratio_analysis --version

Arguments:
  CSV                   Path to o200k_vocab_detailed.csv

Options:
  -W WORDS --words=WORDS   Comma-separated words to tokenize and analyze
  -h --help                Show this help message and exit
  --version                Show version information and exit
"""
import sys
from docopt import docopt
import pandas as pd
import numpy as np
import tiktoken
import unicodedata
from typing import List, Tuple

__version__ = '0.1.0'


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
    return pd.read_csv(path, dtype=dtypes)


def summary_stats(df: pd.DataFrame) -> None:
    """Print overall token counts and percentages."""
    total = len(df)
    def pct(col): return df[col].sum() / total * 100
    print("\n=== Overall Vocabulary Stats ===")
    print(f"Total tokens:        {total}")
    print(f"Special tokens:      {df['is_special'].sum()} ({pct('is_special'):.2f}%)")
    print(f"Printable tokens:    {df['is_printable'].sum()} ({pct('is_printable'):.2f}%)")
    print(f"ASCII tokens:        {df['is_ascii'].sum()} ({pct('is_ascii'):.2f}%)")


def length_distribution(df: pd.DataFrame) -> None:
    """Print descriptive stats and extremes for token lengths."""
    lengths = df['token_length']
    desc = lengths.describe().astype(int)
    print("\n=== Token Length Distribution ===")
    print(desc.to_string())
    counts = lengths.value_counts().sort_index()
    print("\nLength → Count:")
    print(counts.to_string())
    top_long = df.nlargest(10, 'token_length')
    top_short = df.nsmallest(10, 'token_length')
    print("\nTop 10 longest tokens:")
    for _, r in top_long.iterrows():
        print(f"  ID {r.token_id:5d}: {r.token_length:2d} bytes → {r.token_string!r}")
    print("\nTop 10 shortest tokens:")
    for _, r in top_short.iterrows():
        print(f"  ID {r.token_id:5d}: {r.token_length:2d} bytes → {r.token_string!r}")


def categorize_tokens(df: pd.DataFrame) -> None:
    """Bucket tokens by unicode category or whitespace prefix."""
    df['token_string'] = df['token_string'].fillna('')
    def bucket(s: str) -> str:
        if not isinstance(s, str) or s == '':
            return 'empty'
        if s.startswith('Ġ'):
            return 'whitespace_prefixed'
        cat = unicodedata.category(s[0])
        if cat.startswith('L'): return 'letter'
        if cat.startswith('N'): return 'number'
        if cat.startswith('P'): return 'punctuation'
        if cat.startswith('Z'): return 'space'
        return 'other'
    df['bucket'] = df['token_string'].map(bucket)
    print("\n=== Unicode Buckets ===")
    print(df['bucket'].value_counts().to_string())


def analyze_words(words: List[str], df: pd.DataFrame, enc) -> None:
    """Tokenize each word and show ID, string, count, avg byte-len, and token/char ratios."""
    print("\n=== Per-Word Tokenization & Ratios ===")
    for w in words:
        ids = enc.encode(w)
        sub = df.set_index('token_id').loc[ids]
        token_count = len(ids)
        char_count = len(w)
        avg_len = sub['token_length'].mean()
        ratio_t2c = token_count / char_count if char_count else float('nan')
        ratio_c2t = char_count / token_count if token_count else float('nan')
        print(f"\n>> '{w}': {token_count} tokens, {char_count} chars, "
              f"avg {avg_len:.2f} bytes/token, "
              f"{ratio_t2c:.2f} tokens/char, {ratio_c2t:.2f} chars/token")
        for tid, txt in zip(ids, sub['token_string']):
            print(f"   • ID {tid:5d}: {txt!r}")


def main():
    args = docopt(__doc__, version=__version__)
    df = load_vocab(args['CSV'])
    enc = tiktoken.get_encoding("o200k_base")

    summary_stats(df)
    length_distribution(df)
    categorize_tokens(df)

    words = args.get('--words')
    if words:
        analyze_words(words.split(','), df, enc)


if __name__ == '__main__':
    main()
