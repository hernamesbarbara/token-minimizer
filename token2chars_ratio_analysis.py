#! /usr/bin/env python3
"""token2chars_ratio_analysis

Usage:
  token2chars_ratio_analysis CSV [-W WORDS | --words=WORDS] [--words_file=FILE]
  token2chars_ratio_analysis -h | --help
  token2chars_ratio_analysis --version

Arguments:
  CSV                   Path to o200k_vocab_detailed.csv

Options:
  -W WORDS --words=WORDS        Comma-separated words to tokenize and analyze
  --words_file=FILE             Path or URL to newline-separated words file
  -h --help                     Show this help message and exit
  --version                     Show version information and exit
"""
import sys
from docopt import docopt
import pandas as pd
import numpy as np
import tiktoken
import unicodedata
import urllib.request
import urllib.parse
import os
from typing import List

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
    df_o200_vocab = pd.read_csv(path, dtype=dtypes)
    return df_o200_vocab


def summary_stats(df_o200_vocab: pd.DataFrame) -> None:
    """Print overall token counts and percentages for o200k_base vocab."""
    total = len(df_o200_vocab)
    def pct(col): return df_o200_vocab[col].sum() / total * 100
    print("\n=== Overall Vocabulary Stats ===")
    print(f"Total tokens:        {total}")
    print(f"Special tokens:      {df_o200_vocab['is_special'].sum()} ({pct('is_special'):.2f}%)")
    print(f"Printable tokens:    {df_o200_vocab['is_printable'].sum()} ({pct('is_printable'):.2f}%)")
    print(f"ASCII tokens:        {df_o200_vocab['is_ascii'].sum()} ({pct('is_ascii'):.2f}%)")


def length_distribution(df_o200_vocab: pd.DataFrame) -> None:
    """Print descriptive stats and extremes for token lengths for o200k_base vocab."""
    lengths = df_o200_vocab['token_length']
    desc = lengths.describe().astype(int)
    print("\n=== Token Length Distribution ===")
    print(desc.to_string())
    counts = lengths.value_counts().sort_index()
    print("\nLength → Count:")
    print(counts.to_string())
    top_long = df_o200_vocab.nlargest(10, 'token_length')
    top_short = df_o200_vocab.nsmallest(10, 'token_length')
    print("\nTop 10 longest tokens:")
    for _, row in top_long.iterrows():
        print(f"  ID {row.token_id:5d}: {row.token_length:2d} bytes → {row.token_string!r}")
    print("\nTop 10 shortest tokens:")
    for _, row in top_short.iterrows():
        print(f"  ID {row.token_id:5d}: {row.token_length:2d} bytes → {row.token_string!r}")


def categorize_tokens(df_o200_vocab: pd.DataFrame) -> None:
    """Bucket tokens by unicode category or whitespace prefix for o200k_base vocab."""
    df_o200_vocab['token_string'] = df_o200_vocab['token_string'].fillna('')
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
    df_o200_vocab['bucket'] = df_o200_vocab['token_string'].map(bucket)
    print("\n=== Unicode Buckets ===")
    print(df_o200_vocab['bucket'].value_counts().to_string())


def analyze_words(words: List[str], df_o200_vocab: pd.DataFrame, enc) -> None:
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


def main():
    args = docopt(__doc__, version=__version__)
    df_o200_vocab = load_vocab(args['CSV'])
    enc = tiktoken.get_encoding("o200k_base")

    summary_stats(df_o200_vocab)
    length_distribution(df_o200_vocab)
    categorize_tokens(df_o200_vocab)

    words: List[str] = []
    words_file = args.get('--words_file')
    if words_file:
        if words_file.startswith(('http://', 'https://')):
            print(f"Downloading word list from URL: {words_file}")
            with urllib.request.urlopen(words_file) as resp:
                text = resp.read().decode('utf-8')
            parsed = urllib.parse.urlparse(words_file)
            filename = os.path.basename(parsed.path) or 'words.txt'
            with open(filename, 'w', encoding='utf-8') as out:
                out.write(text)
            print(f"Saved downloaded word file to: {filename}")
            lines = text.splitlines()
        else:
            print(f"Loading word list from local file: {words_file}")
            with open(words_file, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
        words = [line.strip() for line in lines if line.strip()]
        print(f"Loaded {len(words)} words")
    elif args.get('--words'):
        words = args['--words'].split(',')
        print(f"Using {len(words)} provided words")

    if words:
        total = len(words)
        print(f"\nAnalyzing {total} words...")
        records = []
        for idx, word in enumerate(words, start=1):
            if total > 100 and idx % 100 == 0:
                print(f"  Processed {idx}/{total} words")
            token_ids = enc.encode(word)
            sub = df_o200_vocab.set_index('token_id').loc[token_ids]
            tc = len(token_ids)
            cc = len(word)
            avg_len = sub['token_length'].mean()
            t2c = tc / cc if cc else float('nan')
            c2t = cc / tc if tc else float('nan')
            records.append({
                'word': word,
                'token_count': tc,
                'char_count': cc,
                'avg_byte_length': avg_len,
                'tokens_per_char': t2c,
                'chars_per_token': c2t
            })
        out_df = pd.DataFrame(records)
        if total > 5:
            report_file = 'word_summary.csv'
            out_df.to_csv(report_file, index=False)
            print(f"Saved summary for {total} words to {report_file}")
        else:
            print("\n=== Word Summary ===")
            print(out_df.to_string(index=False))


if __name__ == '__main__':
    main()
