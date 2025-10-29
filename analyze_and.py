#!/usr/bin/env python3
"""Analyze 'and' contexts in training data to understand I-NP vs O patterns."""

from pathlib import Path
from collections import Counter

def read_wsj(path):
    sents = []
    cur = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                w, pos, bio = parts[0], parts[1], parts[2]
                cur.append((w, pos, bio))
    if cur:
        sents.append(cur)
    return sents

def analyze_and(sents):
    and_as_I = []
    and_as_O = []

    for sent in sents:
        for i, (w, pos, bio) in enumerate(sent):
            if w.lower() in {"and", "or"} and pos == "CC":
                # Get context
                p2 = sent[i-2] if i >= 2 else None
                p1 = sent[i-1] if i >= 1 else None
                n1 = sent[i+1] if i+1 < len(sent) else None
                n2 = sent[i+2] if i+2 < len(sent) else None

                context = {
                    'word': w,
                    'p2': p2,
                    'p1': p1,
                    'n1': n1,
                    'n2': n2,
                    'bio': bio
                }

                if bio == "I-NP":
                    and_as_I.append(context)
                elif bio == "O":
                    and_as_O.append(context)

    return and_as_I, and_as_O

def print_contexts(contexts, label, limit=20):
    print(f"\n{'='*80}")
    print(f"{label} (showing {min(limit, len(contexts))} of {len(contexts)} total)")
    print('='*80)

    for ctx in contexts[:limit]:
        p2_str = f"{ctx['p2'][0]}/{ctx['p2'][1]}/{ctx['p2'][2]}" if ctx['p2'] else "BOS"
        p1_str = f"{ctx['p1'][0]}/{ctx['p1'][1]}/{ctx['p1'][2]}" if ctx['p1'] else "BOS"
        n1_str = f"{ctx['n1'][0]}/{ctx['n1'][1]}/{ctx['n1'][2]}" if ctx['n1'] else "EOS"
        n2_str = f"{ctx['n2'][0]}/{ctx['n2'][1]}/{ctx['n2'][2]}" if ctx['n2'] else "EOS"

        print(f"  {p2_str:25} {p1_str:25} [{ctx['word']}/CC/{ctx['bio']}] {n1_str:25} {n2_str:25}")

def analyze_patterns(contexts):
    """Look for patterns in the contexts."""
    n1_pos = Counter()
    n1_bio = Counter()
    p1_pos = Counter()
    p1_bio = Counter()

    for ctx in contexts:
        if ctx['n1']:
            n1_pos[ctx['n1'][1]] += 1
            n1_bio[ctx['n1'][2]] += 1
        if ctx['p1']:
            p1_pos[ctx['p1'][1]] += 1
            p1_bio[ctx['p1'][2]] += 1

    return {
        'n1_pos': n1_pos,
        'n1_bio': n1_bio,
        'p1_pos': p1_pos,
        'p1_bio': p1_bio
    }

def main():
    train_path = Path("WSJ_02-21.pos-chunk")

    print("Reading training data...")
    sents = read_wsj(train_path)

    print("Analyzing 'and/or' contexts...")
    and_as_I, and_as_O = analyze_and(sents)

    print_contexts(and_as_I, "'and/or' as I-NP", limit=30)
    print_contexts(and_as_O, "'and/or' as O", limit=30)

    print("\n" + "="*80)
    print("PATTERN ANALYSIS: 'and/or' as I-NP")
    print("="*80)
    patterns_I = analyze_patterns(and_as_I)
    print(f"\nNext word POS (top 10):")
    for pos, count in patterns_I['n1_pos'].most_common(10):
        print(f"  {pos:10} {count:5} ({100*count/len(and_as_I):.1f}%)")

    print(f"\nNext word BIO (top 5):")
    for bio, count in patterns_I['n1_bio'].most_common(5):
        print(f"  {bio:10} {count:5} ({100*count/len(and_as_I):.1f}%)")

    print("\n" + "="*80)
    print("PATTERN ANALYSIS: 'and/or' as O")
    print("="*80)
    patterns_O = analyze_patterns(and_as_O)
    print(f"\nNext word POS (top 10):")
    for pos, count in patterns_O['n1_pos'].most_common(10):
        print(f"  {pos:10} {count:5} ({100*count/len(and_as_O):.1f}%)")

    print(f"\nNext word BIO (top 5):")
    for bio, count in patterns_O['n1_bio'].most_common(5):
        print(f"  {bio:10} {count:5} ({100*count/len(and_as_O):.1f}%)")

if __name__ == "__main__":
    main()
