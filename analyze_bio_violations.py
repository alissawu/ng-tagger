#!/usr/bin/env python3
"""
Analyze BIO constraint violations in predictions.
Specifically, finds illegal O -> I-NP transitions and shows context.
"""

import sys
from pathlib import Path
from collections import Counter

def read_pos_file(path):
    """Read .pos file (word + POS)."""
    sentences = []
    current = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                sentences.append([])  # blank line marker
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                current.append((parts[0], parts[1]))
    if current:
        sentences.append(current)
    return sentences

def read_bio_file(path):
    """Read .pos-chunk file (word + POS + BIO) or response.chunk (word + ... + BIO)."""
    sentences = []
    current = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                sentences.append([])  # blank line marker
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                # Last field is BIO tag
                current.append(parts[-1])
    if current:
        sentences.append(current)
    return sentences

def main():
    pos_path = Path('WSJ_24.pos')
    gold_path = Path('WSJ_24.pos-chunk')
    pred_path = Path('response.chunk')

    if not pos_path.exists() or not gold_path.exists() or not pred_path.exists():
        print("Error: Need WSJ_24.pos, WSJ_24.pos-chunk, and response.chunk")
        sys.exit(1)

    print("Reading files...")
    pos_sents = read_pos_file(pos_path)
    gold_sents = read_bio_file(gold_path)
    pred_sents = read_bio_file(pred_path)

    # Find O -> I-NP violations
    violations = []

    sent_idx = 0
    for pos_sent, gold_sent, pred_sent in zip(pos_sents, gold_sents, pred_sents):
        if not pos_sent:  # blank line
            sent_idx += 1
            continue

        prev_pred = '@@'  # Start of sentence marker
        for i, ((word, pos), gold_bio, pred_bio) in enumerate(zip(pos_sent, gold_sent, pred_sent)):
            # Check for O -> I-NP violation
            if prev_pred == 'O' and pred_bio == 'I-NP':
                violations.append({
                    'sent_idx': sent_idx,
                    'token_idx': i,
                    'word': word,
                    'pos': pos,
                    'gold': gold_bio,
                    'pred': pred_bio,
                    'sentence': pos_sent,
                    'gold_tags': gold_sent,
                    'pred_tags': pred_sent
                })
            prev_pred = pred_bio

        sent_idx += 1

    print(f"\n{'='*80}")
    print(f"BIO CONSTRAINT VIOLATIONS: O -> I-NP")
    print(f"{'='*80}\n")
    print(f"Found {len(violations)} violations\n")

    # Show first 30 violations with context
    for idx, v in enumerate(violations[:30], 1):
        sent = v['sentence']
        gold_tags = v['gold_tags']
        pred_tags = v['pred_tags']
        tok_idx = v['token_idx']

        print(f"#{idx}. Sentence {v['sent_idx']}, Token {tok_idx}")
        print(f"   Violation: '{v['word']}' Gold={v['gold']} Pred=I-NP (but prev was O)")
        print()

        # Show context (±3 tokens)
        start = max(0, tok_idx - 3)
        end = min(len(sent), tok_idx + 4)

        for i in range(start, end):
            word, pos = sent[i]
            gold = gold_tags[i]
            pred = pred_tags[i]

            marker = "   >>>" if i == tok_idx else "   ✓  " if gold == pred else "   ✗  "
            status = "" if gold == pred else f"  <<< ERROR: Gold={gold} Pred={pred}"

            print(f"{marker} {word:20s} {pos:10s} Gold:{gold:8s} Pred:{pred:8s}{status}")
        print()

    if len(violations) > 30:
        print(f"... and {len(violations) - 30} more violations\n")

    # Analyze patterns
    print(f"{'='*80}")
    print("PATTERN ANALYSIS")
    print(f"{'='*80}\n")

    # Count POS of violating tokens
    pos_counter = Counter(v['pos'] for v in violations)
    print("POS tags of tokens with O->I-NP violations:")
    for pos, count in pos_counter.most_common(10):
        print(f"  {pos:10s} {count:4d}")
    print()

    # Count what the gold tag actually was
    gold_counter = Counter(v['gold'] for v in violations)
    print("What should these tokens have been? (Gold tags):")
    for gold, count in gold_counter.most_common():
        print(f"  {gold:10s} {count:4d}")
    print()

    # Check what comes after the violation
    next_pred_counter = Counter()
    for v in violations:
        tok_idx = v['token_idx']
        pred_tags = v['pred_tags']
        if tok_idx + 1 < len(pred_tags):
            next_pred_counter[pred_tags[tok_idx + 1]] += 1

    print("What comes AFTER the I-NP violation? (Next predicted tag):")
    for tag, count in next_pred_counter.most_common():
        print(f"  {tag:10s} {count:4d}")
    print()

    print(f"{'='*80}")
    print("RECOMMENDATION:")
    print(f"{'='*80}")
    print("Look at the examples above to decide:")
    print("  Option 1: Always change I-NP -> B-NP (start new phrase)")
    print("  Option 2: Always change I-NP -> O (no phrase)")
    print("  Option 3: Context-based rules (e.g., based on POS or next tag)")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
