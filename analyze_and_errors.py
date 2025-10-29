#!/usr/bin/env python3
"""
Analyze specific 'and' errors on dev set to understand the pattern.
Show gold vs predicted, with full context.
"""

from pathlib import Path
from collections import Counter

def read_file_lines(path):
    with open(path) as f:
        return [line.strip() for line in f]

def main():
    # Read files
    pos_lines = read_file_lines("WSJ_24.pos")
    gold_lines = read_file_lines("WSJ_24.pos-chunk")
    pred_lines = read_file_lines("response.chunk")

    # Build sentences
    sents = []
    cur_sent = []

    for i in range(len(pos_lines)):
        if pos_lines[i] == "":
            if cur_sent:
                sents.append(cur_sent)
            cur_sent = []
        else:
            pos_parts = pos_lines[i].split("\t")
            gold_parts = gold_lines[i].split("\t")
            pred_parts = pred_lines[i].split("\t")

            if len(pos_parts) >= 2 and len(gold_parts) >= 2 and len(pred_parts) >= 2:
                word = pos_parts[0]
                pos = pos_parts[1]
                gold_bio = gold_parts[1]
                pred_bio = pred_parts[1]
                cur_sent.append((word, pos, gold_bio, pred_bio))

    if cur_sent:
        sents.append(cur_sent)

    # Find ALL "and" errors
    and_errors = []

    for sent_idx, sent in enumerate(sents):
        for i, (word, pos, gold, pred) in enumerate(sent):
            if word.lower() in {"and", "or"} and gold != pred:
                # Get context window
                context_start = max(0, i-3)
                context_end = min(len(sent), i+4)
                context = sent[context_start:context_end]

                and_errors.append({
                    'sent_idx': sent_idx,
                    'token_idx': i,
                    'word': word,
                    'pos': pos,
                    'gold': gold,
                    'pred': pred,
                    'context': context,
                    'context_offset': i - context_start
                })

    print("="*80)
    print(f"FOUND {len(and_errors)} 'AND/OR' ERRORS ON DEV SET")
    print("="*80)
    print()

    # Group by error type
    by_pattern = Counter()
    for err in and_errors:
        pattern = f"Gold:{err['gold']} → Pred:{err['pred']}"
        by_pattern[pattern] += 1

    print("ERROR PATTERNS:")
    for pattern, count in by_pattern.most_common():
        print(f"  {pattern:30s} {count:3d} errors")
    print()

    # Show first 20 examples with full context
    print("="*80)
    print("FIRST 20 'AND/OR' ERRORS WITH CONTEXT")
    print("="*80)
    print()

    for rank, err in enumerate(and_errors[:20], 1):
        print(f"#{rank}. Sentence {err['sent_idx']}, Token {err['token_idx']}")
        print(f"   Error: '{err['word']}' Gold={err['gold']} Pred={err['pred']}")
        print()

        # Print context with alignment
        for j, (w, p, g, pr) in enumerate(err['context']):
            if j == err['context_offset']:
                print(f"   >>> {w:20s} {p:8s}  Gold:{g:6s}  Pred:{pr:6s}  <<< ERROR")
            else:
                marker = "✓" if g == pr else "✗"
                print(f"   {marker:3s} {w:20s} {p:8s}  Gold:{g:6s}  Pred:{pr:6s}")
        print()
        print("-"*80)
        print()

    # Analyze: What's BEFORE and AFTER the "and" errors?
    print("="*80)
    print("CONTEXT ANALYSIS: What surrounds 'and' errors?")
    print("="*80)
    print()

    # Previous token analysis
    prev_gold_counter = Counter()
    prev_pred_counter = Counter()
    next_gold_counter = Counter()
    next_pred_counter = Counter()
    prev_pos_counter = Counter()
    next_pos_counter = Counter()

    for err in and_errors:
        sent_idx = err['sent_idx']
        tok_idx = err['token_idx']
        sent = sents[sent_idx]

        if tok_idx > 0:
            prev_gold_counter[sent[tok_idx-1][2]] += 1  # prev gold BIO
            prev_pred_counter[sent[tok_idx-1][3]] += 1  # prev pred BIO
            prev_pos_counter[sent[tok_idx-1][1]] += 1   # prev POS

        if tok_idx < len(sent) - 1:
            next_gold_counter[sent[tok_idx+1][2]] += 1  # next gold BIO
            next_pred_counter[sent[tok_idx+1][3]] += 1  # next pred BIO
            next_pos_counter[sent[tok_idx+1][1]] += 1   # next POS

    print("PREVIOUS TOKEN (before 'and' error):")
    print(f"  Gold BIO: {prev_gold_counter.most_common(5)}")
    print(f"  Pred BIO: {prev_pred_counter.most_common(5)}")
    print(f"  POS:      {prev_pos_counter.most_common(5)}")
    print()

    print("NEXT TOKEN (after 'and' error):")
    print(f"  Gold BIO: {next_gold_counter.most_common(5)}")
    print(f"  Pred BIO: {next_pred_counter.most_common(5)}")
    print(f"  POS:      {next_pos_counter.most_common(5)}")
    print()

    # Key question: Are we systematically over-predicting I-NP or O?
    gold_I = sum(1 for e in and_errors if e['gold'] == 'I-NP')
    gold_O = sum(1 for e in and_errors if e['gold'] == 'O')
    pred_I = sum(1 for e in and_errors if e['pred'] == 'I-NP')
    pred_O = sum(1 for e in and_errors if e['pred'] == 'O')

    print("="*80)
    print("SYSTEMATIC BIAS?")
    print("="*80)
    print(f"When we're WRONG about 'and':")
    print(f"  Gold says I-NP: {gold_I:3d} times")
    print(f"  Gold says O:    {gold_O:3d} times")
    print(f"  We predict I-NP: {pred_I:3d} times")
    print(f"  We predict O:    {pred_O:3d} times")
    print()

    if pred_I > gold_I:
        print(f"→ We OVER-predict I-NP for 'and' by {pred_I - gold_I} errors")
    elif pred_O > gold_O:
        print(f"→ We OVER-predict O for 'and' by {pred_O - gold_O} errors")
    print()

if __name__ == "__main__":
    main()
