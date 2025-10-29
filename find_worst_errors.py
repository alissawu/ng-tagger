#!/usr/bin/env python3
"""
Find the actual contexts where we're making errors on dev set.
Show specific sentences/patterns we get wrong.
"""

from pathlib import Path
from collections import Counter

def read_file_lines(path):
    with open(path) as f:
        return [line.strip() for line in f]

def main():
    # Read files
    pos_lines = read_file_lines("WSJ_24.pos")  # WORD\tPOS
    gold_bio_lines = read_file_lines("WSJ_24.pos-chunk")  # WORD\tBIO
    pred_bio_lines = read_file_lines("response.chunk")  # WORD\tBIO

    # Combine into sentences
    sents = []
    cur_sent = []

    for i in range(len(pos_lines)):
        if pos_lines[i] == "":
            if cur_sent:
                sents.append(cur_sent)
            cur_sent = []
        else:
            pos_parts = pos_lines[i].split("\t")
            gold_parts = gold_bio_lines[i].split("\t")
            pred_parts = pred_bio_lines[i].split("\t")

            if len(pos_parts) >= 2 and len(gold_parts) >= 2 and len(pred_parts) >= 2:
                word = pos_parts[0]
                pos = pos_parts[1]
                gold_bio = gold_parts[1]
                pred_bio = pred_parts[1]
                cur_sent.append((word, pos, gold_bio, pred_bio))

    if cur_sent:
        sents.append(cur_sent)

    # Find sentences with most errors
    sent_errors = []
    for idx, sent in enumerate(sents):
        errors = sum(1 for _, _, g, p in sent if g != p)
        if errors > 0:
            sent_errors.append((idx, errors, sent))

    # Sort by error count
    sent_errors.sort(key=lambda x: x[1], reverse=True)

    print("="*80)
    print(f"TOP 15 WORST SENTENCES ({len(sent_errors)} total sentences with errors)")
    print("="*80)
    print()

    for rank, (idx, err_count, sent) in enumerate(sent_errors[:15], 1):
        print(f"#{rank}. Sentence {idx} - {err_count} errors:")
        print()

        for i, (word, pos, gold, pred) in enumerate(sent):
            if gold != pred:
                print(f"  [{i:2d}] {word:20s} {pos:8s}  Gold: {gold:6s}  Pred: {pred:6s}  <-- ERROR")
            else:
                print(f"  [{i:2d}] {word:20s} {pos:8s}  {gold:6s}")
        print()
        print("-"*80)
        print()

    # Error pattern analysis
    print("\n" + "="*80)
    print("MOST COMMON ERROR PATTERNS")
    print("="*80)

    error_contexts = Counter()
    for idx, err_count, sent in sent_errors:
        for i, (word, pos, gold, pred) in enumerate(sent):
            if gold != pred:
                prev_pos = sent[i-1][1] if i > 0 else "BOS"
                next_pos = sent[i+1][1] if i+1 < len(sent) else "EOS"
                error_contexts[(word, pos, prev_pos, next_pos, gold, pred)] += 1

    print(f"\n{'Word':<15} {'POS':<8} {'PrevPOS':<8} {'NextPOS':<8} {'Gold':<8} {'Pred':<8} {'Count'}")
    print("-"*80)
    for (word, pos, prev_pos, next_pos, gold, pred), count in error_contexts.most_common(40):
        print(f"{word:<15} {pos:<8} {prev_pos:<8} {next_pos:<8} {gold:<8} {pred:<8} {count}")

if __name__ == "__main__":
    main()
