#!/usr/bin/env python3
"""
Dev error analysis for BIO NG chunking.

Usage:
  python3 error_analysis.py WSJ_24.pos-chunk response.chunk WSJ_24.pos out_dir

Outputs:
  - out_dir/summary.txt
  - out_dir/top_error_tokens.tsv
  - out_dir/top_contexts.tsv
  - out_dir/boundary_errors.tsv
  - out_dir/conj_punct_errors.tsv

Notes:
  - Expects key and response to have identical line structure (as enforced by score script).
  - If DEV POS file is omitted, POS-sensitive stats will include 'NA' for POS fields.
"""
import os
import sys
from collections import Counter, defaultdict


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [ln.rstrip('\n') for ln in f]


def parse_bio(tag):
    # keep only B/I/O
    return tag.strip().split('-')[0]


def main():
    if len(sys.argv) not in (3, 4, 5):
        print("Usage: python3 error_analysis.py KEY.pos-chunk RESPONSE.chunk [DEV.pos] [OUT_DIR]", file=sys.stderr)
        sys.exit(1)

    key_path = sys.argv[1]
    resp_path = sys.argv[2]
    dev_pos_path = sys.argv[3] if len(sys.argv) >= 4 and not os.path.isdir(sys.argv[3]) else None
    out_dir = sys.argv[4] if len(sys.argv) == 5 else (sys.argv[3] if len(sys.argv) == 4 and os.path.isdir(sys.argv[3]) else 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    key = read_lines(key_path)
    resp = read_lines(resp_path)
    if len(key) != len(resp):
        print("length mismatch between key and response", file=sys.stderr)
        sys.exit(2)

    dev_pos = None
    if dev_pos_path:
        dev_pos = read_lines(dev_pos_path)
        if len(dev_pos) != len(key):
            print("length mismatch between key and DEV.pos", file=sys.stderr)
            dev_pos = None

    correct = 0
    incorrect = 0
    conf = Counter()  # (gold, pred)
    top_tok = Counter()  # (tok_lower, pos, gold, pred)
    top_ctx = Counter()  # (p_pos, tok, pos, n_pos, gold, pred)
    boundary = Counter()  # boundary-confusions only
    conj_punct = Counter()  # and/or/&/, specific tokens

    # helpers to get DEV.pos POS tag
    def get_pos(line):
        if not dev_pos:
            return 'NA'
        return (line.split('\t')[1] if '\t' in line and line else 'NA')

    for i in range(len(key)):
        k = key[i]
        r = resp[i]
        if k == '':
            if r != '':
                print(f"sentence break expected at line {i}", file=sys.stderr)
                sys.exit(3)
            continue
        k_tok, k_tag_full = k.split('\t')
        r_tok, r_tag_full = (r.split('\t') + [''])[:2]
        k_tag = parse_bio(k_tag_full)
        r_tag = parse_bio(r_tag_full)
        if k_tok != r_tok:
            print(f"Token mismatch at line {i}: key='{k_tok}' resp='{r_tok}'", file=sys.stderr)
            continue

        if k_tag == r_tag:
            correct += 1
        else:
            incorrect += 1
            conf[(k_tag, r_tag)] += 1

            # POS for current, prev, next from DEV.pos if available
            pos_cur = get_pos(dev_pos[i]) if dev_pos else 'NA'
            pos_prev = 'BOS'
            pos_next = 'EOS'
            if i > 0 and key[i-1] != '':
                pos_prev = get_pos(dev_pos[i-1]) if dev_pos else 'NA'
            if i < len(key)-1 and key[i+1] != '':
                pos_next = get_pos(dev_pos[i+1]) if dev_pos else 'NA'

            top_tok[(k_tok.lower(), pos_cur, k_tag, r_tag)] += 1
            top_ctx[(pos_prev, k_tok, pos_cur, pos_next, k_tag, r_tag)] += 1

            # boundary-only confusions
            if (k_tag, r_tag) in {('B', 'O'), ('O', 'B'), ('I', 'O'), ('O', 'I')}:
                boundary[(k_tok.lower(), pos_cur, k_tag, r_tag)] += 1

            if k_tok.lower() in {'and', 'or'} or k_tok in {',', '&'}:
                conj_punct[(k_tok, pos_cur, pos_prev, pos_next, k_tag, r_tag)] += 1

    # Write outputs
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as out:
        total = correct + incorrect
        acc = 100.0 * correct / total if total else 0.0
        # precision/recall/F on groups unavailable here; rely on score script for that
        out.write(f"Tags correct: {correct} / {total} (acc={acc:.2f}%)\n")
        out.write("Confusions (gold→pred), sorted by count:\n")
        for (g, p), c in conf.most_common():
            out.write(f"  {g}->{p}: {c}\n")

    def dump(counter, path, headers):
        with open(path, 'w', encoding='utf-8') as out:
            out.write('\t'.join(headers) + '\n')
            for key, c in counter.most_common():
                out.write('\t'.join(map(str, key)) + f"\t{c}\n")

    dump(top_tok, os.path.join(out_dir, 'top_error_tokens.tsv'),
         ['token', 'pos', 'gold', 'pred', 'count'])
    dump(top_ctx, os.path.join(out_dir, 'top_contexts.tsv'),
         ['prev_pos', 'token', 'pos', 'next_pos', 'gold', 'pred', 'count'])
    dump(boundary, os.path.join(out_dir, 'boundary_errors.tsv'),
         ['token', 'pos', 'gold', 'pred', 'count'])
    dump(conj_punct, os.path.join(out_dir, 'conj_punct_errors.tsv'),
         ['token', 'pos', 'prev_pos', 'next_pos', 'gold', 'pred', 'count'])

    print(f"Wrote analysis to: {out_dir}")


if __name__ == '__main__':
    main()

