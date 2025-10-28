# build_features_min.py
#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Tuple, Optional

def read_wsj(path: Path, has_bio: bool):
    sents = []
    cur = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                # push current sentence if it has tokens
                if cur:
                    sents.append(cur)
                    cur = []
                # also push an *empty* sentence to mark the blank line
                sents.append([])
                continue
            parts = line.split("\t")
            if has_bio:
                if len(parts) < 3:
                    raise ValueError(f"Expected WORD\\tPOS\\tBIO, got: {line}")
                w, pos, bio = parts[0], parts[1], parts[2]
                cur.append((w, pos, bio))
            else:
                if len(parts) < 2:
                    raise ValueError(f"Expected WORD\\tPOS, got: {line}")
                w, pos = parts[0], parts[1]
                cur.append((w, pos, None))
    # flush any remaining sentence
    if cur:
        sents.append(cur)
    return sents
def read_wsj(path: Path, has_bio: bool):
    sents, cur = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                sents.append(cur); cur=[]
                continue
            parts = line.split("\t")
            if has_bio:
                w, pos, bio = parts[0], parts[1], parts[2]
                cur.append((w, pos, bio))
            else:
                w, pos = parts[0], parts[1]
                cur.append((w, pos, None))
    if cur: sents.append(cur)
    return sents
def read_wsj(path: Path, has_bio: bool):
    sents = []
    cur = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                # push current sentence if it has tokens
                if cur:
                    sents.append(cur)
                    cur = []
                # also push an *empty* sentence to mark the blank line
                sents.append([])
                continue
            parts = line.split("\t")
            if has_bio:
                if len(parts) < 3:
                    raise ValueError(f"Expected WORD\\tPOS\\tBIO, got: {line}")
                w, pos, bio = parts[0], parts[1], parts[2]
                cur.append((w, pos, bio))
            else:
                if len(parts) < 2:
                    raise ValueError(f"Expected WORD\\tPOS, got: {line}")
                w, pos = parts[0], parts[1]
                cur.append((w, pos, None))
    # flush any remaining sentence
    if cur:
        sents.append(cur)
    return sents

def token_feats(sent, i, include_prev_bio=True):
    w, pos, _ = sent[i]
    feats = [
        f"w={w}",
        f"wl={w.lower()}",
        f"pos={pos}",
    ]

    def tok(j):
        return sent[j] if 0 <= j < len(sent) else None

    # context ±1 only
    p1 = tok(i-1); n1 = tok(i+1)
    if p1 is None:
        feats.append("p1_BOS")
    else:
        pw, ppos, _ = p1
        feats.append(f"p1w={pw}")
        feats.append(f"p1pos={ppos}")
    if n1 is None:
        feats.append("n1_EOS")
    else:
        nw, npos, _ = n1
        feats.append(f"n1w={nw}")
        feats.append(f"n1pos={npos}")

    if include_prev_bio:
        feats.append("PrevBIO=@@")
    return feats

def write_features(sents, out_path: Path, is_training: bool):
    with out_path.open("w", encoding="utf-8") as out:
        for sent in sents:
            if not sent:
                out.write("\n"); continue
            for i, (w, pos, bio) in enumerate(sent):
                feats = token_feats(sent, i, include_prev_bio=True)
                fields = [w] + feats
                if is_training:
                    fields.append(bio)
                out.write("\t".join(fields) + "\n")

def main():
    if len(sys.argv) != 4:
        print("Usage: python build_features_min.py MODE INPUT OUTPUT")
        print("MODE: train (pos-chunk) | dev/test (pos)")
        sys.exit(1)
    mode, inp, out = sys.argv[1].lower(), Path(sys.argv[2]), Path(sys.argv[3])
    if mode == "train":
        sents = read_wsj(inp, has_bio=True)
        write_features(sents, out, is_training=True)
    elif mode in ("dev","test"):
        sents = read_wsj(inp, has_bio=False)
        write_features(sents, out, is_training=False)
    else:
        sys.exit("Mode must be train/dev/test")

    # sanity: line counts must match
    in_lines = sum(1 for _ in inp.open(encoding="utf-8"))
    out_lines = sum(1 for _ in out.open(encoding="utf-8"))
    if in_lines != out_lines:
        print(f"[WARN] line mismatch {in_lines} vs {out_lines}")
    else:
        print(f"[OK] {out} lines={out_lines}")

if __name__ == "__main__":
    main()
