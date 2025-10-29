#!/usr/bin/env python3
import sys, re
from pathlib import Path
from typing import List, Tuple, Optional

DET_WORDS = {"the","a","an","this","that","these","those"}
PREP_WORDS = {"of","in","on","at","for","with","by","to","from","as","into","over","under"}

# --- utilities ---------------------------------------------------------------

def read_wsj(path: Path, has_bio: bool):
    """Read WSJ files preserving blank lines as sentence separators.
    Returns a list of sentences; each sentence is a list of (w,pos,bio|None)."""
    sents, cur = [], []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "":
                if cur:
                    sents.append(cur)
                    cur = []
                sents.append([])  # represent the blank line explicitly
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
    if cur:
        sents.append(cur)
    return sents

_digit_re = re.compile(r"\d")
_number_re = re.compile(r"^[\+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_punct_re = re.compile(r"^[^\w\s]+$")  # all non-alnum/underscore

def is_punct(w: str) -> bool:
    return bool(_punct_re.match(w))

def has_digit(w: str) -> bool:
    return bool(_digit_re.search(w))

def is_number(w: str) -> bool:
    return bool(_number_re.match(w))

def word_shape(w: str) -> str:
    """Canonically collapse letters/digits to X/x/d with simple runs and keep hyphens/apostrophes."""
    out = []
    prev = None
    for ch in w:
        if ch.isupper():
            cur = "X"
        elif ch.islower():
            cur = "x"
        elif ch.isdigit():
            cur = "d"
        elif ch in "-'/":
            cur = ch
        else:
            cur = "_"
        # compress runs: X+, x+, d+
        if cur in {"X","x","d"} and prev == cur:
            continue
        out.append(cur)
        prev = cur
    return "".join(out)

def pre(w: str, n: int) -> str:
    wl = w.lower()
    return wl[:n] if len(wl) >= n else wl

def suf(w: str, n: int) -> str:
    wl = w.lower()
    return wl[-n:] if len(wl) >= n else wl

def coarse_pos(pos: str) -> str:
    """Coarse mapping tuned for NP chunking."""
    if pos.startswith("NN"): return "N"
    if pos.startswith("JJ"): return "J"
    if pos.startswith("RB"): return "R"
    if pos.startswith("VB"): return "V"
    if pos == "DT": return "DT"
    if pos in {"PRP","PRP$"}: return "PRP"
    if pos in {"IN"}: return "IN"
    if pos in {"CD"}: return "CD"
    if pos in {"CC"}: return "CC"
    if pos in {"TO"}: return "TO"
    if pos in {"POS"}: return "POS"
    if pos.startswith("W"): return "W"
    if pos in {",",".",":",";","-LRB-","-RRB-","``","''"}: return "PUNCT"
    return pos  # leave others as-is

def safe_tok(sent, i) -> Optional[Tuple[str,str,Optional[str]]]:
    return sent[i] if 0 <= i < len(sent) else None

def nouny(pos: str) -> bool:
    return pos.startswith("NN") or pos in {"PRP","PRP$","CD"}

# --- feature extraction ------------------------------------------------------

def token_feats(sent, i, include_prev_bio=True) -> List[str]:
    w, pos, _ = sent[i]
    wl = w.lower()

    feats = [
        f"w={w}",
        f"wl={wl}",
        f"pos={pos}",
        f"cpos={coarse_pos(pos)}",
        f"shape={word_shape(w)}",
        f"suf2={suf(w,2)}",
        f"suf3={suf(w,3)}",
        f"isTitle={'true' if (w[:1].isupper() and wl[1:]!=wl[1:].upper()) else 'false'}",
        f"isAllCaps={'true' if w.isupper() and any(c.isalpha() for c in w) else 'false'}",
        f"hasHyphen={'true' if '-' in w else 'false'}",
        f"hasDigit={'true' if has_digit(w) else 'false'}",
        f"isNumber={'true' if is_number(w) else 'false'}",
        f"isDet={'true' if wl in DET_WORDS else 'false'}",
        f"isPrep={'true' if wl in PREP_WORDS else 'false'}",
    ]

    # Context tokens
    p2 = safe_tok(sent, i-2)
    p1 = safe_tok(sent, i-1)
    n1 = safe_tok(sent, i+1)
    n2 = safe_tok(sent, i+2)

    # BOS/EOS features (distance 1 and 2)
    if p1 is None: feats.append("BOS")
    if p2 is None: feats.append("BOS2")
    if n1 is None: feats.append("EOS")
    if n2 is None: feats.append("EOS2")

    # Neighbor word (±1) light identity
    if p1 is not None:
        pw, ppos, _ = p1
        feats.append(f"p1w={pw}")
        feats.append(f"p1wl={pw.lower()}")
        feats.append(f"p1pos={ppos}")
        feats.append(f"p1cpos={coarse_pos(ppos)}")
    if n1 is not None:
        nw, npos, _ = n1
        feats.append(f"n1w={nw}")
        feats.append(f"n1pos={npos}")
        feats.append(f"n1cpos={coarse_pos(npos)}")

    # POS/cPOS window ±2
    if p2 is not None:
        _, p2pos, _ = p2
        feats.append(f"p2pos={p2pos}")
        feats.append(f"p2cpos={coarse_pos(p2pos)}")
    if n2 is not None:
        _, n2pos, _ = n2
        feats.append(f"n2pos={n2pos}")
        

    # POS n-grams (compact, high impact)
    if p1 is not None:
        feats.append(f"p1pos+pos={p1[1]}+{pos}")
    if n1 is not None:
        feats.append(f"pos+n1pos={pos}+{n1[1]}")
    if p1 is not None and n1 is not None:
        feats.append(f"p1pos+pos+n1pos={p1[1]}+{pos}+{n1[1]}")
    if p2 is not None and p1 is not None:
        feats.append(f"p2pos+p1pos={p2[1]}+{p1[1]}")
        feats.append(f"p2pos+p1pos+pos={p2[1]}+{p1[1]}+{pos}")
    if n1 is not None and n2 is not None:
        feats.append(f"n1pos+n2pos={n1[1]}+{n2[1]}")
        feats.append(f"pos+n1pos+n2pos={pos}+{n1[1]}+{n2[1]}")

    # Proper-noun runs (common NP)
    if (pos == "NNP" and ((p1 is not None and p1[1] == "NNP") or (n1 is not None and n1[1] == "NNP"))):
        feats.append("pnRun=true")

    # Sequence dependency hooks (MEMM style)
    if include_prev_bio:
        feats.append("PrevBIO=@@")
        # Conjunctions with POS cues
        if p1 is not None:
            feats.append(f"PrevBIO+p1pos=@@+{p1[1]}")
        feats.append(f"PrevBIO+pos=@@+{pos}")

    # ===== COORDINATION FEATURES (for 'and' error reduction) =====

    # Proper noun coordination (NNP and NNP pattern - strong O signal)
    if (p1 is not None and n1 is not None and
        p1[1].startswith('NNP') and n1[1].startswith('NNP')):
        feats.append("nnp_and_nnp")

    # Adjective coordination before noun (JJ and JJ N - phrase-level coordination)
    if (p1 is not None and n1 is not None and n2 is not None and
        p1[1].startswith('JJ') and n1[1].startswith('JJ') and n2[1].startswith('NN')):
        feats.append("adj_coord_before_n")

    # Quantifier in previous 3 tokens (scope indicator for list NPs)
    has_quant = False
    for j in range(max(0, i-3), i):
        if sent[j][1] in {'CD', 'DT', 'PRP$'}:
            has_quant = True
            break
    if has_quant:
        feats.append("quant_in_prev3")

    # ===== CC_F1: Noun after CC at phrase boundary (+1 group) =====
    # Targets B→I errors after coordination when followed by punct/verb
    if pos in {'NN', 'NNS', 'NNP', 'NNPS'}:
        if p1 is not None and p1[0].lower() in {'and', 'or', '&'}:
            if n1 is not None and n1[1] in {'.', ',', ';', ':', 'VBD', 'VBZ', 'VBP', 'VBN', 'MD', 'IN'}:
                feats.append("noun_after_cc_at_boundary")  # Strong B-NP signal

    # ===== LINGUISTIC FEATURES (winners from ablation testing) =====

    # Previous token is currency/number symbol (for numeric NPs)
    if p1 is not None and p1[0].lower() in {'$', '#', '%'}:
        feats.append("prev_is_symbol")

    # Determiner/possessive patterns (typically start NPs)
    if pos in {'DT', 'PDT', 'WDT', 'PRP$'}:
        feats.append("is_determiner")

    # Attributive adjectives: JJ before noun (should be inside NP)
    if pos.startswith('JJ'):
        if n1 is not None and n1[1].startswith('NN'):
            feats.append("jj_before_noun")

    # More/most as NP starters (comparative/superlative)
    if wl in {'more', 'most'} and pos in {'RBR', 'RBS', 'JJR', 'JJS'}:
        if n1 is not None and (n1[1].startswith('JJ') or n1[1].startswith('NN')):
            feats.append("more_most_before_np")

    return feats

def write_features(sents, out_path: Path, is_training: bool):
    with out_path.open("w", encoding="utf-8") as out:
        for sent in sents:
            if not sent:
                out.write("\n")
                continue
            for i, (w, pos, bio) in enumerate(sent):
                feats = token_feats(sent, i, include_prev_bio=True)
                fields = [w] + feats
                if is_training:
                    fields.append(bio)
                out.write("\t".join(fields) + "\n")

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 build_features.py MODE INPUT_PATH OUTPUT_PATH")
        print("MODE: train (expects .pos-chunk) | dev/test (expects .pos)")
        sys.exit(1)

    mode = sys.argv[1].lower()
    in_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    if mode == "train":
        sents = read_wsj(in_path, has_bio=True)
        write_features(sents, out_path, is_training=True)
    elif mode in {"dev", "test"}:
        sents = read_wsj(in_path, has_bio=False)
        write_features(sents, out_path, is_training=False)
    else:
        sys.exit("Mode must be train/dev/test")

    # sanity: 1:1 line mapping with input
    with in_path.open(encoding="utf-8") as f: in_lines = sum(1 for _ in f)
    with out_path.open(encoding="utf-8") as f: out_lines = sum(1 for _ in f)
    if in_lines != out_lines:
        print(f"[WARN] line mismatch {in_lines} vs {out_lines}")
    else:
        print(f"[OK] {out_path} lines={out_lines}")

if __name__ == "__main__":
    main()
