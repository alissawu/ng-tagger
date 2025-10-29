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

    # ===== FEATURE 2: QUOTATION BOUNDARIES ===== # NEW_F2
    # Opening quote → B-NP signal for following token # NEW_F2
    if p1 is not None and p1[0] in {'``', '"', '`'}: # NEW_F2
        feats.append("after_open_quote") # NEW_F2
    # Quote before noun # NEW_F2
    if pos in {'``', '"', '`'} and n1 is not None and n1[1].startswith('NN'): # NEW_F2
        feats.append("quote_before_nn") # NEW_F2

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

    # ===== FEATURE 3: "THAT" COMPLEMENTIZER ===== # NEW_F3
    # "that" (IN) as complementizer → following token likely B-NP # NEW_F3
    if p1 is not None and p1[0].lower() == 'that' and p1[1] == 'IN': # NEW_F3
        feats.append("after_that_comp") # NEW_F3

    # ===== COORDINATION FEATURES (for 'and' error reduction) =====

    # ===== FEATURE 1: COORDINATION AFTER PUNCTUATION ===== # NEW_F1
    # Coordination after comma/semicolon (list coordination → strong O signal) # NEW_F1
    if wl in {'and', 'or', '&'} and p1 is not None and p1[0] in {',', ';'}: # NEW_F1
        feats.append("coord_after_punct") # NEW_F1

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

    # ===== NEW CC/COORDINATION FEATURES (targeting 51 B→I errors after CC) ===== # CC_FEATS

    # CC_F1: Noun after CC at phrase boundary (punct/verb follows) # CC_F1
    if pos in {'NN', 'NNS', 'NNP', 'NNPS'}: # CC_F1
        if p1 is not None and p1[0].lower() in {'and', 'or', '&'}: # CC_F1
            if n1 is not None and n1[1] in {'.', ',', ';', ':', 'VBD', 'VBZ', 'VBP', 'VBN', 'MD', 'IN'}: # CC_F1
                feats.append("noun_after_cc_at_boundary")  # Strong B-NP signal # CC_F1

    # CC_F2: List coordination - comma in context + after CC # CC_F2
    if p1 is not None and p1[0].lower() in {'and', 'or'}: # CC_F2
        has_prev_comma = False # CC_F2
        for j in range(max(0, i-5), i): # CC_F2
            if sent[j][0] == ',': # CC_F2
                has_prev_comma = True # CC_F2
                break # CC_F2
        if has_prev_comma and pos in {'NN', 'NNS', 'NNP', 'NNPS'}: # CC_F2
            feats.append("list_coord_after_comma")  # B-NP signal # CC_F2

    # CC_F3: Distinguish phrase-level vs sentence-level coordination # CC_F3
    if wl in {'and', 'or', '&'}: # CC_F3
        if p1 is not None and n1 is not None: # CC_F3
            p_is_noun = p1[1] in {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'CD'} # CC_F3
            n_is_noun = n1[1] in {'NN', 'NNS', 'NNP', 'NNPS'} # CC_F3
            if p_is_noun and n_is_noun: # CC_F3
                # Check what follows the noun after CC # CC_F3
                if n2 is not None: # CC_F3
                    n2_is_noun = n2[1] in {'NN', 'NNS', 'NNP', 'NNPS'} # CC_F3
                    if n2_is_noun: # CC_F3
                        feats.append("cc_phrase_level")  # I-NP ok (e.g., "red and blue cars") # CC_F3
                    else: # CC_F3
                        feats.append("cc_sentence_level")  # O signal (e.g., "housing and inflation .") # CC_F3
                else: # CC_F3
                    feats.append("cc_sentence_level")  # End of sentence # CC_F3

    # CC_F4: Simple - noun after CC gets B-NP boost # CC_F4
    if p1 is not None and p1[1] == 'CC' and pos in {'NN', 'NNS', 'NNP', 'NNPS'}: # CC_F4
        feats.append("noun_after_cc")  # General B-NP signal # CC_F4

    # ===== FEATURE 4: COMMA CONTEXT IN NNP SEQUENCES ===== # NEW_F4
    # Distinguish list commas (I-NP) from appositive commas (O) # NEW_F4
    if pos == ',': # NEW_F4
        if p1 is not None and n1 is not None: # NEW_F4
            # NNP , NNP pattern - check if it's a long list # NEW_F4
            if p1[1] == 'NNP' and n1[1] == 'NNP': # NEW_F4
                # Count NNPs in context window (±3) # NEW_F4
                nnp_count = 0 # NEW_F4
                for j in range(max(0, i-3), min(len(sent), i+4)): # NEW_F4
                    if j != i and sent[j][1] == 'NNP': # NEW_F4
                        nnp_count += 1 # NEW_F4
                if nnp_count >= 3: # NEW_F4
                    feats.append("comma_in_nnp_list")  # Long list → I-NP # NEW_F4
            # Appositive detection: NNP , (WP|WDT|VB|RB|IN) # NEW_F4
            elif p1[1] == 'NNP' and n1[1] in {'WP', 'WDT', 'WRB', 'VBD', 'VBZ', 'VBP', 'RB', 'IN'}: # NEW_F4
                feats.append("comma_appositive")  # Appositive → O # NEW_F4

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
