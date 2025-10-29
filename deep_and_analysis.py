#!/usr/bin/env python3
"""
Deep analysis: What structural patterns distinguish "and" as I-NP vs O?
Focus on features we DON'T currently have.
"""

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

def analyze_and_context(sents):
    """Look at structural patterns around 'and' that we DON'T capture yet."""

    and_as_I_patterns = []
    and_as_O_patterns = []

    for sent in sents:
        for i, (w, pos, bio) in enumerate(sent):
            if w.lower() not in {"and", "or"} or pos != "CC":
                continue

            # Get extended context
            p3 = sent[i-3] if i >= 3 else None
            p2 = sent[i-2] if i >= 2 else None
            p1 = sent[i-1] if i >= 1 else None
            n1 = sent[i+1] if i+1 < len(sent) else None
            n2 = sent[i+2] if i+2 < len(sent) else None
            n3 = sent[i+3] if i+3 < len(sent) else None

            context = {
                'p3': p3, 'p2': p2, 'p1': p1,
                'n1': n1, 'n2': n2, 'n3': n3,
                'bio': bio
            }

            # Compute features we DON'T have yet:

            # 1. POS parallelism: Do p1 and n1 have same coarse POS?
            context['parallel_coarse'] = False
            if p1 and n1:
                def coarse(pos):
                    if pos.startswith("NN"): return "N"
                    if pos.startswith("JJ"): return "J"
                    if pos.startswith("RB"): return "R"
                    if pos.startswith("VB"): return "V"
                    return pos
                context['parallel_coarse'] = (coarse(p1[1]) == coarse(n1[1]))

            # 2. Previous BIO tag (do they match?)
            context['prev_bio'] = p1[2] if p1 else None

            # 3. Is p1 a verb?
            context['prev_is_verb'] = p1 and p1[1].startswith("VB")

            # 4. Distance to last O tag (how deep in NP are we?)
            context['dist_from_O'] = 0
            for j in range(i-1, max(0, i-5), -1):
                if sent[j][2] == "O":
                    break
                context['dist_from_O'] += 1

            # 5. Count nearby commas (list indicator)
            context['nearby_commas'] = 0
            for j in range(max(0, i-3), min(len(sent), i+4)):
                if sent[j][0] == ',':
                    context['nearby_commas'] += 1

            if bio == "I-NP":
                and_as_I_patterns.append(context)
            elif bio == "O":
                and_as_O_patterns.append(context)

    return and_as_I_patterns, and_as_O_patterns

def print_insights(and_I, and_O):
    """Print comparative statistics."""

    print("="*80)
    print("STRUCTURAL PATTERNS: 'and' as I-NP vs O")
    print("="*80)

    # 1. Parallelism
    I_parallel = sum(1 for c in and_I if c['parallel_coarse']) / len(and_I) * 100
    O_parallel = sum(1 for c in and_O if c['parallel_coarse']) / len(and_O) * 100
    print(f"\n1. POS PARALLELISM (p1 coarse POS == n1 coarse POS):")
    print(f"   I-NP: {I_parallel:.1f}% have parallel structure")
    print(f"   O:    {O_parallel:.1f}% have parallel structure")
    print(f"   → Insight: {'Parallelism predicts I-NP' if I_parallel > O_parallel + 5 else 'Not discriminative'}")

    # 2. Previous BIO
    I_prevBIO = Counter(c['prev_bio'] for c in and_I if c['prev_bio'])
    O_prevBIO = Counter(c['prev_bio'] for c in and_O if c['prev_bio'])
    print(f"\n2. PREVIOUS BIO TAG:")
    print(f"   I-NP: {I_prevBIO.most_common(3)}")
    print(f"   O:    {O_prevBIO.most_common(3)}")

    # 3. Previous verb
    I_prevVerb = sum(1 for c in and_I if c['prev_is_verb']) / len(and_I) * 100
    O_prevVerb = sum(1 for c in and_O if c['prev_is_verb']) / len(and_O) * 100
    print(f"\n3. PREVIOUS TOKEN IS VERB:")
    print(f"   I-NP: {I_prevVerb:.1f}% after verb")
    print(f"   O:    {O_prevVerb:.1f}% after verb")
    print(f"   → Insight: {'Verbs before and → O' if O_prevVerb > I_prevVerb + 5 else 'Not strong signal'}")

    # 4. Distance from last O
    I_avgDist = sum(c['dist_from_O'] for c in and_I) / len(and_I)
    O_avgDist = sum(c['dist_from_O'] for c in and_O) / len(and_O)
    print(f"\n4. DISTANCE FROM LAST O TAG (depth in NP):")
    print(f"   I-NP: Average {I_avgDist:.2f} tokens from last O")
    print(f"   O:    Average {O_avgDist:.2f} tokens from last O")
    print(f"   → Insight: {'Deeper in NP → I-NP' if I_avgDist > O_avgDist + 0.5 else 'Not discriminative'}")

    # 5. Nearby commas
    I_withCommas = sum(1 for c in and_I if c['nearby_commas'] > 0) / len(and_I) * 100
    O_withCommas = sum(1 for c in and_O if c['nearby_commas'] > 0) / len(and_O) * 100
    print(f"\n5. NEARBY COMMAS (within ±3 tokens):")
    print(f"   I-NP: {I_withCommas:.1f}% have nearby commas")
    print(f"   O:    {O_withCommas:.1f}% have nearby commas")
    print(f"   → Insight: {'Nearby commas → list → I-NP' if I_withCommas > O_withCommas + 5 else 'Not discriminative'}")

    print("\n" + "="*80)
    print("RECOMMENDATION: Add features with strongest discriminative power")
    print("="*80)

def main():
    train_path = Path("WSJ_02-21.pos-chunk")
    print("Analyzing 'and/or' structural patterns in training data...\n")

    sents = read_wsj(train_path)
    and_I, and_O = analyze_and_context(sents)

    print(f"Found {len(and_I)} 'and/or' as I-NP")
    print(f"Found {len(and_O)} 'and/or' as O\n")

    print_insights(and_I, and_O)

if __name__ == "__main__":
    main()
