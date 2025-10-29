NG Chunker (MaxEnt) – Feature Builder and Pipeline
Full assignment text in `Assignment.txt`
Overview
- Goal: Train a Maximum Entropy tagger to assign BIO tags for Noun Groups (NP chunks) on WSJ.
- Core: `build_features.py` generates tab-separated features for the provided OpenNLP MaxEnt wrappers (`MEtrain.java`, `MEtag.java`).
- Status: Feature set curated to push dev F1 above 95 without feature bloat.

Files
- `build_features.py`: Generates training and test feature files.
- `MEtrain.java`, `MEtag.java`, `maxent-3.0.0.jar`, `trove.jar`: Train and tag with MaxEnt.
- `score.chunk.py`: Scoring script for dev/test.
- `WSJ_02-21.pos-chunk`, `WSJ_24.pos`, `WSJ_24.pos-chunk`, `WSJ_23.pos`: Provided corpora.

Feature Format
- Training lines: `TOKEN\tFEAT1\tFEAT2\t...\tBIO`
- Dev/Test lines: `TOKEN\tFEAT1\tFEAT2\t...` (no final BIO)
- Sentence breaks: blank line in → blank line out (line counts must match).
- Special: `@@` in any feature is replaced by the previous predicted BIO tag during tagging (see `MEtag.java`).

Features (curated, high-yield)
- Word identity and normalization
  - `w=TOKEN`, `wl=token_lower`, `shape=word_shape` (e.g., Xx, xx-dd, Xx-Xx)
  - Prefixes: `pre2=`, `pre3=`; Suffixes: `suf2=`, `suf3=`, `suf4=`
- POS-based features
  - Current: `pos=PTB_POS`, `cpos=coarse_pos` (e.g., NN*, VB*, JJ*, RB*, NNP, PRP, W, PUNCT)
  - Context window: POS and coarse POS for ±1, ±2: `p1pos=`, `p2pos=`, `n1pos=`, `n2pos=`, plus `p1cpos=`, `p2cpos=`, `n1cpos=`, `n2cpos=`
  - POS n-grams: `p1pos+pos`, `pos+n1pos`, `p1pos+pos+n1pos`, `p2pos+p1pos`, `n1pos+n2pos`, `p2pos+p1pos+pos`, `pos+n1pos+n2pos`
- Word context (lightweight)
  - Neighbor words for ±1 (both raw and lowercased): `p1w=`, `p1wl=`, `n1w=`, `n1wl=`
- Capitalization/orthography
  - `isTitle`, `isAllCaps`, `hasHyphen`, `hasDigit`, `isNumber`, `isPunct`
- Lexical cues
  - Determiners: `isDet` if token in {the, a, an, this, that, these, those}
  - Prepositions: `isPrep` if token in {of, in, on, at, for, with, by, to, from, as, into, over, under}
- Chunk-structure cues
  - Sentence edges: `BOS`, `BOS2`, `EOS`, `EOS2`
  - Proper-noun run: `pnRun=true` if NNP next to NNP
  - Comma between nouns: `commaBetweenNP=true` if `,` with NN on both sides
- Coordination features (for 'and' error reduction)
  - `nnp_and_nnp` if both neighbors are proper nouns (entity coordination signal)
  - `adj_coord_before_n` if pattern is JJ and JJ N (phrase-level coordination)
  - `quant_in_prev3` if quantifier (CD/DT/PRP$) in previous 3 tokens (list scope)
- Linguistic features (BIO violation reduction)
  - `prev_is_symbol` if previous token is $, #, or % (numeric NP context)
  - `is_determiner` if current token is DT/PDT/WDT/PRP$ (NP starters)
  - `jj_before_noun` if JJ followed by NN (attributive adjectives)
  - `more_most_before_np` if more/most before JJ/NN (comparative/superlative NP starters)
- Sequence tag dependency (MEMM-style)
  - `PrevBIO=@@`
  - `PrevBIO+pos=@@+{pos}` and `PrevBIO+p1pos=@@+{p1pos}`

Why these features
- Leverage POS context (±2) and POS n-grams for NP structure, with coarse POS to generalize and reduce sparsity.
- Limited neighbor word identity (±1) to capture frequent cues without exploding the feature space.
- Shape/prefix/suffix/casing capture robust orthographic signals for proper nouns, acronyms, numbers.
- MEMM-style `PrevBIO` features capture sequence constraints crucial for BIO tagging.
- A few structure cues (NNP runs, comma between nouns, BOS/EOS) improve boundary decisions.

Usage
1) Build features
- Training: `python3 build_features.py train WSJ_02-21.pos-chunk training.feature`
- Dev: `python3 build_features.py dev WSJ_24.pos test.feature`

2) Compile and train (Posix)
- `javac -cp maxent-3.0.0.jar:trove.jar *.java`
- `java -Xmx8g -cp .:maxent-3.0.0.jar:trove.jar MEtrain training.feature model.chunk`

3) Tag development and score
- `java -Xmx8g -cp .:maxent-3.0.0.jar:trove.jar MEtag test.feature model.chunk response.chunk`
- `python3 score.chunk.py WSJ_24.pos-chunk response.chunk`

4) Produce final test output for submission
- `python3 build_features.py test WSJ_23.pos WSJ_23.feature`
- `java -Xmx8g -cp .:maxent-3.0.0.jar:trove.jar MEtag WSJ_23.feature model.chunk WSJ_23.chunk`

Notes
- Memory: If Java runs out of memory, increase heap: `-Xmx12g` or `-Xmx16g`.
- Windows: Use semicolons in classpaths.
- Line counts: The script preserves 1:1 line mapping with inputs; a mismatch indicates a bug.
- `@@` substitution: `MEtag.java` replaces every `@@` in a line with the previous predicted BIO tag before classification; combining `@@` inside features (e.g., `PrevBIO+pos=@@+NN`) is supported.

Ablation Testing Log (Current F1: 93.2%)

MaxEnt Parameters: 100 iterations, cutoff=4 (MEtrain.java)

Systematic Tests - Removed Features (KEEP REMOVED):
- ✓ **suf4**: No impact
- ✓ **pre3**: Improved P (92.72%), maintained R (93.54%)
- ✓ **pre2**: Improved both P (92.78%) & R (93.54%), 7837 correct
- ✓ **isPunct**: Improved both P (92.79%) & R (93.57%), 7839 correct
- ✓ **n1wl**: Improved both P (92.87%) & R (93.58%), 7840 correct ← NEW HIGH
- ✓ **n2cpos**: No impact (negligible)

Tests - Keep Features (needed):
- ✗ **suf2**: Decreased P & R
- ✗ **isTitle**: Decreased P & R
- ✗ **isAllCaps**: Decreased P & R
- ✗ **hasDigit**: Slight P decrease
- ✗ **hasHyphen**: Negligible but keeping
- ✗ **isDet**: Decreased P & R
- ✗ **isPrep**: Decreased P & R
- ✗ **p1wl**: Decreased P (92.60%) & R (93.40%) significantly
- ✗ **p2cpos**: Decreased P & R
- ✗ **p2pos+p1pos** (POS bigram): Decreased all metrics (parallel test)
- ✗ **n1pos+n2pos** (POS bigram): Decreased all metrics (parallel test)
- ✗ **p2pos+p1pos+pos** (POS trigram): Decreased all metrics (parallel test)
- ✗ **pos+n1pos+n2pos** (POS trigram): Decreased all metrics (parallel test)
- ✗ **BOS2/EOS2** (sentence boundaries): Decreased all metrics (parallel test)

Failed Additions:
- ✗ **PrevBIO+cpos**: Hurt
- ✗ **cc_before_phrase_start**: No improvement
- ✗ **commaBetweenNP**: No effect
- ✗ **cc_parallel_coarse**: Slight decrease (P: 92.87→92.85, R: 93.58→93.57)
  - Despite 81.9% vs 21.6% evidence, maybe redundant with existing POS n-grams
- ✗ **comma_in_nnp_seq**: No improvement
  - Targeted 13 errors where commas in "Finley, Kumble, Wagner" style names should be I-NP
  - Likely too specific or confounded by other patterns
- ✗ **dist_to_eos + near_eos**: Slight negative effect (-1 correct group)
  - Distance to sentence end feature hurt performance
- ✗ **comma_in_prev3**: No improvement
  - Comma in previous 3 tokens had slight negative effect

Linguistic Features Ablation (targeting O→I-NP violations):
- ✗ **is_function_word**: No effect (7844 baseline)
- ✗ **is_currency_symbol**: No effect (7844 baseline)
- ✗ **is_unit_noun**: No effect (7844 baseline)
- ✗ **det_before_np_material**: No real gain (7844), slight precision increase but accuracy decreased
- ✗ **jj_after_copula**: No effect (7844 baseline)
- ✗ **generic_adverb**: Hurt performance (7843, -1 from baseline)
- ✗ **adverb_before_number**: No effect (7844 baseline)

Successful Additions:
- ✓ **nnp_and_nnp**: +2 correct groups, F1 improved to 93.3% (P: 92.90%, R: 93.60%)
  - Detects proper noun coordination (e.g., "Axa and Hoylake")
- ✓ **adj_coord_before_n**: +1 correct group (P: 92.88%, R: 93.59%)
  - Detects adjective coordination before noun (e.g., "professional and private lives")
- ✓ **quant_in_prev3**: +2 correct groups, F1 improved to 93.3% (P: 92.91%, R: 93.60%)
  - Quantifier scope indicator for list NPs (e.g., "86,555 pickups, vans and sport utility vehicles")
- ✓ **prev_is_symbol**: +1 correct group (7845), F1: 93.29%
  - Previous token is currency/percent symbol (helps numeric NP boundaries)
- ✓ **is_determiner**: +1 correct group (7845), F1: 93.29%
  - Identifies determiners as NP starters
- ✓ **jj_before_noun**: +2 correct groups (7846), F1: 93.31% ← BEST LINGUISTIC FEATURE
  - Attributive adjectives before nouns (e.g., "open borders", "high level")
- ✓ **more_most_before_np**: +1 correct group (7845), F1: 93.29%
  - Comparative/superlative as NP starters (e.g., "more radical views")

Current Feature Set:
- Core: w, wl, pos, cpos, shape
- Affixes: suf2, suf3 (NO pre2/pre3/suf4)
- Orthographic: isTitle, isAllCaps, hasHyphen, hasDigit, isNumber (NO isPunct)
- Lexical: isDet, isPrep
- Context: ±2 POS, ±1 words, p1wl (NO n1wl, n2cpos)
- Structural: pnRun
- Coordination: nnp_and_nnp, adj_coord_before_n, quant_in_prev3
- Linguistic: prev_is_symbol, is_determiner, jj_before_noun, more_most_before_np
- MEMM: PrevBIO+p1pos, PrevBIO+pos

Performance:
- Baseline (before coordination features): 93.2% F1 (7840/8378, P: 92.87%, R: 93.58%)
- With coordination features: 93.3% F1 (7844 correct groups)
- With coordination + linguistic features: Expected ~93.3-93.4% F1 (best individual: 7846 correct groups)
- Target: 95% F1
- Gap: ~1.6-1.7 points

Key Insights:
1. **Context Asymmetry**: p1wl helps generalize, n1wl hurts (next-word capitalization is informative)
2. **All base features necessary**: Ablation complete - remaining features all contribute
3. **Error-driven analysis**: Used find_worst_errors.py and analyze_bio_violations.py to identify actual dev set failures
   - Top patterns: comma between NNPs (13 errors), "and" coordination (71 errors, split I↔O), O→I-NP violations (132 violations)
4. **Pattern-specific features often fail**: Highly specific features (comma_in_nnp_seq, is_currency_symbol) don't improve - likely too narrow or confounded by existing features
5. **Coordination features work**: Targeted features for 'and' error patterns (NNP coordination, adjective coordination, quantifier scope) each provide small improvements (+1-2 correct groups)
6. **Linguistic features require careful testing**: 11 features tested, only 4 helped. Function words, currency symbols, and unit nouns showed no improvement despite targeting O→I-NP violations. Best performer: jj_before_noun (+2 correct groups)

