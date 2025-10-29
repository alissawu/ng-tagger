#!/bin/bash
# Parallel feature addition testing - testing 'and' error reduction features

set -e

echo "==================================================================="
echo "Starting Parallel Feature Addition Tests"
echo "Testing 6 new features for 'and' coordination errors"
echo "==================================================================="
echo ""

# Clean up old experiments
rm -rf exp1 exp2 exp3 exp4 exp5 exp6 2>/dev/null || true

# Function to run one experiment
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    shift 2
    local remove_patterns=("$@")

    echo "[$exp_name] Starting..."

    # Create experiment directory
    mkdir -p exp${exp_num}

    # Create modified build_features.py by removing specified features
    local grep_chain="cat build_features.py"
    for pattern in "${remove_patterns[@]}"; do
        grep_chain="$grep_chain | grep -v '$pattern'"
    done
    eval "$grep_chain" > exp${exp_num}/build_features_mod.py

    # Build features
    python3 exp${exp_num}/build_features_mod.py train WSJ_02-21.pos-chunk exp${exp_num}/training.feature > exp${exp_num}/build_train.log 2>&1
    python3 exp${exp_num}/build_features_mod.py dev WSJ_24.pos exp${exp_num}/test.feature > exp${exp_num}/build_dev.log 2>&1

    # Train model
    java -Xmx12g -cp .:maxent-3.0.0.jar:trove.jar MEtrain exp${exp_num}/training.feature exp${exp_num}/model.chunk > exp${exp_num}/train.log 2>&1

    # Tag
    java -Xmx12g -cp .:maxent-3.0.0.jar:trove.jar MEtag exp${exp_num}/test.feature exp${exp_num}/model.chunk exp${exp_num}/response.chunk > exp${exp_num}/tag.log 2>&1

    # Score
    python3 score.chunk.py WSJ_24.pos-chunk exp${exp_num}/response.chunk > exp${exp_num}/results.txt 2>&1

    echo "[$exp_name] Complete!"
}

# Run in 2 batches to reduce memory pressure (3 then 3)
echo "Batch 1: Running experiments 1-3..."
# Baseline: Remove ALL new features
run_experiment 1 "Baseline (no new features)" "FEATURE_DIST_TO_EOS" "FEATURE_COMMA" "FEATURE_NNP" "FEATURE_ADJ" "FEATURE_QUANT" &
# Keep only dist_to_eos: Remove the other 4
run_experiment 2 "Add: dist_to_eos + near_eos" "FEATURE_COMMA" "FEATURE_NNP" "FEATURE_ADJ" "FEATURE_QUANT" &
# Keep only comma: Remove the other 4
run_experiment 3 "Add: comma_in_prev3" "FEATURE_DIST_TO_EOS" "FEATURE_NNP" "FEATURE_ADJ" "FEATURE_QUANT" &
wait

echo "Batch 2: Running experiments 4-6..."
# Keep only nnp: Remove the other 4
run_experiment 4 "Add: nnp_and_nnp" "FEATURE_DIST_TO_EOS" "FEATURE_COMMA" "FEATURE_ADJ" "FEATURE_QUANT" &
# Keep only adj: Remove the other 4
run_experiment 5 "Add: adj_coord_before_n" "FEATURE_DIST_TO_EOS" "FEATURE_COMMA" "FEATURE_NNP" "FEATURE_QUANT" &
# Keep only quant: Remove the other 4
run_experiment 6 "Add: quant_in_prev3" "FEATURE_DIST_TO_EOS" "FEATURE_COMMA" "FEATURE_NNP" "FEATURE_ADJ" &
wait

echo ""
echo "==================================================================="
echo "ALL EXPERIMENTS COMPLETE - RESULTS SUMMARY"
echo "==================================================================="
echo ""

# Parse and display results
print_result() {
    local exp_num=$1
    local exp_name=$2
    local results_file="exp${exp_num}/results.txt"

    if [ -f "$results_file" ]; then
        local accuracy=$(grep "accuracy:" $results_file | awk '{print $2}')
        local precision=$(grep "precision:" $results_file | awk '{print $2}')
        local recall=$(grep "recall:" $results_file | awk '{print $2}')
        local f1=$(grep "F1:" $results_file | awk '{print $2}')
        local correct=$(grep "correct groups" $results_file | awk '{print $1}')

        printf "%-35s  Acc: %-6s  P: %-6s  R: %-6s  F1: %-6s  Correct: %s\n" "$exp_name" "$accuracy" "$precision" "$recall" "$f1" "$correct"
    else
        printf "%-35s  ERROR: Results not found\n" "$exp_name"
    fi
}

print_result 1 "Baseline (no new features)"
print_result 2 "Add: dist_to_eos + near_eos"
print_result 3 "Add: comma_in_prev3"
print_result 4 "Add: nnp_and_nnp"
print_result 5 "Add: adj_coord_before_n"
print_result 6 "Add: quant_in_prev3"

echo ""
echo "==================================================================="
echo "Feature descriptions:"
echo "  dist_to_eos: Distance to sentence end (helps sentence-final 'and')"
echo "  comma_in_prev3: Comma in prev 3 tokens (list indicator)"
echo "  nnp_and_nnp: Both neighbors proper nouns (entity coordination)"
echo "  adj_coord_before_n: JJ and JJ N pattern (phrase-level coord)"
echo "  quant_in_prev3: Quantifier in prev 3 tokens (scope indicator)"
echo "==================================================================="
echo ""
echo "Detailed results in: exp1/results.txt, exp2/results.txt, etc."
echo "To analyze 'and' errors: python3 analyze_and_errors.py"
echo "==================================================================="
