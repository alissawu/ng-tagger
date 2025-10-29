#!/bin/bash
# Parallel feature ablation testing - testing 11 linguistic features individually

set -e

echo "==================================================================="
echo "Linguistic Features Ablation Test"
echo "Testing 11 features individually (12 experiments total)"
echo "==================================================================="
echo ""

# Clean up old experiments
rm -rf exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 exp9 exp10 exp11 exp12 2>/dev/null || true

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

# Run in 4 batches of 3, 3, 3, 3
echo "Batch 1: Running experiments 1-3..."
# Baseline: Remove ALL linguistic features
run_experiment 1 "Baseline (no ling features)" "LING_FUNC" "LING_CURR" "LING_UNIT" "LING_PREV_SYM" "LING_DET1" "LING_DET2" "LING_JJ_N" "LING_JJ_COP" "LING_MORE" "LING_ADV" "LING_ADV_NUM" &
# Keep only function word
run_experiment 2 "Add: is_function_word" "LING_CURR" "LING_UNIT" "LING_PREV_SYM" "LING_DET1" "LING_DET2" "LING_JJ_N" "LING_JJ_COP" "LING_MORE" "LING_ADV" "LING_ADV_NUM" &
# Keep only currency symbol
run_experiment 3 "Add: is_currency_symbol" "LING_FUNC" "LING_UNIT" "LING_PREV_SYM" "LING_DET1" "LING_DET2" "LING_JJ_N" "LING_JJ_COP" "LING_MORE" "LING_ADV" "LING_ADV_NUM" &
wait

echo "Batch 2: Running experiments 4-6..."
# Keep only unit noun
run_experiment 4 "Add: is_unit_noun" "LING_FUNC" "LING_CURR" "LING_PREV_SYM" "LING_DET1" "LING_DET2" "LING_JJ_N" "LING_JJ_COP" "LING_MORE" "LING_ADV" "LING_ADV_NUM" &
# Keep only prev_is_symbol
run_experiment 5 "Add: prev_is_symbol" "LING_FUNC" "LING_CURR" "LING_UNIT" "LING_DET1" "LING_DET2" "LING_JJ_N" "LING_JJ_COP" "LING_MORE" "LING_ADV" "LING_ADV_NUM" &
# Keep only is_determiner
run_experiment 6 "Add: is_determiner" "LING_FUNC" "LING_CURR" "LING_UNIT" "LING_PREV_SYM" "LING_DET2" "LING_JJ_N" "LING_JJ_COP" "LING_MORE" "LING_ADV" "LING_ADV_NUM" &
wait

echo "Batch 3: Running experiments 7-9..."
# Keep only det_before_np_material
run_experiment 7 "Add: det_before_np_material" "LING_FUNC" "LING_CURR" "LING_UNIT" "LING_PREV_SYM" "LING_DET1" "LING_JJ_N" "LING_JJ_COP" "LING_MORE" "LING_ADV" "LING_ADV_NUM" &
# Keep only jj_before_noun
run_experiment 8 "Add: jj_before_noun" "LING_FUNC" "LING_CURR" "LING_UNIT" "LING_PREV_SYM" "LING_DET1" "LING_DET2" "LING_JJ_COP" "LING_MORE" "LING_ADV" "LING_ADV_NUM" &
# Keep only jj_after_copula
run_experiment 9 "Add: jj_after_copula" "LING_FUNC" "LING_CURR" "LING_UNIT" "LING_PREV_SYM" "LING_DET1" "LING_DET2" "LING_JJ_N" "LING_MORE" "LING_ADV" "LING_ADV_NUM" &
wait

echo "Batch 4: Running experiments 10-12..."
# Keep only more_most
run_experiment 10 "Add: more_most_before_np" "LING_FUNC" "LING_CURR" "LING_UNIT" "LING_PREV_SYM" "LING_DET1" "LING_DET2" "LING_JJ_N" "LING_JJ_COP" "LING_ADV" "LING_ADV_NUM" &
# Keep only generic_adverb
run_experiment 11 "Add: generic_adverb" "LING_FUNC" "LING_CURR" "LING_UNIT" "LING_PREV_SYM" "LING_DET1" "LING_DET2" "LING_JJ_N" "LING_JJ_COP" "LING_MORE" "LING_ADV_NUM" &
# Keep only adverb_before_number
run_experiment 12 "Add: adverb_before_number" "LING_FUNC" "LING_CURR" "LING_UNIT" "LING_PREV_SYM" "LING_DET1" "LING_DET2" "LING_JJ_N" "LING_JJ_COP" "LING_MORE" "LING_ADV" &
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

        printf "%-40s  Acc: %-6s  P: %-6s  R: %-6s  F1: %-6s  Correct: %s\n" "$exp_name" "$accuracy" "$precision" "$recall" "$f1" "$correct"
    else
        printf "%-40s  ERROR: Results not found\n" "$exp_name"
    fi
}

print_result 1 "Baseline (no ling features)"
print_result 2 "Add: is_function_word"
print_result 3 "Add: is_currency_symbol"
print_result 4 "Add: is_unit_noun"
print_result 5 "Add: prev_is_symbol"
print_result 6 "Add: is_determiner"
print_result 7 "Add: det_before_np_material"
print_result 8 "Add: jj_before_noun"
print_result 9 "Add: jj_after_copula"
print_result 10 "Add: more_most_before_np"
print_result 11 "Add: generic_adverb"
print_result 12 "Add: adverb_before_number"

echo ""
echo "==================================================================="
echo "Detailed results in: exp1/results.txt, exp2/results.txt, etc."
echo "==================================================================="
