#!/usr/bin/env python3
"""
Batch feature ablation testing - run multiple experiments in parallel
"""
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

# Define experiments: (name, lines_to_comment_out)
EXPERIMENTS = {
    "baseline": [],  # current state
    "no_p1wl_n1wl": ["p1wl=", "n1wl="],  # remove lowercase neighbor words
    "no_p2cpos_n2cpos": ["p2cpos=", "n2cpos="],  # remove coarse POS at ±2
    "no_BOS2_EOS2": ["BOS2", "EOS2"],  # remove distance-2 sentence boundaries
    "no_p2pos_p1pos": ["p2pos+p1pos=", 'f"p2pos={p2pos}"'],  # remove 2-back POS bigram and p2pos
    "no_n1pos_n2pos": ["n1pos+n2pos=", 'f"n2pos={n2pos}"'],  # remove 2-forward POS bigram and n2pos
}

def modify_features(experiment_name, lines_to_comment):
    """Create a modified build_features.py for this experiment"""
    src = Path("build_features.py")
    dst = Path(f"build_features_{experiment_name}.py")

    with src.open() as f:
        lines = f.readlines()

    # Comment out lines containing any of the target strings
    modified = []
    for line in lines:
        should_comment = any(target in line for target in lines_to_comment)
        if should_comment and not line.strip().startswith("#"):
            # Comment out this line
            modified.append("# [ABLATION] " + line)
        else:
            modified.append(line)

    with dst.open("w") as f:
        f.writelines(modified)

    return dst

def run_experiment(experiment_name, lines_to_comment):
    """Run full train/test pipeline for one experiment"""
    try:
        # Create modified feature builder
        if experiment_name == "baseline":
            builder = "build_features.py"
        else:
            builder = modify_features(experiment_name, lines_to_comment)

        # Create experiment directory
        exp_dir = Path(f"exp_{experiment_name}")
        exp_dir.mkdir(exist_ok=True)

        # Build features
        subprocess.run([
            "python3", builder, "train", "WSJ_02-21.pos-chunk",
            str(exp_dir / "training.feature")
        ], check=True, capture_output=True)

        subprocess.run([
            "python3", builder, "dev", "WSJ_24.pos",
            str(exp_dir / "test.feature")
        ], check=True, capture_output=True)

        # Train model
        subprocess.run([
            "java", "-Xmx16g", "-cp", ".:maxent-3.0.0.jar:trove.jar",
            "MEtrain", str(exp_dir / "training.feature"), str(exp_dir / "model.chunk")
        ], check=True, capture_output=True, text=True)

        # Tag dev set
        subprocess.run([
            "java", "-Xmx16g", "-cp", ".:maxent-3.0.0.jar:trove.jar",
            "MEtag", str(exp_dir / "test.feature"), str(exp_dir / "model.chunk"),
            str(exp_dir / "response.chunk")
        ], check=True, capture_output=True, text=True)

        # Score
        result = subprocess.run([
            "python3", "score.chunk.py", "WSJ_24.pos-chunk", str(exp_dir / "response.chunk")
        ], check=True, capture_output=True, text=True)

        # Parse results
        lines = result.stdout.strip().split("\n")
        accuracy = precision = recall = f1 = "?"
        for line in lines:
            if "accuracy:" in line:
                accuracy = line.split(":")[-1].strip()
            elif "precision:" in line:
                precision = line.split(":")[-1].strip()
            elif "recall:" in line:
                recall = line.split(":")[-1].strip()
            elif "F1:" in line:
                f1 = line.split(":")[-1].strip()

        return {
            "name": experiment_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "output": result.stdout
        }

    except Exception as e:
        return {
            "name": experiment_name,
            "error": str(e)
        }

def main():
    print("Starting parallel batch testing...")
    print(f"Running {len(EXPERIMENTS)} experiments in parallel\n")

    results = []

    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=min(len(EXPERIMENTS), 4)) as executor:
        futures = {
            executor.submit(run_experiment, name, lines): name
            for name, lines in EXPERIMENTS.items()
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✓ Completed: {name}")
            except Exception as e:
                print(f"✗ Failed: {name} - {e}")

    # Display results
    print("\n" + "="*80)
    print("BATCH TESTING RESULTS")
    print("="*80)
    print(f"{'Experiment':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*80)

    # Sort by F1 (descending)
    results.sort(key=lambda x: float(x.get('f1', '0').split()[0]) if 'error' not in x else 0, reverse=True)

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<20} ERROR: {r['error']}")
        else:
            print(f"{r['name']:<20} {r['accuracy']:<10} {r['precision']:<10} {r['recall']:<10} {r['f1']:<10}")

    print("="*80)

if __name__ == "__main__":
    main()
