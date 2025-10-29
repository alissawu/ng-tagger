#!/bin/bash
# Quick test script for feature ablation

echo "Building features..."
python3 build_features.py train WSJ_02-21.pos-chunk training.feature
python3 build_features.py dev WSJ_24.pos test.feature

echo "Training model..."
java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtrain training.feature model.chunk

echo "Tagging dev set..."
java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtag test.feature model.chunk response.chunk

echo "Scoring..."
python3 score.chunk.py WSJ_24.pos-chunk response.chunk
