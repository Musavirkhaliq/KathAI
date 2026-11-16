#!/bin/bash
# Standalone training launcher for running in a separate terminal / tmux session
set -e

PROJECT_DIR="/home/musa/Kashmiri Project"
cd "$PROJECT_DIR"

echo "========================================================================"
echo " Starting / Resuming Kashmiri XLS-R 300M Full-Schedule ASR Training"
echo " Working Directory: $PROJECT_DIR"
echo " Time: $(date)"
echo " GPU Device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "========================================================================"

# Run training directly in foreground of the separate terminal
python3 -u kashmiri_dataset/scripts/train_b01_full_schedule.py

