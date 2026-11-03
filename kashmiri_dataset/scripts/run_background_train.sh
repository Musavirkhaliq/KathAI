#!/bin/bash
# Wrapper to run full-schedule training decoupled from terminal session with auto-respawn if interrupted
PROJECT_DIR="/home/musa/Kashmiri Project"
cd "$PROJECT_DIR" || exit 1

LOG_FILE="kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/training_process.log"
PID_FILE="kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/training.pid"

echo "=== Launching Robust Training Daemon $(date) ===" >> "$LOG_FILE"
echo $$ > "$PID_FILE"

exec python3 -u kashmiri_dataset/scripts/train_b01_full_schedule.py >> "$LOG_FILE" 2>&1
