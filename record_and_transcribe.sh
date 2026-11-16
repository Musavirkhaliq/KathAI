#!/bin/bash
# Interactive Kashmiri ASR Record & Transcribe CLI
OUT_DIR="kashmiri_dataset/user_recordings"
mkdir -p "$OUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
AUDIO_FILE="$OUT_DIR/recording_${TIMESTAMP}.wav"

DURATION=${1:-5}

echo "=========================================================="
echo "         NATIVE KASHMIRI ASR: RECORD & TRANSCRIBE"
echo "=========================================================="
echo "Recording for $DURATION seconds from default microphone..."
echo "Speak in Kashmiri now..."
echo "----------------------------------------------------------"

# Record 16kHz Mono 16-bit PCM WAV using arecord
arecord -D default -f S16_LE -r 16000 -c 1 -d "$DURATION" "$AUDIO_FILE" 2>/dev/null

if [ ! -f "$AUDIO_FILE" ] || [ ! -s "$AUDIO_FILE" ]; then
    echo "Error: Recording failed or no audio captured."
    exit 1
fi

echo "Recording captured: $AUDIO_FILE"
echo "Transcribing using B-01 best_model..."
echo "----------------------------------------------------------"

python3 kashmiri_dataset/scripts/transcribe_audio.py "$AUDIO_FILE"
