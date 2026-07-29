#!/usr/bin/env bash
set -euo pipefail
VERSION="${1:-silero-v6.2.0}"
OUTPUT_DIR="${2:-models}"
mkdir -p "$OUTPUT_DIR"
BASE="https://huggingface.co/ggml-org/whisper-vad/resolve/main"
FILE="ggml-${VERSION}.bin"
URL="$BASE/$FILE"
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 "$URL" -o "$OUTPUT_DIR/$FILE"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUTPUT_DIR/$FILE" "$URL"
else
  echo "curl or wget is required" >&2; exit 2
fi
echo "$OUTPUT_DIR/$FILE"
