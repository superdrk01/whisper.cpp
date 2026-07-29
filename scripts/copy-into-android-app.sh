#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <unpacked-artifact-dir> <android-app-module-dir>" >&2
  exit 2
fi

ARTIFACT_DIR=$(cd "$1" && pwd)
APP_DIR=$(cd "$2" && pwd)

for ABI_DIR in "$ARTIFACT_DIR"/*; do
  [[ -d "$ABI_DIR/lib" ]] || continue
  ABI=$(basename "$ABI_DIR")
  DEST="$APP_DIR/src/main/jniLibs/$ABI"
  mkdir -p "$DEST"
  cp "$ABI_DIR/lib/libwhisper_android.so" "$DEST/"
  cp "$ABI_DIR/lib/libc++_shared.so" "$DEST/"
  echo "Installed $ABI into $DEST"
done
