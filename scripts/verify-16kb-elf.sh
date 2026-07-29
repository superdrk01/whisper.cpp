#!/usr/bin/env bash
set -euo pipefail
READELF=${READELF:-llvm-readelf}
(( $# > 0 )) || { echo "usage: $0 library.so [...]" >&2; exit 2; }
for lib in "$@"; do
  [[ -f "$lib" ]] || { echo "missing: $lib" >&2; exit 2; }
  mapfile -t aligns < <("$READELF" -lW "$lib" | awk '$1 == "LOAD" {print $NF}')
  ((${#aligns[@]} > 0)) || { echo "No LOAD segments: $lib" >&2; exit 1; }
  for a in "${aligns[@]}"; do
    value=$((a))
    if (( value < 16384 )); then
      echo "FAIL: $lib has LOAD alignment $a (< 0x4000)" >&2
      exit 1
    fi
  done
  echo "PASS 16KB ELF alignment: $lib (${aligns[*]})"
done
