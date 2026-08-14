#!/usr/bin/env bash
# Zip IJMLC project for Overleaf upload (keeps figures/ subfolder)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
OUT="IJMLC_overleaf.zip"

zip -r "$OUT" \
  main.tex \
  IJMLC_TriFuse_sn.tex \
  IJMLC_TriFuse.tex \
  IJMLC_body.tex \
  IJMLC_bibliography.tex \
  IJMLC_preamble.tex \
  COVER_LETTER.txt \
  OVERLEAF.md \
  README.md \
  figures/ \
  -x "*.zip"

[[ -f sn-jnl.cls ]] && zip -u "$OUT" sn-jnl.cls
[[ -f sn-mathphys-num.bst ]] && zip -u "$OUT" sn-mathphys-num.bst

echo "Created: $DIR/$OUT"
unzip -l "$OUT" | head -30
