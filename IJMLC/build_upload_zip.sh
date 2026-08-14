#!/usr/bin/env bash
# Build flat zip for Springer Editorial Manager (no subfolders in zip root)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
OUT="IJMLC_submission.zip"
STAGING=$(mktemp -d)

cp IJMLC_TriFuse_sn.tex IJMLC_body.tex IJMLC_bibliography.tex IJMLC_preamble.tex "$STAGING/"
cp figures/*.tex "$STAGING/"
[[ -f sn-jnl.cls ]] && cp sn-jnl.cls "$STAGING/"
[[ -f sn-mathphys-num.bst ]] && cp sn-mathphys-num.bst "$STAGING/"
[[ -f IJMLC_TriFuse_sn.pdf ]] && cp IJMLC_TriFuse_sn.pdf "$STAGING/"

cd "$STAGING"
zip -j "$DIR/$OUT" ./*
cd "$DIR"
rm -rf "$STAGING"

echo "Created: $DIR/$OUT"
unzip -l "$OUT"
