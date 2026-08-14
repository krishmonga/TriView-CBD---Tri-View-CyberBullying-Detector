#!/usr/bin/env bash
# Refresh IJMLC content from master manuscript
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER="$DIR/../paper/IEEE_TCSS_TriFuse.tex"

if [[ ! -f "$MASTER" ]]; then
  echo "Master file not found: $MASTER" >&2
  exit 1
fi

awk '/^\\section\{Introduction\}/,/^\\section\*\{Statements and Declarations\}/ { if (!/^\\section\*\{Statements/) print }' "$MASTER" > "$DIR/IJMLC_body.tex"
awk '/^\\begin\{thebibliography\}/,/^\\end\{thebibliography\}/' "$MASTER" > "$DIR/IJMLC_bibliography.tex"

echo "Synced from: $MASTER"
echo "  IJMLC_body.tex        ($(wc -l < "$DIR/IJMLC_body.tex") lines)"
echo "  IJMLC_bibliography.tex ($(wc -l < "$DIR/IJMLC_bibliography.tex") lines)"
echo "Recompile: pdflatex IJMLC_TriFuse_sn.tex"
