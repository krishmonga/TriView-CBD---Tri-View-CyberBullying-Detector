#!/usr/bin/env bash
# Sync IJMLC submission files from master manuscript (IEEE_TCSS_TriFuse.tex)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER="$DIR/../../IEEE_TCSS_TriFuse.tex"

if [[ ! -f "$MASTER" ]]; then
  echo "Master file not found: $MASTER" >&2
  exit 1
fi

awk '/^\\section\{Introduction\}/,/^\\section\*\{Statements and Declarations\}/ { if (!/^\\section\*\{Statements/) print }' "$MASTER" > "$DIR/IJMLC_body.tex"
awk '/^\\begin\{thebibliography\}/,/^\\end\{thebibliography\}/' "$MASTER" > "$DIR/IJMLC_bibliography.tex"

echo "Synced IJMLC_body.tex ($(wc -l < "$DIR/IJMLC_body.tex") lines)"
echo "Synced IJMLC_bibliography.tex ($(wc -l < "$DIR/IJMLC_bibliography.tex") lines)"
echo "Done. Recompile: pdflatex IJMLC_TriFuse.tex"
