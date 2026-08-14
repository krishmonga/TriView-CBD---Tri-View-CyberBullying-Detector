#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHARED="$(cd "$(dirname "$0")/shared" && pwd)"
MASTER="$ROOT/IEEE_TCSS_TriFuse.tex"
# Body without IEEE-only commands
{ sed -n '76,760p' "$MASTER"; sed -n '764,765p' "$MASTER"; } > "$SHARED/TriFuse_body.tex"
sed -n '767,799p' "$MASTER" > "$SHARED/TriFuse_bibliography.tex"
echo "Synced shared content from $MASTER"
