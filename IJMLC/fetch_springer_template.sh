#!/usr/bin/env bash
# Download Springer Nature LaTeX journal template (sn-jnl.cls)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

URL="https://resource-cms.springernature.com/springer-cms/rest/v1/content/19238648/data/v12"
ZIP="springer-sn-jnl-template.zip"

echo "Downloading Springer Nature LaTeX template..."
if curl -fsSL -o "$ZIP" "$URL" && file "$ZIP" | grep -qi zip; then
  rm -rf springer-template-extract
  unzip -o "$ZIP" -d springer-template-extract
  find springer-template-extract -name 'sn-jnl.cls' -exec cp {} . \;
  find springer-template-extract -name 'sn-mathphys-num.bst' -exec cp {} . \;
  echo "Done. Files copied to $DIR"
  ls -la sn-jnl.cls sn-mathphys-num.bst 2>/dev/null || true
else
  echo "Automatic download failed."
  echo "Download manually from:"
  echo "  https://www.springernature.com/gp/authors/campaigns/latex-author-support"
  echo "Copy sn-jnl.cls and sn-mathphys-num.bst into: $DIR"
  exit 1
fi
