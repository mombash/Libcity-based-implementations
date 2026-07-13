#!/usr/bin/env bash
# Download Zenodo artifact bundle once DOI is published.
# Usage: ./artifacts/download.sh [DEST_DIR]
# Set ZENODO_RECORD_ID or edit ZENODO_URL below after deposit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${1:-$REPO_ROOT/artifacts/expected_layout}"

# Placeholder until Zenodo deposit is published:
#   Record: https://zenodo.org/records/TBD
#   DOI:    https://doi.org/10.5281/zenodo.TBD
ZENODO_RECORD_ID="${ZENODO_RECORD_ID:-TBD}"
ZENODO_URL="${ZENODO_URL:-https://zenodo.org/records/${ZENODO_RECORD_ID}/files/zenodo-bundle.tar.gz}"

if [[ "$ZENODO_RECORD_ID" == "TBD" ]]; then
  echo "Zenodo record not configured yet (placeholder)."
  echo "  Record page: https://zenodo.org/records/TBD"
  echo "  DOI:         https://doi.org/10.5281/zenodo.TBD"
  echo ""
  echo "1. Upload zenodo-bundle.tar.gz to Zenodo"
  echo "2. Set ZENODO_RECORD_ID (or ZENODO_URL) in this script / env"
  echo "3. Re-run: ./artifacts/download.sh"
  echo ""
  echo "Expected unpack layout: artifacts/expected_layout/README.md"
  exit 1
fi

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading from $ZENODO_URL ..."
curl -L -o "$TMP/zenodo-bundle.tar.gz" "$ZENODO_URL"
echo "Extracting to $DEST ..."
tar -xzf "$TMP/zenodo-bundle.tar.gz" -C "$DEST"
echo "Done. Verify checksums against artifacts/MANIFEST.json"
