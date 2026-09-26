#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${1:-$REPO_ROOT/artifacts}"
FILE="sparsity-traffic-forecasting-artifacts.tar.gz"
RECORD="https://zenodo.org/api/records/21944002"
URL="https://zenodo.org/records/21944002/files/${FILE}?download=1"
SHA256="cdbc6293586ffb4fca2006004c7b323aedde5fd037cf00e48151b9800aa8cad4"

if ! curl --fail --silent --show-error "$RECORD" >/dev/null; then
  echo "Zenodo record 21944002 is not public; this repository is not release-ready." >&2
  exit 1
fi
if [[ -e "$DEST/paper-release" ]]; then
  echo "Refusing to overwrite existing $DEST/paper-release" >&2
  exit 1
fi
mkdir -p "$DEST"
TMP_ROOT="${TMPDIR:-/tmp}"
TMP="$(mktemp -d "$TMP_ROOT/sparsity-paper.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
curl --fail --location --output "$TMP/$FILE" "$URL"
printf '%s  %s
' "$SHA256" "$TMP/$FILE" | sha256sum --check
tar -xzf "$TMP/$FILE" -C "$DEST"
echo "Artifacts unpacked to $DEST/paper-release"
