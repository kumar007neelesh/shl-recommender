#!/usr/bin/env bash
set -euo pipefail
# Build the catalog on first boot if it isn't present (free hosts allow ~2 min
# for the first /health). Idempotent: skipped once the file exists.
if [ ! -f data/catalog_normalized.json ]; then
  echo "Catalog not found; running ingest..."
  python -m scripts.ingest || echo "Ingest failed; serving with empty catalog."
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
