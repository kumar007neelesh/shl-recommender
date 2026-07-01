"""Download the real SHL catalog and write the normalized file the service uses.

Run on a machine with internet access (this sandbox can't reach shl.com):

    python -m scripts.ingest

It fetches the catalog JSON, normalizes records, filters to Individual Test
Solutions, and writes data/catalog_normalized.json. It prints a summary and a
sample so you can sanity-check the field mapping against the real schema and tweak
app/catalog.py:_FIELD_ALIASES if any field didn't map."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.catalog import normalize_catalog  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    url = settings.catalog_source_url
    print(f"Fetching catalog: {url}")
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        import json
        raw = json.loads(resp.text, strict=False)

    all_items = normalize_catalog(raw, individual_only=False)
    individual = normalize_catalog(raw, individual_only=True)
    print(f"Normalized records (all): {len(all_items)}")
    print(f"Individual Test Solutions kept: {len(individual)}")
    print(f"Excluded (job solutions / unmapped): {len(all_items) - len(individual)}")

    settings.catalog_path.parent.mkdir(parents=True, exist_ok=True)
    settings.catalog_path.write_text(
        json.dumps(individual, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {settings.catalog_path}")

    if individual:
        print("\nSample normalized record:")
        print(json.dumps(individual[0], ensure_ascii=False, indent=2))
        types = {}
        for it in individual:
            for c in (it["test_type"] or "?"):
                types[c] = types.get(c, 0) + 1
        print("\nTest-type distribution:", dict(sorted(types.items())))
    else:
        print("\nWARNING: no items kept. Inspect the raw schema below and adjust "
              "_FIELD_ALIASES in app/catalog.py.")
        print(json.dumps(raw, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
