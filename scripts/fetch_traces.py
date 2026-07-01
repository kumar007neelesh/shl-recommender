"""Download the 10 public conversation traces and unzip them into data/traces/.

    python -m scripts.fetch_traces
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, get_settings  # noqa: E402


def main() -> None:
    url = get_settings().traces_source_url
    out = DATA_DIR / "traces"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Fetching traces: {url}")
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(out)
        names = zf.namelist()
    print(f"Extracted {len(names)} files into {out}")
    for n in names[:20]:
        print(" -", n)


if __name__ == "__main__":
    main()
