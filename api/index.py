"""Vercel serverless entrypoint.

Vercel's Python runtime detects the exported ASGI `app`. All routes are rewritten
to this function via vercel.json, so /health and /chat work at the root."""
import sys
from pathlib import Path

# Ensure the project root is importable when Vercel runs this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402,F401
