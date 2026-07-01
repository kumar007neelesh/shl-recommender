"""Catalog loading and normalization.

The raw SHL JSON field names are not fully known ahead of time, so normalization
is intentionally defensive: it probes a list of likely keys for each field and
falls back gracefully. ingest.py runs this once and writes a clean file; the
service loads only the clean file at startup.

Grounding rule for the whole system: a recommendation is valid ONLY if its URL is
present in `Catalog.urls`. This is the anti-hallucination backstop."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# SHL standard test-type codes, used to backfill test_type when the raw record
# only carries a human-readable category.
TEST_TYPE_KEYWORDS = {
    "A": ["ability", "aptitude", "cognitive", "reasoning", "verify"],
    "B": ["biodata", "situational judgement", "situational judgment", "sjt"],
    "C": ["competency", "competencies", "behavioural", "behavioral"],
    "D": ["development", "360", "multi-rater", "feedback"],
    "E": ["exercise", "assessment exercise", "in-tray", "case study"],
    "K": ["knowledge", "skill", "technical", "coding", "programming", "language test"],
    "P": ["personality", "opq", "motivation", "behavioral style", "behavioural style"],
    "S": ["simulation", "job simulation", "virtual assessment"],
}

_FIELD_ALIASES = {
    "name": ["name", "title", "product_name", "assessment_name", "product", "solution"],
    "url": ["url", "link", "product_url", "href", "page_url", "catalog_url"],
    "description": ["description", "desc", "summary", "details", "about", "overview", "text"],
    "test_type": ["test_type", "test_types", "type", "test_type_code", "category_code"],
    "category": ["category", "solution_type", "product_type", "group", "section"],
    "job_levels": ["job_levels", "job_level", "level", "levels", "seniority"],
    "languages": ["languages", "language", "available_languages"],
    "duration": ["duration", "assessment_length", "length", "time", "minutes"],
    "remote": ["remote", "remote_testing", "remote_support", "is_remote"],
}


def _first(record: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in record and record[k] not in (None, "", []):
            return record[k]
        # case-insensitive fallback
        for rk in record:
            if rk.lower() == k.lower() and record[rk] not in (None, "", []):
                return record[rk]
    return None


def _as_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return ", ".join(str(x) for x in val)
    return str(val).strip()


def _as_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    return [s.strip() for s in str(val).split(",") if s.strip()]


def _infer_test_type(text: str) -> str:
    low = text.lower()
    hits = [code for code, kws in TEST_TYPE_KEYWORDS.items() if any(k in low for k in kws)]
    return "".join(hits)


def normalize_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map one raw catalog record into the normalized shape. Returns None if it
    lacks the minimum (name + url) needed to be recommendable."""
    name = _as_str(_first(record, _FIELD_ALIASES["name"]))
    url = _as_str(_first(record, _FIELD_ALIASES["url"]))
    if not name or not url:
        return None

    description = _as_str(_first(record, _FIELD_ALIASES["description"]))
    category = _as_str(_first(record, _FIELD_ALIASES["category"]))
    test_type = _as_str(_first(record, _FIELD_ALIASES["test_type"]))
    if not test_type:
        test_type = _infer_test_type(f"{name} {category} {description}")

    job_levels = _as_list(_first(record, _FIELD_ALIASES["job_levels"]))
    languages = _as_list(_first(record, _FIELD_ALIASES["languages"]))
    duration = _as_str(_first(record, _FIELD_ALIASES["duration"]))
    remote = _as_str(_first(record, _FIELD_ALIASES["remote"]))

    search_text = " | ".join(
        p for p in [name, test_type, category, description, " ".join(job_levels)] if p
    )
    return {
        "name": name,
        "url": url,
        "test_type": test_type,
        "description": description,
        "category": category,
        "job_levels": job_levels,
        "languages": languages,
        "duration": duration,
        "remote": remote,
        "search_text": search_text,
    }


def _is_individual_test_solution(record: Dict[str, Any], normalized: Dict[str, Any]) -> bool:
    """The assignment restricts scope to Individual Test Solutions and excludes
    Pre-packaged Job Solutions. Detect both raw flags and category text."""
    blob = json.dumps(record).lower() + " " + normalized["category"].lower()
    if "pre-packaged" in blob or "prepackaged" in blob or "job solution" in blob:
        # explicit job-solution marker -> excluded, unless also flagged individual
        if "individual" not in blob:
            return False
    # If a record explicitly says individual, keep it; otherwise default to keep
    # (many catalogs only tag the excluded group). ingest.py logs the split so the
    # user can verify the filter against the real data.
    return True


def normalize_catalog(
    raw: Any, individual_only: bool = True
) -> List[Dict[str, Any]]:
    # Accept either a bare list or a dict wrapping the list under a common key.
    if isinstance(raw, dict):
        for key in ("products", "items", "catalog", "data", "assessments", "results"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raw = [raw]
    out: List[Dict[str, Any]] = []
    seen = set()
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        norm = normalize_record(rec)
        if norm is None:
            continue
        if individual_only and not _is_individual_test_solution(rec, norm):
            continue
        if norm["url"] in seen:
            continue
        seen.add(norm["url"])
        out.append(norm)
    return out


@dataclass
class Catalog:
    items: List[Dict[str, Any]]
    by_url: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    urls: set = field(default_factory=set)

    @classmethod
    def from_file(cls, path: Path) -> "Catalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_items(data)

    @classmethod
    def from_items(cls, items: List[Dict[str, Any]]) -> "Catalog":
        by_url = {it["url"]: it for it in items}
        return cls(items=items, by_url=by_url, urls=set(by_url.keys()))

    def __len__(self) -> int:
        return len(self.items)

    def is_valid_url(self, url: str) -> bool:
        return url in self.urls

    def find_by_name(self, query: str) -> Optional[Dict[str, Any]]:
        q = query.strip().lower()
        if not q:
            return None
        # exact, then substring either direction
        for it in self.items:
            if it["name"].lower() == q:
                return it
        for it in self.items:
            n = it["name"].lower()
            if q in n or n in q:
                return it
        return None
