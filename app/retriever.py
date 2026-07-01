"""Retrieval over the catalog.

Default is a lexical BM25 retriever with light, documented query expansion
(users speak in *roles*, the catalog speaks in *skills*). Lexical retrieval is the
pragmatic choice here: the catalog is small, it needs no model download, it is fast
enough for the 30s/turn cap on free hosting, and it is fully deterministic which
makes Recall@10 reproducible. Embeddings are an optional, RRF-fused enhancement
(disabled by default) so the stack stays free and dependency-light."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from rank_bm25 import BM25Okapi

from .catalog import Catalog

_TOKEN_RE = re.compile(r"[a-z0-9+#]+")

# Modest role/intent -> vocabulary expansion. Boosts recall by bridging the gap
# between how recruiters describe a hire and how the catalog labels a test.
QUERY_EXPANSION = {
    "java": ["java", "programming", "coding", "developer", "software"],
    "python": ["python", "programming", "coding", "developer", "software"],
    "developer": ["programming", "coding", "software", "technical"],
    "engineer": ["technical", "programming", "engineering"],
    "frontend": ["javascript", "html", "css", "web", "programming"],
    "backend": ["programming", "api", "database", "sql"],
    "sql": ["sql", "database", "data"],
    "data": ["data", "analytics", "sql", "numerical"],
    "personality": ["personality", "opq", "behavioral", "behavioural", "motivation"],
    "cognitive": ["cognitive", "ability", "reasoning", "verify", "aptitude"],
    "leadership": ["leadership", "management", "manager", "competency"],
    "manager": ["management", "leadership", "competency", "supervisor"],
    "sales": ["sales", "customer", "service", "persuasion"],
    "stakeholder": ["communication", "interpersonal", "collaboration", "teamwork"],
    "graduate": ["graduate", "entry", "early career", "campus"],
    "verbal": ["verbal", "reasoning", "comprehension"],
    "numerical": ["numerical", "reasoning", "numeracy"],
}


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def expand(tokens: Sequence[str]) -> List[str]:
    out = list(tokens)
    for t in tokens:
        out.extend(QUERY_EXPANSION.get(t, []))
    return out


@dataclass
class ScoredItem:
    item: Dict
    score: float


class Retriever:
    def __init__(self, catalog: Catalog, use_embeddings: bool = False):
        self.catalog = catalog
        self.use_embeddings = use_embeddings
        self._corpus_tokens = [tokenize(it["search_text"]) for it in catalog.items]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        self._embedder = None
        self._embeddings = None
        if use_embeddings:
            self._init_embeddings()

    def _init_embeddings(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # lazy, optional
            import numpy as np  # noqa: F401

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [it["search_text"] for it in self.catalog.items]
            self._embeddings = self._embedder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            # Any failure (missing package, no model) -> silently fall back to BM25.
            self.use_embeddings = False

    # -- scoring ---------------------------------------------------------------
    def _bm25_scores(self, query: str) -> List[float]:
        if not self._bm25:
            return []
        q = expand(tokenize(query))
        return list(self._bm25.get_scores(q))

    def _emb_ranking(self, query: str) -> List[int]:
        if not (self.use_embeddings and self._embedder is not None):
            return []
        import numpy as np

        qv = self._embedder.encode([query], normalize_embeddings=True)[0]
        sims = self._embeddings @ qv
        return list(np.argsort(-sims))

    @staticmethod
    def _rrf(rankings: List[List[int]], n: int, k: int = 60) -> List[float]:
        scores = [0.0] * n
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] += 1.0 / (k + rank + 1)
        return scores

    def search(
        self,
        query: str,
        k: int = 10,
        test_types: Optional[Sequence[str]] = None,
    ) -> List[ScoredItem]:
        if not self.catalog.items:
            return []
        bm25 = self._bm25_scores(query)
        order = sorted(range(len(bm25)), key=lambda i: -bm25[i])

        if self.use_embeddings:
            emb_order = self._emb_ranking(query)
            bm25_order = order
            fused = self._rrf([bm25_order, emb_order], n=len(self.catalog.items))
            order = sorted(range(len(fused)), key=lambda i: -fused[i])
            score_lookup = fused
        else:
            score_lookup = bm25

        wanted = {t.upper() for t in test_types} if test_types else None
        results: List[ScoredItem] = []
        for i in order:
            it = self.catalog.items[i]
            if wanted and not (set(it.get("test_type", "").upper()) & wanted):
                continue
            results.append(ScoredItem(item=it, score=float(score_lookup[i])))
            if len(results) >= k:
                break
        return results
