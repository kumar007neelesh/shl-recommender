"""Local replay harness mirroring the grader's design.

For each trace it spins up a *simulated user* (LLM-driven if a key is set, else a
rule-based answerer) that knows the persona's facts, answers the agent's questions
truthfully, says "no preference" for anything outside its facts, and stops when the
agent returns a shortlist. We then score Recall@10 of the final shortlist against
the trace's labeled expected assessments.

Trace schema is unknown ahead of time, so loading probes a list of likely keys and
prints what it found. Run after ingest + fetch_traces:

    python -m eval.harness
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import Agent  # noqa: E402
from app.catalog import Catalog  # noqa: E402
from app.config import DATA_DIR, get_settings  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.retriever import Retriever  # noqa: E402
from app.schemas import Message  # noqa: E402

_NORM = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _NORM.sub(" ", s.lower()).strip()


def _get(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default


def load_traces(folder: Path) -> List[dict]:
    traces = []
    for p in sorted(folder.glob("**/*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if isinstance(it, dict):
                it["_file"] = p.name
                traces.append(it)
    return traces


def expected_keys(trace: dict) -> Tuple[set, set]:
    """Return (expected_names, expected_urls) however the trace encodes them."""
    raw = _get(trace, ["expected", "expected_shortlist", "labels", "relevant",
                       "gold", "ground_truth", "expected_recommendations"], [])
    names, urls = set(), set()
    if isinstance(raw, dict):
        raw = raw.get("recommendations") or raw.get("items") or list(raw.values())
    for x in raw or []:
        if isinstance(x, str):
            (urls if x.startswith("http") else names).add(
                x if x.startswith("http") else _norm(x))
        elif isinstance(x, dict):
            if x.get("url"):
                urls.add(x["url"])
            if x.get("name"):
                names.add(_norm(x["name"]))
    return names, urls


def recall_at_k(recs, exp_names: set, exp_urls: set, k: int = 10) -> Optional[float]:
    total = len(exp_names | exp_urls)
    if total == 0:
        return None  # unlabeled trace -> excluded from mean
    got_names = {_norm(r.name) for r in recs[:k]}
    got_urls = {r.url for r in recs[:k]}
    hit = 0
    for u in exp_urls:
        if u in got_urls:
            hit += 1
    for n in exp_names:
        if n in got_names or any(n in gn or gn in n for gn in got_names):
            hit += 1
    return hit / total


class SimulatedUser:
    """Answers the agent from the persona's facts; 'no preference' otherwise."""

    def __init__(self, trace: dict, llm: LLMClient):
        self.persona = _get(trace, ["persona", "role", "description"], "")
        self.facts = _get(trace, ["facts", "fact_set", "attributes"], {}) or {}
        self.opening = _get(trace, ["opening", "first_message", "initial_query",
                                    "query", "intent"], None)
        self.llm = llm

    def first_message(self) -> str:
        if self.opening:
            return self.opening
        if isinstance(self.facts, dict) and self.facts.get("intent"):
            return str(self.facts["intent"])
        return f"I'm hiring for: {self.persona or 'a role'}."

    def reply(self, agent_question: str) -> str:
        if self.llm.available:
            out = self.llm.complete_json(
                "You simulate a hiring manager. Answer the assistant's question "
                "truthfully and briefly using ONLY these facts. If the answer isn't "
                "in the facts, reply that you have no preference. "
                'Return JSON {"answer":"..."}.',
                f"Facts: {json.dumps(self.facts)}\nPersona: {self.persona}\n"
                f"Assistant asked: {agent_question}",
            )
            if out and out.get("answer"):
                return out["answer"]
        # rule-based: match question words to fact values
        q = agent_question.lower()
        if isinstance(self.facts, dict):
            for key, val in self.facts.items():
                if str(key).lower() in q:
                    return str(val)
        return "No strong preference."


def run_trace(agent: Agent, trace: dict, llm: LLMClient,
              max_turns: int = 8) -> List:
    user = SimulatedUser(trace, llm)
    messages: List[Message] = [Message(role="user", content=user.first_message())]
    last_recs: List = []
    for _ in range(max_turns):
        resp = agent.respond(messages)
        messages.append(Message(role="assistant", content=resp.reply))
        if resp.recommendations:
            last_recs = resp.recommendations
        if resp.end_of_conversation or resp.recommendations:
            break
        if len(messages) >= max_turns:
            break
        messages.append(Message(role="user", content=user.reply(resp.reply)))
    return last_recs


def main() -> None:
    settings = get_settings()
    catalog = Catalog.from_file(settings.catalog_path)
    agent = Agent(catalog, Retriever(catalog, settings.use_embeddings),
                  LLMClient(settings), settings)
    llm = LLMClient(settings)

    traces = load_traces(DATA_DIR / "traces")
    if not traces:
        print("No traces found. Run scripts/fetch_traces.py first.")
        return

    scores, scored = [], 0
    for t in traces:
        recs = run_trace(agent, t, llm, settings.max_turns)
        en, eu = expected_keys(t)
        r = recall_at_k(recs, en, eu, settings.top_k)
        label = t.get("_file", "trace")
        if r is None:
            print(f"[{label}] unlabeled -> skipped (returned {len(recs)} recs)")
        else:
            scored += 1
            scores.append(r)
            print(f"[{label}] Recall@10 = {r:.2f} ({len(recs)} recs)")
    if scores:
        print(f"\nMean Recall@10 over {scored} labeled traces: "
              f"{sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    main()
