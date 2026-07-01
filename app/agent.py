"""The agent: turns a stateless conversation history into a schema-valid reply.

Layering (outer guards always win over the LLM):
  1. Injection hard-override (deterministic) -> refuse.
  2. LLM controller decides action + search query (or deterministic fallback).
  3. Turn-1-vague guard: never recommend on the first turn for a vague query.
  4. Turn-budget guard: near the cap, commit to a shortlist instead of stalling.
  5. Grounding: recommendations come from the retriever and are filtered to URLs
     that actually exist in the catalog. Capped at 10.
This keeps a non-deterministic conversation from breaking the contract."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .catalog import Catalog
from .config import Settings
from .llm import LLMClient
from .prompts import (
    COMPARE_SYSTEM,
    CONTROLLER_SYSTEM,
    compare_user,
    controller_user,
)
from .retriever import Retriever
from .schemas import ChatResponse, Message, Recommendation
from .scope import REFUSAL_TEXT, classify_scope, detect_injection

_ACTIONABLE_KEYWORDS = {
    "developer", "engineer", "manager", "sales", "analyst", "java", "python",
    "javascript", "sql", "data", "personality", "cognitive", "leadership",
    "graduate", "verbal", "numerical", "coding", "programming", "technical",
    "accountant", "nurse", "designer", "consultant", "customer", "support",
    "marketing", "finance", "hr", "recruiter", "administrator", "clerical",
    "supervisor", "executive", "intern", "contact centre", "call center",
}
_COMPARE_RE = re.compile(
    r"(difference between|compare|vs\.?|versus|which is better)", re.I
)


class Agent:
    def __init__(self, catalog: Catalog, retriever: Retriever, llm: LLMClient,
                 settings: Settings):
        self.catalog = catalog
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    # -- public ---------------------------------------------------------------
    def respond(self, messages: List[Message]) -> ChatResponse:
        user_msgs = [m for m in messages if m.role == "user"]
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        last_user = user_msgs[-1].content if user_msgs else ""
        turns_used = len(user_msgs) + len(assistant_msgs)

        # 1. Injection hard-override.
        if detect_injection(last_user):
            return self._refuse("injection")

        decision = self._decide(messages, last_user, turns_used)
        action = decision.get("action", "clarify")

        # 3. Don't recommend on turn 1 for a vague query.
        is_first_user_turn = len(user_msgs) == 1
        if is_first_user_turn and self._looks_vague(last_user) and action in {
            "recommend", "refine"
        }:
            action = "clarify"
            decision.setdefault(
                "reply",
                "Happy to help. What role are you hiring for, and which skills or "
                "qualities matter most?",
            )

        # 4. Turn-budget guard: if we're about to run out of turns, commit.
        # Assistant reply we are about to send is one more turn.
        remaining = self.settings.max_turns - (turns_used + 1)
        if action == "clarify" and remaining <= 1 and self._has_signal(user_msgs):
            action = "recommend"
            decision["search_query"] = decision.get("search_query") or self._cumulative_query(user_msgs)

        if action == "refuse":
            return self._refuse(decision.get("refuse_reason") or "offtopic",
                                reply=decision.get("reply"))
        if action == "compare":
            return self._compare(decision, last_user)
        if action == "clarify":
            reply = decision.get("reply") or (
                "Could you tell me more about the role and the skills you want to "
                "assess?"
            )
            return ChatResponse(reply=reply, recommendations=[],
                                end_of_conversation=False)
        # recommend / refine
        return self._recommend(decision, user_msgs)

    # -- decision -------------------------------------------------------------
    def _decide(self, messages: List[Message], last_user: str,
                turns_used: int) -> Dict:
        # deterministic scope first (cheap, reliable)
        scope = classify_scope(last_user)
        if scope:
            return {"action": "refuse", "refuse_reason": scope}

        if self.llm.available:
            convo = self._render_conversation(messages)
            out = self.llm.complete_json(
                CONTROLLER_SYSTEM,
                controller_user(convo, turns_used, self.settings.max_turns),
            )
            if out and out.get("action") in {
                "clarify", "recommend", "refine", "compare", "refuse"
            }:
                return out
        # deterministic fallback
        return self._fallback_decision(messages)

    def _fallback_decision(self, messages: List[Message]) -> Dict:
        user_msgs = [m for m in messages if m.role == "user"]
        last = user_msgs[-1].content if user_msgs else ""
        if _COMPARE_RE.search(last):
            return {"action": "compare", "compare_targets": self._guess_targets(last)}
        if len(user_msgs) == 1 and self._looks_vague(last):
            return {
                "action": "clarify",
                "reply": "What role are you hiring for, and which skills matter most?",
            }
        if self._has_signal(user_msgs):
            return {"action": "recommend",
                    "search_query": self._cumulative_query(user_msgs)}
        return {
            "action": "clarify",
            "reply": "Tell me about the role and the skills you'd like to assess.",
        }

    # -- branches -------------------------------------------------------------
    def _recommend(self, decision: Dict, user_msgs: List[Message]) -> ChatResponse:
        query = decision.get("search_query") or self._cumulative_query(user_msgs)
        test_types = decision.get("test_types") or None
        hits = self.retriever.search(
            query, k=self.settings.top_k, test_types=test_types
        )
        if not hits:  # filter too strict -> retry unfiltered
            hits = self.retriever.search(query, k=self.settings.top_k)
        recs = self._ground(hits)
        if not recs:
            return ChatResponse(
                reply="I couldn't find a confident match yet. Could you add a "
                      "little more detail about the role or skills?",
                recommendations=[], end_of_conversation=False,
            )
        reply = decision.get("reply") or (
            f"Here are {len(recs)} SHL assessments that fit what you described."
        )
        return ChatResponse(reply=reply, recommendations=recs,
                            end_of_conversation=True)

    def _compare(self, decision: Dict, last_user: str) -> ChatResponse:
        targets = decision.get("compare_targets") or self._guess_targets(last_user)
        items = [it for it in (self.catalog.find_by_name(t) for t in targets) if it]
        if len(items) < 2:
            # fall back to retrieval if names not resolvable
            hits = self.retriever.search(last_user, k=2)
            items = [h.item for h in hits]
        if len(items) < 2:
            return ChatResponse(
                reply="I couldn't identify both assessments in the catalog. Which "
                      "two would you like compared?",
                recommendations=[], end_of_conversation=False,
            )
        reply = self._compare_text(last_user, items)
        recs = self._ground([type("S", (), {"item": it})() for it in items])
        return ChatResponse(reply=reply, recommendations=recs,
                            end_of_conversation=True)

    def _compare_text(self, question: str, items: List[Dict]) -> str:
        block = "\n".join(
            f"- {it['name']} (type {it['test_type'] or 'n/a'}): "
            f"{it['description'] or 'no description in catalog'}"
            for it in items
        )
        if self.llm.available:
            out = self.llm.complete_json(
                COMPARE_SYSTEM + " Return JSON {\"text\": \"...\"}.",
                compare_user(question, block),
            )
            if out and out.get("text"):
                return out["text"]
        # deterministic grounded fallback
        return "Here's how they compare, based on the catalog:\n" + block

    def _refuse(self, reason: str, reply: Optional[str] = None) -> ChatResponse:
        return ChatResponse(
            reply=reply or REFUSAL_TEXT.get(reason, REFUSAL_TEXT["offtopic"]),
            recommendations=[], end_of_conversation=False,
        )

    # -- helpers --------------------------------------------------------------
    def _ground(self, hits) -> List[Recommendation]:
        """Only keep items whose URL is genuinely in the catalog. Dedupe + cap 10."""
        recs: List[Recommendation] = []
        seen = set()
        for h in hits:
            it = h.item
            url = it.get("url", "")
            if not self.catalog.is_valid_url(url) or url in seen:
                continue
            seen.add(url)
            recs.append(Recommendation(
                name=it["name"], url=url, test_type=it.get("test_type", "")
            ))
            if len(recs) >= self.settings.top_k:
                break
        return recs

    @staticmethod
    def _render_conversation(messages: List[Message]) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in messages)

    @staticmethod
    def _cumulative_query(user_msgs: List[Message]) -> str:
        # Joining ALL user turns means added constraints (refine) are included.
        return " ".join(m.content for m in user_msgs)

    @staticmethod
    def _looks_vague(text: str) -> bool:
        t = text.lower().strip()
        if len(t) > 140:  # a pasted JD is not vague
            return False
        if any(k in t for k in _ACTIONABLE_KEYWORDS):
            return False
        return True

    def _has_signal(self, user_msgs: List[Message]) -> bool:
        return not self._looks_vague(self._cumulative_query(user_msgs))

    @staticmethod
    def _guess_targets(text: str) -> List[str]:
        # split on "between X and Y", "X vs Y", "X versus Y"
        m = re.search(r"between (.+?) and (.+?)[\?\.]?$", text, re.I)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)[\?\.]?$", text, re.I)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        return []
