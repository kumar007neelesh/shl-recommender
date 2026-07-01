"""Scope and prompt-injection guard.

The agent must stay on-topic (SHL assessments only) and resist injection. The LLM
does nuanced classification, but these deterministic checks are a safety net that
works even if the LLM is unavailable, and injection detection here HARD-overrides
to a refusal regardless of what the LLM says."""
from __future__ import annotations

import re
from typing import Optional

_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier)",
    r"disregard (all |the )?(previous|prior|above|instructions)",
    r"forget (your|all|the|previous) (instructions|rules|prompt)",
    r"reveal (your|the) (system )?(prompt|instructions)",
    r"what (is|are) your (system )?(prompt|instructions)",
    r"you are now\b",
    r"act as (?!a hiring|an? recruiter)",   # roleplay hijack, not legitimate persona
    r"developer mode",
    r"jailbreak",
    r"print (your|the) (prompt|system)",
    r"repeat (the|your) (system )?prompt",
]

_LEGAL_PATTERNS = [
    r"\bis it legal\b", r"\blawsuit\b", r"\bsue\b", r"\blegally\b",
    r"discrimination (law|case|claim)", r"\beeoc\b", r"\bgdpr\b",
    r"adverse impact (lawsuit|liability)", r"can i be sued",
]

# General hiring/HR advice that is NOT about choosing an SHL assessment.
_HIRING_ADVICE_PATTERNS = [
    r"how (do|should) i (interview|onboard|fire|negotiate)",
    r"write (a |me a )?(job description|offer letter|rejection)",
    r"how much should i pay", r"what salary", r"salary range",
    r"how to (conduct|run) an interview",
    r"should i hire (him|her|them|this)",
]

# Clearly unrelated domains.
_OFFTOPIC_PATTERNS = [
    r"\bweather\b", r"\bstock price\b", r"\brecipe\b", r"\btranslate\b",
    r"\bwrite (me )?(a|an) (poem|essay|story|code|python|sql)\b",
    r"\bwho (won|is the president)\b", r"\bcapital of\b",
]

_COMPILED = {
    "injection": [re.compile(p, re.I) for p in _INJECTION_PATTERNS],
    "legal": [re.compile(p, re.I) for p in _LEGAL_PATTERNS],
    "hiring_advice": [re.compile(p, re.I) for p in _HIRING_ADVICE_PATTERNS],
    "offtopic": [re.compile(p, re.I) for p in _OFFTOPIC_PATTERNS],
}

REFUSAL_TEXT = {
    "injection": (
        "I can only help you find SHL assessments from the catalog, so I can't "
        "follow that instruction. Tell me about the role you're hiring for and I'll "
        "suggest relevant assessments."
    ),
    "legal": (
        "I'm not able to give legal advice. I can only help you choose SHL "
        "assessments for a role. What role are you hiring for?"
    ),
    "hiring_advice": (
        "I focus on recommending SHL assessments rather than general hiring advice. "
        "If you tell me about the role and skills you're screening for, I can suggest "
        "assessments that fit."
    ),
    "offtopic": (
        "I can only help with selecting SHL assessments. Describe the role or skills "
        "you want to evaluate and I'll recommend assessments from the catalog."
    ),
}


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED["injection"])


def classify_scope(text: str) -> Optional[str]:
    """Return a refusal category if the text is out of scope, else None.
    Injection is checked first and always wins."""
    if detect_injection(text):
        return "injection"
    for cat in ("legal", "hiring_advice", "offtopic"):
        if any(p.search(text) for p in _COMPILED[cat]):
            return cat
    return None
