"""Wire schemas. The assignment says the schema is non-negotiable, so these models
are the single source of truth and every response is validated against them before
it leaves the service."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages must contain at least one item")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    # SHL single-letter test-type code (A,B,C,D,E,K,P,S). Kept as-is from catalog.
    test_type: str = ""


class ChatResponse(BaseModel):
    reply: str
    # EMPTY while clarifying or refusing; 1..10 items once committed to a shortlist.
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def _cap_ten(cls, v: List[Recommendation]) -> List[Recommendation]:
        # Hard guard: never exceed 10, even if an upstream bug tries to.
        return v[:10]


class HealthResponse(BaseModel):
    status: str = "ok"
