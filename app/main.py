"""FastAPI service exposing GET /health and POST /chat.

Stateless: every /chat call carries the full history; we hold no per-conversation
state. The handler is wrapped so that ANY internal error still returns a
schema-valid ChatResponse (empty recommendations) — schema compliance on every
response is a hard-eval requirement."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .agent import Agent
from .catalog import Catalog, normalize_catalog
from .config import get_settings
from .llm import LLMClient
from .retriever import Retriever
from .schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("shl-recommender")

_STATE: dict = {}


def _load_catalog(settings) -> Catalog:
    """Prefer the pre-built local file (fast, no network). If it's absent — e.g. on
    a serverless platform with an ephemeral filesystem where the file wasn't
    bundled — fetch and normalize the catalog from the source URL once, in memory."""
    try:
        catalog = Catalog.from_file(settings.catalog_path)
        log.info("Loaded catalog: %d items from %s", len(catalog),
                 settings.catalog_path)
        return catalog
    except FileNotFoundError:
        log.warning("Local catalog missing; fetching from %s",
                    settings.catalog_source_url)
    try:
        import httpx
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(settings.catalog_source_url)
            resp.raise_for_status()
            raw = resp.json()
        items = normalize_catalog(raw, individual_only=True)
        log.info("Fetched catalog: %d Individual Test Solutions", len(items))
        return Catalog.from_items(items)
    except Exception:
        log.exception("Catalog fetch failed; serving with empty catalog "
                      "(clarify/refuse still work).")
        return Catalog.from_items([])


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    catalog = _load_catalog(settings)
    retriever = Retriever(catalog, use_embeddings=settings.use_embeddings)
    llm = LLMClient(settings)
    _STATE["agent"] = Agent(catalog, retriever, llm, settings)
    _STATE["ready"] = True
    log.info("Service ready. LLM provider=%s available=%s",
             settings.llm_provider, llm.available)
    yield
    _STATE.clear()


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    agent: Agent = _STATE.get("agent")
    if agent is None:
        return ChatResponse(
            reply="The service is still starting up. Please retry in a moment.",
            recommendations=[], end_of_conversation=False,
        )
    try:
        return agent.respond(req.messages)
    except Exception:  # never leak a 500 with a non-conforming body
        log.exception("chat handler error")
        return ChatResponse(
            reply="Sorry, something went wrong on my side. Could you rephrase what "
                  "role or skills you'd like to assess?",
            recommendations=[], end_of_conversation=False,
        )


@app.exception_handler(Exception)
async def _schema_safe_errors(_, exc):  # validation/other -> still valid-ish body
    log.exception("unhandled error: %s", exc)
    return JSONResponse(
        status_code=200,
        content=ChatResponse(
            reply="I couldn't process that request. Please describe the role or "
                  "skills you'd like assessed.",
            recommendations=[], end_of_conversation=False,
        ).model_dump(),
    )
