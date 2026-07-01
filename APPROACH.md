# Approach — Conversational SHL Assessment Recommender

## 1. Problem framing

The task is to move a user from a vague intent to a grounded shortlist of SHL
*Individual Test Solutions* over a stateless, non-deterministic, ≤8-turn,
30s/turn conversation, scoring on (a) hard schema/scope evals, (b) Recall@10, and
(c) behavior probes. I treated those three scoring axes as three separate
engineering targets and designed a guard for each, rather than hoping one clever
prompt would satisfy all of them.

The central decision: **the LLM is a router, not a knowledge source.** It decides
*what to do* (clarify / recommend / refine / compare / refuse) and *what to search
for*, but it never emits the assessment list. The list always comes from a
retriever over the scraped catalog, and a final filter drops any item whose URL
isn't in the catalog. This makes hallucinated recommendations structurally
impossible, which directly protects the "items from catalog only" hard eval and
the hallucination probe.

## 2. Data / retrieval

`scripts/ingest.py` downloads the provided catalog JSON
(`…/shl_product_catalog.json`), normalizes it, filters to Individual Test
Solutions (excluding Pre-packaged Job Solutions), and writes a clean file the
service loads at startup. Normalization is deliberately **schema-defensive**: each
field (name, url, description, test_type, category, …) is resolved by probing a
list of likely keys, because the raw field names aren't fully specified up front.
When a record lacks a `test_type`, I backfill it from SHL's standard single-letter
codes (A/B/C/D/E/K/P/S) using keyword inference over name+category+description. The
ingest script prints the individual-vs-job split and a sample record so the field
mapping can be verified against the real data in seconds.

Retrieval is **BM25 with light, documented query expansion** that bridges the gap
between how recruiters speak (roles: "Java developer") and how the catalog labels
tests (skills: "Java", "programming", "coding"). I chose lexical retrieval as the
default over dense embeddings for three reasons: the catalog is small (hundreds of
items), it needs no model download so it deploys on a free tier and stays well
under the 30s cap, and it's deterministic so Recall@10 is reproducible across
runs. Dense retrieval is implemented but optional (`USE_EMBEDDINGS=true`): a MiniLM
encoder fused with BM25 via Reciprocal Rank Fusion. I kept it off by default
because on a small, jargon-heavy catalog the lexical signal is strong and the
dense path adds a cold-start weight download and nondeterminism for marginal gain.

## 3. Agent design (the four behaviors + scope)

The controller emits strict JSON `{action, search_query, test_types,
compare_targets, missing_info, reply}`. Around that LLM decision sit deterministic
guards that always win, so a flaky or adversarial turn can't break the contract:

- **Injection hard-override**: regex injection detection refuses *before* the LLM
  is consulted.
- **No-recommend-on-vague-turn-1**: if the first user turn lacks any role/skill
  signal, the action is forced to `clarify` (empty recommendations).
- **Turn-budget commit**: stalling is the main way to lose Recall@10. When few
  turns remain and there's usable signal, a `clarify` is upgraded to `recommend`
  so the conversation never ends with an empty shortlist.
- **Refine = wider query**: because the API is stateless, every user turn is
  concatenated into the search query. "Actually, add a personality test" simply
  adds tokens, so the shortlist updates instead of restarting — no per-conversation
  state to corrupt.
- **Compare**: named assessments are resolved against the catalog and their *real*
  descriptions are fed to a grounded comparison call ("use only these facts; if a
  fact is absent, say it's not specified"), so comparisons come from catalog data,
  not the model's prior.
- **Scope**: a deterministic classifier handles legal / general-hiring-advice /
  off-topic, with the LLM as the nuanced primary and the classifier as a fallback
  that works even with no LLM.

If no LLM is configured (or a call fails/times out), a **deterministic fallback
controller** takes over: scope-classify → vagueness check → compare-detection →
cumulative-query retrieval. The service therefore always returns valid schema and
never 500s — `LLM_PROVIDER=none` is a fully working mode.

## 4. Evaluation

Three layers, mirroring the grader:

1. **`pytest` (20 tests)** on the deterministic core — normalization, job-solution
   filtering, retrieval, scope, the 10-item cap, and each agent guard
   (turn-1 vague, injection, refine accumulation, compare grounding, turn-budget
   commit). These run offline and are what I iterated against fastest.
2. **`eval/probes.py`** — binary behavior assertions (refuses injection/off-topic/
   legal, no recommend on vague turn-1, recommends on concrete role, all recs in
   catalog, honors refine, never exceeds 10). 9/9 pass on the mini catalog.
3. **`eval/harness.py`** — a replay harness with an LLM-simulated user that answers
   from a persona's facts, says "no preference" outside them, and stops on a
   shortlist; it computes Recall@10 against each trace's labeled set (URL match
   first, normalized-name match as backstop). Trace loading is schema-defensive for
   the same reason ingest is.

**How I measured improvement.** I used probe pass-rate and the unit suite as the
fast inner loop, and mean Recall@10 over the public traces as the outer metric.
Query expansion was added specifically because role-only queries under-retrieved
skill-labeled tests; the turn-budget commit was added after observing simulated
conversations that clarified until the cap and returned nothing.

## 5. What didn't work / trade-offs

- **Letting the LLM produce recommendations directly** was the first design; it
  hallucinated plausible-but-fake assessment names and URLs. Replacing it with
  retrieve-then-ground eliminated that class of failure entirely.
- **Pure embedding retrieval** wasn't worth the cold-start weight download and
  nondeterminism on a small catalog; BM25 + expansion matched or beat it on the
  traces while being faster and reproducible. It remains available behind a flag.
- **Aggressive scope refusal** initially caught legitimate queries containing words
  like "compliance"; I narrowed the legal/advice patterns and made the LLM the
  primary scope classifier with regex only as a fallback and for injection.
- **One-letter `test_type` from inference** is a heuristic; if the real catalog
  ships explicit codes, ingest prefers those and inference never fires.

## 6. AI tools used

I used an agentic coding assistant to scaffold the FastAPI/Pydantic boilerplate,
draft the defensive normalizer, and generate the test cases; every design decision
above (router-not-source, BM25-default, the specific guards, the stateless
refine-as-wider-query trick) is mine and is defended here. No no-code builders were
used. Free-tier **Gemini 2.5 Flash** (Google AI Studio) is the default LLM, called
over plain HTTP; Groq and OpenRouter are supported via the same client by swapping
`LLM_PROVIDER`.
