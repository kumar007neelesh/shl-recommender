# SHL Conversational Assessment Recommender

A stateless FastAPI agent that takes a hiring manager from a vague intent
("I'm hiring a Java developer") to a grounded shortlist of SHL **Individual Test
Solutions**, through dialogue. It clarifies, recommends, refines, compares, and
refuses out-of-scope requests — and **never recommends anything that isn't in the
scraped catalog**. LLM: **Google Gemini 2.5 Flash** (free tier).

---

## 1. Get a free Gemini API key (2 minutes)

1. Go to **https://aistudio.google.com/apikey** and sign in with a Google account.
2. Click **Create API key** (no credit card needed). Copy the key (starts with `AIza…`).

Free-tier `gemini-2.5-flash` allows ~10 requests/min and ~250/day — plenty for
local testing and the grader. If you run the eval harness a lot, switch to
`gemini-2.5-flash-lite` (higher daily limit) via `LLM_MODEL`.

---

## 2. Run locally

```bash
# from the project root:
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# configure the LLM
cp .env.example .env
#   then edit .env and paste your key into LLM_API_KEY

# build the catalog from the real SHL dataset (needs internet)
python -m scripts.ingest          # writes data/catalog_normalized.json
#   -> prints how many Individual Test Solutions were kept + a sample record.
#      If 0 kept, adjust _FIELD_ALIASES in app/catalog.py per the printed schema.

# run the API
uvicorn app.main:app --reload     # serves on http://127.0.0.1:8000
```

Test it:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl -X POST http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]}'
```

Interactive docs: `http://127.0.0.1:8000/docs`.

> No key? It still runs. With `LLM_PROVIDER=none` (or a missing key) a deterministic
> controller still clarifies, refuses, retrieves, and obeys the schema — lower
> quality, but nothing crashes and every response is valid.

---

## 3. Evaluate

```bash
python -m scripts.fetch_traces    # downloads the 10 public traces -> data/traces/
python -m eval.harness            # Recall@10 via an LLM-simulated user
python -m eval.probes             # behavior assertions (refusal, turn-1, refine…)
pytest -q                         # 20 offline unit tests on the deterministic core
```

Tip: for the harness (many LLM calls) set `LLM_MODEL=gemini-2.5-flash-lite` to
avoid hitting the daily free-tier limit.

---

## 4. Deploy on Vercel (to test the public endpoint)

Vercel runs FastAPI as a single Python serverless function. Two files make this
work and are included: **`api/index.py`** (exports the ASGI `app`) and
**`vercel.json`** (routes every path to it and sets a 60s timeout, above the
grader's 30s cap).

### Steps

1. **Build the catalog and commit it** so it's bundled into the deploy:

   ```bash
   python -m scripts.ingest
   git init && git add . && git commit -m "SHL recommender"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   `.gitignore` is set so `data/catalog_normalized.json` **is** committed. (If you
   skip it, the app fetches the catalog from the source URL on first cold start —
   slower first request, still works.)

2. **Import on Vercel:** https://vercel.com → **Add New… → Project** → import your
   GitHub repo. It auto-detects Python/FastAPI. Leave build settings default.

3. **Add environment variables** (Project → Settings → Environment Variables):

   | Key            | Value              |
   |----------------|--------------------|
   | `LLM_PROVIDER` | `gemini`           |
   | `LLM_MODEL`    | `gemini-2.5-flash` |
   | `LLM_API_KEY`  | *(your AIza… key)* |

4. **Deploy.** You get a URL like `https://<project>.vercel.app`.

5. **Verify** (first hit may be a slow cold start; the grader allows up to 2 min on
   the first `/health`):

   ```bash
   curl https://<project>.vercel.app/health
   curl -X POST https://<project>.vercel.app/chat \
     -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"Hiring a Java developer"}]}'
   ```

Submit `https://<project>.vercel.app` as your endpoint.

### CLI alternative

```bash
npm i -g vercel
vercel                       # first deploy (follow prompts)
vercel env add LLM_API_KEY   # paste key; repeat for LLM_PROVIDER, LLM_MODEL
vercel --prod                # promote to the production URL
```

### Honest caveat about Vercel

Vercel is serverless: **cold starts add latency** and the filesystem is ephemeral.
It works fine here (catalog is committed / loaded into memory, Gemini Flash is
fast), but if a cold start ever risks the 30s cap, the assignment's suggested hosts
— **Render, Railway, Fly, Modal, HF Spaces** — keep a warm process and are steadier.
`render.yaml` and `Dockerfile` are included. Render: push to GitHub → New Web
Service → pick the repo → set `LLM_API_KEY` secret → deploy (`buildCommand` runs
ingest; `healthCheckPath` is `/health`).

### Keep-alive (optional, prevents Render idle-sleep)

`.github/workflows/keep-alive.yml` pings `/health` every 5 minutes so the free
Render service doesn't cold-start during grading. To enable it: set a repository
**variable** `SERVICE_URL` = your `https://<service>.onrender.com` URL under
**Settings → Secrets and variables → Actions → Variables**. Read the header
comment in that file first — on a **private** repo, run it only during the grading
window (Actions minutes round up to 1/run and would exceed the free cap if left on
permanently); on a **public** repo it's free to leave running.

---

## Architecture (one screen)

```
POST /chat (stateless history)
        |
        v
   injection? ---------------> REFUSE (deterministic, hard-override)
        |
   LLM controller (Gemini) decides action + search query   [fallback: rules]
        | clarify / recommend / refine / compare / refuse
        v
   guards: no-recommend-on-vague-turn-1 . turn-budget commit . cap@10
        |
        v
   Retriever (BM25 + role->skill expansion, optional embeddings)
        | candidates
        v
   Grounding: keep ONLY URLs present in the catalog --> recommendations[]
```

The LLM decides *what to do* and *what to search for*; the retriever supplies the
items; a final filter drops any URL not in the catalog. Hallucinated
recommendations are therefore structurally impossible. See `APPROACH.md`.

## Layout

```
api/        Vercel entrypoint (exports the ASGI app)
app/        FastAPI service, agent, retriever, catalog, scope guard, LLM client
scripts/    ingest.py (catalog), fetch_traces.py
eval/       harness.py (Recall@10), probes.py (behavior assertions)
tests/      offline unit tests + mini catalog fixture
vercel.json / Dockerfile / render.yaml   deploy configs
```
