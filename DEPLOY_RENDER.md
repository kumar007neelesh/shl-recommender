# Deploying the SHL Recommender on Render — Detailed Guide

This walks through deploying the FastAPI service to Render's free tier end to end,
using **Gemini** as the LLM. Two methods are covered: **Blueprint** (uses the
included `render.yaml`, fastest) and **Manual Web Service** (full dashboard control).
Pick one.

---

## 0. Prerequisites

1. **A Gemini API key.** Get one free at https://aistudio.google.com/apikey
   (no credit card). It starts with `AIza…`.
2. **The code on GitHub.** Render deploys from a Git repo. From the project root:
   ```bash
   python -m scripts.ingest            # build data/catalog_normalized.json (needs internet)
   git init
   git add .
   git commit -m "SHL recommender"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
   Committing the catalog (`.gitignore` allows it) makes the first request fast. If
   you skip it, the service fetches the catalog from the source URL on first boot.
3. **A Render account** — sign in with GitHub at https://dashboard.render.com so
   Render can see your repos.

---

## Method A — Blueprint (recommended)

The repo already contains `render.yaml`, which declares the service, the free plan,
the build/start commands, the health-check path, the pinned Python version, and the
Gemini env vars (with the API key marked as a prompt-for secret).

1. Render Dashboard → **New +** → **Blueprint**.
2. Select your repository. Render parses `render.yaml` and shows a service named
   **`shl-recommender`**.
3. It prompts for the one secret value (`LLM_API_KEY`, marked `sync: false` in the
   YAML). Paste your Gemini key.
4. Click **Apply**. Render runs the build (`pip install` + catalog ingest) and then
   starts Uvicorn.
5. When the deploy finishes, your service is live at
   `https://shl-recommender-XXXX.onrender.com`.

Skip to **Verify the deployment** below.

---

## Method B — Manual Web Service

Use this if you'd rather configure everything in the dashboard.

1. Render Dashboard → **New +** → **Web Service**.
2. Connect the repo and pick the **`main`** branch.
3. Fill in the settings:

   | Field | Value |
   |---|---|
   | **Name** | `shl-recommender` (or anything) |
   | **Language / Runtime** | `Python` |
   | **Branch** | `main` |
   | **Build Command** | `pip install -r requirements.txt && python -m scripts.ingest` |
   | **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
   | **Instance Type** | **Free** |

4. Expand **Advanced** → set **Health Check Path** to `/health`.
5. Add **Environment Variables**:

   | Key | Value |
   |---|---|
   | `LLM_PROVIDER` | `gemini` |
   | `LLM_MODEL` | `gemini-2.5-flash` |
   | `LLM_API_KEY` | *(your `AIza…` key)* |
   | `PYTHON_VERSION` | `3.12.7` |

6. Click **Create Web Service**. Render builds and deploys; watch the **Logs** tab.

---

## Verify the deployment

Once the dashboard shows **Live**, from your terminal (replace the host):

```bash
# readiness
curl https://<your-service>.onrender.com/health
# -> {"status":"ok"}

# a full turn
curl -X POST https://<your-service>.onrender.com/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with stakeholders"}]}'
```

Open `https://<your-service>.onrender.com/docs` for the Swagger UI.

**Confirm the catalog loaded.** In the Render **Logs** tab you should see one of:
- `Loaded catalog: N items from …/catalog_normalized.json` (committed file), or
- `Fetched catalog: N Individual Test Solutions` (runtime fallback).

If N looks right and `/chat` returns grounded `recommendations`, you're done.

**Submit** `https://<your-service>.onrender.com` as your endpoint URL (both
`/health` and `/chat` live there).

---

## Keep the service awake during grading (optional)

Render's free tier **sleeps after ~15 min of inactivity**; the next request triggers
a 30–60s cold start. The SHL grader allows up to 2 minutes on the first `/health`,
so this is tolerable — but to be safe:

- **Quick manual option:** hit `/health` once right before you submit to pre-warm it.
- **Automated option:** the repo includes `.github/workflows/keep-alive.yml`, which
  pings `/health` every 5 minutes. Enable it by adding a repository **variable**
  `SERVICE_URL = https://<your-service>.onrender.com` under **Settings → Secrets and
  variables → Actions → Variables**. On a **private** repo, only run it during the
  grading window (Actions minutes round up per run and would exceed the free cap if
  left on forever); on a **public** repo it's free to leave running. An external
  uptime pinger (e.g. UptimeRobot) is an even more reliable alternative that uses no
  Actions minutes.

---

## Redeploys and updates

Auto-deploy is on: every push to `main` rebuilds and redeploys. You can also click
**Manual Deploy → Deploy latest commit** in the dashboard, or **Clear build cache &
deploy** if a build misbehaves.

---

## Troubleshooting

- **Build fails on the ingest step.** The Blueprint's build command tolerates this
  and the app fetches the catalog at runtime instead. On a manual service, change the
  build command to `pip install -r requirements.txt && (python -m scripts.ingest || true)`
  to make ingest non-fatal.
- **Python version error at build.** Bump `PYTHON_VERSION` (e.g. `3.12.8`) or edit
  `.python-version`. The service was developed and tested on Python 3.12.
- **`/chat` returns empty `recommendations` for a clear role.** Check the logs for
  the catalog count. If it's 0, the raw catalog schema didn't map — run
  `python -m scripts.ingest` locally, read the printed sample record, and adjust
  `_FIELD_ALIASES` in `app/catalog.py`.
- **LLM not being used** (agent feels rigid). Confirm `LLM_API_KEY` is set and the
  logs show `LLM provider=gemini available=True`. If it says `available=False`, the
  key is missing/empty — the service still works via the deterministic fallback, but
  quality is lower.
- **429 / rate limit from Gemini.** Free `gemini-2.5-flash` is ~10 req/min, ~250/day.
  For heavy testing set `LLM_MODEL=gemini-2.5-flash-lite` (higher daily limit).
- **First request after idle times out.** That's the cold start waking the service;
  retry once, or use the keep-alive above.
