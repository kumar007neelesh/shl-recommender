FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Bake the catalog into the image so cold starts are instant (needs network at
# build time). If your build has no network, delete this line and rely on start.sh.
RUN python -m scripts.ingest || echo "Build-time ingest skipped; will run at boot."
EXPOSE 8000
CMD ["./start.sh"]
