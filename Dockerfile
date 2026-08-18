# Dockerfile
# -----------
# Single image that can run EITHER the FastAPI backend OR the Streamlit
# dashboard, selected by the CMD you use at `docker run` time. This keeps
# one Dockerfile to maintain instead of two near-duplicates.
#
# Build:
#   docker build -t bi-assistant .
#
# Run FastAPI (default):
#   docker run --rm -p 8000:8000 --env-file .env bi-assistant
#
# Run Streamlit instead:
#   docker run --rm -p 8501:8501 --env-file .env bi-assistant \
#       streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

FROM python:3.12-slim

# Prevents Python from buffering stdout/stderr - logs show up immediately
# in `docker logs`, which matters for debugging during grading/demo.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (separate layer) so code changes don't force
# a full dependency reinstall on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code + local knowledge base (no sample/demo
# data is bundled - the app requires a real upload before analysis).
COPY app/ ./app/
COPY streamlit_app.py .
COPY data/ ./data/

# Create writable dirs the app needs at runtime (uploads, logs, faiss cache).
RUN mkdir -p /app/data/uploads /app/logs /app/data/faiss_index

EXPOSE 8000 8501

# Basic container-level healthcheck for the FastAPI service (a no-op if you
# override CMD to run Streamlit instead - Streamlit has its own /_stcore/health).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
