# # Phase 1 — Dockerfile
# # Multi-stage build: keeps the final image lean (~200MB vs ~1GB)

# # ── Stage 1: install deps ────────────────────────────────────────────────────
# FROM python:3.11-slim AS builder

# WORKDIR /install
# COPY requirements.txt .
# RUN pip install --no-cache-dir --prefix=/install/packages -r requirements.txt


# # ── Stage 2: runtime image ───────────────────────────────────────────────────
# FROM python:3.11-slim

# # Non-root user for security
# RUN useradd -m -u 1000 appuser

# WORKDIR /app

# # Copy installed packages from builder stage
# COPY --from=builder /install/packages /usr/local

# # Copy app code and the pre-trained model
# COPY app/ .

# # Switch to non-root
# USER appuser

# EXPOSE 8000

# # Health check so Docker knows when the container is ready
# HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
#   CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

EXPOSE 3000
CMD ["python", "bot.py"]