# ============================================================
# 1) BUILDER STAGE
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installer deps globalement (PAS --user)
RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# 2) FINAL STAGE
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

# Copier libs du builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copier code complet
COPY --chown=appuser:appuser . .

ENV PYTHONUNBUFFERED=1 \
    MODEL_VERSION=1.0.0

EXPOSE 8000

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
