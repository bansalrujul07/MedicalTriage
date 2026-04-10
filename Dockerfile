FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt

COPY triage_env /app/triage_env
COPY README.md /app/README.md
COPY openenv.yaml /app/openenv.yaml
COPY graders /app/graders
COPY server /app/server
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

RUN pip install -e /app/triage_env

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "triage_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
