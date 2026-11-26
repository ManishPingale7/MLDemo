FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

# LightGBM needs OpenMP; keep all three model types loadable.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serving.txt ./
RUN python -m pip install --no-cache-dir -r requirements-serving.txt && python -m pip check
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DO_NOT_TRACK=true
COPY src/ ./src/
COPY web/ ./web/
COPY config.yaml ./
COPY .container/ ./
RUN useradd --uid 10001 --create-home demo && chmod -R a+rX /app
USER demo
EXPOSE 8765
HEALTHCHECK --interval=5s --timeout=3s --start-period=15s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2)"
CMD ["python", "-m", "uvicorn", "fraud.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765"]
