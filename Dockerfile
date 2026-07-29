FROM python:3.11-slim

# Fail fast on unbuffered output; don't write .pyc into the image layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy requirements first so the dependency layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY assistant/ ./assistant/

# Run as a non-root user. SQLite needs write access to the working directory.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8080

CMD ["uvicorn", "assistant.api:app", "--host", "0.0.0.0", "--port", "8080"]
