FROM python:3.12-slim

WORKDIR /app

# libgomp1: required by PyTorch CPU ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    build-essential \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Copy dependency manifest first so this layer is cached unless deps change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source and prompt templates
COPY app/ ./app/
COPY prompts/ ./prompts/

# Directories populated at runtime via bind mounts
RUN mkdir -p ai_models output logs

EXPOSE 8000

# Invoke uvicorn directly so the service binds to 0.0.0.0 inside the container
CMD ["uv", "run", "uvicorn", "app.api.main_service:app", "--host", "0.0.0.0", "--port", "8000"]
