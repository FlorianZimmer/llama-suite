FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime deps from pyproject.toml
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install .

# Run from source tree so static assets are always available.
ENV PYTHONPATH=/app/src \
    LLAMA_SUITE_ROOT=/data \
    HOST=0.0.0.0 \
    PORT=8088

RUN useradd -u 10001 -m appuser
USER 10001

EXPOSE 8088

CMD ["python", "-m", "llama_suite.webui.server"]

