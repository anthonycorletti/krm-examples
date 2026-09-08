FROM ghcr.io/astral-sh/uv:0.12.7-python3.13-trixie-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
USER 65532:65532
EXPOSE 8000 8001
CMD ["/srv/.venv/bin/uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
