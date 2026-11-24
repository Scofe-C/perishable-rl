# syntax=docker/dockerfile:1
# Personal DE platform image: Python 3.12 + the exact versions in uv.lock.
#   docker compose -f infra/compose.yaml build
#   MSYS_NO_PATHCONV=1 docker compose -f infra/compose.yaml run --rm frn python -m frn.ingest --data-root /data/freshretailnet
# (On Git Bash for Windows, MSYS_NO_PATHCONV=1 stops it from rewriting /data/... into C:/Program Files/Git/data/...)
# Base images are pinned by tag *and* digest: the tag says what it is, the digest guarantees the same bytes.

FROM ghcr.io/astral-sh/uv:0.13.8@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 AS uv

FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS base
COPY --from=uv /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DBT_SEND_ANONYMOUS_USAGE_STATS=false \
    DAGSTER_HOME=/app/dagster \
    FRN_DATA_ROOT=/data/freshretailnet
WORKDIR /app
# Dependencies first (cached layer), project code second. The uv download cache lives in a BuildKit cache
# mount: rebuilds reuse it, but it never ends up inside the image (it doubled the image size before).
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY configs ./configs
COPY dbt ./dbt
COPY dagster ./dagster
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
RUN useradd --create-home --uid 1000 frn && chown -R frn /app
USER frn

FROM base AS dev
USER root
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen
COPY tests ./tests
RUN chown -R frn /app
USER frn
CMD ["pytest", "-q"]
