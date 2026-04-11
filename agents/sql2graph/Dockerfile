FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the released package from PyPI
RUN uv pip install --system structured2graph

# Copy .env.example as reference; users supply real values at runtime
# via: docker run -it --env-file .env memgraph/structured2graph
COPY .env.example /app/.env.example

ENTRYPOINT ["structured2graph"]
