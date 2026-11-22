# One image, six entrypoints (selected by the Deployment's command). Built in CI and pushed to ACR.
FROM python:3.12-slim AS build
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --prerelease=allow .

FROM python:3.12-slim
RUN useradd --system --uid 10001 app
WORKDIR /app
COPY --from=build /usr/local /usr/local
COPY migrations ./migrations
COPY alembic.ini ./
USER 10001
EXPOSE 8000
# Overridden per Deployment, e.g.:
#   fat-backend:   uvicorn support_assistant.backend.app:app --host 0.0.0.0 --port 8000
#   orchestrator:  uvicorn support_assistant.orchestrator.service:app --host 0.0.0.0 --port 8000
#   agent-*:       AGENT=director uvicorn support_assistant.agents.service:app --host 0.0.0.0 --port 8000
#   mcp-servers:   python -m support_assistant.mcp_servers.main
CMD ["uvicorn", "support_assistant.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
