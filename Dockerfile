# One image, six entrypoints (selected by each Deployment's command). Built and signed in CI
# (deploy.yml), pushed to ACR, scanned by Trivy (ci.yml). Every service is still its own container:
# the Helm chart runs this image once per service with a different command and identity (D17).
FROM python:3.12-slim AS build
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
# Install EXACTLY the versions the tests ran against (uv.lock), in a layer that is cached until
# the lock changes. Without this the image could resolve different versions than CI tested.
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt \
    && uv pip install --system --no-cache -r requirements.txt
COPY src ./src
RUN uv pip install --system --no-cache --no-deps . \
    # Pre-compile now: the running container has a read-only root filesystem and cannot write .pyc.
    && python -m compileall -q /usr/local/lib/python3.12/site-packages

FROM python:3.12-slim
LABEL org.opencontainers.image.title="support-assistant" \
      org.opencontainers.image.description="LangGraph + Microsoft Agent Framework support assistant"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --system --uid 10001 app
WORKDIR /app
COPY --from=build /usr/local /usr/local
COPY migrations ./migrations
COPY alembic.ini ./
COPY seed ./seed
COPY eval ./eval
USER 10001
EXPOSE 8000
# Overridden per Deployment, e.g.:
#   fat-backend:   uvicorn support_assistant.backend.app:app --host 0.0.0.0 --port 8000
#   orchestrator:  uvicorn support_assistant.orchestrator.service:app --host 0.0.0.0 --port 8000
#   agent-*:       AGENT=director uvicorn support_assistant.agents.service:app --host 0.0.0.0 --port 8000
#   mcp-servers:   python -m support_assistant.mcp_servers.main
#   worker / jobs: python -m support_assistant.worker | support_assistant.jobs | support_assistant.smoke
CMD ["uvicorn", "support_assistant.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
