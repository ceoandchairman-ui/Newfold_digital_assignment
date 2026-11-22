# 1. Setup and running the code

## What runs where (read this first)

This project was built with the rule **"nothing runs locally"**: the real system needs Azure (Postgres with Entra-only auth, three Redis clusters, Azure OpenAI, Service Bus). So there are two ways to run it:

| Way | What you get | Needs |
|---|---|---|
| **A. The test suite** (start here) | Every main journey and failure case, running the real graph, agents, harness, payment state machine and guards against in-memory fakes | Python 3.12 and `uv` only |
| **B. Integration tests in CI** | The same logic against a real Postgres (pgvector) and Redis Stack | A GitHub repo (the CI job starts throwaway containers) |
| **C. The deployed system** | The full service on Azure Kubernetes Service (AKS) | An Azure subscription, see the checklist at the end |

## A. Run it in two minutes

Prerequisites: Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd ai-support-assistant
uv sync --extra dev --prerelease=allow --python 3.12   # --prerelease: Microsoft Agent Framework ships pre-release deps

uv run pytest -q                       # 307 passed, 18 skipped (the 18 are the integration tests)
uv run ruff check src tests scripts    # lint
uv run mypy src                        # strict type check
```

Expected: `307 passed, 18 skipped`. The 18 skips are the integration tests, which need a database (see B).

**See the main journeys run, with names:**

```bash
uv run pytest tests/test_journeys.py tests/test_red_team.py -v
```

Each test name reads as a sentence, for example `test_gateway_timeout_never_claims_success_and_holds_the_bill`. `docs/03_tests_and_examples.md` maps every journey and failure case to its test.

**Read an example conversation:** `docs/example_transcripts.md` walks through each journey step by step, and each one matches an automated test.

## B. Integration tests against real Postgres and Redis (CI)

The `integration` job in `.github/workflows/ci.yml` starts `pgvector/pgvector:pg16` and `redis/redis-stack-server` as service containers, runs the real Alembic migrations, and runs `tests/integration/`. The job **fails if any integration test is skipped**, so a broken setup cannot pass quietly.

To run them yourself you need your own Postgres with the `vector` extension and a Redis Stack (RediSearch and RedisJSON):

```bash
export IT_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/it
export IT_REDIS_URL=redis://localhost:6379/0
uv run pytest tests/integration -v
```

> These integration tests have **not been run yet** (nothing runs locally here). Their first run is in CI, and I expect it to find a few bugs. That is their purpose.

## C. Deploying to Azure

One-time setup (these are decisions or access I did not have):

1. **Terraform state storage account:** create it, then fill in `infra/terraform/envs/dev.backend.hcl`.
2. **Entra admin group:** set `pg_admin_object_id` in `envs/dev.tfvars`.
3. **Embedder provider** (BGE-M3, 1024 dimensions) and its URL, and the region and Azure OpenAI quota.
4. **GitHub environments:** `dev`, `production`, `plan-dev`, `plan-prod`, with Azure OIDC variables (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `ACR_NAME`, `RESOURCE_GROUP`, `AKS_NAME`, `SMOKE_PHONE`, `SMOKE_DOB`) and required reviewers on `production`.
5. **Key Vault secrets:** `database-url`, `redis-url`, `checkpoint-redis-url`, `tool-cache-redis-url`, `embedder-api-key`.

Then merging to `main` runs `.github/workflows/deploy.yml`:

```
build image -> sign + SBOM -> [plan] -> (approval) -> apply the reviewed plan
   -> verify signature -> migrations -> helm upgrade --atomic -> smoke test -> dashboards
```

A failed smoke test rolls the release back and stops promotion to prod.

**Seed data and try it:**

```bash
python -m support_assistant.seed seed/example_seed.toml      # C1002 owes bill B1005, etc.
python -m support_assistant.cli https://<dev-backend-host>   # verify, chat, approve a payment
python -m support_assistant.loadtest https://<dev> +15550001 1990-05-01 50 10   # latency baseline
```

## Configuration

Every setting is an environment variable, validated at startup (`src/support_assistant/config.py`). `.env.example` lists them. The services refuse to start if a required value is missing, for example the orchestrator without `CHECKPOINT_REDIS_URL`.

## Repository map

| Path | What |
|---|---|
| `src/support_assistant/backend/` | Fat backend: identity, memory, budget, Prompt Shields |
| `src/support_assistant/orchestrator/` | LangGraph graph, Redis checkpointer, run lock |
| `src/support_assistant/agents/` | The three agents, the harness, and the MCP client registry |
| `src/support_assistant/mcp_servers/` | `database`, `comms` and `device` MCP servers |
| `src/support_assistant/domain/`, `services/` | Business rules and payment/retrieval services |
| `migrations/`, `seed/` | Schema and demo data |
| `infra/terraform/`, `deploy/helm/`, `.github/workflows/` | Infrastructure, charts, pipelines |
| `tests/` | Unit and journey tests; `tests/integration/` needs real services |
