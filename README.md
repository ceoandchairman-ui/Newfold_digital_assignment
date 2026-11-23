# AI Support Assistant

A LangGraph orchestrator with Microsoft Agent Framework agents, MCP servers, PostgreSQL with pgvector, and Redis, on Azure Kubernetes Service. See `docs/product_backlog.md` for scope and decisions.

## Layout

| Path | What |
|---|---|
| `src/support_assistant/backend/` | Fat backend: verification, memory, budget, Prompt Shields, query rewrite |
| `src/support_assistant/orchestrator/` | LangGraph graph, Redis checkpointer, run lock, `AgentClient` |
| `src/support_assistant/agents/` | One service per agent, the harness, and `mcp_client/` (the tool registry) |
| `src/support_assistant/mcp_servers/` | The `database`, `comms` and `device` servers, run from one launcher |
| `src/support_assistant/mcp_catalog.py` | The contract both sides share: servers, ports and tools |
| `src/support_assistant/services/`, `domain/` | Payments, retrieval, episodes; pure business rules |
| `migrations/` | Alembic migrations (plain SQL) |
| `infra/terraform/` | All Azure resources, with a single environment, `envs/prod` |
| `deploy/helm/support/` | One Deployment per service, plus policies, quota, ingress and CronJobs |
| `docs/` | Backlog, requirements, DR, spikes, example transcripts |

## Development workflow (E10-15)

Nothing runs locally. Code is written and reviewed on your machine, checked in CI, and runs in Azure. There is one environment, production.

1. **Before pushing:** run the same checks CI runs. Tests use fakes and need no services.
   ```
   uv sync --extra dev --prerelease=allow
   uv run ruff check src tests && uv run mypy src && uv run pytest
   ```
2. **Pull request:** `ci.yml` runs lint, types, tests, `terraform validate`, `helm lint` and an image build.
3. **Merge to `main`:** `deploy.yml` builds one image, then:
   - plans the Terraform change and shows it for review;
   - after a required reviewer approves the `production` environment, applies exactly that plan, runs migrations, then Helm, then a smoke test.
4. **Try it:** run `python -m support_assistant.cli https://<backend>`.

`.env.example` lists every setting the services read from their environment.

## Before the first deployment

- Create the Terraform state storage account, and fill in `envs/prod.backend.hcl`.
- Set the repository variables `PG_ADMIN_OBJECT_ID`, `PG_ADMIN_NAME` and (for a group) `PG_ADMIN_TYPE=Group`: the Postgres Entra administrator.
- Name the embedder and reranker provider (D3, D18). The migrations need `EMBEDDING_DIM=1024`.
- Set the region (D15) and confirm the Azure OpenAI quota there.
- Create the GitHub `build`, `plan-prod` and `production` environments, with the variables `AZURE_CLIENT_ID` (per environment), `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` and `ACR_NAME`. `production` needs required reviewers.
