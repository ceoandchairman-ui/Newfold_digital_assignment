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

uv run pytest -q                       # 389 passed, 28 skipped (the 28 are the integration tests)
uv run ruff check src tests scripts    # lint
uv run mypy src                        # strict type check
```

Expected: `389 passed, 28 skipped`. The 28 skips are the integration tests, which need a database (see B).

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

**One-time setup** (about 15 minutes; everything after it is the pipeline):

1. **Bootstrap** (as an Azure Owner, once): `cd infra/bootstrap && terraform init && terraform apply -var github_repository=<owner/repo>`.
   It creates the Terraform state storage, a shared container registry, and two GitHub OIDC identities (`plan`: read-only, `apply`: can deploy) with no stored secrets.
2. **Copy its outputs:**
   - `backend_config` into `infra/terraform/envs/prod.backend.hcl`;
   - `main_tfvars` into `envs/*.tfvars` (and set `pg_admin_object_id` there, an Entra group for Postgres admins);
   - `github_variables` into GitHub: create the environments `build`, `plan-prod`, `production`, set the variables (`AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `ACR_NAME`, `AZURE_CLIENT_ID` per environment, `PG_ADMIN_OBJECT_ID`, `PG_ADMIN_NAME`; optionally `SEED_DEMO_DATA=true` with `SMOKE_PHONE`, `SMOKE_DOB`), and add required reviewers on `production`.
3. **One secret:** add `EMBEDDER_API_KEY` as a GitHub secret (it comes from the embedder provider).
4. **Decisions** I did not have: the embedder provider URL, the region and Azure OpenAI quota (`envs/*.tfvars`), and the CODEOWNERS teams.

**Then merging to `main` does everything.** Nothing is typed in by hand after step 3: Terraform creates the Key Vault secrets (addresses only, no passwords, since Postgres and Redis use Entra tokens), and the pipeline passes Terraform's outputs (identities, Key Vault, prefix) straight into Helm.

```
build image -> sign + SBOM -> [plan] -> (approval) -> apply the reviewed plan
   -> verify signature -> helm upgrade --atomic, in the PRIVATE cluster via `az aks command invoke`
        pre-deploy hook:  migrations -> a Postgres role per service (least privilege) -> demo seed (only if SEED_DEMO_DATA=true)
   -> smoke test -> dashboards
```

A failed smoke test rolls the release back and stops promotion to prod. Secrets reach pods through the Key Vault CSI driver. The cluster API is private, so the runner never talks to it directly.

**Try it** (with SEED_DEMO_DATA=true the demo data is loaded, so C1002 owes bill B1005):

```bash
python -m support_assistant.cli https://<backend-host>   # verify, chat, approve a payment
python -m support_assistant.loadtest https://<backend-host> +15550001 1990-05-01 50 10   # latency baseline
```

> **Not yet run:** none of this has executed against a real subscription. Expect the first pipeline run to need fixes. The bootstrap, Terraform, chart and workflows have been validated (`terraform validate`, `helm lint` and render, workflow parsing) but not applied.

## Documents and uploads (two different paths)

| | **Team documents** (the support manual, policy) | **Customer uploads** (a screenshot, a PDF bill, a text file) |
|---|---|---|
| Lands in | private blob container `knowledge-base` | private blob container `customer-uploads` (separate account) |
| Trust | trusted reference material | **untrusted**: validated, quarantined, malware-scanned |
| Then | the `indexer` CronJob mirrors it into pgvector every 6 hours (and right after each deploy) | held for a human agent; **never given to a model** |
| Retention | kept (versioned, soft delete) | deleted automatically after 30 days (configurable); removed on an erasure request |

**Team uploads a document:** put `.md` or `.txt` files in the `knowledge-base` container (Azure portal, Storage Explorer or `az storage blob upload --auth-mode login`). The account is private, so reach it from a network listed in `docs_allowed_ip_ranges` (your office or VPN egress IPs) or from inside the VNet, and be in the Entra group set in `docs_writers_object_id`. Edited documents replace their old text; deleted documents are removed from the knowledge base at the next run. Trigger a run now with `kubectl -n support create job --from=cronjob/indexer indexer-now`. Only Markdown and text are indexed today; other files are skipped and reported.

**A customer uploads a file** (a verified session is required):

```bash
curl -F session_id=<session> -F file=@screenshot.png https://<backend>/upload     # -> status PENDING_SCAN
curl "https://<backend>/uploads/<id>?session_id=<session>"                         # -> CLEAN once scanned
```

Accepted: PNG, JPEG, PDF and UTF-8 text, up to 10 MB, 10 uploads per customer per day. The file's real bytes must match its extension.

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
