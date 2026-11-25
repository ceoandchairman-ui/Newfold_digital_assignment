# 1. Setup and running the code

## What runs where (read this first)

This project was built with the rule **"nothing runs locally"**: the real system needs Azure (Postgres with Entra-only auth, three Redis clusters, Azure OpenAI, Service Bus). So there are three ways to run it:

| Way | What you get | Needs |
|---|---|---|
| **A. The test suite** (start here) | Every main journey and failure case, running the real graph (three agent subgraphs), agents, harness, payment state machine and guards against in-memory fakes. Third-party providers (Twilio, Stripe, Gmail, Drive) run as stubs or mock HTTP transports. | Python 3.12 and `uv` only |
| **B. Integration tests in CI** | The same logic against a real Postgres (pgvector) and Redis Stack | A GitHub repo (the CI job starts throwaway containers) |
| **C. The deployed system** | The full service on Azure Kubernetes Service (AKS) | An Azure subscription, see the checklist at the end |

## A. Run it in two minutes

Prerequisites: Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd ai-support-assistant
uv sync --extra dev --prerelease=allow --python 3.12   # --prerelease: Microsoft Agent Framework ships pre-release deps

uv run pytest -q                       # 535 passed, 28 skipped (the 28 are the integration tests)
uv run ruff check src tests scripts    # lint
uv run mypy src                        # strict type check
```

Expected: `535 passed, 28 skipped`. The 28 skips are the integration tests, which need a database (see B).

**See the main journeys run, with names:**

```bash
uv run pytest tests/test_journeys.py tests/test_red_team.py tests/test_graph_topology.py -v
```

Each test name reads as a sentence, for example `test_gateway_timeout_never_claims_success_and_holds_the_bill`. `docs/03_tests_and_examples.md` maps every journey and failure case to its test.

**Read an example conversation:** `docs/example_transcripts.md` walks through each journey step by step, and each one matches an automated test.

**See the architecture:** `docs/diagrams/architecture.html` shows six current diagrams on one page: Azure infrastructure, delivery pipelines, RAG and background workloads, one request end to end, agent orchestration, and retriever metadata. Their Mermaid sources are in `docs/diagrams/current/`.

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
4. **Decisions** I did not have: the embedder provider URL, the region (D15), **quota for the five model deployments** (`envs/*.tfvars`, see "Models" below), the Stripe customer-to-payment-method mapping, the Twilio and Google credentials, and the CODEOWNERS teams.

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

## Models: one deployment per job

Each job has its own Azure OpenAI deployment and reasoning effort, so any one can change without touching the others (`src/support_assistant/models.py`). The deployment names are settings. The model behind each name is chosen in `infra/terraform/envs/prod.tfvars`, so swapping a model is a Terraform edit, not a code change.

| Job | Settings (deployment / effort) | Default | Starting model | Used by |
|---|---|---|---|---|
| **CHAT** | `LLM_CHAT_DEPLOYMENT` / `LLM_CHAT_EFFORT` | `chat-support` / low | gpt-5.4-mini | `support_agent` router and `normal_resp_gen` (generic answers, the conversation) |
| **PLANNER** | `LLM_PLANNER_DEPLOYMENT` / `LLM_PLANNER_EFFORT` | `chat-planner` / medium | gpt-5.4 | `instruct_gen` (reasoning) and `billing_gen` (task questions, tools) |
| **REWRITE** | `LLM_REWRITE_DEPLOYMENT` / `LLM_REWRITE_EFFORT` | `chat-rewrite` / minimal | gpt-5.4-nano | the fat backend's query rewrite. `REWRITE_TIMEOUT_SECONDS` (default 1.5), then the customer's own words are used. |
| **TROUBLESHOOTER** | `LLM_TROUBLESHOOTER_DEPLOYMENT` / `LLM_TROUBLESHOOTER_EFFORT` | `chat-troubleshooter` / medium | gpt-5.4 | `troubleshoot_agent`. `TROUBLESHOOTER_TOOL_STEPS` (default 3) bounds the device-check loop; 0 turns it off. |
| **JUDGE** | `LLM_JUDGE_DEPLOYMENT` / `LLM_JUDGE_EFFORT` | `chat-judge` / low | gpt-5.4-mini | `eval_agent` groundedness check. `ONLINE_JUDGE_ENABLED`, `JUDGE_TIMEOUT_SECONDS` (default 4). |

Effort is one of `none`, `minimal`, `low`, `medium`, `high`. The payment verdict never involves a model. The GPT-4.1 family is not used: Microsoft lists it as Deprecated (retiring 2027-04-14).

> **Not verified, and the first things to check:** that these models are offered in your region with quota (`capacity_k` in the tfvars is a placeholder), whether the region needs `sku_name = "GlobalStandard"`, and the API versions. `AZURE_OPENAI_API_VERSION` (the agents, Responses API) and `AZURE_OPENAI_CHAT_API_VERSION` (the query rewrite, Chat Completions) both default to `2025-04-01-preview`, which is a guess. Reasoning effort through the Agent Framework has not been run against a real deployment. The final model choice should come from running each candidate through the journey, red-team and retrieval evaluations.

## Third-party providers: placeholders until you configure them

Every provider defaults to a stub, so nothing outside the cluster is contacted until you switch it on. Secrets come from Key Vault as environment variables, never from a file in the repo.

| Provider | Used for | To switch on | Status |
|---|---|---|---|
| **Twilio SMS** | Notification SMS (a provider beside Azure Communication Services) and the `twilio_sms` MCP server | `SMS_PROVIDER=twilio`, `USE_STUB_TWILIO=false`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_SMS_FROM` | REST client written from the documented API, **unverified** |
| **Twilio chat channel** | Customers chat by SMS or WhatsApp | `TWILIO_CHANNEL_ENABLED=true`, `TWILIO_WEBHOOK_URL` (the exact public URL), the same token. Point the Twilio webhook at `POST /channels/twilio/inbound`. | Written and tested, **unverified**. Inbound only; the reply goes back as TwiML. |
| **Stripe** | The real payment gateway behind `process_payment`, and the `stripe` MCP server (payment status, reconcile) | `USE_STUB_GATEWAY=false`, `STRIPE_SECRET_KEY`, `STRIPE_CURRENCY` | REST client written from the documented API, **unverified**. **Placeholder:** the mapping from our customer to a Stripe customer and saved payment method is not decided, so a real charge refuses until it exists. |
| **Gmail** | The company support mailbox: `send_email` (template only), `get_support_thread` | `USE_STUB_GMAIL=false`, `GOOGLE_CREDENTIALS_JSON`, `GOOGLE_SUPPORT_MAILBOX` | **Provider not implemented**: it raises until written. The stub runs. |
| **Google Drive** | Company policy documents: `search_documents`, `read_document` (read-only, returned as untrusted text) | `USE_STUB_DRIVE=false`, `GOOGLE_CREDENTIALS_JSON`, `GOOGLE_DRIVE_FOLDER_ID` | **Provider not implemented**: it raises until written. The stub runs. |
| **Embedder** | bge-m3 embeddings | `EMBEDDER_URL`, `EMBEDDER_API_KEY` | Provider **not chosen** |

Gmail and Drive are about the **company's** mailbox and folder, never a customer's own account. SMS and email text is never written by a model: the tools take a template name, and the recipient comes from the customer's record.

**The Twilio chat channel** is an inbound adapter in the fat backend, not an MCP server. Each request must carry a valid Twilio signature. A repeated delivery is answered once. The sender's phone number identifies the customer but does not prove identity: the customer still verifies with a date of birth (`YYYY-MM-DD`) before any data is shown, and a payment confirmation is a text reply of YES or NO.

## Documents and uploads (two different paths)

| | **Team documents** (the support manual, policy) | **Customer uploads** (a screenshot, a PDF bill, a text file) |
|---|---|---|
| Lands in | private blob container `knowledge-base` | private blob container `customer-uploads` (separate account) |
| Trust | trusted reference material | **untrusted**: validated, quarantined, malware-scanned |
| Then | the `index-worker` notices the change within about a minute and mirrors it into pgvector. The `indexer` CronJob (every 6 hours, and right after each deploy) is the safety net. | held for a human agent; **never given to a model** |
| Retention | kept (versioned, soft delete) | deleted automatically after 30 days (configurable); removed on an erasure request |

**Team uploads a document:** put `.md` or `.txt` files in the `knowledge-base` container (Azure portal, Storage Explorer or `az storage blob upload --auth-mode login`). The account is private, so reach it from a network listed in `docs_allowed_ip_ranges` (your office or VPN egress IPs) or from inside the VNet, and be in the Entra group set in `docs_writers_object_id`. Edited documents replace their old text; deleted documents are removed from the knowledge base on the next run.

**No cluster access is needed.** The `index-worker` lists the container every 30 seconds (`INDEX_POLL_SECONDS`) and indexes once the listing has stayed the same for two looks (`INDEX_SETTLE_POLLS`), so a 20-file upload is indexed once, after it finishes. The worker and the CronJob share a Postgres advisory lock, so they never run at the same time. To force a run: `kubectl -n support create job --from=cronjob/indexer indexer-now`. Only Markdown and text are indexed today; other files are skipped and reported.

**A customer uploads a file** (a verified session is required):

```bash
curl -F session_id=<session> -F file=@screenshot.png https://<backend>/upload     # -> status PENDING_SCAN
curl "https://<backend>/uploads/<id>?session_id=<session>"                         # -> CLEAN once scanned
```

Accepted: PNG, JPEG, PDF and UTF-8 text, up to 10 MB, 10 uploads per customer per day. The file's real bytes must match its extension.

## Configuration

Every setting is an environment variable, validated at startup (`src/support_assistant/config.py`). `.env.example` lists them. The services refuse to start if a required value is missing, for example the orchestrator without `CHECKPOINT_REDIS_URL`.

Beyond the model and provider settings above, the ones you are most likely to change:

| Setting | Default | What it does |
|---|---|---|
| `HISTORY_DAYS`, `HISTORY_MAX_SESSIONS` | 5, 5 | On a session's first turn the fat backend adds a short note about the customer's other conversations from the last days. 0 days turns it off. |
| `DAILY_TOKEN_BUDGET` | 50000 | Per-customer daily cap on estimated input tokens. |
| `MAX_TROUBLESHOOT_TURNS`, `MAX_EVAL_RETRIES` | 3, 2 | Bounds before the graph escalates. |

## Repository map

| Path | What |
|---|---|
| `src/support_assistant/backend/` | Fat backend: identity, memory and the 5-day history, budget, Prompt Shields, query rewrite. `channels.py` is the Twilio chat channel. |
| `src/support_assistant/orchestrator/` | The LangGraph graph (three agent subgraphs and the code around them), Redis checkpointer, run lock |
| `src/support_assistant/agents/` | The three agent services (`support_agent`, troubleshooter, evaluator), the LLM judge, the harness, and the MCP client registry |
| `src/support_assistant/models.py` | Which model does which job |
| `src/support_assistant/mcp_servers/` | Nine MCP servers: `database`, `comms`, `device_diagnostics`, `account_management`, `crm`, `twilio_sms`, `stripe`, `gmail`, `google_drive` |
| `src/support_assistant/integrations/` | Twilio, Stripe and Google clients, each with a stub |
| `src/support_assistant/index_worker.py`, `indexer.py` | Knowledge-base indexing: the always-on worker and the CronJob |
| `src/support_assistant/domain/`, `services/` | Business rules and payment/retrieval services |
| `migrations/`, `seed/` | Schema and demo data |
| `infra/terraform/`, `deploy/helm/`, `.github/workflows/` | Infrastructure, charts, pipelines |
| `docs/diagrams/` | `architecture.html` and, in `current/`, the Mermaid sources and images. The `*.mmd` files directly in `docs/diagrams/` are older versions |
| `tests/` | Unit and journey tests; `tests/integration/` needs real services |
