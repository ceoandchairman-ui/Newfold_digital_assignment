"""Typed, validated, fail-fast configuration (E0-2). Values come from the environment only."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# How hard a reasoning model thinks before answering. More effort is slower and costs more.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Data stores ---------------------------------------------------------
    database_url: SecretStr
    # Entra-only Postgres in Azure: each new connection gets a fresh token as its password (db.py).
    database_entra_auth: bool = False
    # The Postgres role to connect as under Entra auth: the pod's identity NAME (id-<env>-<service>).
    database_entra_user: str | None = None
    # Three Redis clusters, one per purpose (D9). Each has its own eviction policy in Terraform.
    #   redis_url             "cache": conversation window and verified flag. LRU eviction, no persistence.
    #   checkpoint_redis_url  LangGraph checkpoints, graph state, run locks. NoEviction + AOF + modules.
    #   tool_cache_redis_url  tool-call results. LRU eviction, no persistence. Unset disables the cache.
    redis_url: SecretStr
    # Entra auth for Azure Managed Redis: the pod's identity object id is the username (redis_factory).
    redis_entra_auth: bool = False
    redis_entra_object_id: str | None = None
    checkpoint_redis_url: SecretStr | None = (
        None  # required by the orchestrator, checked at startup
    )
    tool_cache_redis_url: SecretStr | None = None
    tool_cache_account_ttl_s: int = Field(60, ge=1)
    tool_cache_bills_ttl_s: int = Field(30, ge=1)

    # LangGraph checkpoints and graph state live in Redis (D9). They must outlive a customer
    # stepping away mid-payment, so the TTL is long and Redis must not evict (see Terraform).
    checkpoint_ttl_minutes: int = Field(1440, ge=5, description="Idle checkpoint expiry")
    run_lock_ttl_ms: int = Field(120_000, ge=1000, description="Safety expiry for a crashed run")

    # Postgres pool (E1-4)
    db_pool_min_size: int = Field(2, ge=1)
    db_pool_max_size: int = Field(10, ge=1)
    db_pool_timeout_s: float = Field(10.0, gt=0)

    # Short-term memory window in Redis (E2-6, D6)
    redis_window_size: int = Field(20, ge=1, description="Max messages kept in the hot window")
    redis_session_ttl_s: int = Field(1800, ge=60, description="Short TTL; Postgres is the truth")
    # Earlier conversations (other sessions) offered as context on a session's first turn.
    history_days: int = Field(5, ge=0, le=30, description="0 turns cross-session history off")
    history_max_sessions: int = Field(5, ge=1, le=10)

    # --- Embeddings / LLM ----------------------------------------------------
    # D3: BGE-M3 via a managed OpenAI-compatible API. 1024 dims fits pgvector HNSW (limit 2000).
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    # Raise this to force every document to be embedded again with the same model name.
    embedding_version: str = "1"
    # The metadata registry (registry.py), Azure Cosmos DB with Entra auth only. Unset = no
    # registry: nothing is checked or recorded. Production sets REGISTRY_REQUIRED so that a
    # missing endpoint stops a service instead of silently skipping the checks.
    cosmos_endpoint: str | None = None
    cosmos_database: str = "registry"
    registry_required: bool = False
    embedder_url: str = ""
    # D18: open reranker on the same host, applied to the top candidates only.
    rerank_enabled: bool = False
    reranker_url: str = ""
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = Field(20, ge=1)
    # Cosine similarity a vector match needs. Unset = no floor. Set it only from measurements on the
    # real embedder (evaluation.py): the right value differs from model to model.
    retrieval_min_similarity: float | None = Field(None, ge=-1.0, le=1.0)
    azure_openai_endpoint: str | None = None
    # PLACEHOLDER, NOT VERIFIED: GPT-5.x needs a newer version than 2024-10-21. Two settings because
    # the agents (Agent Framework, Responses API) and the query rewrite (LangChain, Chat Completions)
    # are different clients, and one may need a different version than the other.
    azure_openai_api_version: str = "2025-04-01-preview"  # the agents
    azure_openai_chat_api_version: str = "2025-04-01-preview"  # the query rewrite
    # Shared small tier for easy turns (D19). Empty disables tiering: the agent's own model is used.
    # --- Model roles (models.py): one deployment per job, each with its own reasoning effort. ---
    # Terraform creates the deployments and pins the model behind each name. Effort applies to
    # reasoning models; set it to None for a model that has no such setting.
    llm_chat_deployment: str = "chat-support"  # support_agent: conversation, generic questions
    llm_chat_effort: ReasoningEffort | None = "low"
    llm_planner_deployment: str = "chat-planner"  # support_agent: reasoning and task questions
    llm_planner_effort: ReasoningEffort | None = "medium"
    llm_rewrite_deployment: str = "chat-rewrite"  # fat backend: query rewrite
    llm_rewrite_effort: ReasoningEffort | None = "minimal"
    llm_troubleshooter_deployment: str = "chat-troubleshooter"  # troubleshoot_agent
    llm_troubleshooter_effort: ReasoningEffort | None = "medium"
    llm_judge_deployment: str = "chat-judge"  # eval_agent: LLM-as-judge groundedness check
    llm_judge_effort: ReasoningEffort | None = "low"
    # If a role's model fails or times out, this deployment answers instead (a quota limit is per
    # deployment, so a 429 on one is usually survivable on another). None means no fallback.
    llm_chat_fallback_deployment: str | None = "chat-planner"
    llm_planner_fallback_deployment: str | None = "chat-support"
    llm_troubleshooter_fallback_deployment: str | None = "chat-planner"
    # The rewrite and the judge sit on the customer's wait. Past the timeout each falls back
    # (the raw message; "no objection") instead of delaying or failing the turn.
    rewrite_timeout_seconds: float = Field(1.5, gt=0)
    online_judge_enabled: bool = True
    judge_timeout_seconds: float = Field(4.0, gt=0)

    # --- MCP boundary --------------------------------------------------------
    mcp_transport: Literal["http", "sse", "stdio"] = "http"
    mcp_database_url: str = "http://mcp-servers:8001/mcp"
    mcp_comms_url: str = "http://mcp-servers:8002/mcp"
    mcp_device_url: str = "http://mcp-servers:8003/mcp"
    mcp_account_url: str = "http://mcp-servers:8004/mcp"
    mcp_crm_url: str = "http://mcp-servers:8005/mcp"
    mcp_twilio_url: str = "http://mcp-servers:8006/mcp"
    mcp_stripe_url: str = "http://mcp-servers:8007/mcp"
    mcp_gmail_url: str = "http://mcp-servers:8008/mcp"
    mcp_drive_url: str = "http://mcp-servers:8009/mcp"

    # --- Service-to-service (ClusterIP DNS names) ----------------------------
    orchestrator_url: str = "http://orchestrator:8000"
    # The retrieval service (search + reranker). One attempt may take this long; one retry follows.
    retrieval_url: str = "http://retrieval:8000"
    retrieval_timeout_seconds: float = Field(8.0, gt=0)
    agent_director_url: str = "http://agent-director:8000"
    agent_troubleshooter_url: str = "http://agent-troubleshooter:8000"
    agent_evaluator_url: str = "http://agent-evaluator:8000"
    max_eval_retries: int = Field(2, ge=0)
    max_troubleshoot_turns: int = Field(3, ge=1)
    # Device checks and fixes the troubleshooter may run in one turn (0 turns the tool loop off).
    troubleshooter_tool_steps: int = Field(3, ge=0, le=5)
    # Per-customer daily cap on estimated input tokens (E6-5). 0 disables the cap.
    daily_token_budget: int = Field(50_000, ge=0)

    # --- Security (see docs/04_resiliency_and_security.md) --------------------------------------
    # A session id is a bearer token: it must be long and random. 0 disables the rule (tests only).
    min_session_id_length: int = Field(32, ge=0)
    # Identity-guessing lockout: failed /verify attempts per session, per person being guessed
    # (phone or name) and per caller address, within one window. The caller waits out the window.
    verify_max_failures_session: int = Field(5, ge=1)
    verify_max_failures_subject: int = Field(10, ge=1)
    verify_max_failures_ip: int = Field(30, ge=1)
    verify_window_seconds: int = Field(900, ge=1)
    # Request rates (per minute). Chat is capped per customer and per caller address.
    chat_rate_per_minute: int = Field(30, ge=1)
    ip_rate_per_minute: int = Field(120, ge=1)
    # How many reverse proxies we run in front of the API; only then is X-Forwarded-For believed.
    trusted_proxy_hops: int = Field(0, ge=0, le=5)

    # --- Resilience -------------------------------------------------------------------------------
    # Turns in flight per backend pod before it answers 503 instead of queueing.
    max_inflight_turns: int = Field(64, ge=1)
    # The whole graph run for one message must finish in this long (a payment resume is exempt).
    turn_timeout_seconds: float = Field(45.0, gt=0)
    # Calls from the orchestrator to an agent service: a breaker opens after this many failures in a
    # row and stays open for the cooldown, so a dead agent costs microseconds instead of timeouts.
    agent_circuit_threshold: int = Field(5, ge=1)
    agent_circuit_cooldown_seconds: float = Field(30.0, gt=0)
    # One model call must answer in this long, else its fallback deployment (if any) is tried.
    llm_timeout_seconds: float = Field(30.0, gt=0)
    # A client-supplied id that makes a retried /chat message run once.
    idempotency_ttl_seconds: int = Field(300, ge=1)
    # --- Document storage (E2-2, E13). Private blob accounts, Entra auth, no keys. ----------------
    # Team documents the indexer reads (trusted reference material).
    docs_storage_account: str | None = None
    docs_container: str = "knowledge-base"
    # index-worker: how often to look for changed documents, and how many looks in a row must see
    # the same state before it indexes (so a multi-file upload is indexed once, after it finishes).
    index_poll_seconds: int = Field(30, ge=1)
    index_settle_polls: int = Field(2, ge=1)
    # Files customers upload (untrusted: validated, malware-scanned, expiring).
    uploads_storage_account: str | None = None
    uploads_container: str = "customer-uploads"
    upload_max_bytes: int = Field(10_000_000, ge=1)
    upload_daily_limit: int = Field(10, ge=1, description="Uploads per customer per 24 h")
    # Azure AI Content Safety Prompt Shields (E10-9). Unset = local pattern check only.
    content_safety_endpoint: str | None = None

    acs_endpoint: str | None = None  # https://<name>.communication.azure.com
    acs_sms_from: str | None = None  # purchased ACS phone number
    acs_email_from: str | None = None  # verified ACS sender address, e.g. DoNotReply@<domain>
    servicebus_namespace_fqdn: str | None = None  # e.g. sb-support-dev-xxxxx.servicebus.windows.net

    # --- Third-party providers. PLACEHOLDERS: none is configured yet, every one defaults to a stub.
    # Secrets belong in Key Vault (mounted as environment variables), never in this file.
    # Twilio: outbound SMS (a provider beside ACS) and the inbound chat channel.
    sms_provider: Literal["acs", "twilio"] = "acs"  # who the notification worker sends SMS with
    twilio_account_sid: str | None = None  # PLACEHOLDER: ACxxxxxxxx
    twilio_auth_token: SecretStr | None = None  # PLACEHOLDER: from Key Vault
    twilio_sms_from: str | None = None  # PLACEHOLDER: the purchased Twilio number, E.164
    twilio_channel_enabled: bool = False  # the inbound chat webhook /channels/twilio/inbound
    twilio_webhook_url: str | None = None  # PLACEHOLDER: the exact public URL Twilio calls
    # Stripe: the real payment gateway behind process_payment (replaces the stub gateway).
    stripe_secret_key: SecretStr | None = None  # PLACEHOLDER: sk_live_... from Key Vault
    stripe_api_base: str = "https://api.stripe.com"
    stripe_currency: str = "usd"  # PLACEHOLDER until the billing currency is decided
    # Google Workspace: the company support mailbox (Gmail) and company documents (Drive).
    google_credentials_json: SecretStr | None = None  # PLACEHOLDER: service account key
    google_support_mailbox: str | None = None  # PLACEHOLDER: support@<company domain>
    google_drive_folder_id: str | None = None  # PLACEHOLDER: the shared policy folder

    # --- Feature flags (stubs for local and CI) ------------------------------
    use_stub_sms: bool = True
    use_stub_email: bool = True
    use_stub_gateway: bool = True
    use_stub_twilio: bool = True
    use_stub_gmail: bool = True
    use_stub_drive: bool = True

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("db_pool_min_size must be <= db_pool_max_size")
        return self


@lru_cache
def get_settings() -> Settings:
    """Fail fast at startup if required values are missing or invalid."""
    return Settings()
