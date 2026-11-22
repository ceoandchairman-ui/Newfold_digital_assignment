"""Typed, validated, fail-fast configuration (E0-2). Values come from the environment only."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Data stores ---------------------------------------------------------
    database_url: SecretStr
    # Entra-only Postgres in Azure: each new connection gets a fresh token as its password (db.py).
    database_entra_auth: bool = False
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

    # --- Embeddings / LLM ----------------------------------------------------
    # D3: BGE-M3 via a managed OpenAI-compatible API. 1024 dims fits pgvector HNSW (limit 2000).
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedder_url: str = ""
    # D18: open reranker on the same host, applied to the top candidates only.
    rerank_enabled: bool = False
    reranker_url: str = ""
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = Field(20, ge=1)
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    # Shared small tier for easy turns (D19). Empty disables tiering: the agent's own model is used.
    azure_openai_small_deployment: str = "chat-small"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None
    # Local development only; production authenticates with Managed Identity.
    azure_openai_api_key: SecretStr | None = None

    # --- MCP boundary --------------------------------------------------------
    mcp_transport: Literal["http", "sse", "stdio"] = "http"
    mcp_database_url: str = "http://mcp-servers:8001/mcp"
    mcp_comms_url: str = "http://mcp-servers:8002/mcp"
    mcp_device_url: str = "http://mcp-servers:8003/mcp"

    # --- Service-to-service (ClusterIP DNS names) ----------------------------
    orchestrator_url: str = "http://orchestrator:8000"
    agent_director_url: str = "http://agent-director:8000"
    agent_troubleshooter_url: str = "http://agent-troubleshooter:8000"
    agent_evaluator_url: str = "http://agent-evaluator:8000"
    max_eval_retries: int = Field(2, ge=0)
    max_troubleshoot_turns: int = Field(3, ge=1)
    # Per-customer daily cap on estimated input tokens (E6-5). 0 disables the cap.
    daily_token_budget: int = Field(50_000, ge=0)
    # Model-driven tool calling in the Director (E4-3/E5-7). Off until verified end to end (E4-0).
    director_tool_calling: bool = False
    # Azure AI Content Safety Prompt Shields (E10-9). Unset = local pattern check only.
    content_safety_endpoint: str | None = None

    acs_endpoint: str | None = None  # https://<name>.communication.azure.com
    acs_sms_from: str | None = None  # purchased ACS phone number
    acs_email_from: str | None = None  # verified ACS sender address, e.g. DoNotReply@<domain>
    servicebus_namespace_fqdn: str | None = None  # e.g. sb-support-dev-xxxxx.servicebus.windows.net

    # --- Feature flags (stubs for local and CI) ------------------------------
    use_stub_sms: bool = True
    use_stub_email: bool = True
    use_stub_gateway: bool = True

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("db_pool_min_size must be <= db_pool_max_size")
        return self


@lru_cache
def get_settings() -> Settings:
    """Fail fast at startup if required values are missing or invalid."""
    return Settings()
