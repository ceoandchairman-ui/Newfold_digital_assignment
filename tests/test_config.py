import pytest
from pydantic import ValidationError

from support_assistant.config import Settings


def make(**overrides):
    base = {
        "database_url": "postgresql://u:p@localhost/db",
        "redis_url": "redis://localhost:6379/0",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_defaults_load():
    s = make()
    assert s.embedding_dim == 1024
    assert s.redis_session_ttl_s == 1800


def test_missing_required_fails_fast(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_secrets_are_masked():
    s = make()
    assert "u:p" not in repr(s)
    assert "u:p" not in str(s.database_url)


def test_pool_bounds_validated():
    with pytest.raises(ValidationError):
        make(db_pool_min_size=20, db_pool_max_size=5)
