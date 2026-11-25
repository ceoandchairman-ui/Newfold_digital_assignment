"""Grants for the document paths: the indexer and customer uploads."""

from support_assistant import dbroles


def _text(workload: str) -> str:
    return " | ".join(dbroles.grant_statements(f"id-support-dev-{workload}", workload))


def test_the_indexer_can_add_and_remove_knowledge_chunks_and_touch_nothing_else():
    text = _text("indexer")
    # UPDATE is for re-embedding a chunk when the embedding model changes.
    assert 'GRANT SELECT, INSERT, UPDATE, DELETE ON "knowledge_base_chunks"' in text
    for other in ("customers", "bills", "transaction_history", "customer_uploads", "audit_events"):
        assert f'"{other}"' not in text


def test_the_indexer_has_its_sequence_because_it_inserts():
    assert "GRANT USAGE, SELECT ON ALL SEQUENCES" in _text("indexer")


def test_only_the_backend_can_touch_customer_uploads():
    for workload in dbroles.GRANTS:
        has_access = '"customer_uploads"' in _text(workload)
        assert has_access == (workload == "fat-backend"), workload


def test_the_backend_can_manage_uploads_for_accept_scan_status_and_erasure():
    assert 'GRANT SELECT, INSERT, UPDATE, DELETE ON "customer_uploads"' in _text("fat-backend")


def test_only_the_indexers_and_the_retrieval_service_touch_the_manual():
    readers = {w for w in dbroles.GRANTS if '"knowledge_base_chunks"' in _text(w)}
    assert readers == {"indexer", "index-worker", "retrieval"}


def test_the_index_worker_has_exactly_the_indexers_rights():
    assert dbroles.GRANTS["index-worker"] == dbroles.GRANTS["indexer"]
    assert "GRANT USAGE, SELECT ON ALL SEQUENCES" in _text("index-worker")
