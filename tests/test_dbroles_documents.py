"""Grants for the document paths: the indexer and customer uploads."""

from support_assistant import dbroles


def _text(workload: str) -> str:
    return " | ".join(dbroles.grant_statements(f"id-support-dev-{workload}", workload))


def test_the_indexer_can_add_and_remove_knowledge_chunks_and_touch_nothing_else():
    text = _text("indexer")
    assert 'GRANT SELECT, INSERT, DELETE ON "knowledge_base_chunks"' in text
    assert "UPDATE" not in text
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


def test_no_service_other_than_the_indexer_and_orchestrator_and_troubleshooter_reads_the_manual():
    readers = {w for w in dbroles.GRANTS if '"knowledge_base_chunks"' in _text(w)}
    assert readers == {"indexer", "orchestrator", "agent-troubleshooter"}
