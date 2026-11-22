import pytest

from support_assistant.evaluation import Case, evaluate


def _retriever(table: dict[str, list[str]]):
    async def retrieve(query: str) -> list[str]:
        return table[query]

    return retrieve


async def test_perfect_retrieval_scores_one():
    cases = [Case("q1", ("Guide > No signal",))]
    r = await evaluate(cases, _retriever({"q1": ["Guide > No signal", "Guide > Slow data"]}))
    assert (r.recall_at_k, r.mrr) == (1.0, 1.0)


async def test_subsection_counts_as_a_hit_for_its_parent():
    cases = [Case("q", ("Guide > No signal",))]
    r = await evaluate(cases, _retriever({"q": ["Guide > No signal > SIM"]}))
    assert r.recall_at_k == 1.0


async def test_mrr_uses_first_relevant_rank_and_misses_score_zero():
    cases = [Case("a", ("A",)), Case("b", ("B",)), Case("c", ("Z",))]
    table = {"a": ["X", "A"], "b": ["B > sub"], "c": ["X", "Y"]}
    r = await evaluate(cases, _retriever(table))
    assert r.recall_at_k == pytest.approx(2 / 3)
    assert r.mrr == pytest.approx((0.5 + 1.0 + 0.0) / 3)


async def test_similar_heading_prefix_is_not_a_hit():
    # "Guide > No signal" must not match "Guide > No signal at all" (not a child).
    r = await evaluate(
        [Case("q", ("Guide > No signal",))], _retriever({"q": ["Guide > No signal at all"]})
    )
    assert r.recall_at_k == 0.0


async def test_k_limits_the_window_and_empty_set_is_rejected():
    r = await evaluate([Case("q", ("C",))], _retriever({"q": ["A", "B", "C"]}), k=2)
    assert r.recall_at_k == 0.0
    with pytest.raises(ValueError):
        await evaluate([], _retriever({}))
