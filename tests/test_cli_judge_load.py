import httpx
import pytest

from support_assistant import cli
from support_assistant.evaluation_judge import JudgeError, build_judge_chain, judge, judge_many
from support_assistant.loadtest import percentile, summarize


# --- judge (E9-6) -------------------------------------------------------------------------------------------
class FixedLLM:
    """Stands in for a chat model: a Runnable-compatible callable returning fixed text."""

    def __init__(self, text: str):
        self.text = text

    def __call__(self, _input):
        return self.text


def chain(text: str):
    return build_judge_chain(FixedLLM(text))


GOOD = '{"helpfulness": 5, "groundedness": 4, "tone": 5, "safety": 5, "reasons": "ok"}'


async def test_judge_parses_a_valid_verdict_even_with_surrounding_text():
    v = await judge(chain(f"Here you go:\n{GOOD}\nThanks"), "transcript")
    assert (v.helpfulness, v.minimum) == (5, 4)


@pytest.mark.parametrize(
    "bad",
    [
        "no json",
        "{not json}",
        '{"helpfulness": 9, "groundedness": 1, "tone": 1, "safety": 1}',
        '{"helpfulness": 3}',
    ],
)
async def test_unparseable_or_out_of_range_verdicts_are_errors_not_passes(bad):
    with pytest.raises(JudgeError):
        await judge(chain(bad), "t")


async def test_aggregate_counts_judge_errors_as_failures():
    low = '{"helpfulness": 2, "groundedness": 5, "tone": 5, "safety": 5}'
    out = await judge_many(chain(GOOD), ["a", "b"], min_score=4)
    assert out["pass_rate"] == 1.0
    mixed = await judge_many(chain(low), ["a"], min_score=4)
    assert mixed["below_threshold"] == 1 and mixed["pass_rate"] == 0.0
    broken = await judge_many(chain("garbage"), ["a", "b"], min_score=4)
    assert broken["judge_errors"] == 2 and broken["judged"] == 0


# --- load test maths (E9-8) -------------------------------------------------------------------------------------
def test_percentiles_use_nearest_rank():
    data = [float(i) for i in range(1, 101)]
    assert (percentile(data, 50), percentile(data, 95), percentile(data, 100)) == (
        50.0,
        95.0,
        100.0,
    )
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile(data, 0)


def test_summary_reports_error_rate_and_milliseconds():
    s = summarize([0.1, 0.2, 0.3, 0.4], failures=1, wall_s=2.0)
    assert s["requests"] == 5 and s["error_rate"] == 0.2 and s["p50_ms"] == 200.0
    assert s["throughput_rps"] == 2.5
    assert summarize([], 3, 1.0)["error_rate"] == 1.0


# --- CLI (E6-4) ---------------------------------------------------------------------------------------------------
def _scripted(answers):
    it = iter(answers)
    return lambda _prompt="": next(it)


def test_cli_verifies_chats_and_confirms_a_payment(monkeypatch, capsys):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/verify":
            return httpx.Response(200, json={"verified": True})
        if request.url.path == "/chat":
            return httpx.Response(200, json={"pending_approval": {"question": "Pay 5 for B1?"}})
        return httpx.Response(200, json={"reply": "Your payment was successful."})

    real = httpx.Client
    monkeypatch.setattr(
        cli.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    cli.run("http://x", ask=_scripted(["+1", "1990-05-01", "pay my bill", "yes", "quit"]))
    out = capsys.readouterr().out
    assert (
        calls == ["/verify", "/chat", "/approve"] and "Pay 5 for B1?" in out and "successful" in out
    )


def test_cli_stops_when_verification_fails(monkeypatch, capsys):
    real = httpx.Client
    handler = lambda r: httpx.Response(401, json={"message": "I need to verify your identity."})  # noqa: E731
    monkeypatch.setattr(
        cli.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    cli.run("http://x", ask=_scripted(["+1", "1990-05-01"]))
    assert "verify your identity" in capsys.readouterr().out
