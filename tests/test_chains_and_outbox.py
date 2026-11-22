from support_assistant.chains import rewrite_query
from support_assistant.notifications import LogOutbox


class _Boom:
    async def ainvoke(self, _):
        raise RuntimeError("model down")


class _Fixed:
    def __init__(self, text: str) -> None:
        self.text = text

    async def ainvoke(self, _):
        return self.text


async def test_rewrite_without_chain_returns_original():
    assert await rewrite_query(None, [], "my phone has no signal") == "my phone has no signal"


async def test_rewrite_falls_back_when_model_fails():
    assert await rewrite_query(_Boom(), [], "it is slow") == "it is slow"  # type: ignore[arg-type]


async def test_rewrite_uses_model_output_and_ignores_blank():
    assert (
        await rewrite_query(_Fixed("  slow mobile data  "), [], "it is slow") == "slow mobile data"
    )  # type: ignore[arg-type]
    assert await rewrite_query(_Fixed("   "), [], "it is slow") == "it is slow"  # type: ignore[arg-type]


async def test_log_outbox_accepts_messages():
    await LogOutbox().enqueue("C1", "sms", "payment_received")
