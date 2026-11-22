import pytest

from support_assistant.errors import ConflictError
from support_assistant.locks import RunLock


class FakeRedis:
    """Implements just the two calls the lock uses, with the same semantics."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def set(self, key, value, nx=False, px=None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def eval(self, script, numkeys, key, token):
        if self.data.get(key) == token:
            del self.data[key]
            return 1
        return 0


async def test_second_run_for_the_same_session_is_refused():
    lock = RunLock(FakeRedis())
    async with lock.hold("S1"):
        with pytest.raises(ConflictError):
            async with lock.hold("S1"):
                pass


async def test_different_sessions_do_not_block_each_other():
    lock = RunLock(FakeRedis())
    async with lock.hold("S1"):
        async with lock.hold("S2"):
            pass


async def test_lock_is_released_after_success_and_after_error():
    redis = FakeRedis()
    lock = RunLock(redis)
    async with lock.hold("S1"):
        pass
    assert redis.data == {}
    with pytest.raises(RuntimeError):
        async with lock.hold("S1"):
            raise RuntimeError("boom")
    assert redis.data == {}
    async with lock.hold("S1"):  # can be taken again
        pass


async def test_a_run_never_releases_a_lock_that_now_belongs_to_another_run():
    redis = FakeRedis()
    lock = RunLock(redis)
    async with lock.hold("S1"):
        redis.data["lock:run:S1"] = "someone-elses-token"  # our lock expired; another run took it
    assert redis.data["lock:run:S1"] == "someone-elses-token"
