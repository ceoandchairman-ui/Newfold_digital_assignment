"""One long-lived MCP connection per server URL (E8-2): no handshake per tool call.

Opening a connection and running `initialize` for every call costs several round trips. Here a
single owner task holds the connection open and serves requests concurrently (an MCP session is
multiplexed). The connection is owned by ONE task because the underlying client uses cancel scopes
that must be entered and exited in the same task.

If the connection breaks, the owner exits, the next request opens a fresh one, and the failed
request is retried once. Persistent failure surfaces as ExternalServiceUnavailable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import structlog

from support_assistant.errors import ExternalServiceUnavailable
from support_assistant.logs import error_fields

log = structlog.get_logger()

Opener = Callable[[str], AbstractAsyncContextManager[Any]]
_Item = tuple[str, tuple[Any, ...], "asyncio.Future[Any]"]


@asynccontextmanager
async def default_opener(url: str) -> AsyncIterator[Any]:  # pragma: no cover - needs a real server
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


class McpConnection:
    def __init__(self, url: str, opener: Opener = default_opener, timeout_s: float = 10.0) -> None:
        self._url, self._open, self._timeout = url, opener, timeout_s
        self._queue: asyncio.Queue[_Item | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._broken = asyncio.Event()  # set by the owner when its connection dies
        self.connects = 0  # how many times a connection was opened (observability and tests)

    # --- public requests -----------------------------------------------------------------------
    async def call(self, tool: str, args: dict[str, Any], *, retry: bool = True) -> Any:
        """Run a tool. Returns the raw MCP result. `retry=False` for tools that change state."""
        return await self._submit("call_tool", (tool, args), retry=retry)

    async def list_tools(self) -> Any:
        """Ask the server which tools it serves (used by the registry's readiness check)."""
        return await self._submit("list_tools", ())

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            await self._queue.put(None)
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (TimeoutError, asyncio.CancelledError) as exc:
                log.warning("mcp_close_forced", **error_fields(exc))
                self._task.cancel()
        self._task = None

    # --- internals ---------------------------------------------------------------------------------
    async def _submit(self, method: str, args: tuple[Any, ...], *, retry: bool = True) -> Any:
        try:
            return await self._once(method, args)
        except Exception as first:  # noqa: BLE001 - one retry on a fresh connection, reads only
            log.warning(
                "mcp_request_failed", operation=method, will_retry=retry, **error_fields(first)
            )
            if not retry:
                raise ExternalServiceUnavailable(f"mcp {method} failed") from first
            task = self._task
            if task is not None and self._broken.is_set():
                await asyncio.wait({task}, timeout=5)  # let the dead owner finish, then reconnect
            try:
                return await self._once(method, args)
            except Exception as exc:  # noqa: BLE001
                log.error("mcp_unreachable", operation=method, **error_fields(exc))
                raise ExternalServiceUnavailable(f"mcp {method} unreachable") from exc

    async def _once(self, method: str, args: tuple[Any, ...]) -> Any:
        await self._ensure()
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._queue.put((method, args, fut))
        try:
            return await asyncio.wait_for(fut, timeout=self._timeout)  # never hang a turn
        except TimeoutError as exc:
            raise ExternalServiceUnavailable(f"mcp {method} timed out") from exc

    async def _ensure(self) -> None:
        async with self._lock:
            if self._task is None or self._task.done():
                self._queue = asyncio.Queue()
                self._broken = asyncio.Event()
                self._task = asyncio.create_task(self._own())

    async def _own(self) -> None:
        """Owns the connection for its whole life; exits when it breaks or on close."""
        pending: set[asyncio.Task[None]] = set()
        broken = self._broken
        queue = self._queue
        try:
            async with self._open(self._url) as session:
                self.connects += 1

                async def one(method: str, args: tuple[Any, ...], fut: asyncio.Future[Any]) -> None:
                    try:
                        result = await getattr(session, method)(*args)
                    except Exception as exc:  # noqa: BLE001 - the connection itself failed
                        log.warning("mcp_connection_broken", operation=method, **error_fields(exc))
                        broken.set()
                        if not fut.done():
                            fut.set_exception(exc)
                        return
                    if not fut.done():  # the caller may have timed out; a late reply is fine
                        fut.set_result(result)

                while not broken.is_set():
                    getter = asyncio.create_task(queue.get())
                    waiter = asyncio.create_task(broken.wait())
                    done, _ = await asyncio.wait(
                        {getter, waiter}, return_when=asyncio.FIRST_COMPLETED
                    )
                    waiter.cancel()
                    if getter not in done:
                        getter.cancel()
                        break
                    item = getter.result()
                    if item is None:
                        break
                    t = asyncio.create_task(one(*item))
                    pending.add(t)
                    t.add_done_callback(pending.discard)
        except Exception as exc:  # noqa: BLE001 - failed to connect or the connection died
            log.warning("mcp_connection_lost", **error_fields(exc))
            self._fail_queued(queue, exc)
        finally:
            for t in pending:
                t.cancel()
            self._fail_queued(queue, ExternalServiceUnavailable("mcp connection closed"))

    @staticmethod
    def _fail_queued(queue: asyncio.Queue[Any], exc: Exception) -> None:
        while not queue.empty():
            item = queue.get_nowait()
            if item is not None and not item[2].done():
                item[2].set_exception(exc)
