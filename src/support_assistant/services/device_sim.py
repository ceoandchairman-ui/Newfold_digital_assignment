"""Device simulation (E3-8): deterministic device state and the troubleshooting tool catalog.

The full catalog in the support manual is not available, so this implements the tools the backlog
names (check_*, toggle_*, reseat_sim_card, set/reset_apn_settings, reboot_device, can_send_mms).
Extend the catalog when the manual is recovered.

Every tool is idempotent: `toggle_*` takes the wanted state (`enabled`) instead of flipping, so a
retried call can never undo itself. State is per (customer, device) and lives in a store.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

DEFAULT_APN = "internet"


@dataclass(frozen=True)
class DeviceState:
    powered: bool = True
    airplane_mode: bool = False
    mobile_data: bool = True
    wifi: bool = False
    roaming: bool = False
    sim_seated: bool = True
    sim_locked: bool = False
    apn: str = DEFAULT_APN
    signal_bars: int = 4

    @property
    def has_service(self) -> bool:
        return self.powered and not self.airplane_mode and self.sim_seated and not self.sim_locked

    @property
    def data_works(self) -> bool:
        return self.has_service and self.mobile_data and self.apn == DEFAULT_APN

    def report(self) -> dict[str, Any]:
        return {**asdict(self), "has_service": self.has_service, "data_works": self.data_works}


class DeviceStateStore(Protocol):
    async def get(self, customer_id: str, device_id: str) -> DeviceState: ...

    async def put(self, customer_id: str, device_id: str, state: DeviceState) -> None: ...


class InMemoryDeviceStore:
    def __init__(self, initial: dict[tuple[str, str], DeviceState] | None = None) -> None:
        self._states = dict(initial or {})

    async def get(self, customer_id: str, device_id: str) -> DeviceState:
        return self._states.get((customer_id, device_id), DeviceState())

    async def put(self, customer_id: str, device_id: str, state: DeviceState) -> None:
        self._states[(customer_id, device_id)] = state


class RedisDeviceStore:
    """Shared across MCP replicas. Key `device:{customer}:{device}`, expires after a day."""

    def __init__(self, redis: Any, ttl_s: int = 86_400) -> None:
        self._redis, self._ttl = redis, ttl_s

    async def get(self, customer_id: str, device_id: str) -> DeviceState:
        raw = await self._redis.get(f"device:{customer_id}:{device_id}")
        return DeviceState(**json.loads(raw)) if raw else DeviceState()

    async def put(self, customer_id: str, device_id: str, state: DeviceState) -> None:
        await self._redis.set(
            f"device:{customer_id}:{device_id}", json.dumps(asdict(state)), ex=self._ttl
        )


class DeviceSimulation:
    def __init__(self, store: DeviceStateStore) -> None:
        self._store = store

    async def _update(self, customer_id: str, device_id: str, **changes: Any) -> dict[str, Any]:
        state = replace(await self._store.get(customer_id, device_id), **changes)
        await self._store.put(customer_id, device_id, state)
        return state.report()

    async def check_status(self, customer_id: str, device_id: str) -> dict[str, Any]:
        return (await self._store.get(customer_id, device_id)).report()

    async def toggle_airplane_mode(
        self, customer_id: str, device_id: str, enabled: bool
    ) -> dict[str, Any]:
        return await self._update(customer_id, device_id, airplane_mode=enabled)

    async def toggle_mobile_data(
        self, customer_id: str, device_id: str, enabled: bool
    ) -> dict[str, Any]:
        return await self._update(customer_id, device_id, mobile_data=enabled)

    async def toggle_wifi(self, customer_id: str, device_id: str, enabled: bool) -> dict[str, Any]:
        return await self._update(customer_id, device_id, wifi=enabled)

    async def toggle_roaming(
        self, customer_id: str, device_id: str, enabled: bool
    ) -> dict[str, Any]:
        return await self._update(customer_id, device_id, roaming=enabled)

    async def reseat_sim_card(self, customer_id: str, device_id: str) -> dict[str, Any]:
        return await self._update(customer_id, device_id, sim_seated=True)

    async def set_apn_settings(self, customer_id: str, device_id: str, apn: str) -> dict[str, Any]:
        return await self._update(customer_id, device_id, apn=apn.strip())

    async def reset_apn_settings(self, customer_id: str, device_id: str) -> dict[str, Any]:
        return await self._update(customer_id, device_id, apn=DEFAULT_APN)

    async def reboot_device(self, customer_id: str, device_id: str) -> dict[str, Any]:
        # A reboot clears airplane mode and restores the radio; it cannot fix a locked SIM.
        return await self._update(customer_id, device_id, powered=True, airplane_mode=False)

    async def can_send_mms(self, customer_id: str, device_id: str) -> dict[str, Any]:
        state = await self._store.get(customer_id, device_id)
        return {"can_send_mms": state.data_works, "reason": _mms_reason(state)}


def _mms_reason(s: DeviceState) -> str:
    if not s.powered:
        return "device is off"
    if s.airplane_mode:
        return "airplane mode is on"
    if not s.sim_seated:
        return "SIM card not detected"
    if s.sim_locked:
        return "SIM is locked"
    if not s.mobile_data:
        return "mobile data is off"
    if s.apn != DEFAULT_APN:
        return "APN settings are wrong"
    return "ok"
