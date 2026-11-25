"""Converts the supplied telecom data (data/corpus/db.toml) into our seed format (seed.py).

The file is used as supplied and never edited. What differs from our tables:
  - identifiers are `customer_id`, `plan_id`, ... and statuses read "Overdue", "Active"
  - payment methods are nested inside customers, and bill line items inside bills
  - a customer lists its `line_ids`, and a LINE names its device (our devices name their line)
  - data is in GB. It is stored in MB at 1 GB = 1000 MB, so 3.2 GB reads back as exactly 3200 MB
  - the postal address is not imported: nothing needs it

Anything inconsistent (an unknown status, a line no customer owns, a bill its customer does not
list) stops the load. Guessing about someone's bill is worse than not loading.
"""

from __future__ import annotations

from typing import Any

MB_PER_GB = 1000


class CorpusError(ValueError):
    """The supplied file cannot be converted without guessing."""


def is_corpus(data: dict[str, Any]) -> bool:
    plans = data.get("plans")
    return bool(plans) and isinstance(plans, list) and "plan_id" in plans[0]


def _status(value: str) -> str:
    """ "Awaiting Payment" -> AWAITING_PAYMENT. Unknown values are rejected by the seed models."""
    return value.strip().upper().replace(" ", "_")


def _mb(gigabytes: float) -> int:
    return round(float(gigabytes) * MB_PER_GB)


def _pick(row: dict[str, Any], *names: str) -> dict[str, Any]:
    return {n: row[n] for n in names if n in row}


def convert_corpus(data: dict[str, Any]) -> dict[str, Any]:
    try:
        return _convert(data)
    except KeyError as exc:
        raise CorpusError(f"a required field is missing: {exc.args[0]}") from exc


def _convert(data: dict[str, Any]) -> dict[str, Any]:
    customers, methods = [], []
    owner: dict[str, str] = {}  # line id -> customer id
    listed_bills: dict[str, str] = {}  # bill id -> customer id
    for c in data["customers"]:
        cid = c["customer_id"]
        customers.append(
            {
                "id": cid,
                "phone": c["phone_number"],
                "account_status": _status(c["account_status"]),
                "goodwill_credit_used_this_year": str(c.get("goodwill_credit_used_this_year", 0)),
                **_pick(c, "full_name", "date_of_birth", "email", "last_extension_date"),
            }
        )
        for n, m in enumerate(c.get("payment_methods", []), start=1):
            methods.append(
                {
                    "id": f"PM-{cid}-{n}",
                    "customer_id": cid,
                    "kind": m["method_type"],
                    "last4": m.get("account_number_last_4"),
                    "expiration": m.get("expiration_date"),
                }
            )
        for line_id in c.get("line_ids", []):
            if line_id in owner:
                raise CorpusError(f"line {line_id} belongs to two customers")
            owner[line_id] = cid
        for bill_id in c.get("bill_ids", []):
            listed_bills[bill_id] = cid

    plans = [
        {
            "id": p["plan_id"],
            "name": p["name"],
            "monthly_price": str(p["price_per_month"]),
            "data_limit_mb": _mb(p["data_limit_gb"]),
            "refuel_price_per_gb": str(p["data_refueling_price_per_gb"]),
        }
        for p in data["plans"]
    ]

    lines = []
    line_of_device: dict[str, str] = {}
    for ln in data["lines"]:
        line_id = ln["line_id"]
        if line_id not in owner:
            raise CorpusError(f"line {line_id} belongs to no customer")
        if device_id := ln.get("device_id"):
            if device_id in line_of_device:
                raise CorpusError(f"device {device_id} is on two lines")
            line_of_device[device_id] = line_id
        lines.append(
            {
                "id": line_id,
                "customer_id": owner[line_id],
                "status": _status(ln["status"]),
                "data_used_mb": _mb(ln.get("data_used_gb", 0)),
                "data_refueling_mb": _mb(ln.get("data_refueling_gb", 0)),
                **_pick(
                    ln,
                    "plan_id",
                    "phone_number",
                    "roaming_enabled",
                    "contract_end_date",
                    "last_plan_change_date",
                    "last_sim_replacement_date",
                    "suspension_start_date",
                ),
            }
        )
    if unknown := set(owner) - {ln["id"] for ln in lines}:
        raise CorpusError(f"customers list lines that do not exist: {sorted(unknown)}")

    devices = []
    for d in data["devices"]:
        device_id = d["device_id"]
        if device_id not in line_of_device:
            raise CorpusError(f"device {device_id} is on no line")
        devices.append(
            {
                "id": device_id,
                "line_id": line_of_device[device_id],
                "esim_capable": d.get("is_esim_capable"),
                **_pick(
                    d,
                    "model",
                    "imei",
                    "device_type",
                    "activated",
                    "activation_date",
                    "last_esim_transfer_date",
                ),
            }
        )
    if unknown := set(line_of_device) - {d["id"] for d in devices}:
        raise CorpusError(f"lines name devices that do not exist: {sorted(unknown)}")

    bills, items = [], []
    for b in data["bills"]:
        bill_id = b["bill_id"]
        if listed_bills.get(bill_id) != b["customer_id"]:
            raise CorpusError(f"bill {bill_id} and its customer disagree about who owes it")
        bills.append(
            {
                "id": bill_id,
                "amount_due": str(b["total_due"]),
                "status": _status(b["status"]),
                **_pick(b, "customer_id", "due_date", "period_start", "period_end", "issue_date"),
            }
        )
        items += [
            {"bill_id": bill_id, "description": i["description"], "amount": str(i["amount"])}
            for i in b.get("line_items", [])
        ]
    if unknown := set(listed_bills) - {b["id"] for b in bills}:
        raise CorpusError(f"customers list bills that do not exist: {sorted(unknown)}")

    return {
        "customers": customers,
        "plans": plans,
        "payment_methods": methods,
        "lines": lines,
        "devices": devices,
        "bills": bills,
        "bill_line_items": items,
    }
