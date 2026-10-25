"""Spend ledger. Every paid call appends one row to runs/spend.jsonl. Hard stop before the cap."""

from __future__ import annotations

import fcntl
import json
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "runs" / "spend.jsonl"
PRICES_FILE = ROOT / "runs" / "router_models_2026-10-25.json"
BASELINE_USD = 0.0  # spend before this ledger started
CAP_USD = 8.00
STOP_USD = 7.70  # stop new calls here so in-flight calls cannot cross the cap
FALLBACK = {"input": 3.0, "output": 15.0}  # conservative if a provider price is unknown
# OpenAI is a separate credit with its own cap. Published standard prices (developers.openai.com/api/docs/pricing,
# checked 2026-10-25), per 1M tokens. All input is billed at the uncached rate here, which over-counts slightly.
OPENAI_CAP_USD = 25.00
OPENAI_STOP_USD = 24.00
OPENAI_PRICES = {
    "gpt-6.1-sol": {"input": 2.0, "output": 10.0},
    "gpt-6-sol": {"input": 2.0, "output": 10.0},
    "gpt-6-astra": {"input": 10.0, "output": 50.0},
    "gpt-6-luna": {"input": 0.1, "output": 0.5},
    "gpt-5.6-sol": {"input": 4.0, "output": 20.0},
    "gpt-5.6-terra": {"input": 2.0, "output": 12.0},
    "gpt-5.6-luna": {"input": 0.2, "output": 1.2},
    "gpt-5.5": {"input": 5.0, "output": 30.0},
    "gpt-5.4": {"input": 2.5, "output": 15.0},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.5},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "o3": {"input": 2.0, "output": 8.0},
    "o4-mini": {"input": 1.1, "output": 4.4},
}

_lock = threading.Lock()
_prices: dict | None = None


class BudgetStop(RuntimeError):
    pass


def _load_prices() -> dict:
    global _prices
    if _prices is None:
        data = json.loads(PRICES_FILE.read_text())["data"]
        _prices = {}
        for model in data:
            for provider in model.get("providers", []):
                if provider.get("pricing"):
                    _prices[(model["id"], provider["provider"])] = provider["pricing"]
    return _prices


def price(model: str, provider: str | None) -> dict:
    if model.startswith("openai:"):
        return OPENAI_PRICES.get(model.split(":", 1)[1], {"input": 10.0, "output": 50.0})
    base = model.split(":")[0]
    table = _load_prices()
    if provider and (base, provider) in table:
        return table[(base, provider)]
    known = [v for (m, _), v in table.items() if m == base]
    if known:
        return {"input": max(v["input"] for v in known), "output": max(v["output"] for v in known)}
    return FALLBACK


def total(credit: str = "hf") -> float:
    spent = 0.0
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                row = json.loads(line)
                is_openai = row.get("provider") in ("openai", "azure")
                if is_openai == (credit == "openai"):
                    spent += row["usd"]
            except (ValueError, KeyError):
                pass
    return (BASELINE_USD if credit == "hf" else 0.0) + spent


def check(model: str = "") -> None:
    with _lock:
        if model.startswith(("openai:", "azure:")):
            if total("openai") >= OPENAI_STOP_USD:
                raise BudgetStop(f"openai spend {total('openai'):.3f} >= stop {OPENAI_STOP_USD}")
        elif total("hf") >= STOP_USD:
            raise BudgetStop(f"hf spend {total('hf'):.3f} >= stop {STOP_USD}")


def record(model: str, provider: str | None, prompt_tokens: int, completion_tokens: int, tag: str) -> float:
    p = price(model, provider)
    usd = (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1e6
    row = {
        "t": time.time(), "model": model, "provider": provider, "in": prompt_tokens,
        "out": completion_tokens, "usd": usd, "tag": tag,
    }
    with _lock:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(json.dumps(row) + "\n")
            fcntl.flock(fh, fcntl.LOCK_UN)
    return usd
