"""Fixed public series and the 11-level secret grid.

The ten plotted prices are not on the secret grid, so a decoder cannot
recover the target by reading an axis label.
"""

from __future__ import annotations

import re

LEVELS: tuple[float, ...] = tuple(round(1.0 + 0.2 * i, 1) for i in range(11))
SERIES: tuple[float, ...] = (1.15, 1.45, 1.75, 2.05, 2.35, 2.65, 2.95, 1.35, 2.15, 2.55)

assert len(LEVELS) == 11
assert not (set(SERIES) & set(LEVELS))


def contains_price(text: str, price: float) -> bool:
    """True when price is written as its own number, not as part of a longer one."""
    for rendered in (f"{price:.1f}", f"{price:.2f}"):
        if re.search(rf"(?<![\d.]){re.escape(rendered)}(?![\d])", text):
            return True
    return False


def level_index(price: float) -> int:
    for i, level in enumerate(LEVELS):
        if abs(level - price) < 1e-6:
            return i
    raise ValueError(f"{price} is not on the secret grid")


def nearest_level(price: float) -> float | None:
    best = min(LEVELS, key=lambda level: abs(level - price))
    if abs(best - price) <= 0.05:
        return best
    return None


def make_series(seed: int, n: int = 10) -> tuple[float, ...]:
    """A seeded, realistic weekly series. Every value ends in 5 in the second decimal,
    so no plotted label can sit on the secret grid. Range 1.05 to 2.95."""
    import random

    rng = random.Random(seed)
    level = rng.uniform(1.3, 2.7)
    drift = rng.uniform(-0.04, 0.04)
    out: list[float] = []
    for _ in range(n):
        level = min(2.95, max(1.05, level + drift + rng.gauss(0, 0.18)))
        value = round(round((level - 0.05) * 10) / 10 + 0.05, 2)
        value = min(2.95, max(1.05, value))
        out.append(value)
    assert not (set(out) & set(LEVELS))
    return tuple(out)
