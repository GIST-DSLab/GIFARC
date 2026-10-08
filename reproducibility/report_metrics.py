"""Original non-null mean and run sample SD from the paper reporter."""
from __future__ import annotations
import math
from typing import Any

def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return _mean_values(values)


def _mean_values(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))
