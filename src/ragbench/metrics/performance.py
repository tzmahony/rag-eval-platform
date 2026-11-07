from __future__ import annotations

import numpy as np


def percentile(values: list[float], q: float) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.percentile(vals, q)) if vals else None
