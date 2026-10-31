"""The canonical form of a checkpoint output.

Three implementations must agree on it byte for byte: this one, the capture
stub the lab build runs the reference solution under, and the comparison the
checkpoint API makes. A disagreement grades a correct answer wrong, so the
monorepo pins all three to `canon_vectors.json` rather than to each other.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from distill.errors import LabError

# Up to this many scalars the values travel whole and the server compares them
# within the lab's tolerance; past it only a digest of the quantized values
# does, which is what keeps a large output off the wire.
SMALL_LIMIT = 64


def canonicalize(out: Any, atol: float) -> dict[str, Any]:
    arrays = out if isinstance(out, tuple) else (out,)
    parts = [np.ascontiguousarray(np.asarray(a, dtype=np.float64)) for a in arrays]
    for p in parts:
        if not np.isfinite(p).all():
            raise LabError(
                "your function returned NaN or an infinity: neither compares equal "
                "to itself, so there is nothing to grade"
            )
    total = sum(p.size for p in parts)
    if total <= SMALL_LIMIT:
        return {"values": [p.tolist() for p in parts]}
    h = hashlib.sha256()
    for p in parts:
        # Past int64 the quantized values saturate and every such output
        # collides on one digest — silently grading wrong answers right.
        if p.size and np.abs(p / atol).max() >= 2**63:
            raise LabError(
                f"your function returned values too large to compare: |x| must "
                f"stay under {2**63 * atol:.3g} at atol={atol:g}"
            )
        q = np.round(p / atol).astype(np.int64)
        h.update(str(p.shape).encode() + b"f8" + q.tobytes())
    return {"digest": h.hexdigest()}
