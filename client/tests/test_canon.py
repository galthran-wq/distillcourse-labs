"""The client canonicalizes to the bytes the build captured the reference in.

The vectors are the monorepo's single statement of that format — the same file
`check_canon_vectors.py` holds the capture stub to and `tests/labs/test_verify`
holds the server compare to. Reading them here is what makes the three one
contract instead of three implementations that agree today.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from distill.canon import canonicalize
from distill.errors import LabError

VECTORS = Path(__file__).parents[2] / "frontend/lib/labs/python/canon_vectors.json"

# JSON has no literal for these, and the vectors stay parsable by every client.
SPECIAL = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}

CASES: list[dict[str, Any]] = json.loads(VECTORS.read_text())["cases"]


def revive(value: Any) -> Any:
    if isinstance(value, str):
        return SPECIAL[value]
    if isinstance(value, list):
        return [revive(item) for item in value]
    return value


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_a_committed_vector(case: dict[str, Any]) -> None:
    arrays = revive(case["arrays"])
    output = tuple(arrays) if case["as_tuple"] else arrays[0]
    if case.get("rejects"):
        with pytest.raises(LabError):
            canonicalize(output, case["atol"])
        return
    assert canonicalize(output, case["atol"]) == case["expected"]


def test_the_vectors_cover_both_halves_of_the_contract() -> None:
    """A file that lost its digest cases would still pass every assertion
    above, and the half of the format that travels as a digest is the half a
    client is most likely to get wrong."""
    forms = {next(iter(case["expected"])) for case in CASES if not case.get("rejects")}

    assert forms == {"values", "digest"}
