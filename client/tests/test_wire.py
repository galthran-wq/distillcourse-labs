"""The fake server's half of the shapes `python/wire_shapes.json` pins.

Every test in this package answers from `conftest.Server`, which restates by
hand what the real backend serves. A fake that drifts from it is a suite
passing against a server that does not exist — so the shapes it answers with
are held to the same committed file the backend's own suite is held to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import distill
from tests.conftest import DEVICE, MANIFEST, TOKEN_ANSWER, Server, verdict

SHAPES: dict[str, Any] = json.loads(
    (
        Path(__file__).parents[2] / "frontend/lib/labs/python/wire_shapes.json"
    ).read_text()
)


def test_the_manifest_the_fake_serves_is_the_one_the_api_serves() -> None:
    assert set(MANIFEST) == set(SHAPES["manifest"])
    for checkpoint in MANIFEST["checkpoints"]:
        assert set(checkpoint) == set(SHAPES["manifest_checkpoint"])


def test_the_device_answers_the_fake_serves_are_the_ones_the_api_serves() -> None:
    assert set(DEVICE) == set(SHAPES["device"])
    assert set(TOKEN_ANSWER) == set(SHAPES["token"])


def test_the_verdict_the_fake_answers_with_is_the_one_the_api_answers_with() -> None:
    answer = verdict()

    assert set(answer) == set(SHAPES["submission_result"])
    assert set(answer["progress"]) == set(SHAPES["completion"])


def test_the_body_this_client_posts_is_the_one_the_api_takes(paired: Server) -> None:
    """The producing side, which no fake can vouch for."""
    distill.check("surprisal", lambda p: p)

    assert set(paired.posted[0]) == set(SHAPES["submission"])
