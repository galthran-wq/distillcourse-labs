"""The review under a check's verdict: asked for after the verdict is printed,
and never a reason the verdict is lost."""

from __future__ import annotations

import inspect
from typing import Any

import httpx
import numpy as np
import pytest

import distill
from tests.conftest import CHAT_ID, CHAT_URL, Server, verdict


def surprisal(p: Any) -> Any:
    return -np.log2(p)


VERDICT = "✓ surprisal (1/2 required)\n"


@pytest.mark.usefixtures("paired")
def test_the_verdict_is_printed_before_the_review_streams_below_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    distill.check("surprisal", surprisal)

    assert capsys.readouterr().out == (
        f"{VERDICT}Correct. Nothing to change.\n\ncontinue on the site → {CHAT_URL}\n"
    )


def test_the_submission_carries_the_source_and_the_review_names_it(
    paired: Server,
) -> None:
    distill.check("surprisal", surprisal)

    [posted] = paired.posted
    assert posted["source"] == inspect.getsource(surprisal)
    assert paired.reviewed == [{"submission_id": posted["id"]}]


def test_a_function_without_source_is_checked_and_not_reviewed(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    # A ufunc has no Python source to read.
    distill.check("surprisal", np.negative)

    assert paired.posted[0]["source"] is None
    assert paired.reviewed == []
    assert capsys.readouterr().out == VERDICT


def test_a_chat_still_answering_skips_the_review(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    paired.review_refusal = httpx.Response(409, json={"detail": "busy"})

    distill.check("surprisal", surprisal)

    assert capsys.readouterr().out == f"{VERDICT}{distill.REVIEW_SKIPPED}\n"


@pytest.mark.parametrize("status", [429, 500, 404])
def test_a_refused_review_is_one_line_under_the_verdict(
    paired: Server, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    paired.review_refusal = httpx.Response(status, json={"detail": "no"})

    distill.check("surprisal", surprisal)

    assert capsys.readouterr().out == f"{VERDICT}{distill.REVIEW_UNAVAILABLE}\n"


def test_an_unreachable_review_is_one_line_under_the_verdict(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    paired.review_unreachable = True

    distill.check("surprisal", surprisal)

    assert capsys.readouterr().out == f"{VERDICT}{distill.REVIEW_UNAVAILABLE}\n"


def test_a_review_that_breaks_off_ends_in_one_line_and_raises_nothing(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    paired.review_events = [
        {"type": "text", "delta": "Corr"},
        {"type": "error", "detail": "the model is down"},
    ]

    distill.check("surprisal", surprisal)

    out = capsys.readouterr().out
    assert out.startswith(VERDICT)
    assert out.endswith(f"{distill.REVIEW_UNAVAILABLE}\n")


def test_a_failed_check_is_reviewed_too(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    paired.answer = verdict(passed=False, hint="Natural log, not base two.", done=0)

    distill.check("surprisal", surprisal)

    assert len(paired.reviewed) == 1
    assert capsys.readouterr().out.startswith(
        "✗ surprisal — Natural log, not base two. (0/2 required)\n"
    )


def test_the_next_question_continues_the_chat_the_review_is_in(
    paired: Server,
) -> None:
    distill.check("surprisal", surprisal)
    distill.ask("show me the groupby version")

    assert paired.asked[-1]["chat_id"] == CHAT_ID
