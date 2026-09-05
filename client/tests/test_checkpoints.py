from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import distill
from distill.errors import LabError
from tests.conftest import ATOL, LAB, LESSON, VERSION, Server, verdict

SEEDED = np.random.default_rng(1337).random(4)


def test_the_function_is_called_on_the_inputs_the_reference_was(
    paired: Server,
) -> None:
    """The seed is the whole agreement between the two runs: the build captured
    the expected output from these numbers, so anything else is graded against
    a different question."""
    seen: list[Any] = []

    def record(p: Any) -> Any:
        seen.append(p)
        return np.zeros(2)

    distill.check("surprisal", record)

    assert np.array_equal(seen[0], SEEDED)
    assert paired.posted[0]["payload"] == {"values": [[0.0, 0.0]]}


def test_a_small_output_travels_whole_and_a_large_one_as_a_digest(
    paired: Server,
) -> None:
    distill.check("surprisal", lambda p: -np.log2(p))
    distill.check("surprisal", lambda p: np.zeros(100))

    small, large = (posted["payload"] for posted in paired.posted)
    assert small == {"values": [(-np.log2(SEEDED)).tolist()]}
    assert list(large) == ["digest"]


def test_a_submission_says_which_lab_and_which_build_it_answers(
    paired: Server,
) -> None:
    distill.check("surprisal", lambda p: p)
    distill.check("surprisal", lambda p: p)

    first, second = paired.posted
    assert first["lesson"] == LESSON
    assert first["content_version"] == VERSION
    assert first["checkpoint"] == "surprisal"
    # A retry reuses an id to claim the verdict it already earned; two attempts
    # are two submissions and must not collide on the log's primary key.
    assert first["id"] != second["id"]


def test_a_pass_prints_the_standing(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    paired.answer = verdict(passed=True, done=3, required=5)

    distill.check("surprisal", lambda p: p)

    assert capsys.readouterr().out.endswith("✓ surprisal (3/5 required)\n")


def test_a_declared_mistake_prints_the_hint_that_names_it(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    paired.answer = verdict(passed=False, hint="Natural log, not base two.", done=0)

    distill.check("surprisal", lambda p: -np.log(p))

    assert capsys.readouterr().out.endswith(
        "✗ surprisal — Natural log, not base two. (0/2 required)\n"
    )


def test_predictions_are_posted_as_plain_numbers_and_scored(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    paired.answer = verdict(passed=True, score=0.91, done=2)

    distill.submit_predictions("beat-baseline", np.array([0.1, 0.9, 0.5]))

    assert paired.posted[0]["payload"] == [0.1, 0.9, 0.5]
    assert capsys.readouterr().out.endswith(
        "✓ beat-baseline — roc_auc 0.9100 vs 0.8500 (2/2 required)\n"
    )


def test_an_error_measure_is_printed_as_the_error_it_measures(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    """Scores are compared "higher is better", so rmse travels negated. The
    sign is the wire format; what a learner is shown is the error."""
    paired.answer = verdict(passed=True, score=-0.31, done=2)
    paired.spec("beat-baseline").update({"metric": "rmse", "threshold": -0.5})
    distill.open_lab(LAB)

    distill.submit_predictions("beat-baseline", np.array([0.1, 0.9, 0.5]))

    assert capsys.readouterr().out.endswith(
        "✓ beat-baseline — rmse 0.3100 vs 0.5000 (2/2 required)\n"
    )


def test_a_review_the_judge_could_not_answer_says_to_post_it_again(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    """A review is judged while the request is open and nothing re-grades a
    stored verdict, so a pending one is not a queue — it is the server saying
    it could not decide, in its own words."""
    paired.answer = verdict(
        passed=None, pending=True, hint="Judging is unavailable right now."
    )

    distill.submit_review("why-uniform", "Because no outcome is favoured.")

    assert paired.posted[0]["payload"] == "Because no outcome is favoured."
    assert capsys.readouterr().out.endswith(
        "… why-uniform — Judging is unavailable right now. (1/2 required)\n"
    )


def test_a_judged_review_prints_what_the_judge_said(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    paired.answer = verdict(passed=True, hint="Names the constraint.", done=2)

    distill.submit_review("why-uniform", "Because no outcome is favoured.")

    assert capsys.readouterr().out.endswith(
        "✓ why-uniform — Names the constraint. (2/2 required)\n"
    )


def test_a_call_the_lab_does_not_declare_never_reaches_the_server(
    paired: Server,
) -> None:
    with pytest.raises(LabError, match="no checkpoint called"):
        distill.check("entropy", lambda p: p)
    with pytest.raises(LabError, match="declared metric"):
        distill.check("beat-baseline", lambda p: p)
    with pytest.raises(LabError, match="declared exact"):
        distill.submit_review("surprisal", "…")
    assert paired.posted == []


def test_nothing_can_be_submitted_before_a_lab_is_opened(server: Server) -> None:
    with pytest.raises(LabError, match="no lab is open"):
        distill.check("surprisal", lambda p: p)
    assert server.posted == []


def test_an_output_nothing_can_be_compared_to_is_refused_here(
    paired: Server,
) -> None:
    """A NaN is the shape a broken solution most often takes, and it would be
    posted as a value no reference can match. It is told apart from a wrong
    answer before the attempt is spent."""
    with pytest.raises(LabError, match="NaN"):
        distill.check("surprisal", lambda p: p * np.nan)
    assert paired.posted == []


def test_the_lab_declares_the_tolerance_the_digest_quantizes_by(
    paired: Server,
) -> None:
    """The manifest's atol reaches canonicalization: quantizing by a different
    one produces a digest the server cannot match, and the values half of the
    contract would never notice."""
    distill.check("surprisal", lambda p: np.full(100, ATOL * 1.4))
    coarse = paired.posted[0]["payload"]["digest"]

    distill.check("surprisal", lambda p: np.full(100, ATOL * 1.6))

    assert paired.posted[1]["payload"]["digest"] != coarse
