"""Every way the server can say no, read out in one line.

A learner is at a prompt in a notebook, usually mid-lab: nothing here retries,
queues or writes anything down, so the only thing that matters is that the line
says which of the two things to do — run the cell again, or go back to the
lesson page.
"""

from __future__ import annotations

import httpx
import pytest

import distill
from distill import api
from distill.errors import LabError
from tests.conftest import LAB, Server


def submit() -> None:
    distill.submit_review("why-uniform", "Because no outcome is favoured.")


def test_an_unreachable_server_is_not_a_lost_answer(paired: Server) -> None:
    paired.unreachable = True

    with pytest.raises(LabError, match=api.UNREACHABLE):
        submit()


def test_a_rebuilt_lab_sends_the_learner_back_for_a_new_notebook(
    paired: Server,
) -> None:
    """409 is the version guard: the notebook was generated from a build whose
    answer key is gone, and no amount of retrying makes it gradeable."""
    paired.refusal = httpx.Response(409, json={"detail": "this lab has been rebuilt"})

    with pytest.raises(LabError, match=api.STALE):
        submit()


def test_a_revoked_token_reads_as_pairing_rather_than_as_a_failure(
    paired: Server,
) -> None:
    """The token outlives the kernel, so a rejected one was ended by the
    account — a password change or signing out everywhere revokes every
    notebook pairing. Pairing again is the whole fix, so the line says so."""
    paired.refusal = httpx.Response(401, json={"detail": "not paired"})

    with pytest.raises(LabError, match="distill.login"):
        submit()
    with pytest.raises(LabError, match="password change"):
        submit()


def test_a_spent_budget_says_how_long_to_wait_when_the_server_says(
    paired: Server,
) -> None:
    paired.refusal = httpx.Response(
        429,
        json={"detail": "too many requests, try again later"},
        headers={"retry-after": "45"},
    )

    with pytest.raises(LabError, match="try again in 45s"):
        submit()


def test_a_spent_daily_budget_is_a_wait_measured_in_hours(paired: Server) -> None:
    """What the review budget refuses with: the seconds the server counts in
    are not what a learner deciding whether to wait needs to read."""
    paired.refusal = httpx.Response(
        429,
        json={"detail": "too many requests, try again later"},
        headers={"retry-after": "86400"},
    )

    with pytest.raises(LabError, match="try again in 24 hours"):
        submit()


def test_a_refusal_without_a_delay_is_read_out_as_the_server_worded_it(
    paired: Server,
) -> None:
    paired.refusal = httpx.Response(
        429, json={"detail": "too many requests, try again later"}
    )

    with pytest.raises(LabError, match="too many requests, try again later"):
        submit()


def test_a_delay_given_as_a_date_is_not_read_out_as_a_delay(paired: Server) -> None:
    """RFC 7231 allows an HTTP-date there. Nothing this client talks to sends
    one, and printing "try again in Wed, 21 Oct 2015 07:28:00 GMTs" would be
    worse than the wording the server already supplied."""
    paired.refusal = httpx.Response(
        429,
        json={"detail": "too many requests, try again later"},
        headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"},
    )

    with pytest.raises(LabError, match="too many requests, try again later"):
        submit()


def test_a_refusal_meant_for_this_client_is_not_read_out_to_the_learner(
    paired: Server,
) -> None:
    """FastAPI answers a malformed body with a list of validation errors under
    the same key as its human-readable ones; printing that at a learner tells
    them nothing they can act on."""
    paired.refusal = httpx.Response(
        422, json={"detail": [{"loc": ["body", "payload"]}]}
    )

    with pytest.raises(LabError, match=r"the server refused that \(422\)"):
        submit()


def test_a_lab_that_does_not_exist_is_reported_when_it_is_opened(
    server: Server,
) -> None:
    server.refusal = httpx.Response(404, json={"detail": "no lab at that lesson"})

    with pytest.raises(LabError, match="no lab at that lesson"):
        distill.open_lab(LAB)
