from __future__ import annotations

import sys
import types

import httpx
import pytest

import distill
from distill import api
from distill.errors import LabError
from tests.conftest import CODE, DEVICE, LAB, TOKEN, Server


@pytest.fixture
def instant(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The naps the poll loop would have taken, recorded instead of slept."""
    naps: list[float] = []
    monkeypatch.setattr("time.sleep", naps.append)
    return naps


def colab_with_secrets(
    monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str]
) -> None:
    """A kernel that is Colab: `from google.colab import userdata` succeeds and
    answers from `secrets`, raising — as the real one does — on anything else."""
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    # noqa B010 twice: these are modules being furnished, not objects whose
    # attributes mypy could know about.
    setattr(colab, "userdata", types.SimpleNamespace(get=lambda name: secrets[name]))  # noqa: B010
    setattr(google, "colab", colab)  # noqa: B010
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def test_a_code_becomes_the_token_every_submission_carries(
    server: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    distill.login(CODE)
    distill.open_lab(LAB)
    distill.submit_review("why-uniform", "Because no outcome is favoured.")

    assert server.bearers == [f"Bearer {TOKEN}"]
    assert TOKEN not in capsys.readouterr().out


def test_an_unpaired_kernel_is_told_how_to_pair(
    server: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    """Opening a lab needs no token — the manifest is public — so this is where
    a learner who skipped the login cell finds out, rather than at the first
    checkpoint they have just solved."""
    distill.open_lab(LAB)

    assert api.UNPAIRED in capsys.readouterr().out
    with pytest.raises(LabError, match="not paired"):
        distill.submit_review("why-uniform", "Because no outcome is favoured.")
    assert server.posted == []


def test_a_token_in_the_environment_stands_in_for_pairing(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(api.TOKEN_VAR, "a-token-issued-by-hand")
    distill.open_lab(LAB)
    distill.submit_review("why-uniform", "Because no outcome is favoured.")

    assert server.bearers == ["Bearer a-token-issued-by-hand"]


def test_no_argument_login_walks_the_device_flow(
    server: Server, instant: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    server.polls_before_approval = 2

    distill.login()
    distill.open_lab(LAB)
    distill.submit_review("why-uniform", "Because no outcome is favoured.")

    out = capsys.readouterr().out
    assert DEVICE["approve_url"] in out
    assert TOKEN not in out
    assert server.bearers == [f"Bearer {TOKEN}"]
    # Two "not yet" answers cost two naps; the release costs none.
    assert instant == [distill.POLL_SECONDS, distill.POLL_SECONDS]


def test_polling_stops_when_the_approval_expires(
    server: Server, instant: list[float]
) -> None:
    server.device_refusal = httpx.Response(
        410, json={"detail": "the approval expired — run distill.login() again"}
    )

    with pytest.raises(LabError, match="expired"):
        distill.login()
    assert instant == []


def test_a_token_already_at_hand_skips_the_device_flow(
    server: Server, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(api.TOKEN_VAR, "a-token-issued-by-hand")

    distill.login()

    assert "already paired" in capsys.readouterr().out
    assert server.urls == []


def test_a_colab_secret_stands_in_for_pairing(
    server: Server, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    colab_with_secrets(monkeypatch, {api.TOKEN_VAR: "from-the-secrets-panel"})

    distill.login()
    distill.open_lab(LAB)
    distill.submit_review("why-uniform", "Because no outcome is favoured.")

    assert "already paired" in capsys.readouterr().out
    assert server.bearers == ["Bearer from-the-secrets-panel"]


def test_the_environment_outranks_the_colab_secret(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    colab_with_secrets(monkeypatch, {api.TOKEN_VAR: "from-the-secrets-panel"})
    monkeypatch.setenv(api.TOKEN_VAR, "from-the-environment")

    distill.open_lab(LAB)
    distill.submit_review("why-uniform", "Because no outcome is favoured.")

    assert server.bearers == ["Bearer from-the-environment"]


def test_running_the_login_cell_again_replaces_a_revoked_pairing(
    server: Server, instant: list[float]
) -> None:
    """A pairing made earlier in this kernel never short-circuits: the server
    may have revoked it, and re-running the cell is the advertised recovery."""
    distill.login(CODE)

    distill.login()

    assert any("/labs/device" in str(url) for url in server.urls)
    # Approved before the first poll, so the loop never napped.
    assert instant == []


def test_a_kernel_that_is_colab_without_the_secret_still_pairs(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stand-in raises on an absent name exactly like Colab's own
    `userdata`, and the flow falls through to the device pairing."""
    colab_with_secrets(monkeypatch, {})

    distill.login()

    assert any("/labs/device" in str(url) for url in server.urls)
