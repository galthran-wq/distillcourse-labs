"""Where a notebook's submissions go, and when that is decided.

The generated notebook's first cell is `import distill`, so anything the
learner or a checkout sets afterwards has to still be heard: the host is read
where the request is made rather than when the module loads.
"""

from __future__ import annotations

import pytest

import distill
from distill import api
from tests.conftest import LAB, Server


def test_the_host_is_read_when_the_request_is_made(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(api.HOST_VAR, "http://elsewhere.test/")

    distill.open_lab(LAB)

    assert str(server.urls[-1]) == f"http://elsewhere.test/api/v1/labs/{LAB}/manifest"


def test_without_the_variable_it_talks_to_the_deployment(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(api.HOST_VAR)

    distill.open_lab(LAB)

    assert str(server.urls[-1]).startswith(f"{api.DEFAULT_HOST}/api/v1/")
