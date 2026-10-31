"""A checkpoint API that answers without a backend.

The shapes are the ones `apps/backend/tests/labs/conftest.py` builds a real lab
out of: a manifest whose spec halves are what the build publishes, and a
verdict with every field the submission endpoint returns. A fake that drifts
from them is a suite passing against a server that does not exist.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import httpx
import pytest

import distill
from distill import api

HOST = "http://labs.test"
LAB = "information-theory/surprisal-lab"
LESSON = f"lesson:{LAB}"
VERSION = "a1" * 32
TOKEN = "the-long-lived-bearer-token"
CODE = "K7xR2mQ9pLw4"
DEVICE_CODE = "dQ4nXw8pR2mL"

# The two device-flow answers, module-level so `test_wire` can hold their
# shapes to the same pinned file the real API's suite is held to.
DEVICE = {
    "code": DEVICE_CODE,
    "approve_url": f"{HOST}/labs/pair/{DEVICE_CODE}",
    "expires_at": "2026-08-14T12:10:00Z",
}
TOKEN_ANSWER = {"token": TOKEN}

CHAT_ID = "5b1c3a52-8f0e-4d55-9a49-0d6b1c2e7f10"
CHAT_URL = f"{HOST}/courses/{LAB.partition('/')[0]}/lessons/{LAB.partition('/')[2]}?chat={CHAT_ID}"
NEW_CHAT_ID = "9e7d2c41-3b6a-4f88-8c15-2a4e6f0b9d33"
LAB_CHAT = {"chat_id": NEW_CHAT_ID, "url": CHAT_URL.replace(CHAT_ID, NEW_CHAT_ID)}
# One answer as the ask endpoint streams it: every kind of event but `error`,
# which a failed run sends in place of `done`.
ANSWER: list[dict[str, Any]] = [
    {"type": "step", "label": "checking surprisal"},
    {"type": "text", "delta": "The log goes "},
    {"type": "text", "delta": "the other way round."},
    {"type": "done", "chat_id": CHAT_ID, "url": CHAT_URL},
]

# The review a check's verdict is followed by, streamed as an answer is.
REVIEW: list[dict[str, Any]] = [
    {"type": "text", "delta": "Correct. "},
    {"type": "text", "delta": "Nothing to change."},
    {"type": "done", "chat_id": CHAT_ID, "url": CHAT_URL},
]

INPUTS_CODE = "p = rng.random(4)\ninputs = (p,)\n"
ATOL = 1e-06
THRESHOLD = 0.85

MANIFEST: dict[str, Any] = {
    "lesson": LESSON,
    "content_version": VERSION,
    "data_files": [],
    "checkpoints": [
        {
            "id": "surprisal",
            "kind": "exact",
            "text": "Return the surprisal of each probability.",
            "required": True,
            "spec": {"seed": 1337, "inputs_code": INPUTS_CODE, "atol": ATOL},
        },
        {
            "id": "beat-baseline",
            "kind": "metric",
            "text": "Beat the baseline on the held-out set.",
            "required": True,
            "spec": {
                "metric": "roc_auc",
                "threshold": THRESHOLD,
                "holdout": "data/holdout.csv",
            },
        },
        {
            "id": "why-uniform",
            "kind": "review",
            "text": "Say why the uniform distribution maximises entropy.",
            "required": False,
            "spec": {},
        },
    ],
}


def verdict(
    *,
    passed: bool | None = True,
    hint: str | None = None,
    score: float | None = None,
    pending: bool = False,
    done: int = 1,
    required: int = 2,
) -> dict[str, Any]:
    return {
        "passed": passed,
        "hint": hint,
        "score": score,
        "pending": pending,
        "progress": {"passed_required": done, "total_required": required},
    }


def _streamed(events: list[dict[str, Any]]) -> httpx.Response:
    """An answer as the server streams one: an event per `data:` line."""
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
    )


class Server:
    """What the client talked to, and what it will be told next."""

    def __init__(self) -> None:
        self.posted: list[dict[str, Any]] = []
        self.bearers: list[str | None] = []
        self.urls: list[httpx.URL] = []
        # A copy, so a test that needs another lab's manifest edits its own.
        self.manifest = deepcopy(MANIFEST)
        self.answer: dict[str, Any] = verdict()
        self.refusal: httpx.Response | None = None
        self.unreachable = False
        # The device flow's script: how many polls answer "not yet" before the
        # token, and a refusal that pre-empts the poll entirely (an expiry).
        self.polls_before_approval = 0
        self.device_refusal: httpx.Response | None = None
        self._device_polls = 0
        self.asked: list[dict[str, Any]] = []
        self.answer_events: list[dict[str, Any]] = deepcopy(ANSWER)
        self.reviewed: list[dict[str, Any]] = []
        self.review_events: list[dict[str, Any]] = deepcopy(REVIEW)
        # A refusal or an outage of the review alone: the verdict before it
        # was answered.
        self.review_refusal: httpx.Response | None = None
        self.review_unreachable = False

    def spec(self, checkpoint: str) -> dict[str, Any]:
        """The public half of one checkpoint, to edit before `open_lab`."""
        entry = next(c for c in self.manifest["checkpoints"] if c["id"] == checkpoint)
        published: dict[str, Any] = entry["spec"]
        return published

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(request.url)
        if self.unreachable:
            raise httpx.ConnectError("no route to host", request=request)
        if self.refusal is not None:
            return self.refusal
        if request.url.path.endswith("/pair"):
            return httpx.Response(200, json=TOKEN_ANSWER)
        if request.url.path.endswith("/labs/device"):
            return httpx.Response(201, json=DEVICE)
        if f"/labs/device/{DEVICE_CODE}" in request.url.path:
            if self.device_refusal is not None:
                return self.device_refusal
            self._device_polls += 1
            if self._device_polls <= self.polls_before_approval:
                return httpx.Response(428, json={"detail": "not approved yet"})
            return httpx.Response(200, json=TOKEN_ANSWER)
        if request.url.path.endswith("/manifest"):
            return httpx.Response(200, json=self.manifest)
        if request.url.path.endswith(f"/labs/{LAB}/chat"):
            self.bearers.append(request.headers.get("authorization"))
            return httpx.Response(201, json=LAB_CHAT)
        if request.url.path.endswith(f"/labs/{LAB}/ask"):
            self.bearers.append(request.headers.get("authorization"))
            self.asked.append(json.loads(request.content))
            return _streamed(self.answer_events)
        if request.url.path.endswith(f"/labs/{LAB}/review"):
            self.bearers.append(request.headers.get("authorization"))
            self.reviewed.append(json.loads(request.content))
            if self.review_unreachable:
                raise httpx.ConnectError("no route to host", request=request)
            if self.review_refusal is not None:
                return self.review_refusal
            return _streamed(self.review_events)
        self.bearers.append(request.headers.get("authorization"))
        self.posted.append(json.loads(request.content))
        return httpx.Response(200, json=self.answer)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Server:
    fake = Server()
    monkeypatch.delenv(api.TOKEN_VAR, raising=False)
    monkeypatch.setenv(api.HOST_VAR, HOST)
    monkeypatch.setattr(distill, "_lab", None)
    monkeypatch.setattr(
        distill,
        "_api",
        api.Api(httpx.Client(transport=httpx.MockTransport(fake.handle))),
    )
    return fake


@pytest.fixture
def paired(server: Server) -> Server:
    """A kernel that has run the login cell and opened the lab."""
    distill.login(CODE)
    distill.open_lab(LAB)
    return server
