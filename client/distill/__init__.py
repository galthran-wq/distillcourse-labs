"""Submit distillcourse lab checkpoints from the notebook you solve them in.

    import distill

    distill.login()
    distill.open_lab("<course>/<lesson>")

Everything after that is one call per checkpoint. The server grades and keeps
the score; this package computes the inputs, canonicalizes what your function
returned, posts it and prints the verdict — and, for `distill.check`, the
tutor's short review of the function below it. It remembers nothing between
kernels.

`distill.ask("why is my IV negative?", woe_table)` asks the tutor, with the
objects named after the question attached; `distill.clear_chat()` starts the
next question in a fresh conversation.
"""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from distill import api, tutor
from distill.canon import canonicalize
from distill.errors import LabError

__all__ = [
    "LabError",
    "ask",
    "check",
    "clear_chat",
    "login",
    "open_lab",
    "submit_predictions",
    "submit_review",
]

_api = api.connect()


@dataclass(frozen=True)
class _Lab:
    """The manifest, which is also the build this notebook was generated from:
    submissions carry that version, and the server refuses the ones it has
    outlived rather than grading them against a newer answer key."""

    lesson: str
    content_version: str
    checkpoints: dict[str, dict[str, Any]]


_lab: _Lab | None = None
# The chat the last answer came from. Only a shortcut: a question without one
# continues the learner's latest chat in the lab, which the server keeps, so a
# restarted kernel picks the conversation up where it was.
_chat: str | None = None


# How often the kernel asks whether the learner has clicked Approve. The
# server allows 60 polls per minute per code, so this spends a third of it.
POLL_SECONDS = 3

PAIRED = "paired — checkpoints from this kernel count towards your progress"


def login(code: str | None = None) -> None:
    """Pair this kernel with your account.

    With no argument, prints a link to approve this notebook from the browser
    you are signed in on, and waits for the click. With a code from the lesson
    page, exchanges it directly — the fallback for a kernel whose learner has
    no browser session anywhere.

    Skipped entirely when a token is already at hand: `DISTILL_TOKEN` in the
    environment, or a Colab Secret of that name you stored yourself.
    """
    if code is not None:
        _api.pair(code)
        print(PAIRED)
        return
    # Only a token from outside the kernel short-circuits: one paired earlier
    # here may have been revoked since, and re-running this cell is the
    # advertised way to pair again.
    if _api.provisioned is not None:
        print("already paired — this kernel holds a token")
        return
    started = _api.begin_device()
    print(f"→ Open {started['approve_url']} and approve this notebook")
    # The wait ends by the server's word, not a clock here: approval releases
    # the token, and an expired code answers with what to do about it.
    while not _api.poll_device(str(started["code"])):
        time.sleep(POLL_SECONDS)
    print(PAIRED)


def open_lab(lab: str) -> None:
    """Fetch what `<course>/<lesson>` asks for. Public: no pairing needed."""
    global _lab, _chat
    _chat = None
    manifest = _api.manifest(lab)
    _lab = _Lab(
        lesson=manifest["lesson"],
        content_version=manifest["content_version"],
        checkpoints={entry["id"]: entry for entry in manifest["checkpoints"]},
    )
    required = sum(1 for entry in _lab.checkpoints.values() if entry["required"])
    print(f"{lab} — {len(_lab.checkpoints)} checkpoints, {required} required")
    if _api.token is None:
        print(api.UNPAIRED)


REVIEW_UNAVAILABLE = "(review unavailable)"
REVIEW_SKIPPED = "(review skipped: the tutor is answering)"


def check(checkpoint: str, fn: Callable[..., Any]) -> None:
    """Run `fn` on this checkpoint's inputs and submit what it returned; then,
    below the verdict, a short review of `fn` from the tutor, written into this
    lab's chat so `distill.ask` can continue from it."""
    spec = _spec(checkpoint, "exact")
    payload = canonicalize(fn(*_inputs(spec)), float(spec["atol"]))
    source = tutor.source(fn)
    attempt = str(uuid.uuid4())
    _report(checkpoint, _submit(checkpoint, payload, attempt=attempt, source=source))
    if source is not None:
        _review(attempt)


def _review(attempt: str) -> None:
    """The review of a verdict already printed. Nothing about it can reach the
    verdict, so whatever goes wrong is one line rather than an error."""
    global _chat
    try:
        _chat = tutor.show(_api.review(_path(_opened()), {"submission_id": attempt}))
    except api.Busy:
        print(REVIEW_SKIPPED)
    except LabError:
        print(REVIEW_UNAVAILABLE)


# Scores travel "higher is better", so an error measure travels negated — the
# server compares one way for every metric. What is printed is the error.
LOWER_IS_BETTER = frozenset({"rmse"})


def submit_predictions(checkpoint: str, predictions: Any) -> None:
    """Submit predictions for the held-out set; the server scores them."""
    spec = _spec(checkpoint, "metric")
    result = _submit(checkpoint, np.asarray(predictions, dtype=np.float64).tolist())
    score = result["score"]
    _report(checkpoint, result, "" if score is None else _scored(spec, score))


def _scored(spec: dict[str, Any], score: float) -> str:
    sign = -1 if spec["metric"] in LOWER_IS_BETTER else 1
    threshold = sign * float(spec["threshold"])
    return f"{spec['metric']} {sign * score:.4f} vs {threshold:.4f}"


def submit_review(checkpoint: str, text: str) -> None:
    """Submit a written answer for the judge to read against its rubric.

    Judged while the request is open, so there is nothing to wait for: a
    verdict that comes back pending is one the server could not reach, and its
    own hint says to post the answer again. Nothing re-grades it later.
    """
    _spec(checkpoint, "review")
    _report(checkpoint, _submit(checkpoint, text))


def ask(text: str, *objects: Any) -> None:
    """Ask the tutor about this lab, in this lab's chat.

    Each object is sent along — a function as its source, a table as its shape,
    dtypes and first rows, an array as its first rows, anything else as its
    repr — and so is the error your last cell raised, if it raised one.
    """
    global _chat
    lab = _opened()
    if len(text) > tutor.QUESTION_CHARS:
        raise LabError(
            f"a question is at most {tutor.QUESTION_CHARS} characters — "
            "attach long material as an object instead"
        )
    _chat = tutor.show(
        _api.ask(
            _path(lab),
            {
                "text": text,
                "chat_id": _chat,
                "attachments": tutor.attachments(objects),
            },
        )
    )
    # Sent and answered: the next question must not carry the same error.
    sys.last_value = None


def clear_chat() -> None:
    """Start a new conversation with the tutor in this lab. The old one is kept,
    and listed on the site."""
    global _chat
    _chat = str(_api.new_chat(_path(_opened()))["chat_id"])
    print("new chat started")


def _path(lab: _Lab) -> str:
    """`<course>/<lesson>`, the lab as the API's paths name it."""
    return lab.lesson.removeprefix("lesson:")


def _opened() -> _Lab:
    if _lab is None:
        raise LabError(
            'no lab is open — run distill.open_lab("<course>/<lesson>") first'
        )
    return _lab


def _spec(checkpoint: str, kind: str) -> dict[str, Any]:
    declared = _opened().checkpoints.get(checkpoint)
    if declared is None:
        raise LabError(f'this lab has no checkpoint called "{checkpoint}"')
    if declared["kind"] != kind:
        raise LabError(
            f'"{checkpoint}": declared {declared["kind"]}, submitted as {kind}'
        )
    spec: dict[str, Any] = declared["spec"]
    return spec


def _inputs(spec: dict[str, Any]) -> Any:
    """The challenge inputs, generated here rather than downloaded.

    The same three lines run at build time in the capture stub, so the learner's
    function is called on exactly what the reference was called on. The seed is
    the whole agreement, and the code that consumes it is short enough to state
    twice across the two runtimes rather than shipped between them.
    """
    namespace: dict[str, Any] = {"rng": np.random.default_rng(spec["seed"]), "np": np}
    exec(spec["inputs_code"], namespace)
    return namespace["inputs"]


def _submit(
    checkpoint: str,
    payload: Any,
    *,
    attempt: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    lab = _opened()
    return _api.submit(
        {
            # One id per attempt, minted here because the submissions log is
            # keyed by it. Nothing in this client re-sends one — a cell run
            # again is a new attempt, and the learner meant it to be. `check`
            # mints its own, which its review then names.
            "id": attempt or str(uuid.uuid4()),
            "lesson": lab.lesson,
            "checkpoint": checkpoint,
            "content_version": lab.content_version,
            "payload": payload,
            "source": source,
        }
    )


def _report(checkpoint: str, result: dict[str, Any], detail: str = "") -> None:
    if result["pending"]:
        mark = "…"
    else:
        mark = "✓" if result["passed"] else "✗"
    said = detail or result["hint"]
    progress = result["progress"]
    line = f"{mark} {checkpoint}"
    if said:
        line += f" — {said}"
    print(
        f"{line} ({progress['passed_required']}/{progress['total_required']} required)"
    )
