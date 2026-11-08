"""The two ends of a question to the tutor that happen in the kernel: what is
sent along with it, and how the answer is shown while it arrives."""

from __future__ import annotations

import inspect
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from distill.errors import LabError

# What one attached object may weigh, and all of them together. The server
# holds the first one too; past either, the text is cut and says so.
ATTACHMENT_CHARS = 20_000
ATTACHMENTS_CHARS = 60_000
# How many objects one question carries, the error included; the server's cap.
ATTACHMENTS = 16
QUESTION_CHARS = 8_000
TRUNCATED = "\n[… truncated]"
TABLE_ROWS = 5
TRACEBACK_LINES = 30
# How often the answer under the cell is redrawn while text arrives: each
# redraw re-renders the whole Markdown, and a token at a time is a frontend
# that stops painting.
REDRAW_SECONDS = 0.2


def source(fn: Any) -> str | None:
    """A function's source within an attachment's cap, or None where Python
    cannot find it: a builtin, or a function defined somewhere its source is
    gone."""
    try:
        return _cut(inspect.getsource(fn), ATTACHMENT_CHARS)
    except (OSError, TypeError):
        return None


def _rendered(obj: Any) -> tuple[str, str]:
    """An object as the tutor reads it, and the kind that says how."""
    if inspect.isroutine(obj) or inspect.isclass(obj):
        text = source(obj)
        return "function", repr(obj) if text is None else text
    # A DataFrame, recognised by what it can do: this package does not
    # depend on pandas, and neither need the table.
    if all(hasattr(obj, name) for name in ("shape", "dtypes", "head")):
        return "table", (
            f"shape: {obj.shape}\ndtypes:\n{obj.dtypes}\n\n"
            f"{obj.head(TABLE_ROWS).to_string()}"
        )
    if isinstance(obj, np.ndarray):
        rows = obj[:TABLE_ROWS] if obj.ndim else obj
        return "array", (
            f"shape: {obj.shape}\ndtype: {obj.dtype}\n\n{np.array2string(rows)}"
        )
    return "value", repr(obj)


def _last_error() -> dict[str, str] | None:
    """The error the learner's last cell died of: IPython leaves it in
    `sys.last_value` until the next one replaces it. `distill.ask` clears it
    once a question carrying it has been answered."""
    error = getattr(sys, "last_value", None)
    if not isinstance(error, BaseException):
        return None
    lines = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    ).splitlines()
    return {
        "name": type(error).__name__,
        "kind": "error",
        "text": "\n".join(lines[-TRACEBACK_LINES:]),
    }


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(TRUNCATED))] + TRUNCATED


def attachments(objects: Iterable[Any]) -> list[dict[str, str]]:
    """What goes with a question: each object, then the last error if there is
    one, within the caps."""
    found = []
    for position, obj in enumerate(objects, start=1):
        name = getattr(obj, "__name__", None)
        kind, text = _rendered(obj)
        found.append(
            {
                "name": name if isinstance(name, str) else f"arg{position}",
                "kind": kind,
                "text": text,
            }
        )
    error = _last_error()
    if error is not None:
        found.append(error)
    if len(found) > ATTACHMENTS:
        raise LabError(f"at most {ATTACHMENTS} objects go with one question")
    left = ATTACHMENTS_CHARS
    for attachment in found:
        attachment["text"] = _cut(attachment["text"], min(ATTACHMENT_CHARS, left))
        left = max(0, left - len(attachment["text"]))
    return found


def _drawer() -> Callable[[str], None] | None:
    """What redraws the answer under the cell, or None outside a notebook."""
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
        from IPython.display import Markdown, display  # type: ignore[import-not-found]
    except ImportError:
        return None
    if get_ipython() is None:
        return None
    handle = display(Markdown(""), display_id=True)
    return lambda text: handle.update(Markdown(text))


def _markdown(steps: list[str], text: str, url: str | None) -> str:
    shown = f"*{' · '.join(steps)}*\n\n{text}" if steps else text
    return f"{shown}\n\n[continue on the site →]({url})" if url else shown


def show(events: Iterator[dict[str, Any]]) -> str | None:
    """Draw an answer as its events arrive, and return the chat it is in.

    In a notebook it is Markdown under the cell, redrawn as it grows. Anywhere
    else there is nothing to redraw, so it is printed once, whole, at the end.
    """
    # Made on the first event rather than up front: a question the server
    # refuses outright must not leave an empty output above its error.
    draw: Callable[[str], None] | None = None
    started = False
    steps: list[str] = []
    parts: list[str] = []
    url: str | None = None
    chat: str | None = None
    failure: str | None = None
    drawn = 0.0
    for event in events:
        if not started:
            draw, started = _drawer(), True
        if event["type"] == "step":
            steps.append(event["label"])
        elif event["type"] == "text":
            parts.append(event["delta"])
        elif event["type"] == "done":
            chat, url = event["chat_id"], event["url"]
        elif event["type"] == "error":
            failure = event["detail"]
        if draw is not None and time.monotonic() - drawn >= REDRAW_SECONDS:
            draw(_markdown(steps, "".join(parts), url))
            drawn = time.monotonic()
    if draw is not None:
        draw(_markdown(steps, "".join(parts), url))
    else:
        text = "".join(parts)
        print(f"{text}\n\ncontinue on the site → {url}" if url else text)
    if failure is not None:
        raise LabError(f"the tutor could not finish: {failure}")
    return chat
