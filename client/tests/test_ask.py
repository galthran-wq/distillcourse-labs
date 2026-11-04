"""Asking the tutor from the notebook: what is sent, and how the answer shows."""

from __future__ import annotations

import re
import sys
import time
import types
from typing import Any

import httpx
import numpy as np
import pytest

import distill
from distill import api, tutor
from distill.errors import LabError
from tests.conftest import (
    CHAT_ID,
    CHAT_URL,
    LAB,
    LAB_CHAT,
    NEW_CHAT_ID,
    TOKEN,
    Server,
)


def woe(table: Any) -> Any:
    return table


class Frame:
    """Enough of a DataFrame to be read as one, without pandas."""

    shape = (3, 2)
    dtypes = "bin    object\nwoe    float64"

    def head(self, rows: int) -> Frame:
        assert rows == tutor.TABLE_ROWS
        return self

    def to_string(self) -> str:
        return "  bin   woe\n0  a  0.12"


def test_a_question_carries_its_text_and_continues_the_labs_latest_chat(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    distill.ask("why is my IV negative?")

    assert paired.asked == [
        {"text": "why is my IV negative?", "chat_id": None, "attachments": []}
    ]
    assert paired.bearers[-1] == f"Bearer {TOKEN}"
    assert capsys.readouterr().out == (
        f"The log goes the other way round.\n\ncontinue on the site → {CHAT_URL}\n"
    )


def test_the_next_question_names_the_chat_the_last_answer_came_from(
    paired: Server,
) -> None:
    distill.ask("why is my IV negative?")
    distill.ask("and is my binning ok?")

    assert paired.asked[-1]["chat_id"] == CHAT_ID


def test_a_restarted_kernel_names_no_chat_and_leaves_the_choice_to_the_server(
    paired: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    distill.ask("why is my IV negative?")
    monkeypatch.setattr(distill, "_chat", None)

    distill.ask("and is my binning ok?")

    assert paired.asked[-1]["chat_id"] is None


def test_another_lab_does_not_inherit_the_chat(paired: Server) -> None:
    distill.ask("why is my IV negative?")
    distill.open_lab(LAB)

    distill.ask("and is my binning ok?")

    assert paired.asked[-1]["chat_id"] is None


def test_clearing_starts_a_chat_the_next_question_continues(
    paired: Server, capsys: pytest.CaptureFixture[str]
) -> None:
    distill.clear_chat()
    distill.ask("and is my binning ok?")

    assert capsys.readouterr().out.startswith("new chat started\n")
    assert any(url.path.endswith(f"/labs/{LAB}/chat") for url in paired.urls)
    assert paired.asked[-1]["chat_id"] == NEW_CHAT_ID == LAB_CHAT["chat_id"]


def test_each_object_is_attached_the_way_its_kind_reads(paired: Server) -> None:
    distill.ask("what is wrong?", woe, Frame(), np.arange(20).reshape(10, 2), 0.5)

    attached = paired.asked[-1]["attachments"]
    assert [(item["name"], item["kind"]) for item in attached] == [
        ("woe", "function"),
        ("arg2", "table"),
        ("arg3", "array"),
        ("arg4", "value"),
    ]
    assert attached[0]["text"].startswith("def woe(table: Any) -> Any:")
    assert attached[1]["text"] == (
        "shape: (3, 2)\ndtypes:\nbin    object\nwoe    float64\n\n"
        "  bin   woe\n0  a  0.12"
    )
    assert attached[2]["text"] == (
        "shape: (10, 2)\ndtype: int64\n\n[[0 1]\n [2 3]\n [4 5]\n [6 7]\n [8 9]]"
    )
    assert attached[3]["text"] == "0.5"


def test_a_function_without_source_is_attached_as_its_repr(paired: Server) -> None:
    distill.ask("what is this?", len)

    assert paired.asked[-1]["attachments"] == [
        {"name": "len", "kind": "function", "text": repr(len)}
    ]


def test_the_last_error_is_attached_once(
    paired: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        raise ZeroDivisionError("division by zero")
    except ZeroDivisionError as raised:
        monkeypatch.setattr(sys, "last_value", raised, raising=False)

    distill.ask("why did that fail?")
    distill.ask("and now?")

    [error] = paired.asked[0]["attachments"]
    assert (error["name"], error["kind"]) == ("ZeroDivisionError", "error")
    assert error["text"].startswith("Traceback")
    assert error["text"].endswith("ZeroDivisionError: division by zero")
    assert paired.asked[1]["attachments"] == []


def test_a_long_traceback_keeps_its_last_lines(
    paired: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Chained rather than recursive: a traceback folds a repeated frame into
    # one line, and a chain of causes is how a real error gets this long.
    cause: Exception | None = None
    for depth in range(20):
        try:
            raise ValueError(f"layer {depth}") from cause
        except ValueError as error:
            cause = error
    monkeypatch.setattr(sys, "last_value", cause, raising=False)

    distill.ask("why?")

    lines = paired.asked[0]["attachments"][0]["text"].splitlines()
    assert len(lines) == tutor.TRACEBACK_LINES
    assert lines[-1] == "ValueError: layer 19"


def test_attachments_are_cut_to_their_caps(paired: Server) -> None:
    distill.ask("too much?", "a" * 30_000, "b" * 30_000, "c" * 30_000, "d")

    texts = [item["text"] for item in paired.asked[-1]["attachments"]]
    assert all(len(text) <= tutor.ATTACHMENT_CHARS for text in texts)
    assert texts[0].endswith(tutor.TRUNCATED)
    assert sum(map(len, texts[:3])) == tutor.ATTACHMENTS_CHARS
    # Nothing left for the last one: it goes with its name and the mark alone.
    assert texts[3] == tutor.TRUNCATED


class Notebook:
    """IPython, as much of it as the answer is drawn with."""

    def __init__(self) -> None:
        self.drawn: list[str] = []
        self.displayed = 0

    def display(self, _markdown: Any, *, display_id: bool) -> Any:
        assert display_id
        self.displayed += 1
        return types.SimpleNamespace(
            update=lambda markdown: self.drawn.append(markdown)
        )


@pytest.fixture
def notebook(monkeypatch: pytest.MonkeyPatch) -> Notebook:
    fake = Notebook()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: object()  # type: ignore[attr-defined]
    display = types.ModuleType("IPython.display")
    display.Markdown = lambda text: text  # type: ignore[attr-defined]
    display.display = fake.display  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    return fake


@pytest.mark.usefixtures("paired")
def test_in_a_notebook_the_answer_is_markdown_redrawn_as_it_grows(
    notebook: Notebook,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The whole answer arrives within one redraw interval: drawn as the first
    # event lands, then not again until it is complete.
    monkeypatch.setattr(time, "monotonic", lambda: 1.0)

    distill.ask("why is my IV negative?")

    assert len(notebook.drawn) == 2
    assert notebook.drawn[0] == "*checking surprisal*\n\n"
    assert notebook.drawn[-1] == (
        "*checking surprisal*\n\nThe log goes the other way round.\n\n"
        f"[continue on the site →]({CHAT_URL})"
    )
    assert capsys.readouterr().out == ""


def test_a_failed_answer_shows_what_came_and_says_it_failed(
    paired: Server, notebook: Notebook
) -> None:
    paired.answer_events = [
        {"type": "text", "delta": "The log"},
        {"type": "error", "detail": "the model is down"},
    ]

    with pytest.raises(LabError, match="the model is down"):
        distill.ask("why is my IV negative?")

    assert notebook.drawn[-1] == "The log"


def test_a_chat_still_answering_says_to_wait(paired: Server) -> None:
    paired.refusal = httpx.Response(409, json={"detail": "busy"})

    with pytest.raises(LabError, match=api.BUSY):
        distill.ask("why is my IV negative?")


def test_a_spent_budget_is_read_out(paired: Server) -> None:
    paired.refusal = httpx.Response(
        429, json={"detail": "monthly budget of $50.00 is spent"}
    )

    with pytest.raises(LabError, match="monthly budget"):
        distill.ask("why is my IV negative?")


def test_an_unreachable_server_is_said_so(paired: Server) -> None:
    paired.unreachable = True

    with pytest.raises(LabError, match=api.UNREACHABLE):
        distill.ask("why is my IV negative?")


@pytest.mark.usefixtures("server")
def test_asking_needs_an_open_lab_and_a_pairing() -> None:
    with pytest.raises(LabError, match="open_lab"):
        distill.ask("why?")
    distill.open_lab(LAB)
    with pytest.raises(LabError, match=re.escape(api.UNPAIRED)):
        distill.ask("why?")
    with pytest.raises(LabError, match=re.escape(api.UNPAIRED)):
        distill.clear_chat()


def test_more_objects_than_the_server_takes_is_refused_here(paired: Server) -> None:
    with pytest.raises(LabError, match="at most"):
        distill.ask("too many?", *range(tutor.ATTACHMENTS + 1))

    assert paired.asked == []


def test_an_error_stays_for_the_retry_when_the_question_was_refused(
    paired: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ValueError("bad bins")
    monkeypatch.setattr(sys, "last_value", error, raising=False)
    paired.refusal = httpx.Response(409, json={"detail": "busy"})

    with pytest.raises(LabError):
        distill.ask("why?")

    assert sys.last_value is error


def test_a_refused_question_leaves_no_empty_output(
    paired: Server, notebook: Notebook
) -> None:
    paired.refusal = httpx.Response(409, json={"detail": "busy"})

    with pytest.raises(LabError):
        distill.ask("why?")

    assert notebook.drawn == []
    assert notebook.displayed == 0


def test_an_overlong_question_is_refused_here(paired: Server) -> None:
    with pytest.raises(LabError, match="at most"):
        distill.ask("x" * (tutor.QUESTION_CHARS + 1))

    assert paired.asked == []
