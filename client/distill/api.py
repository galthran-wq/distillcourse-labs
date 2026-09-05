"""The checkpoint API as a notebook talks to it.

Nothing is retried and nothing is written to disk: the learner is sitting at a
prompt, and running the cell again is the retry. Every refusal the server can
answer with becomes one line telling them what to do about it.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from distill.errors import LabError

# The deployment a notebook opened from the public mirror submits to.
DEFAULT_HOST = "https://distillcourse.com"
# DISTILL_API points a checkout at a backend of its own. Read where the request
# is made rather than at import, because the generated notebook's first cell is
# `import distill`: a host set in any cell after it would otherwise be ignored.
HOST_VAR = "DISTILL_API"
# Pairing is skipped entirely when this is set — in the environment, or as a
# Colab Secret of the same name the learner stored by hand (Colab's `userdata`
# is read-only from a notebook, so nothing here ever writes one).
TOKEN_VAR = "DISTILL_TOKEN"

TIMEOUT = 30.0

UNREACHABLE = (
    "could not reach distillcourse — check the connection and run the cell again"
)
STALE = "the lab was updated — reopen the notebook from the course page"
UNPAIRED = "not paired — run distill.login() and approve it from your signed-in browser"
# A token the server will not take. It outlives the kernel, so the likely
# reason is that the account ended it: changing the password or signing out
# everywhere revokes every notebook pairing. Pairing again is the whole fix,
# which is why this says so rather than reporting a failure.
REVOKED = (
    "this notebook's pairing has ended — a password change or signing out "
    "everywhere ends one — run distill.login() to pair again"
)


class Api:
    """One notebook's connection, and the token its submissions carry.

    The token stays here, in this process, for as long as the kernel lives: it
    speaks for the learner's account, so it is never printed and never stored.
    """

    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        self._paired: str | None = None

    @property
    def provisioned(self) -> str | None:
        """A token the learner supplied out of band — the environment, or a
        Colab Secret. What no-argument login defers to. A token paired earlier
        in this kernel is deliberately not one: the pairing cell is what a
        learner re-runs after a revocation, so it must pair again rather than
        report the dead token as already good."""
        return os.environ.get(TOKEN_VAR) or _colab_secret()

    @property
    def token(self) -> str | None:
        return self.provisioned or self._paired

    def pair(self, code: str) -> None:
        self._paired = str(
            self._request("POST", "/labs/pair", body={"code": code})["token"]
        )

    def begin_device(self) -> dict[str, Any]:
        """Start a pairing this kernel will poll for: the server hands back the
        code and the URL the learner approves it at, signed in on the site."""
        return self._request("POST", "/labs/device")

    def poll_device(self, code: str) -> bool:
        """One ask: True once the approval released the token, which is kept
        here and never returned; False while it has not happened yet. A code
        the server is done with — expired, already released — is a LabError
        whose message says what to do."""
        try:
            response = self._client.get(_url(f"/labs/device/{code}"))
        except httpx.HTTPError:
            raise LabError(UNREACHABLE) from None
        if response.status_code == httpx.codes.PRECONDITION_REQUIRED:
            return False
        if not response.is_success:
            raise LabError(_refusal(response))
        self._paired = str(response.json()["token"])
        return True

    def manifest(self, lab: str) -> dict[str, Any]:
        return self._request("GET", f"/labs/{lab}/manifest")

    def submit(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/labs/submissions", body=body, as_learner=True)

    def _request(
        self, method: str, path: str, *, body: Any = None, as_learner: bool = False
    ) -> dict[str, Any]:
        headers = {}
        if as_learner:
            if self.token is None:
                raise LabError(UNPAIRED)
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = self._client.request(
                method, _url(path), json=body, headers=headers
            )
        except httpx.HTTPError:
            raise LabError(UNREACHABLE) from None
        if not response.is_success:
            raise LabError(_refusal(response))
        answer: dict[str, Any] = response.json()
        return answer


def connect() -> Api:
    return Api(httpx.Client(timeout=TIMEOUT))


def _colab_secret() -> str | None:
    """The token a learner stored in Colab's Secrets panel, if this kernel is
    Colab and they granted this notebook access to it."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        value = userdata.get(TOKEN_VAR)
    except Exception:
        # Colab's own SecretNotFoundError / NotebookAccessError, neither of
        # which exists as a type outside Colab: both mean "nothing to use
        # here", and the flow falls through to pairing.
        return None
    return str(value) or None


def _url(path: str) -> str:
    host = os.environ.get(HOST_VAR) or DEFAULT_HOST
    return f"{host.rstrip('/')}/api/v1{path}"


def _refusal(response: httpx.Response) -> str:
    if response.status_code == httpx.codes.UNAUTHORIZED:
        return REVOKED
    if response.status_code == httpx.codes.CONFLICT:
        return STALE
    after = response.headers.get("retry-after")
    # RFC 7231 also allows an HTTP-date there; the server sends a delay, and a
    # date read out as a delay would be worse than not saying one at all.
    if (
        response.status_code == httpx.codes.TOO_MANY_REQUESTS
        and after
        and after.isdigit()
    ):
        return f"too many attempts — try again in {_wait(int(after))}"
    return _detail(response)


def _wait(seconds: int) -> str:
    """The delay as the learner would say it: a daily budget is hours away."""
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} minutes"
    return f"{round(seconds / 3600)} hours"


def _detail(response: httpx.Response) -> str:
    """What the server said, when it said it in words meant for a person.

    FastAPI answers a malformed body with a list of validation errors under the
    same key; that is a bug in this client rather than something to read out.
    """
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str):
        return detail
    return f"the server refused that ({response.status_code})"
