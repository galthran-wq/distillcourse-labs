# distillcourse

The client a lab notebook submits its checkpoints with. The generated notebook
installs it in its setup cell, so a learner's three lines are:

```python
%pip install -q "git+https://github.com/galthran-wq/distillcourse-labs#subdirectory=client"
import distill

distill.open_lab("<course>/<lesson>")
```

`open_lab` needs no account: it fetches the lab's manifest, which is public.
Submitting does, and pairing is the notebook's second generated cell:

```python
distill.login()
```

With no argument it prints an approval link — open it in the browser you are
signed in on, click Approve, and the cell finishes on its own within a few
seconds. `distill.login("<code>")` with a code from the lesson page is the
fallback for a learner with no signed-in browser to click from.

A token already at hand skips pairing entirely, in this order: `DISTILL_TOKEN`
in the environment; the same name stored as a Colab Secret (the Secrets panel
is write-only from outside — this client never puts one there); then the
approval flow; then an explicit code.

The token pairing buys lives in the kernel and nowhere else — not printed,
not written to disk, gone when the runtime restarts. It is also ended by the
account: changing your password or signing out everywhere revokes every
notebook pairing, and the next checkpoint says so. Pair again after either.
`DISTILL_API` points the client at a backend other than
`https://distillcourse.com`; like `DISTILL_TOKEN` it is read when a request is
made, so setting either in a cell below `import distill` works.

## Checkpoints

```python
distill.check("logloss", log_loss)          # runs your function on the lab's inputs
distill.submit_predictions("holdout", p)    # scored against labels you never see
distill.submit_review("why-uniform", text)  # read by the judge against its rubric
```

Each prints one line: `✓`, `✗` or `…`, the checkpoint, whatever the server had
to say about it, and how many of the required checkpoints you have passed. The
server keeps the score — this package has no memory between kernels, and
re-running a cell is always safe. A `…` is not a queue: every kind is decided
while the request is open, including a review, so it means the server could
not decide this one and the line says to post it again.

`check` generates the inputs from the seed the manifest carries, which is what
the reference solution was run on at build time, and posts what your function
returned in the canonical form `canon_vectors.json` pins. Nothing about the
expected answer is downloaded, and nothing about how far off yours was comes
back: a wrong answer earns a hint only where the lab's author anticipated that
particular mistake.

## The review under a check

Every `distill.check(...)` prints its verdict first, then asks for a short
review of the function you checked and streams it below: for a pass, whether
anything is worth changing; for a fail, one hint at where to look. It reads
your function's source (`inspect.getsource`, so a function with no Python
source is checked but not reviewed) and is written into the lab's chat, so a
`distill.ask(...)` after it continues from it. A review that cannot be had —
the tutor is answering in that chat, the service is down — is one line under
the verdict, which it never changes.

## Asking the tutor

```python
distill.ask("why is my IV negative?", woe_table)  # objects after the text go with it
distill.clear_chat()                              # the next question starts a new chat
```

The tutor is the site's own, in a chat that belongs to this lab: it can read
your checkpoint standing, each checkpoint's expected output and the lab's
reference solution, and it decides how much of that to show. Each object is
sent the way it reads best — a function or class as its source, a DataFrame as
its shape, dtypes and first rows, an array as its first rows, anything else as
its `repr` — and so is the traceback of the error your last cell raised, once.
Each is cut at 20,000 characters, all of them together at 60,000.

In a notebook the answer is Markdown under the cell, growing as it arrives;
anywhere else it is printed whole at the end. It ends with a link to the same
chat on the lab's page. The conversation is kept by the server, not the
kernel: a question after a restart continues the lab's latest chat, and
`clear_chat()` is the way to start over.

## Development

Source of truth is `apps/lab-client/` in the monorepo; the mirror's `client/`
is a copy. Python 3.10 — Colab's floor, and what the suite runs on.

```sh
uv run pytest
./scripts/lint.sh
```

Tests answer with `httpx.MockTransport` rather than a backend. Two things
outside this package are contracts it cannot restate: canonicalization, pinned
by `apps/frontend/lib/labs/python/canon_vectors.json` and asserted against
here, and the submission shapes, which mirror `apps/backend/tests/labs/`.
