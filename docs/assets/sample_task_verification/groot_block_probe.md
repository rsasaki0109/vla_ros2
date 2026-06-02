# GR00T Block Probe

Date: 2026-06-03

This probe records the **precise, reproducible** reason the GR00T adapter is
`blocked` in `vla_zoo`, so the evidence-matrix status is backed by an actual
failure rather than a prose assertion. No GR00T inference is implemented and no
task-success claim is made.

## Environment

| Component | Value |
|---|---|
| Package env | `.venv` (base) and `.venv-smolvla` |
| `gr00t` importable | No (`importlib.util.find_spec("gr00t")` is `None` in both envs) |

## Probe 1 — the adapter refuses rather than fabricates

```python
from vla_zoo import load_model
m = load_model("groot")              # constructs an inert scaffold
m.predict(image=None, instruction="walk forward")
```

Result: raises `MissingDependencyError` with the single-source `GROOT_BLOCKED_NOTE`
("GR00T is blocked until the NVIDIA Isaac GR00T stack is wired in…"). The adapter
never returns an action. Even if `gr00t` imported, `predict_observation` raises
`NotImplementedError` — a real serving adapter is still required.

## Probe 2 — there is no pip-installable GR00T package

`import gr00t` raises `ModuleNotFoundError` in every environment here, and the
obvious distribution names do not exist on PyPI:

| Candidate package | PyPI status |
|---|---|
| `gr00t` | Not found (HTTP 404) |
| `isaac-gr00t` | Not found (HTTP 404) |
| `nvidia-gr00t` | Not found (HTTP 404) |

So the block is **not** a missing `pip install` line that `vla_zoo` could add. The
real runtime is the [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
GitHub stack — cloned and built in a dedicated environment with its own weights and
license — not a published Python package.

## Reproduce

```bash
# Probe 1: the adapter refuses
PYTHONPATH=src .venv/bin/python -c "
from vla_zoo import load_model
from vla_zoo.core.errors import MissingDependencyError
try:
    load_model('groot').predict(image=None, instruction='walk forward')
except MissingDependencyError as e:
    print('blocked:', str(e)[:60])
"

# Probe 2: no importable package, none on PyPI
PYTHONPATH=src .venv/bin/python -c "import importlib.util as u; print(u.find_spec('gr00t'))"
```

## Conclusion

GR00T stays `blocked` by design and the status is reproducible: no `gr00t` module
is importable, no GR00T package is published to PyPI, and the adapter raises rather
than fabricating actions. Unblocking requires the external NVIDIA Isaac-GR00T stack
in a dedicated serving environment plus a real serving adapter and a recorded action
probe — see the [GR00T status page](../../groot_remote.md).
