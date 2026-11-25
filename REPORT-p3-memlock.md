# REPORT-p3-memlock — P3: say it when the memlock limit is too low for the RAM tier

## What changed

Two commits on `saguaro/q1-p3-memlock` (base `e2cdfb25b`):

- `a9992be9a` — step 1, saguaro.py only:
  - `memlock_soft_gib()` (after `qfn_ram_said`, saguaro.py:7850): the soft `RLIMIT_MEMLOCK` in GiB via the
    `resource` module, Linux only (non-Linux: `None` — skip; the plan's "non-Linux: skip"). `RLIM_INFINITY`
    (unlimited) maps to `inf`; an unanswerable limit (ImportError/OSError/ValueError) maps to `None` — unknown,
    which never warns.
  - `memlock_tier_warn(planned_gib)` (saguaro.py:7865): when a RAM tier is planned and the soft limit is a number
    under it, one `warn(...)`: the limit (GiB, or KiB as `ulimit -l` shows it when under 1 GiB — a 64 KiB limit is
    named "64 KiB", not "0 GiB"), the planned tier in GB, and the fix the plan asked for — systemd:
    `LimitMEMLOCK=infinity` in the unit's `[Service]` section, `systemctl daemon-reload`, restart the service;
    login shells: a `"<user> - memlock unlimited"` line in `/etc/security/limits.conf`, log out and in.
    No tier planned (`None`), unknown limit, or limit at/over the tier: nothing is said.
  - Two calls, both with `lay["resident_budget_gib"]` (so a page-cache layout — no tier — says nothing):
    - `set_up_qfn` step 2 "your choices" — the fit step, where `QF.plan` lays the tier out (saguaro.py:8088);
    - `main`'s `--check` qfn branch, beside its `qfn_check_line` (saguaro.py:9546).
  - Nothing is pinned differently; engines/, serve/ and the configs' args are untouched.
- `b726e70da` — step 2, tools/test_setup_qfn.py: new `Memlock` class (after `CheckLine`) and a docstring line:
  - limits 64 KiB, 8 GiB (the field case), unlimited (`inf`), unknown (`None`), at the tier, over the tier,
    and no tier planned — warn/no-warn as the plan specifies;
  - the wording checked once: the limit as ``ulimit -l`` names it, the planned tier, `LimitMEMLOCK=infinity`,
    `systemctl daemon-reload`, `"<user> - memlock unlimited"` + `/etc/security/limits.conf, then log out and in`;
  - the reader itself: `memlock_soft_gib()` equals the process's own soft limit (GiB, unlimited as `inf`) on
    Linux; skipped elsewhere;
  - the two call sites pinned: `inspect.getsource` shows the call in `set_up_qfn` and `main`, and a full
    `SetupFlow` run with the reader patched to 8 GiB prints the line (and prints nothing when unlimited).

## Files

- `saguaro.py` (+39: the two functions, the two calls)
- `tools/test_setup_qfn.py` (+70/−1: the `Memlock` class, docstring)
- `REPORT-p3-memlock.md` (this file)

## Tests run (only the modules the plan names, one at a time)

- `python -m unittest tools.test_setup_qfn` — before any change: 82 tests, OK (1 skip); after step 1: 82, OK;
  after step 2: **89 tests, OK (skipped=1)** (the 1 skip is the pre-existing ref/-not-in-checkout skip).
- `python -m unittest tools.test_setup_amd` — before: 34, OK; after both steps: **34, OK**.
- (`/opt/llm/saguaro/.venv/bin/python` used throughout; `py_compile` on saguaro.py as a syntax check.)

## Open questions

- GLM / DS4: the plan hooks the warning in "where the plan knows its size". Those engines size their pinned tier
  at run time (`SAGUARO_GLM_RAM_HEADROOM_GB` headroom of free RAM / the DS4 CPU lane); saguaro.py's setup plans
  never compute a GB number for it, so nothing was hooked there — the two calls are QFN's tier. If a GLM/DS4
  number is wanted, the plan's size would have to be derived (free RAM minus headroom), which felt outside
  "keep saguaro.py edits to this function and its two calls".
- The warning fires from the setup and --check only (as planned); a start of an already-written config does not
  re-check the limit.
- `inspect.getsource` is used to pin the `--check` call (main is not runnable in a unit test); it fails loudly if
  the call is ever removed, but it is a text-level guard, not an executed path.
- On a machine whose soft limit is unlimited (typical bare-metal), the feature is invisible; this worktree's box
  has a 8 MiB soft limit, so the flow tests exercise the warning through the patched reader, not the real one.
