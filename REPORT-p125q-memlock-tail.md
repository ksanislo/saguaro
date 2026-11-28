# P125q - report: the elastic expert tail can be locked

With the elastic K/V host copy the QFN expert complement registers only its hottest share and keeps a cold RAM tail;
on mortar the tail's mlock failed - "FileExpertSource: mapped pinned cache complement ready: resident 68.00 GiB,
pinned 62.20 GiB ...; mlock failed (raise ulimit -l)" - because both start paths (the documented systemd unit and
saguaro.sh's own start) run with the account default memlock limit, so the tail stayed pageable and the kernel could
swap cold experts. Fixed along the plan's three lines.

## What changed

1. **The systemd unit (docs) - commit 447a108fd.** `docs/SETUP.md`, "Running Saguaro as a systemd service": the
   `[Service]` block gains `LimitMEMLOCK=infinity`, and a new **Memory-lock limit.** bullet says why (the engine
   page-locks its expert RAM tier, `ulimit -l` defaults of 64 KiB / 8 MiB cover far less, the tail then stays
   pageable and the engine says `mlock failed (raise ulimit -l)`). The section is the only systemd unit in the
   repo - `setup.py`/`saguaro.py` never write a unit file themselves (searched: no `ExecStart` anywhere else), so
   the docs' unit *is* "setup's" unit. `docs/SETTINGS.md` already carries a Locked-memory-limit note for the DS4
   engram path (pre-existing, untouched).

2. **saguaro.sh - commit c31027c7f.** A `raise_memlock()` function, called at start right after the `cd`:
   Linux only (ulimit -l's unit differs elsewhere, and the limit is a Linux thing). It reads `ulimit -Sl`/`-Hl`,
   raises the soft limit to the hard one when they differ (soft "unlimited" is unreachable, and equal means
   nothing to do). When even the *hard* limit is a finite number under 8 GiB - 8388608 KiB, the smallest pinned
   tier qfn_fit can plan (`max(8, ram_gb - RAM_SPARE_GB)`) - it prints once (three wrapped lines, one message)
   naming the hard limit in KiB and the one-line fix: a `"<user> - memlock unlimited"` line in
   `/etc/security/limits.conf`, then log out and in (and the systemd form, `LimitMEMLOCK=infinity`). Unlimited or
   unreadable limits say nothing. By hand on this box: soft 64 raised to the 8192 KiB hard; soft==hard=8192 warns
   only. (A tier larger than 8 GiB beside a between-8-GiB-and-tier hard limit is still caught at the plan level by
   the existing `saguaro.py` `memlock_tier_warn` (P3), which knows the planned tier - untouched.)

3. **The engine's warning - commit bbf211719.** `engines/qfn/src/platform/memory.cpp` (non-Windows branch, the
   `strata_core` the FileExpertSource line prints): the bare `mlock failed (raise ulimit -l)` note becomes
   `mlock failed (raise ulimit -l): the limit it hit is <soft RLIMIT_MEMLOCK as ulimit -l shows it: 64 KiB /
   8.00 GiB / unlimited / unknown>, all <N GiB> stay pageable` for `lock_resident`, and `... <X GiB> of <Y GiB>
   stay pageable` (X = requested minus the bytes locked so far) for `lock_resident_ranges`. The note flows into the
   existing `FileExpertSource: ... complement ready: ... ; <note>` print unchanged otherwise. `serve/qfn_engine.py`
   and `engines/qfn/serve/server.py` untouched - nothing Python-side changed, so
   `tools/saguaro/extract_qfn_engine.py` had nothing to run. The sibling `src/platform/memory.cpp` (maya_core, the
   GLM/DS4 engines) was left alone: the plan's warning is the QFN FileExpertSource line.

## Files

- `docs/SETUP.md` (unit line + bullet)
- `saguaro.sh` (`raise_memlock`, called at start)
- `engines/qfn/src/platform/memory.cpp` (failure notes; `<sys/resource.h>` added)
- `tools/test_setup_memlock.py` (new test module)

## Tests

- `python -m unittest tools.test_setup_memlock -v` (the only test module run, as the plan names it): **9 tests,
  OK** - `UnitText` (the `[Service]` line, the section's why + the failure string named, unit AND bullet carry the
  line) and `SaguaroShRaises` (the extracted `raise_memlock` driven through a scripted `sh` with fake `uname`/
  `ulimit` builtins, every `ulimit -Sl VALUE` call logged): non-Linux touches nothing; soft 64 under an unlimited
  hard -> `set unlimited`, silent; 8 MiB hard -> raised to it, warned once with the KiB limit and the fix lines;
  hard at 8388608 KiB and over -> raised, silent; 8388607 -> warns; unlimited/unlimited -> nothing; the script
  calls `raise_memlock` at start.
- `g++ -fsyntax-only -std=c++17` on the changed `memory.cpp`: OK (no build, no GPU runs, per the plan).
- saguaro.sh's raise path by hand (scratch/rm.sh, sourced in `sh`): soft 64 -> 8192 raised; 8192 hard warns once.

## Open questions

- The 8 GiB "hard limit too low" floor in saguaro.sh is qfn_fit's smallest tier; it says nothing for a box whose
  hard limit sits between 8 GiB and the *actual* planned tier - the P3 `memlock_tier_warn` covers that case once
  the plan is known (limit < tier), so the two warnings divide the space. If the plan wants saguaro.sh itself to
  read the config's tier, that needs the config parsed in shell - not done.
- The root `src/platform/memory.cpp` (maya_core) keeps the bare note; if mortar-class tails appear on the GLM/DS4
  engines too, mirror the step-3 note there.
- No `CHANGELOG.md` entry: the release notes are written at release time; say the word and one goes in under 1.2.5.
