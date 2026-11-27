# FIX-REPORT - P124x re-review (REVIEW-RESULT.md, base f4b4dac08, prior tip 24c312a2c)

Every finding below, in the review's order, with what changed (commit) or why not. The previous
round's report (F1-F6/N1-N4) is the file as committed at 24c312a2c; this one covers the re-review.

**F-A - rule (d)'s "ends the request, not the engine" claim is hollow (the process exits anyway).**
Corrected the claims to what the engine does (51ece4173); the engine half is carried as the open
item, not shipped blind. Why the doc route: every path that prints a timed-out ERR leaves the
request loop through `return 1` (solo prompt read generate.cpp:21616, solo decode :22665, batch step
:18768, batch_fatal read :21073), and making them "fail the request, return to stdin" is real state
unwinding (lent slots, half-read pieces, and in the batch shape other slots mid-window) in a file
that cannot be GPU-tested here - shipping it unwound and wrong buys a worse abort than the reload
it avoids. What is true now, everywhere the review named:
- REPORT 2d: states the exit (with the four `return 1` sites), that the re-armed verifier never
  gets its next window because the process is gone, and what the rule still delivers - the request
  ends on the engine's own line, not the pump's SIGKILL of a self-ending process - plus a REPORT 4
  open item naming the gated paths for the engine half.
- `engines/qfn/serve/server.py` RECOVERED_267 comment (and `serve/qfn_engine.py` re-extracted, not
  hand-edited): "the shape the 9070 XT box needed" replaced with the same truth; the rule stays
  because it is the protocol's half and it stops the SIGKILL.
- `serve/test_fatal_recovery.py`: header, the fake's recovered-mode comment and the drained-case
  comment now say the fake speaks the protocol the SERVER implements, the real engine exits after
  the line today, and the tests pin the server half (a recovered line never invalidates the
  engine). The assertions themselves stand as protocol tests.
- `tools/card_tests/mortar_pressure.sh` header: a drained #267 fails the run like every #267 (its
  exit prints "the engine had stopped", the grep at :235/:245 catches it); the line is still echoed
  at the verdict so runs can tell the shapes apart.

**F-B - the card test can never run (`awk 'NR==2 {print $3}'` on a comma CSV).** Fixed (d00409720):
`awk -F,` at mortar_pressure.sh:226. Reproduced the review's empty field first: the sampler writes
through `csv.writer` (comma-separated, CRLF line ends), and the default whitespace FS yields nothing.
The `\r` re-check: the CR rides the LAST field (vram_free_mib), not the baseline one, but the read
strips it from `$3` anyway; verified end to end against a real `csv.writer` row (baseline non-empty,
parses as the int the verdict takes). `bash -n` stays clean.

**M1 - rule (b) has no off switch inside auto, and the "every rule has a knob" claim overstates.**
Both halves fixed (29fed29e2): `SAGUARO_VRAM_RESERVE_FLOOR=0` skips the floor and keeps the reserve
as given/default (parsed strictly - only exactly "0"; the probe and the lastError clear are skipped
with it), so an operator can stay on `--expert-cache auto` and keep a measured reserve, which the
"leave auto" opt-out did not allow; the floor message names the knob. The claim at REPORT 2a and the
(a) comment is gone, replaced by the true list: (a) SAGUARO_RESIDENT_BUDGET_CAP, (b)
SAGUARO_VRAM_RESERVE_FLOOR, (c) SAGUARO_PARK_SWAP_MIB, (d) no operator knob - its switch is the
recovered note itself. (The park knob's env_num/atof family has the same "typo reads as 0" property
for SAGUARO_PARK_SWAP_MIB; pre-existing P124a pattern, left as its siblings are, noted here.)

**L1 - the (b) floor can be silently undone later by #496.** Fixed with the note, not the exemption
(bfa290a42): the floor message now says a card too full for any working cache at it may still see
the #496 line lower it, and the comment + REPORT 2b state what the floor really promises (what the
CACHE is sized against). #496's own line already names the reserve it takes, so the two together
read honestly. The 15 GiB+ exemption was declined: on exactly such an over-full card it leaves
cache < min_slots and the prompt path allocating its own buffers over the reserve - the page-out
#496 exists to avoid - untestable here without a GPU.

**L2 - SAGUARO_RESIDENT_BUDGET_CAP parsed with atoi.** Fixed (2baa33867): only exactly "0" turns
the cap off; "off", "no" and typos now leave the protection on. Matches the repo's strict pattern
(`SAGUARO_OWNED_PRICE=exact`).

**L3 - carried nits.** Fixed (a0d78c8db, plus a whitespace follow-up 2848a7622):
- INFO `park_pool`: `park_swap_mib=` prints only while the swap rule is on - with
  SAGUARO_PARK_SWAP_MIB=0 the line is exactly the pre-P124x bytes (report 2c/4 follow; no parser of
  the line exists in the repo, re-checked).
- the (b) probe block is compiled Linux/HIP-only (`#if defined(STRATA_USE_HIP) && !defined(_WIN32)`,
  the UMA block's guard), so CUDA/auto no longer runs cudaMemGetInfo + cudaGetDeviceProperties + a
  cudaGetLastError clear for a floor that never raises there; the guarded block (no ROCm here to
  compile it) was syntax-checked with stubs under `g++ -std=c++20 -Wall -Wextra`, and
  vram_policy_test still drives the header's CUDA/Windows arms directly (12 OK).
- the serve-test runner is committed as `tools/run_qfn_serve_tests.py` (the scratch file it replaces
  is gone); `serve/test_fatal_recovery.py` - the plan's named module - runs from a clean checkout:
  11 OK. REPORT 3 points at the committed path.
- the park tick's second /proc/meminfo read stays declined as in the first FIX-REPORT (N4): the
  reader is the shared host-memory guard, outside the named modules' reach; two reads a second are
  what the OFF-switched swap rule already avoids (no swap read at bar 0).

## Verified after the fixes

Tests, one at a time, rebuilt from current source: `park_policy_test` 56,
`conversation_memory_test` 45, `resident_policy_test` 13, `vram_policy_test` 12 OK (g++ 14.2,
-std=c++20); `serve/test_fatal_recovery.py` 11 OK via the committed runner. No GPU run.
`cmake --build scratch/build-p124x-sm75 --target strata` compiles and links generate.cpp clean
(CUDA sm_75; the one warning at :15768 is the pre-existing batch-telemetry one).
`tools/saguaro/extract_qfn_engine.py` re-run: `serve/qfn_engine.py` byte-identical. `py_compile`
clean on server.py, qfn_engine.py and the test module; `bash -n` clean on the card script.
