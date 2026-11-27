# FIX-REPORT — P124x mortar memory (review REVIEW-RESULT.md, base f4b4dac08)

One commit per rule and per fix; each finding below points at what changed (or why nothing did), with
the commits in order: rule (a) 624f95f58, rule (b) 1c190a03a, rule (c) c54ba610d, rule (d) 652b1cd63,
F3 26feff6db, F4 9ff4ff922, step-3 tests 3b4cae905, card script 08bebe8d6, REPORT d46851693. Tests after
every step, one module at a time: park_policy_test 56, conversation_memory_test 45, resident_policy_test
13, vram_policy_test 12 (g++ 14.2 -std=c++20, scratch/p124x-build), serve test_fatal_recovery 11
(scratch/run_p124x_server_tests.py), CUDA sm_75 build of the engine (scratch/build-p124x-sm75, nvcc 13.1:
strata/saguaro-qfn links whole - re-verified "no work to do" after the last C++ change). No GPU was run.

## High

**F1 — fixes uncommitted, step 3 absent. Fixed.** The whole uncommitted tree is now split into the commits
above (each rule one commit, `git add` of the named files only). Step 3 exists: the swap rule is tested in
`park_policy_test` (swap_high semantics, credit-off pressure_need at the review's own 8-GiB/78-GiB/2-GiB
figures -> 10 GiB, dust/at-bar/off arms) and in `conversation_memory_test` (SwapTotal/SwapFree parsing);
budget_cap and hip_auto_reserve_mib have new header-only modules `resident_policy_test` /
`vram_policy_test` registered in CMakeLists beside `host_limits_test`; `fatal_engine_line` has a
classification table plus two end-to-end protocol cases in `serve/test_fatal_recovery.py`; the card-test
script is committed at `tools/card_tests/mortar_pressure.sh` (3 x 100K-250K conversations for 30 min, RAM /
VRAM sampled every 10 s, PASS = no abort / VRAM never under the floor / no swap growth); the sm_75 CUDA
build was done and is recorded in REPORT section 3 together with the HIP note (no ROCm toolkit in this
session; the card run is the HIP proof). One process note: `serve/test_fatal_recovery.py` could not import
on the base either (engines/qfn/serve/server.py's `from serve import turbo_live` — the file lives in the
root serve package since 936183783), so the module is run through a 20-line scratch harness that registers
that module under the imported name; pre-existing, noted, not "fixed" inside this branch's scope.

**F2 — the -9 story. Fixed (REPORT d46851693).** Section 1.1 now reads the log as it is: both -9s follow
`... did not finish within 5 s (#267)` lines (log:119, log:443) and the engine dies 17 ms / 4.8 s behind
the server echoing them — the pump's fatal-line `proc.kill()`, not the OOM killer; the table rows say so.
The section states plainly that the log shows no timed-out-with-drained event, that the eleven "GPU
finished in 0-9 ms" lines are the watchdog's own pre-kill releases, and that (d) is justified by the drain
evidence plus the code shape, not by a logged killed-while-drained end. Both -9s stay fatal under the new
rule, and the new `test_fatal_recovery` cases pin exactly that (the not-drained note kills; the drained
one does not).

**F3 — re-armed vs still-poisoned. Fixed (26feff6db).** `release_gpu_waits` gained the `re_armed` out
(set true only on drained-and-re-armed); `timed_out_note()` words the three shapes: the bare `released and
the GPU finished (#267)` — the string `RECOVERED_267` matches — is now printed only when the verifier
really took the arm back, and a split/hand-off verifier that stays poisoned prints `... and the GPU
finished, but the verifier cannot take another window (#267)`, which the server treats as fatal (table +
feed() case in the test module). The single-GPU re-arm itself is unchanged from what the review verified
good.

## Medium

**F4 — version coupling. Fixed (9ff4ff922).** The server comment now says the recovered-note rule holds
because qfn_engine.py is extracted from that very file and ships with the engine binary it drives, and
that pairing an older engine with this server buys one extra failed request — the "safer" claim is gone.
REPORT section 4 lists the open alternative (a READY behaviour flag) rather than pretending the comment
enforces anything.

**F5 — the 16 MiB swap bar. Fixed (c54ba610d + REPORT 2c).** The default is **256 MiB**, and the reason is
written where the review asked (REPORT 2c): every mortar abort line carried 392 MiB-5.9 GiB of swap (so
256 MiB still fires on every storm in the log, including the smallest, #8's 392 MiB), while a calm box's
residual swap sits in the tens of MiB — the review's own latch scenario (17 MiB of residual swap emptying
a resident box forever) is now calm, and the test pins both sides. Why still an in-use bar rather than a
growth bar: swap does not drain by itself, so "in use above the bar" means exactly "working pages are on
disk right now", which is the truth the credit was betting against; the plateau case (storm over, pages
never touched again) is left to the raw floor and PSI, and REPORT 4 records the growth bar as the
alternative if the card run disagrees. The interaction with rule (a) is documented: a capped budget keeps
raw MemAvailable above the free floor, so the swap term is the backstop (operator override, foreign RAM
pressure, booted-into-pressure), not the everyday path.

**F6 — the half-written report. Fixed (d46851693).** Sections 2a-2d, 3 and 4 are written (headroom figures
now in prose, not only in code comments), and the dangling "see the commits between this file's first
commit and the branch tip" cross-reference is replaced by the actual commit list.

## Low / nits

**N1 — "eleven ... 416-432 MiB". Fixed (d46851693):** "ten of the eleven", with #6's 528 MiB / log:361
named as the eleventh. The rule (b) code comment says "ten of its eleven aborts" too.

**N2 — clamp opt-out and the cap-0 silence. Fixed (624f95f58):** `SAGUARO_RESIDENT_BUDGET_CAP=0` honours
the operator's `--resident-budget-gib` as asked (the message even names it), and the cap-0 corner is loud:
the engine prints the pool, margin and asked figures and KEEPS the asked budget — silently clamping to
zero would flip the box out of the resident mode its setup named; "said, never silent" is satisfied by the
saying, not by an unasked mode change. resident_policy_test carries the cap-0 and one-byte-over cases.

**N3 — APU. Fixed (1c190a03a):** `hip_auto_reserve_mib` takes `integrated` and returns the given reserve
for an integrated GPU; generate.cpp reads it from `cudaDeviceProp.integrated` (the same probe the io-
prefetch warning and the UMA block use). vram_policy_test pins the arm; the UMA sizing is untouched.

**N4 — style/efficiency. Partly fixed:** the stray space before `if (!o.vram_reserve_given` is gone (the
line is byte-identical to base again); the `park_swap_mib=` token's external-parser note is in the code
comment and REPORT 4. The double /proc/meminfo read was NOT merged: `sample_host_memory()` already opens
the file twice per tick at base (MemAvailable and MemTotal separately), so the park tick adds a third
1.5-KiB procfs read per second — collapsing all three means changing the shared memory-guard function,
which the plan's named test modules do not cover (memory_guard_test is not one of them), for a cost of
three procfs reads a second. Left, with the reason here rather than an unverifiable refactor in.

## Verdict items ("To fix before merge")

1. done: rules a-d are separate commits (624f95f58, 1c190a03a, c54ba610d, 652b1cd63), REPORT filled and
   the -9 story corrected (d46851693).
2. done: F3 note words (26feff6db) and the coupling pinned (9ff4ff922), qfn_engine.py only ever by
   `tools/saguaro/extract_qfn_engine.py` (re-extracted in both commits; py_compile clean).
3. done: the step-3 tests above, run one at a time; card script committed; sm_75 build done and recorded;
   HIP noted (3b4cae905, 08bebe8d6, REPORT 3).
4. done: the 256 MiB threshold's reason and the rule-(a) interaction in REPORT 2c/4 (c54ba610d,
   d46851693).
