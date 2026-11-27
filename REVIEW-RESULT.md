# REVIEW-RESULT - P124x re-review (base f4b4dac08, tip 29aa5310b)

Verdict: **MERGE-WITH-NITS**

The four rules match their code and the REPORT's claims match the branch: the log figures quote
out (verified below), the OFF switches are byte-identical paths, `serve/qfn_engine.py` is the
extractor's own output, and every named test module passes from current source. Nothing found is a
correctness bug; the nits are doc/unit-consistency items that can ride along or land as a follow-up.

## What was checked, with what it showed

- **Tests, one at a time, rebuilt fresh from this checkout (g++ 14.2 -std=c++20, CUDA-free):**
  `park_policy_test` 56 OK, `conversation_memory_test` 45 OK (with `conversation_memory.cpp`),
  `resident_policy_test` 13 OK, `vram_policy_test` 12 OK; `serve/test_fatal_recovery.py` 11 OK via
  the committed `tools/run_qfn_serve_tests.py`; `bash -n tools/card_tests/mortar_pressure.sh` clean;
  `py_compile` clean on server.py / qfn_engine.py / the test module. The sm_75 build directory is
  real and `ninja -n scratch/build-p124x-sm75 strata` reports **no work to do** at this tip - the
  CUDA-build claim holds for the committed sources (no GPU run here, as the report says).
- **Generated file:** re-ran `tools/saguaro/extract_qfn_engine.py` -> `serve/qfn_engine.py`
  byte-identical (diff empty, `git status` clean for the path). The rule's comment and
  `fatal_engine_line` reached it through server.py:899-927 only.
- **OFF paths byte-identical:** (a) `SAGUARO_RESIDENT_BUDGET_CAP=0` skips the whole block
  (generate.cpp:6431-6432); (b) `SAGUARO_VRAM_RESERVE_FLOOR=0` skips the probe, the note and the
  lastError clear inside the Linux/HIP-only `#if` (generate.cpp:6888-6915) - and the block is
  compiled out on CUDA and Windows entirely; (c) `SAGUARO_PARK_SWAP_MIB=0` -> `swap_used` never read
  (generate.cpp:18416), `swap_high` false (park_policy.hpp:105-107), the INFO line prints the exact
  pre-P124x bytes (empty `%s`, generate.cpp:10608-10613), the release notes print `sw=""`
  (generate.cpp:18427-18431), and the pressure test asserts 0 is "exactly the P124a behaviour".
- **Locks/threads/stdout:** no new lock anywhere; the swap read joins the existing once-a-second
  park tick on the serve thread (generate.cpp:18386-18416, guarded by `park_busy`); all new engine
  output is single `fprintf(stderr)`/`printf` calls on paths that already printed there; the INFO
  line keeps its one-printf+fflush shape.
- **Re-arm shape (verify.cpp):** verified the claim against the code - `stage_inputs`/`stage_batch`
  reset `h_seq_`, `h_flag_`, `h_flagA_`, `h_flagB_` per window (verify.cpp:2815-2818, 3998-4001), so
  a drained, non-split re-arm starts clean; split (`next_`) and incoming-hand-off (`hflag_in_h_`)
  verifiers re-poison (verify.cpp:558-560), the not-drained path poisons (verify.cpp:543), the
  watchdog/DMA-failure callers keep poison=true (verify.cpp:467, 3027, 3059, 3406, 4570), and
  `released_note` words the three shapes so `RECOVERED_267` (server.py:919) cannot match the
  still-closed note ("finished, but..." is not "finished (#267)").
- **Log figures (§1.1-1.3):** 13 engine ends, 11x-6 + 2x-9 at the table's times; the two -9s are the
  server's own fatal-line kills 17 ms (log:119 -> 125) and 4.8 s (log:443 -> 445) behind the
  not-drained #267 lines; 10 LOW readings 416-432 MiB + one 528 (log:361, no LOW); 11 watchdog
  releases at 0-9 ms; window captures 500 -> 446 (log:345/352); "resident RAM: 68.00 GiB", 892 MiB /
  45,845 tokens, "snapshot 2618 MiB exceeds", "capture took the serve thread 30881 ms",
  "41453.9 MB ... not 66487.8 MB", "157887.6 MB ... 2967 major faults", RSS 84,840-89,437 MiB - all
  in ref/mortar-aborts.log as quoted.
- **Manager items:** (1) figures quoted from the logs - yes, and the ones spot-checked land.
  (2) headroom rules with reasons - yes: cap = total - 2x pool - max(8 GiB, total/8) with the
  mortar arithmetic (§2a), 512 + 512 floor with the 272 MiB measured spend and the #1376 sibling
  story (§2b), the 256 MiB bar between tens-of-MiB dust and the 392 MiB-5.9 GiB storms (§2c).
  (3) pressure give-back before swap - partly by design: the swap bar fires *after* swap crosses
  256 MiB; the before-swap defence is rule (a) keeping raw available above the raw floor plus PSI,
  and the in-use-vs-growth bar choice is an open question (§4) - see finding N4.
  (4) #267 no abort - honestly partial: the recovered note stops the SIGKILL, but every timed-out
  ERR path still `return 1`s (verified at generate.cpp:18782 batch step, :21087 batch_fatal read,
  :21630-21640 solo read ERR-then-return, decode :22651-22681) so the process exits and reloads -
  the branch says exactly this in REPORT 2d/4 and FIX-REPORT F-A. No hidden overclaim found.

## Findings

**N1 (nit) - SAGUARO_PARK_SWAP_MIB turns itself OFF on any non-numeric value.**
`env_num` (generate.cpp:10569-10572) runs `atof`, so `SAGUARO_PARK_SWAP_MIB=off` / `no` / a typo
reads as 0 and silently disables the release rule - the exact failure mode the branch fixed for
`SAGUARO_RESIDENT_BUDGET_CAP` (L2, generate.cpp:6424-6428 parses only exact "0"). It follows the
existing `SAGUARO_PARK_*` family convention and FIX-REPORT.md:43-44 notes the choice, so this is
consistency debt, not a regression - but the knob's typo direction fails open while its siblings
(a)/(b) fail closed.

**N2 (nit) - the new `Memory::swap_used` field makes park_policy_test warn under -Wall -Wextra.**
Seven `-Wmissing-field-initializers` on the pre-existing `Memory{avail, psi}` inits
(park_policy_test.cpp:33,59,62,68,83,137,145). Behaviour is right (value-init = `nullopt` = "no
swap reading", as the calm tests need) and the module passes, but the file used to be warning-clean
and the smoke test's build flags may trip over it.

**N3 (nit, doc) - two report figures wobble; none change a conclusion.**
§1.2's "~20.4 KiB/token" is ~19.9 (892 MiB / 45,845 tokens; the pool is then ~9.96 GiB, not ~10.2 -
the 63.6-vs-68 cap conclusion is unchanged); "RSS at the aborts 84.8-89.4 GiB" is the log's MiB
figures /1000 (84,840-89,437 MiB = 82.9-87.3 GiB). Likewise REPORT 4 / FIX-REPORT F-A's line cites
for the four `return 1` sites (21616/22665/18768/21073) predate the branch's own line additions -
this tip has them at ~21640 / ~22681 / 18782 / ~21087, and FIX-REPORT:26's "grep at :235/:245" is
:239/:249 in the committed script.

**N4 (observation) - rule (c) answers the storm, not the pre-swap moment, by construction.**
The plan asked for give-back *before* the system swaps; the bar (park_policy.hpp:41, swap in use >
256 MiB) fires once swap exists. Before that it is rule (a)'s cap (raw available stays above the raw
16 GiB floor, so the un-credited floor acts first) plus PSI avg10. A slow swap creep that stays
under the bar while the credit keeps `available` reading calm would stay silent - the in-use-vs-
growth-bar trade-off is exactly REPORT §4's open question 1 and the card script's swap-growth check
(SWAP_GROW_MIB, mortar_pressure.sh:270) is the tripwire for it. Deliberate, documented, env one
line from the other choice - accepted as shipped.

**N5 (observation) - the re-armed verifier is end-to-end unexercised anywhere, today.**
Because every timed-out ERR still exits the process (F-A), no run - unit, server test, or the card
script - ever lets a re-armed verifier take its next window. The state claims that matter when the
engine half lands (the per-window word resets, and the non-word window state behind `staged_` /
`commit_live_` / batch `b_running_`) are verified here only by reading (verify.cpp:2815-2821,
3997-4001, 554-563). REPORT 4 names the gated paths and declines to ship the unwinding untested -
right call on a no-GPU box - but the landing of that half needs its own review with GPU evidence;
the recovered note should not be mistaken today for process survival (the script, server comment
and tests all now say this loudly - checked).

## Not findings (checked and clean)

No nested lock acquire anywhere new; no secrets, no api_key (the card script drops it,
mortar_pressure.sh:94) and no paths outside the repo (script output under `--out`, config read-only);
`fatal_engine_line` keeps every old kill for non-timed-out lines and for note-less lines
(server.py:922-927, table-tested at test_fatal_recovery.py:122-133); the server/engine pairing
coupling is stated where it lives (server.py:903-918, REPORT 2d/F4); #496's lowering can still
undo the (b) floor on an over-full card and says so in both lines (L1, generate.cpp:6904-6911 +
6983-6995); integrated APUs and the UMA block do not double-fire (the floor skips integrated,
generate.cpp:6899-6902); the resident clamp uses the same pool formula as the park pool
(generate.cpp:6433-6435 vs 10584-10585) and errs toward cutting more, never less (REPORT 4);
`conversation_swap_used` fails closed to `nullopt` on malformed/unreadable input
(conversation_memory.cpp:255-285, tested at conversation_memory_test.cpp:65-87).
