# REVIEW-RESULT - P125o (Inkling 2 elastic VRAM, step 2: the K / V grows on demand) - re-review after fixes

Verdict: **MERGE-WITH-NITS**

Range `git diff d210adc76e47b9883484a6b20b32372ff98a0ce1..HEAD` against PLAN-p125o-ik2-elastic-2.md and
REPORT-p125o-ik2-elastic-2.md. The P125e commits through 37c7523ca carry their earlier review (MERGE-WITH-NITS);
this review re-checks the a2 / a3 step plus its six fix commits (e0efdcb7d .. 3a77926da) over the whole range. The
prior FIX-FIRST verdict's six actionable findings are all fixed and verified below; what remains is two NITs and
three INFOs, none blocking.

## The prior round's findings - fixed and verified

1. **CRITICAL (write epilog wrapped `% kv_cap` over the commit) - fixed (e0efdcb7d).** The full-visit prompt path
   passes `write_kv_cap = (g_kv_vmm && !Ly.sliding) ? 0 : Ly.kv_cap` (src/ik2/ik2_main.cu:2128), the token path
   `wcap = g_kv_vmm ? ring : Ly.kv_cap` (:3448) and the block / graph windows the same (:4566); the kernels'
   epilog `slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t)` (src/ik2/ik2_kernels.cu:465, :471) then writes by
   absolute position exactly where the attention reads it (which already passed 0). Sliding layers keep their
   ring; every plain / OFF layout still receives `Ly.kv_cap`, where `pos < kv_cap` made the wrap the identity -
   byte-identical. The untouched `--parallel` / `--tp` prep sites (ik2_main.cu:5113, :5125 in `par_launches`,
   :5929, :5940 in `tp_token_launches`) are safe: the elastic gate excludes those modes (src/ik2/ik2_plan.hpp:38-42,
   passed at ik2_main.cu:8127-8128), so `g_kv_vmm` is false there and `Ly.kv_cap` is never a commit. The card test
   now judges the growth read by its answer ("alpha", tools/card_tests/ik2_kv_start.sh:226-229), not its token
   count, so a regression of exactly this bug can no longer hide.
2. **HIGH (drafting-auto trial wrote before committing) - fixed (5170e0505).** `trial()` commits
   `min(ids.size() + gen_t, ctx)` before its first `forward` (ik2_main.cu:10319-10327), so the prompt pass never
   runs ahead of a commit; VRAM short it stops the measurement with one NOTE and the unfinished request counts no
   tokens and no seconds (`if (out.empty()) break;` at :10353 kills the negative token count); `decode_greedy`
   clamps its own commit to `ctx` (:7193), so an oversized `SAGUARO_IK2_DRAFT_TOKENS` cannot push `g_kv_chunks`
   past `g_kv_n_chunks`. `decode_greedy` has no other caller (only :10350), so its failure `return` cannot strand
   a serve request.
3. **MEDIUM (unpark failed every request with a false OOM) - fixed (41072c757).** `full_rows` and `unpark` now
   return the kind of result (0 done, -1 the pre-existing folder-read / missing-snapshot failures, 1 the grow
   failure; ik2_main.cu:10104-10140, :10217-10245); only the 1 fails the request with the out-of-memory ERR + DONE
   (:10997-11009), and the -1 falls through to the from-scratch re-read the result was ignored for at
   37c7523ca:10802 - the OFF path is byte-identical to P125e again, and a non-VMM card (where VRAM cannot be the
   cause) never sees the false message.
4. **LOW (betting on zeroed fresh pages) - fixed (593527e06).** `kv_grow` memsets the freshly committed span
   `[old, min(bytes, ceiling's chunks))` on `g_s` at the commit (ik2_main.cu:581-595) - one memset a chunk,
   host-side, stream-ordered before the request's kernels (`g_s == g_dev[0].s`, :1846; single GPU is guaranteed by
   the elastic gate, so the `cudaSetDevice` belt at :590-594 is a no-op and no cross-device stream use is
   reachable). The old claim is replaced where it lived (src/ik2/ik2_vmm.hpp:7-9). The load-time DEPTH memset
   (ik2_main.cu:8671-8690, sized by the commit) is unchanged and inside the mapped prefix.
5. **NIT (drafter drops on the OFF path) - fixed (deadadfc4).** All three comparisons gate on `g_kv_vmm`:
   decode_greedy's hook (:7194), the serve `spec_rows` factor (:11038-11039), the CLI's drop and its line
   (:11435-11438).
6. **NIT (`kv cache:` line counts the layout) - fixed (3a77926da).** The comment at ik2_main.cu:8715-8719 says the
   line prints the laid-out rows while the commit rounds up to whole chunks; the printed figure stays (it is what
   the plan's 32768-vs-262144 load-memory comparison compares).

## Manager checks

- **Commit / grow never on the per-token path**: `kv_grow` (ik2_main.cu:566-600) is called at the serve request's
  start (:10927, after the guard, before `busy = true` at :10949 and before any kernel), at kept-conversation
  restore (:10224, once per unpark), at each trial request (:10322), at the CLI's load (:11428) and at the window
  warm-up (:11545), and at `decode_greedy`'s entry (:7193, once per call). None is inside a token loop (the decode
  loops at :7200+ and the serve generation loop hold no `kv_grow`), none in a kernel; drafts stay inside the
  start-time commit (the room clamp keeps `pos < ids + max_new`, already committed).
- **HIP equivalents for every VMM call**: the CUDA entry points (ik2_vmm.hpp:32-41, loaded at :64-68) map to the
  HIP branch one for one - `hipDeviceGetAttribute` / `hipMemGetAllocationGranularity` (:219-227),
  `hipMemAddressReserve` (:246), `hipMemCreate` / `hipMemMap` / `hipMemRelease` (:270-276),
  `hipMemSetAccess` (:284), `hipMemUnmap` (:289, :300), `hipMemAddressFree` (:307). `hipDeviceGet` is legitimately
  absent (HIP's `location.id` is the ordinal, :215, :264). Where the API is missing, `hipDeviceAttribute...Supported`
  / `granule()` returns 0 (or the whole branch is the `#else` stub, :312-332) and the engine takes the a1 fallback
  with one stderr line (ik2_main.cu:8142-8165); the configure probe (CMakeLists.txt:1015-1050) gates
  `SAGUARO_IK2_HIP_VMM` so an old ROCm still builds. The single-device access grant on HIP (:262-265) is documented
  (:254-256) and safe under the elastic gate. Rollback matches on both paths (:178-185 vs :284-291).
- **A grow failure fails the request, never the engine**: `Range::commit` gives back exactly what the failed call
  mapped (:163-185, :270-291); `kv_grow` shrinks every buffer back to the old commit and leaves `g_kv_chunks` and
  the graphs untouched (:575-579, the base never moves); the serve handler answers NOTE + ERR + DONE and continues
  with no kernel run (:10927-10941, before `busy`), the unpark path resets `busy` / `g_wd_busy` before `continue`
  (:11006-11007); a trial shortage stops only the measurement (:10322-10327). The only exits a VMM call causes are
  at the load before READY (reserve / initial-commit `die`, :8653-8660) and the CLI run's load-time `die`
  (:11429-11430, :11546) - neither is a serving engine dying mid-request.

## What checked out (brief items)

- **Feature OFF byte-identical**: `g_kv_commit` defaults to `ctx` (:7560) and every changed print keeps its number
  when the step is not taken (READY / `@READY` :10424-10426 and `INFO ctx=` :10399 print `g_kv_run` = ctx; STAT
  `kv_cap=` prints `g_kv_commit` = the old `g_kv_run` :10492; `draft_room` / `draft::load` take `g_kv_commit`
  = ctx :8293, :9313, :9318); `kv_grow` no-ops without `g_kv_vmm` (:567); `write_kv_cap` / `wcap` degenerate to
  `Ly.kv_cap`; the fallback prints step 1's NOTE verbatim plus the promised stderr line (:8146-8165); `spec_rows`
  is always true without `g_kv_vmm`. The VMM allocation branch is an `else if` behind the plan's elastic gate and
  the streaming / swing / q4 branches precede it (q4 and streaming are themselves gate-excluded; `g_kv_swing` is
  only ever set where `g_kv_streaming` holds, :8109-8112, so no VMM visit can bypass the range).
- **Locks, threads, stdout**: no new mutex anywhere; every `kv_grow` / VMM call runs on the request thread or at
  the load (the only other thread, the stdin reader, touches no CUDA); `api()` is a magic static (ik2_vmm.hpp:61-73);
  the NOTE / ERR / DONE lines are single `printf`s with `fflush` (:10931-10940, :10997-11008), `say()` flushes
  (:9938-9941), so nothing interleaves a line.
- **Generated file**: `serve/qfn_engine.py`, `engines/qfn/serve/server.py` and `tools/saguaro/extract_qfn_engine.py`
  appear in no commit of the range - the engine-side `@END` field order rode in its own commit (7b46f80c2) with no
  generated-file edit.
- **Chunk arithmetic / rollback**: chunk bytes round *up* to the granule (ik2_plan.hpp:57-61, `align_up` at :53);
  `Range::commit` clamps to the reservation (:149, :258) and releases each create handle right after mapping
  (:170, :276); the memset's `zero_hi` is min'd with the ceiling's chunk count (:586). Callers keep `n <= ctx`
  (guards :10903, :11425; clamps :7193, :10322; warm-up gate :11539; the unpark join is a prefix of `ids`), so the
  commit never runs past the reserved range.
- **No paths outside the repo, no secrets**: the card test works ROOT-relative under `scratch/ik2-kv-start-card`
  (tools/card_tests/ik2_kv_start.sh:40-49), drops `api_key` from its config copy (:135), and the diff's code lines
  hold no absolute paths or secrets.
- **Tests really exercise the change**: tests/ik2_plan_test.cpp holds the chunk arithmetic and the ascending commit
  order to Inkling's rows (the 45 checks, incl. the 60K read committing exactly chunks 8 .. 14); the card test now
  fails on a wrong answer, not just a refusal; the GROWABLE flag greps the engine's stdout, which serve pipes to
  the engine's log file (serve/qfn_engine.py:697-703) - the file the script greps (:165).

## Findings, by severity

1. **NIT - the step dropped one of step 1's test checks.** de8a5d9bb deleted
   `expect(!elastic_start(..., true, false), "not --parallel")` from tests/ik2_plan_test.cpp with no replacement:
   the `!parallel` term of the gate (ik2_plan.hpp:39-41) - which the engine still passes correctly
   (`g_par > 1`, ik2_main.cu:8128) - is no longer held by any test ("not two GPUs" and "not --tp" still are, :61-62).
   It also makes REPORT-p125o-ik2-elastic-2.md:90-91 ("the 21 of step 1 (unchanged, still green)") - repeated by
   FIX-REPORT.md:75-76 - untrue of that one check, and the earlier review's "21 unchanged" line missed the deletion
   too. One restored line closes it.
2. **NIT - the card test can die without a verdict when the engine dies** (inherited from P125e step 5, still in
   the range): under `set -eu` (tools/card_tests/ik2_kv_start.sh:39), the bare `kill "$SRV_PID"` at :187 returns
   non-zero if the server already exited - precisely the engine-fault case the card test exists to catch - and the
   script aborts before printing "RESULT:" with no explanation. `|| true` (and reporting a dead `$SRV_PID`) would
   let the verdict run.
3. **INFO - the HIP shortage message always names the OOM case**: any `hipMemCreate` failure reports
   "out of device memory (hipMemCreate)" (ik2_vmm.hpp:270-272) where the CUDA branch distinguishes
   `CUDA_ERROR_OUT_OF_MEMORY` from other codes (:163-165). Cosmetic - the request fails either way, never the
   engine - but a card log could blame VRAM for a non-VRAM error.
4. **INFO - `kv_grow` trusts its callers to clamp `n`**: it clamps nothing itself (ik2_main.cu:567-571); all six
   call sites do clamp (:10903 guard, :10322 and :7193 `min(..., ctx)`, :11425 guard, the :11539 warm gate, and the
   unpark join is a prefix of `ids`). A future unclamped caller past `ctx` would set `g_kv_chunks` past
   `g_kv_n_chunks` and underflow `zero_hi - old` (size_t) on a *second* such grow. `need = min(need,
   g_kv_n_chunks)` inside `kv_grow` would make it self-defending.
5. **INFO (carried, unchanged)** - the grow-failure `@END error out_of_memory %zu 0 0 0` (:10937) keeps the
   sibling `context_length` line's field order (:10909); the legacy `@END` reader lives outside the repo, so the
   ordering rests on that sibling. No defect raised, none found.

## Tests run (plan-named modules only, one at a time)

- `ninja -C scratch/build-ik2-elastic-2 ik2_plan_test && ./scratch/build-ik2-elastic-2/ik2_plan_test`
  -> `ik2_plan_test: ok` (45 checks; ninja reported the target current at HEAD, so this is the tree under review).
- `ninja -C scratch/build-ik2-elastic-2 saguaro-ik2` -> current at HEAD; independently recompiled its fixed
  `ik2_main.cu` with the exact nvcc command from `ninja -t commands`, output redirected to
  scratch/review2_ik2_main.o: exit 0, no new warnings.
- `bash -n tools/card_tests/ik2_kv_start.sh` -> clean. Not executed end to end - no GPU and no Inkling weights on
  this box, as the report states; Ken's cards (watch the `alpha` verdict and the `NOTE context:` wording) decide
  the HIP branch.
