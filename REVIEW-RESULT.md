# REVIEW-RESULT - P125o (Inkling 2 elastic VRAM, step 2: the K / V grows on demand)

Verdict: **FIX-FIRST**

Range `git diff d210adc76e47b9883484a6b20b32372ff98a0ce1..HEAD` against PLAN-p125o-ik2-elastic-2.md and
REPORT-p125o-ik2-elastic-2.md. The P125e commits through 37c7523ca carry their earlier review (MERGE-WITH-NITS);
this review is of 37c7523ca..HEAD (the a2 / a3 step) and re-checks the OFF-path over the whole range. One CRITICAL
and one HIGH must be fixed before merge; both are of the "silently wrong / engine-killing" kind the plan itself
warns about.

## Findings, by severity

1. **CRITICAL - the K/V-write epilogue still wraps past the committed start step; every conversation past the
   commit is silently wrong.** `Vi.kv_cap` is the commit on a VMM-backed start (src/ik2/ik2_main.cu:8536), and the
   write epilogue passes it to kernels that index modulo it: `slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t)`
   (src/ik2/ik2_kernels.cu:465 and :471). All full-visit epilogue call sites pass `Ly.kv_cap`: the prompt path
   (ik2_main.cu:2108-2119), the decode paths (ik2_main.cu:3443-3477), the block / graph windows
   (ik2_main.cu:4555-4632). Attention instead reads by absolute position (`attn_kv_cap = Ly.sliding ? Ly.kv_cap :
   0`, ik2_main.cu:2129, :3423). So from position 32768 (default SAGUARO_IK2_KV_START) on, every written row lands
   at `pos % 32768` - trampling the conversation's own early rows - while attention reads the never-written
   absolute rows as zeros. No fault, no NOTE, wrong answers, exactly for the past-the-start-step reads this step
   exists to enable. The report's claim "The kernels needed nothing ... no wrap to rebase"
   (REPORT-p125o-ik2-elastic-2.md:27-31) and the comment at ik2_main.cu:8600-8605 are true only of the reads; the
   P125e review had already named this very wrap ("the K/V-write epilog wraps % kv_cap over the first rows").
   Committing pages (a3) makes them writable, but the row index stays modulo the commit, so the a3 hooks cannot
   paper over it. Fix direction (not applied here): pass `kv_cap = 0` for non-sliding visits in the `prep_qkv_r_*`
   calls where `g_kv_vmm`, as the reads already do; the DEPTH memset (ik2_main.cu:8635-8652, sized `kv_cap` =
   commit) is the only consumer that needs the capped value and it keeps it.
   The extended card test cannot catch this: its 60 K verdict checks HTTP 200 and `completion_tokens > 0`
   (tools/card_tests/ik2_kv_start.sh:211-220), never that the answer is the first word it asked for - and finding
   1 corrupts precisely those rows while still emitting tokens. The growth read should check the returned text
   equals "alpha".

2. **HIGH - the `--drafting auto` trial writes its prompt rows before committing any of them; the hook in
   `decode_greedy` runs after.** Trial requests are admitted up to the *ceiling* (`v.size() + 256 < ctx`,
   ik2_main.cu:10250), and `trial()` runs the prompt through `forward()` (ik2_main.cu:10269-10281) - writing rows
   [0, ids.size()) - before `decode_greedy` reaches its `kv_grow(P + gen, ...)` (ik2_main.cu:7161, called at
   :10291). On an elastic VMM card a DRAFT_IDS prompt longer than the commit therefore writes past the committed
   prefix with nothing committed: today that lands in wrapped rows (finding 1 - garbage measurement, corrupted
   first rows), and the moment finding 1 is fixed to absolute writes the same prompt pass writes reserved-but-
   unmapped pages - an illegal-address fault that kills the engine before READY, violating "a grow failure fails
   the request, never the engine". The report's "the `--drafting auto` trials (`decode_greedy`, before READY)
   commit their positions too" (REPORT:55-56) covers only the generation, not the trial's prompt pass that
   precedes it. Commit in `trial()` before the forward loop (or bound the :10250 filter by `g_kv_commit`).
   Two side-effects in the same hook: `if (!kv_grow(P + gen, gerr)) return;` leaves `out` empty and the trial then
   counts `made += (int) out.size() - 1` (ik2_main.cu:10294) - a negative token count feeds the tok/s table; and
   `P + gen` is not clamped by `ctx` (SAGUARO_IK2_DRAFT_TOKENS is unbounded, :10256), so an oversized trial sets
   `g_kv_chunks` past `g_kv_n_chunks` (the commit call clamps to the reservation, so nothing faults, but the
   counter overstates).

3. **MEDIUM - `unpark` failures now fail every request with a false OOM on all cards, changing the OFF path.**
   `unpark` already returned false on two pre-existing, non-VMM conditions - the folder read failing
   (ik2_main.cu:10175-10181) and no snapshot at `n` (:10188-10191) - and the caller ignored the result, degrading
   to reading the prompt from scratch (`git show 37c7523ca:src/ik2/ik2_main.cu:10802`: `if (j >= 0) unpark(...)`).
   The new handling turns *every* false into `ERR the GPU ran out of memory taking back the kept conversation's
   context` and drops the request (ik2_main.cu:10937-10950) - including on cards where `g_kv_vmm` is false and VRAM
   cannot be the cause. So the feature-OFF behavior is not byte-identical: a corrupt folder file or a missing
   snapshot used to cost a re-read, now costs the turn, under a misleading message. Distinguish the a3 grow
   failure (the new `full_rows` false from `kv_grow`, :10070-10079) from the old ones, or keep the from-scratch
   fallback for the old ones.

4. **LOW - the growable cache depends on freshly mapped pages reading as zero, which the VMM API does not
   document.** ik2_vmm.hpp:7 and :142 state "fresh physical pages map zeroed - the DEPTH test's semantics", and
   the DEPTH memset only covers the committed rows (`Vi.kv_cap`, ik2_main.cu:8635-8652): `SAGUARO_IK2_DEPTH=N`
   with N past the start step reads positions [32768, N) as zeros only because the commit hook mapped fresh
   pages that were *not* explicitly zeroed. `cuMemCreate` / `hipMemCreate` carry no documented zeroing guarantee
   (`cudaMalloc` explicitly does not initialize). In practice NVIDIA/AMD drivers hand out zeroed pages, but this
   is a correctness dependency on undocumented behavior; a targeted memset at commit time (or a note where the
   claim lives) would make it safe.

5. **NIT - the drafter-drop comparisons can change non-elastic behavior under rare env knobs.**
   `if (drafter && P + gen > ik2::draft::context_rows(g_kv_commit)) drafter = nullptr` (ik2_main.cu:7162) and the
   CLI's twin (:11365-11368) compare the *requested* run length against `context_rows(g_kv_commit)`; with
   SAGUARO_IK2_DRAFT_CTX below the context (ik2_draft.cu:698-702) or a SAGUARO_IK2_DRAFT_TOKENS beyond the trial
   filter's 256 headroom (:10250, :10256), step-1 ran drafted where this step runs undrafted. Served tokens stay
   identical (drafting verifies; `spec_rows` at :10971 is guarded by the request guard at :10843 so it cannot fire
   without DRAFT_CTX), but the `--drafting auto` measurement and the CLI's drafting can move on cards with this
   step OFF.

6. **NIT - the `kv cache:` line counts the layout, not the mapped bytes** (ik2_main.cu:8682-8696): it prints
   `Vi.kv_cap` (the commit) while the VMM commits `chunks_for(commit) * chunk_b`, which rounds above the commit
   when SAGUARO_IK2_KV_START is not a multiple of a chunk (e.g. 10000 -> 12288 rows). Exact at the default 32768
   (8 whole chunks), harmless under the card test's 2 % rule; a one-word comment would save a future reader.

7. **INFO** - the grow-failure `@END error out_of_memory %zu 0 0 0` (ik2_main.cu:10873) has the field order of the
   context_length line (:10850), as commit 7b46f80c2 says; no @END reader exists inside this repo to verify the
   legacy parser against (it lives in the consumer outside), so that ordering rests on the sibling line alone.

## Manager checks

- **Commit / grow never on the per-token path**: `kv_grow` (ik2_main.cu:566-582) is called at request start
  (:10861, before the guard-set `busy = true` at :10889 and before any kernel), at kept-conversation restore
  (:10075), at the CLI start (:11362, load-time), at the window warm-up (:11477) and at the `decode_greedy`
  entry (:7161) - none inside the token loops (:7164-7198, :11205-11290) and none in a kernel. Decode drafts are
  budget-clamped (`room = min({max_new - i - 1, ctx - hist.size(), req_wmax})`, :11230-11231), so a request never
  writes past what its start commit covered. Finding 2's hook ordering does not put a commit in a loop, but does
  put writes ahead of commits - fix it by committing earlier, never per token.
- **HIP equivalents for every VMM call**: cuDeviceGet/cuDeviceGetAttribute/cuMemGetAllocationGranularity/
  cuMemAddressReserve/cuMemAddressFree/cuMemCreate/cuMemRelease/cuMemMap/cuMemUnmap/cuMemSetAccess
  (ik2_vmm.hpp:28-101) have counterparts in the HIP branch (:216-307); `hipDeviceGet` is legitimately unused
  (HIP's `location.id` is the ordinal, :212). The one behavioral difference - CUDA grants access to peer devices
  (:97-115) where HIP grants only its own (:259-262) - is documented (:252-253) and safe: the elastic gate
  excludes two GPUs, --tp, --parallel and EP (src/ik2/ik2_plan.hpp:38-42, ik2_main.cu:8097-8099), and the
  configure probe (CMakeLists.txt:1015-1050) falls back to the a1 start with one stderr line where a piece is
  missing (ik2_main.cu:8110-8132). Reserve-call shape matches the proven src/mm3/mm3_vmm.hpp pattern
  (mm3_vmm.hpp:53,:123 vs ik2_vmm.hpp:62,:134).
- **A grow failure fails the request, never the engine**: on shortage `kv_grow` rolls back exactly what the
  failed call mapped (ik2_vmm.hpp:175-182, :281-288), `shrink`s the buffers that did commit back to the old
  commit (ik2_main.cu:576-578), leaves `g_kv_chunks` and the graphs (base pointer stable) untouched, and the
  serve handler answers NOTE + ERR + DONE and continues with no kernel run (:10859-10877); the unpark path
  (:10937-10950) resets `busy` / `g_wd_busy` before `continue` (busy was set at :10889-10890). The only engine
  exits on VMM failure are at load (:8618-8623) - before service - which matches the rule. (Finding 2 is the
  violation: an unfixed finding 1 lets the *trial* fault the engine before READY.)

## What checked out (brief items)

- **Feature OFF byte-identical**: `g_kv_commit` defaults to `ctx` (:7527-7528); `kv_grow` is a no-op without
  `g_kv_vmm`; the VMM branch is `else if`-guarded off the elastic start; `draft_room`/drafter-load/STAT `kv_cap=`
  /READY/`INFO ctx=` print the same numbers as before (:8261, :9273, :10432, :10339-10366); the fallback NOTE
  keeps step 1's line verbatim plus the promised stderr line (:8114-8132); `req_spec`'s new `spec_rows` factor is
  always true without the feature (guard :10843 vs :10971). Exceptions: findings 3 and 5.
- **Locks, threads, stdout**: the only other thread is the stdin reader queueing lines (:10371-10398); all
  `kv_grow` / VMM calls run on the request thread - no lock is taken anywhere near them, so no nesting question
  arises; `api()`'s static init is a magic static (:58-70). The ERR + DONE pair is one printf (:10867, :10941)
  like the sibling error paths, and `say()` flushes (:9898-9901), so nothing can interleave a line.
- **Generated file**: `serve/qfn_engine.py`, `serve/server.py` and `tools/saguaro/extract_qfn_engine.py` do not
  appear in the range at all - nothing was hand-edited behind the generator's back.
- **Rollback / clamp arithmetic**: `Range::commit` clamps to the reserved cap (:146, :255), releases each
  `cuMemCreate` handle right after mapping (:167, :273 - the mapping holds it), and grants access once per
  committed span; the chunk math in src/ik2/ik2_plan.hpp:51-72 rounds chunk bytes *up* to the granule and the
  prefix-commit self-heals any non-divisible row size.
- **No paths outside the repo, no secrets**: the card test works ROOT-relative under `scratch/ik2-kv-start-card`
  (tools/card_tests/ik2_kv_start.sh:38-47), copies the config with `api_key` dropped (:131), and the diff holds no
  absolute paths or secrets.
- **Tests really exercise the change**: tests/ik2_plan_test.cpp genuinely holds the new chunk arithmetic and the
  ascending commit order to Inkling's rows (45 checks), as the plan asks; the engine integration however is only
  reachable on a card, and the extended card test judges "answered", not "answered correctly" - see finding 1's
  note; strengthen the growth read's verdict to the expected word.

## Tests run (plan-named modules only, one at a time)

- `g++ -std=c++17 -Wall -o scratch/ik2_plan_test_review tests/ik2_plan_test.cpp && scratch/ik2_plan_test_review`
  -> `ik2_plan_test: ok` (45 checks; rebuilt from HEAD, matches the coder's binary in
  scratch/build-ik2-elastic-2).
- `bash -n tools/card_tests/ik2_kv_start.sh` -> clean. Not executed end to end (no GPU, no Inkling weights - as
  the report states). No GPU run anywhere; the build target `saguaro-ik2` was not rebuilt here (the coder's
  scratch/build-ik2-elastic-2 is present and current per the report).
