# FIX report: the P8 re-review (./REVIEW-RESULT.md) - finding -> what changed or why not

Three commits on top of 42150823d, one per actionable finding; the two Info findings needed nothing and are
answered below. All changes are in `src/ik2/ik2_main.cu` (the High fix, comments) and `REPORT-p8-u9-findings.md`
(the wording the review flagged); `serve/qfn_engine.py` / `engines/qfn/serve/server.py` were never touched, so
the generator (`tools/saguaro/extract_qfn_engine.py`) was not run.

## 1. High (FIX-FIRST) - staged head load copies only the q8 (ik2_main.cu:9104) -> FIXED (commit d9a079601)

The `DevScope g0(0)` block now holds both D2Hs, as the old peer pair had: the `q8` into `stage` and the `q8d`
scales into `stage + qb`, before the two H2Ds. The staged branch - the only one that runs today, since
`g_two.link.all_peer` is false at drafter load - no longer folds an uninitialized pinned-memory scale into the
drafter's head through `fold_q8`/`set_head_q8`. Also `REPORT-p8-u9-findings.md` §1: the sentence that spelled
only the `q8` half of the `qb + db` bounce now spells both D2Hs and both H2Ds.

## 2. Low - the kMIX comment leans on a die the PF_GPU1 path never reaches (:3082-3085) -> FIXED (commit d8701d9a7)

Behaviour kept (the `return false` stays - the review calls it "fine as behaviour"); the guarantee was reworded
after re-checking the code: the die is inside `tp_gu` (`:2785`, called only at `:2823`/`:2825`), which lives in
the `else if (g_tp)` branch (`:2735`), while `p1` requires `!g_tp` (`:2453`). The comment now says the refusal
is the guard for the prefill path, not insurance over that die; §4 of the report was corrected the same way and
notes the token path's `two_send` has no kMIX check of its own (out of P8's scope, unchanged).

## 3. Low - the ev_cp record's consequence was not recorded (:1679-1683) -> FIXED (commit 6e9b06a4d)

No code change (the review confirms the record is correct and closes the real race). The report's §5 now states
what it omitted: the new snapshot is newer than the paired `write_context`'s, so the leading wait of the next
`drafter_copy_taps` orders the next copy-down - and the next `propose`'s D2H into `host_anchor` - behind the
drafter's *propose*, not just its `write_context`; near-free in the live propose-verify-write_context pattern,
but one more ordering on the staged path. The open question about the staged branch's missing leading
`if (cp_valid) wait` stays open, as the review agrees it should.

## 4. Info - the staging block's transient ~1.4 GB pinned allocation (:9103) -> NO CHANGE

The review itself calls it acceptable (opt-in only, freed right after, dies via `CU()` on failure) and the
chunked bounce a nicety, not a finding to fix; it is also a one-shot load-time copy, so chunking would buy
peak-RAM kindness only. Left as is; the review's alignment aside (`stage + qb` is 2-byte aligned since `db`
itself requires `C % 32 == 0`) confirms no hidden correctness issue in the fix's pointer arithmetic.

## 5. Info - the named test module is a docs gate -> NO CHANGE

`tools.test_docs_new_engines` is what the plan names ("Run ONLY"); nothing here can extend it without touching
other files, which the plan forbids. The real verification remains what the review lists: the single-TU compile
below and the two-card start check, still unavailable on this machine - recorded as an open question in the
report.

## Verification after the fixes

    /usr/local/cuda-13.1/bin/nvcc -std=c++20 -DNDEBUG -arch=compute_75 -DSAGUARO_IK2_PASCAL=1 \
      -DMAYA_DS4_IQ_ONLY -DDS4_IQ_MIN_FORMS -Iinclude -Isrc/hy4 -Isrc/hy4/ds4 -Isrc/ds4 -Isrc/mm3 \
      -Ithird_party/ggml -Isrc/ik2 -Isrc/strata_fmt -c src/ik2/ik2_main.cu -o scratch/ik2_main_final.o
    -> exit 0; only the two pre-existing unused-declaration warnings (Shards::num, g_apu)

    /opt/llm/saguaro/.venv/bin/python -m unittest tools.test_docs_new_engines -v
    -> Ran 7 tests in 0.018s - OK

The High fix adds only host-side `cudaMemcpy` calls inside the `dg > 0` opt-in block and comments; it launches
no kernel and touches no device code, so the review's device-side "default path untouched" proof still holds.
Working tree clean; the scratch objects are under ./scratch/ only.
