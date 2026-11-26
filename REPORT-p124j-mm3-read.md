# P124j — request priority: MiniMax-M3.1-Flash keeps a stopped prompt read (Saguaro 1.2.4; audit item 3)

Plan: PLAN-p124j-mm3-read.md, per ref/REPORT-p124c-prio-audit.md §3 and §8.3. Engine + driver + tests for the
mm3 prompt-read preemption; card test written for the manager. No GPU runs, no serve/qfn_engine.py touched.

## 1. Verified first (with the exact files the audit's line numbers pointed at: src/mm3/mm3_main.cpp)

(a) **The server never cut an mm3 read.** `Service._sliced`'s read gate is `serve/server.py:5740-5743`: it fires
    only for `(can_pause and multi)` or `(keeps_stopped_read and not batch)`. MM3's READY offers `stop` only
    (`src/mm3/mm3_main.cpp:601` solo `READY <ctx> stop`, `:1042` multi `READY <ctx> stop multi=<n>`) →
    `can_pause` False (`serve/server.py:1520`), and `keeps_stopped_read` was only ever set from `can_pause`
    (`:1523`, the P124h line) → False. So an mm3 read ran to its end even though the engine keeps a cut read.

(b) **A STOP landing on the read's LAST chunk fell through to decode.** Solo: `src/mm3/mm3_main.cpp:740` took the
    STOP only with `&& !last`, so the loop finished the prompt and the decode loop's stop check (`:849`, `:965`)
    ended the request in decode instead. Multi: the `if (last) break;` at `:1448` stood before the per-request
    check `in.stop.load() || in.stopped(g.rid)` at `:1449`, so the same fall-through (the decode `show()` check
    is `:1158`).

(c) **MM3 keeps a cut read — so the work was in scope.** The cancel paths insert every fed chunk
    (`state.insert` per chunk, `:737` solo, `:1442` multi) and never clear: solo `:758-762` DONEs with
    `reused=from` and continues without `state.clear()`; multi `:1473-1478` DONEs, `release(best)` (`:1105-1109`,
    marks the sequence idle only) and `in.forget(g.rid)` — the sequence's `S.state` stands. Only an engine error
    clears (`:754` solo, `:1467` multi). The resumed GEN re-matches the state (`:654`, `:1355` — the last token is
    always read again for its logits) and reads on from the chunk reached. The audit's §10.2 caveat (the cancelled
    DONE reports `reused=from`, the OLD base, not the partial end) stays true and is noted under open questions.

## 2. Engine: a STOP during the last chunk ends the read after that chunk — solo AND multi

`src/mm3/mm3_main.cpp` (6 insertions, 2 deletions; the per-rid STOP machinery untouched):

- Solo `:740`: `if (done < T || (in.stop.load() && !last))` → `if (done < T || in.stop.load())`. The cancel path
  (`:758-762`) then ends the request with its `cancel` DONE exactly as a STOP between chunks does; the resumed
  read re-reads the last token alone. The pipelined TP path (`whole`, progress callback → `done < T`) already
  ended this way.
- Multi `:1448-1452`: moved the `if (in.stop.load() || in.stopped(g.rid)) { cancel = true; break; }` check before
  `if (last) break;`, so the STOP taken after the last chunk sets `cancel` and the keep path
  (`:1473-1478` → DONE `reused=bfrom2`, state kept, `release`, `forget`) runs. Per request: `STOP <rid>` ends
  one conversation's read (`in.stopped`, `:210-216`); a bare STOP (engine protocol, only from the driver's
  shutdown paths) still means all.

**Compile proof (no GPU runs):** fresh build dir `scratch/build-mm3-sm75` (git-ignored scratch), configured with
CMake/Ninja `-DMAYA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` against `/usr/local/cuda-13.1/bin/nvcc`
(CUDA 13.1.115): `[62/62] Linking CXX executable saguaro-mm3`, zero errors in `scratch/mm3-build.log`.

## 3. Driver: keeps_stopped_read for the mm3 engine

`serve/server.py` (+2 lines, in `StrataEngine.__init__`'s READY block, right after the P124h GLM line):

```python
if self.info.get("engine_kind") == "mm3":   # P124j: MM3 keeps a cancelled read too (solo and multi), so a cut read resumes
    self.keeps_stopped_read = True
```

`INFO engine=mm3 engine_kind=mm3 ...` is printed before READY (`src/mm3/mm3_main.cpp:337`), so `self.info` holds
it when READY is parsed. The `_sliced` gate itself is unchanged: mm3 has no PAUSE, so it takes the `cut = True`
branch (`:5753`); GLM's line and the maya-metal exclusion are untouched. The cut on **multi** is per request: the
line driver's close drains with `STOP <rid>` (`:2353`, `_generate_multi`'s finally) — never an engine-wide STOP —
and the new test below pins exactly that. The fair-read timer pause (`SAGUARO_FAIR_READ_S`, `:5723-5728`) stays
can_pause-only, as before.

## 4. Tests (serve/test_priority.py, +80 lines)

- `ReadsPreempt.test_an_mm3_solo_read_is_cut_for_a_normal_and_continues_kept` — an mm3-like fake
  (`UnreportingRead`: keeps_stopped_read, no pause, no reused figure over the pipe — the line driver's shape),
  `multi=0`: a low read is **cut** (STOP — the fake has no `pause()`; the server would crash if it sent one) for
  a waiting normal request; the second call carries the same prompt and reads on from where the cut left it
  (`ticks == CHUNKS`: nothing kept was read again); both log lines checked, the continue line in its numberless
  wording.
- `ReadsPreempt.test_an_mm3_multi_read_is_cut_for_a_normal_and_continues_kept` — the same with `multi=2`.
- `TheMultiCutNamesOneRequest.test_closing_a_multi_request_stops_only_its_rid` — drives the real
  `StrataEngine._generate_multi` with a fake stdin: closing one request's generator (what `_sliced`'s cut does)
  sends `STOP 1` — its rid, never a bare STOP.

Results: `serve.test_priority` **59 tests OK** (56 before, 3 new),
`serve.test_fair` **14 tests OK**, run one at a time with `/opt/llm/saguaro/.venv/bin/python -m unittest`.
No other test module was run (per plan).

## 5. Card test for the manager: scratch/card-test-mm3-read.sh (committed with -f; scratch/ is ignored)

`CFG=/path/to/<mm3>.json scratch/card-test-mm3-read.sh` — a test copy of the card's config on port **8092** with
its OWN session dir, its OWN engine log and **no api_key** (nothing of the live box is written); one phase: a
fresh ~30K-token **low** prompt read, cut **once** at read position 10000 (`CUT_AT`) by a short normal request;
the server must log the pause line naming the normal request and the continue line; PASS when the big prompt's
own end line in the ENGINE log — `prompt N tokens = R reused + K read` (`src/mm3/mm3_main.cpp:1012-1017` solo,
`:1119-1123` multi; both shapes' extraction tested) — reports **R > 0**: the resumed read's DONE reused what the
cut kept. Not run here (no GPU runs); the manager runs it on the card.

## Files changed / added

- `src/mm3/mm3_main.cpp` — solo `:740` and multi `:1448` last-chunk STOP (step 2) — commit 2457e5c1b
- `serve/server.py` — `keeps_stopped_read` for `engine_kind=mm3` (step 3) — commit a97123d20
- `serve/test_priority.py` — the three tests (step 4) — commit 3144c700f
- `scratch/card-test-mm3-read.sh` — the card test (this commit)
- `REPORT-p124j-mm3-read.md` — this report (this commit)

## Open questions

1. The resumed read's sampling state: MM3's cancelled-read DONE reports `reused=from`, the OLD base (audit
   §10.2) — the KV is trusted for the kept prefix, and P124j now lets the server rely on that for priority cuts.
   Penalty windows / drafts rebuild from `state` (the fake-token vector, which DOES hold the kept prefix), so the
   resumed answer should be exact; the card test measures the KV reuse figure, not token-for-token equality with
   an uninterrupted read. A parity run (same seed, cut vs not) would close it.
2. A STOP arriving after the read's last chunk but during the engine's first-token draw still lands in decode
   (`:849` / `:1158` catch it with 0 tokens emitted) — the cut then behaves as an answer cut at 0 written;
   bounded, and the driver's drain is bounded too. Left as is (smallest change).
3. Multi sequences are LRU-picked (`:1343-1369`): a cut read's kept sequence can still be handed to the NEXT
   request if its prompt extends another conversation better — the keep is per sequence, best-effort, as with
   GLM's slots. Fine for the priority feature; worth remembering when judging card-test flakes.
4. hy4 has the identical shape (audit §8.2) and was deliberately NOT opened here — this plan is mm3 only.
5. The card test assumes the config's engine reports via its `log` file (set to the scratch copy); if the card
   runs mm3 with `--parallel` the multi log line is matched too (both awk shapes tested), but with several big
   prompts interleaved its "last big line" reading would need a rid filter — single cut, single big prompt here.
