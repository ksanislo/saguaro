# P124h - GLM's SOLO engine can stop a prompt read too (Saguaro 1.2.4, audit item 1)

Per PLAN-p124h-glm-solo-read.md, from ref/REPORT-p124c-prio-audit.md §1/§8.1. No C++ changed, nothing on a GPU,
serve/qfn_engine.py untouched (it was not needed).

## 1. Verification: does a solo PAUSE inside a read keep what it read? YES (fast path, default settings)

The old gate comment (serve/server.py:5720-5722 before the change: "on a solo can_pause one a PAUSE during a read
may be ignored or lose the read") does NOT hold for the GLM line engine. The chain, with file:line:

1. **Solo GLM offers PAUSE.** `READY <ctx> stop pause` with no `multi=` - src/program/generate.cpp:5008; the driver
   parses `can_pause` from it (serve/server.py:1520). `StrataEngine.pause()` with `multi` false sends a bare
   `PAUSE` line, no rid (serve/server.py:2394-2399).
2. **A solo PAUSE is honored by the read.** The `PAUSE` line sets `pause_req` **and `stop_req`**
   (generate.cpp:1319-1323). The prompt read checks `read_stopped()` (= `stop_req || admit_stop`, :1267):
   inside one `prefill` call at every chunk through the progress callback
   (`return !read_stopped()`, generate.cpp:2200-2204 -> src/core/glm_prefill.cu:4910-4912 one-part path,
   :5022-5025/:5083-5090 pipelined path), and between pieces for solo (`!prompt_between` - that callback is set
   only by the multi path, generate.cpp:3846/:3865 - so the solo check :2224-2227 fires): `err = "cancelled"`.
   The prefill stops every part at the same chunk boundary and leaves the position there
   (glm_prefill.cu:4926-4927, `pos = pos_start + off[limit]` :5091; `err = "cancelled"`, return false
   :5101-5102). The PAUSE is neither ignored nor destructive.
3. **The read is kept.** The `err == "cancelled"` branch (generate.cpp:2256-2305): `SAGUARO_GLM_KEEP_CANCELLED`
   is on by default (:2262); the exact half-read state is snapshotted at the token reached
   (`model.snapshot_save()` at `pos`, `snap_tokens` = the read prefix, log "glm slots: a cancelled prompt keeps
   the N tokens it read (M new)" :2266-2275), else the longest checkpoint of this prompt (:2276-2297).
   `finish = "cancel"` (:2304) and the solo DONE carries finish+reuse (:2622-2625). `pause_snapshot` (:2332-2361,
   called at :2627) is a **no-op for a read pause** - it requires written tokens (`!emitted.empty()`, :2334) - so
   the keep is purely the cancelled-read machinery, the same one a client's STOP during an upload already uses
   and the audit's live GLM log shows (§7: "a cancelled prompt keeps the 152512 tokens it read").
4. **The next GEN of the same prompt continues from there.** `prep_restore` matches the kept `snap_tokens` as a
   strict prefix of the prompt (generate.cpp:1986-1988), restores it and returns `reuse = snap_tokens.size()`
   (:2000); the read goes on from `at = reuse` (:2167) and says so with `RESUME <reuse>` (:2118).
5. **The driver's resume matches.** The pause's DONE with `finish == "cancel"` makes `_sliced` treat it as "not
   finished" (serve/server.py:5797-5799), it queues again (:5851) and re-sends `cur = ids + produced` (:5914) -
   produced is empty mid-read, so the prompt is identical and the engine reuses the kept snapshot.

Two real limits found (documented, not "lost" in the wrong sense):

* **maya-metal is a different engine and DOES drop a cut read.** Its READY also says `stop pause`
  (src/metal/serve.cpp:560), but a STOP between prompt chunks ends with `model.reset()` - "a cut prompt leaves
  no state worth keeping" (serve.cpp:302-306, :328-329); its PAUSE keep is decode-only (`!emitted.empty()`,
  :492). So the gate must not open for it.
* The keep needs the GPU fast path (`model.fast()`, the default: include/maya/core/glm_model.hpp:126-128). With
  `SAGUARO_GLM_SLOW=1` the read runs through the 8-token `prep_tail` path, which stops on the stop flag
  (generate.cpp:2105-2108) but has no keep branch (the cancelled branch sits inside the batched-path block
  :2151/:2256) - a partial read is then re-read, conversation base intact. Likewise
  `SAGUARO_GLM_KEEP_CANCELLED=0` turns the engine-side keep off.

## 2. The gate opened (serve/server.py, one commit)

The `_sliced` read-preemption condition itself is **unchanged**:

```python
((can_pause and multi) or (keeps_stopped_read and not batch))     # serve/server.py:5729-5732
```

Instead the driver now declares the capability it was missing: `StrataEngine` sets, right where it parses READY
(serve/server.py:1522-1523),

```python
# P124h: GLM keeps a cancelled prompt read (#573), so solo reads can pause too; maya-metal drops one, so it does not
self.keeps_stopped_read = self.can_pause and Path(exe).name.lower() != "maya-metal"
```

plus a `keeps_stopped_read = False` class default (serve/server.py:1464). This is the narrowest true condition:
the `pause` READY token is only ever emitted by the GLM line engine (generate.cpp:5007-5008) and maya-metal
(metal/serve.cpp:560), so `can_pause` picks exactly those two and the exe name drops the one that loses reads.
DS4/MM3/IK2/HY4 advertise `stop` only (ds4_main.cpp:751-753, hy4_main.cpp:958, ik2_main.cu:10165-10166,
mm3_main.cpp:601), so they stay shut; MM3/HY4 *do* keep cut reads but that is audit items 2/3, not this plan.
QFN's wrapper pins `can_pause = False` (serve/server.py:9440) and keeps its own keeps_stopped_read+battery of
comments. With the flag set, solo GLM opens the second gate branch (batch is unset on StrataEngine) and at
:5738 `can_pause` is checked first, so it **pauses** - the engine keeps the half-read - it is never cut. The
gate's comment (:5721-5728) and the pause-line comment (:5739) were reworded to the new truth.

Fair reads (the SAGUARO_FAIR_READ_S timer, :5688/:5707-5720) remain multi-only, as before: this change is only the
R122 priority preemption, which needs no timer.

## 3. Tests (serve/test_priority.py, one commit)

* New `ReadsPreempt.test_a_solo_glm_read_pauses_by_its_pause_and_keeps_it`: `PausingRead` with `multi = 0` and
  `keeps_stopped_read = True` (the driver's shape for the GLM line engine); a LOW read and a waiting NORMAL
  (`race_read(engine, PRIORITY_NORMAL)` - a low steps aside for a running/waiting normal, FairLock.step_aside
  :617-623). Asserts: both answers correct and the normal one finishes first; `engine.pauses >= 1` (PAUSE, not a
  cut); the log "priority: a low prompt read paused" and "prompt read continues from where the engine kept it";
  the second call for the long prompt carries **the same ids** and its read finishes its base in exactly
  CHUNKS ticks - the kept read was not re-read.
* Existing `test_a_solo_pause_engine_lets_its_read_finish`: kept, and made the "engine that does NOT keep reads"
  case explicitly (`multi = 0` **and** `keeps_stopped_read = False` - maya-metal's shape): its read still runs to
  its end, no "prompt read paused" in the log, every chunk read exactly once.
* `race_read` gained a `second=` priority parameter (default PRIORITY_HIGH, unchanged behavior); PausingRead's
  docstring now names the flag instead of the old multi-only rule.

Runs (the two the plan names, one at a time):

```
python -m unittest serve.test_priority   -> Ran 54 tests ... OK   (53 before + 1 new; run twice more for the
                                                        new race: stable)
python -m unittest serve.test_fair       -> Ran 14 tests ... OK
```

## 4. Files / commits

* serve/server.py - `keeps_stopped_read` on StrataEngine (class default :1464, set at READY parse :1522-1523),
  gate + pause comments :5721-5728/:5739. Commit 673681d67.
* serve/test_priority.py - PausingRead docstring, `race_read(second=)`, the new solo test, the old solo test made
  the drop-reads case. Commit 1e94324fa.
* src/ untouched; serve/qfn_engine.py untouched (no extract run needed).

## 5. Open questions

1. **maya-metal is excluded by its exe name.** The engines carry no READY capability for "a stopped read is
   kept" (the audit's DS4 item proposes a `keepread` token); if maya-metal later keeps cut reads, or the GLM
   exe is ever renamed, the exe-name test here must change with it. A READY token would be sturdier - engine
   work, out of this plan's scope.
2. The driver advertises the keep unconditionally for the GLM line engine, but the engine only keeps when
   `SAGUARO_GLM_KEEP_CANCELLED=1` (default) and the fast path is on (`SAGUARO_GLM_SLOW=1` loses partial reads -
   re-read, not corruption). Same assumption shape as QFN's keeps_stopped_read; both switches are developer
   tools. Worth a READY fact later?
3. Fair reads (SAGUARO_FAIR_READ_S) are still multi-only (:5688, :5707-5720). Solo GLM now steps aside for
   priority only;
   a solo long read still never pauses on the timer for same-priority waiters. Deliberate here (the plan asked
   for the priority gate), but 1.2.5 could revisit.
4. The audit's open question 1 ("needs a live GLM run") is answered from code + the audit's own live logs (§7);
   the mock test proves the driver half. A one-line live check: start a solo GLM, fire a low request with a
   100K-token prompt, then a normal one, and watch "priority: a low prompt read paused at N of M ..." followed
   by "glm slots: a cancelled prompt keeps the N tokens it read" and a RESUME at that N.
