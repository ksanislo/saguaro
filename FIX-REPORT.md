# FIX-REPORT - P125g review findings -> what changed

Response to ./REVIEW-RESULT.md (the review of `d210adc7..HEAD` against PLAN-p125g-glm-host-cache).
Every FIX-FIRST finding (1-8) is addressed below. This session resumed an interrupted fix pass:
findings 1-6 and 8 were landed and committed earlier on the branch; this session re-verified each
against the tree (not the commit message - the review's own "Checked clean" warns a report can claim
what the code does not do), completed the one finding left open (7), and ran the plan's named module.

The only test module the plan names is the granule arena C++ test; it was rebuilt and run here (GPU
modules are out of the plan's scope and were not run). `granule_arena_test`: **46 checks, exit 0**.

## FIX-FIRST

### 1 (Critical) - slot_put gave a still-bound conversation's host copy back

**Fixed** (`5b4ebedce`, verified in `src/program/generate.cpp:1614`). `slot_put` takes a `detaches`
flag and calls `model.kv_pinned_release()` only when it is set. The three continuing parks the review
named now keep their pins (default `detaches = false`): the mid-read `slot_prefix` park
(generate.cpp:1869, reached from the sys_end / divergence sites and the multi path at :3594/:3596),
the checkpoint continuation (generate.cpp:2012, which then `ckpt_restore`s / `slot_take`s onto the
same host copy), and the paused request's park (generate.cpp:2354). Only the detaching sites give the
granules back - generate.cpp:2040 (an overwriting `slot_take`/reset follows), :3646 and :3689 (each
followed by `seq_release`), and shutdown at :5411/:5415. The `kv_pinned_release` docstring
(include/maya/core/glm_model.hpp:784-792) now states the real contract (call only from a park that
detaches; a continuing park must not), which is what the code does.

### 2 (Major) - mem_vs_ctx no longer saw the GLM declared pin

**Fixed** (`906ac5981`, verified). The lazy load line prints
`... %llu MiB of pinned RAM at start, up to %.2f GB` (src/core/glm_model.cu:1018-1023); the
whole-cache fallback prints `... %.2f GB of pinned RAM at the load, the whole cache` (glm_model.cu:
1013-1017). `PINNED_DECLARED_RES` is now a pair (tools/mem_vs_ctx.py:95-100): the original
`Gi?B of pinned RAM` pattern (which the fallback's whole-cache figure and the pre-P125g / QFN lines
match) plus `pinned latent cache grows with use: .*up to ([\d.]+) GB` for the lazy line's ceiling.
`nums()` (mem_vs_ctx.py:329-330) already iterates a tuple of patterns. The fixtures were moved onto
the new line shapes and the GLM test asserts 2.95 from the lazy line despite its "0 MiB at start"
(tools/test_mem_vs_ctx.py:266, 300-307). The module was not run here (the plan does not name it), but
both patterns were traced against both real line shapes.

### 3 (Major) - seq_release leaked the CUDA device on HIP

**Fixed** (`379f59910`, verified in `src/core/glm_multi.cu:637-674`). The thread's device is saved at
entry (`cudaGetDevice(&cur)` at :637-638) and restored at the very end (`cudaSetDevice(cur)` at :674),
outside the `#if !defined(MAYA_USE_HIP)` block - on every platform. The CUDA-only branch no longer
carries its own restore; the give-back loop (:639-646) runs everywhere. This is the same save/restore
`kv_pinned_release` does.

### 4 (Medium) - GranuleArena::close() unregisters nothing, contrary to its comment

**Fixed** (`ca2a6bff3`, verified). close() takes no `Pinner` and cannot unregister, so the comment now
states the real contract instead of the false one ("close() does NOT release first ... a caller that
still holds pins must call release(p) before close(), or those granules stay registered on a range about
to be unmapped"): include/maya/core/granule_arena.hpp:150-153. `kv_stream_free` does exactly that
release-then-close (src/core/glm_model.cu:1029-1033).

### 5 (Medium) - reset()'s legacy branch unpinned without checking the sync

**Fixed** (`f8326ef35`, verified in `src/core/glm_model.cu:859-865`). The legacy branch now gates
`kv_unpin(kv_)` on `cudaStreamSynchronize(0)` reporting success (`const bool synced = ...; if (synced)
kv_unpin(kv_);`), the same rule as the fast branch above it (:853, :858).

### 6 (Low) - a give-back that then failed left granules pinned (the cheap refusals)

**Fixed** (`2895c25dd`, verified). The pin moved after the cheap checks in all three sites: the file
`slot_load` registers granules only after its "every part ready" loop - the no-fast-path and snapshot
allocations refused nothing before pinning (src/core/glm_fast_path.cu:10036-10057); the RAM `slot_load`
registers each part's granules only after that part's fast_/header/layout checks
(glm_fast_path.cu:10346-10374), so a slot of another model refuses with no pins held; `mtpw_cache_rows`
refuses "more cache entries than the window holds" before pinning (src/core/glm_mtp_rows.cu:841-846).
The one remaining late failure (the snapshot allocation, after a part's pin) is the acknowledged,
self-healing case the review allowed - the rows were partly written there anyway.

### 7 (Low) - the arena's advertised thread-safety was untested; unsynchronized reads undocumented

**Fixed.** The unsynchronized `reserved_bytes()/granule_bytes()/base()/empty()` reads are documented
as benign-by-construction (written only by open()/close()) in `5fb0e9cba` (include/maya/core/
granule_arena.hpp:85-90). The "no test behind it" half - the only finding left when the prior session
was suspended - is closed here (`6899107f9`): `tests/granule_arena_test.cpp` runs pinning and reading
threads on one arena behind a spin barrier. The asserted outcome is interleaving-independent, so a
correctly locked arena pins every granule exactly once (8 granules -> 8 pins, no double-pin by a
racing ensure) and no reader ever sees a pinned figure past the arena's size; removing the mutex would
double-pin a granule and trip it. The recording fake's `pins` vector is only touched under the arena's
own lock, so it is serialized by it too. Threads::Threads is linked for the target (CMakeLists.txt:
344-348) as idle_gate_test does. Result: 46 checks, exit 0, stable over repeated runs.

### 8 (Low) - the eager-fallback load line contradicted itself

**Fixed** (`7b1ba583c`, verified). With no lazy arena, pin0 is the whole cache, so the fallback now
says "pinned latent cache: X GB of pinned RAM at the load, the whole cache (no lazy arena on this
device)" rather than "grows with use ... up to the same" (src/core/glm_model.cu:1013-1017); the
grows-with-use wording stays on the lazy line only. The card-test script reads either shape.

## Not changed

No finding in REVIEW-RESULT.md is disputed; all eight are addressed. The "Checked clean" items were
left untouched (they were already correct), and per the plan the GPU / serve / mem_vs_ctx modules were
not run.
