# FIX-REPORT - REVIEW-RESULT findings, one commit each (`318f5da92..761adcaa7`)

This is round 2 (`git diff 88809c4e..761adcaa7`, `REVIEW-RESULT.md` second round). The review's preamble
already confirms all **seven round-1 findings are genuinely fixed**; their round-1 detail lives in the prior
version of this file (git `318f5da92`). Round 2 opens nine new findings - two blocking HIGH on which mode a
card gets and what memory a park really returns, three MEDIUM in the give-way path and prefix mode, four LOW.
All nine are taken (the nits were cheap). Nothing beyond compilation was testable here (no GPU): the touched
paths need a card + resident tier, so the proof stays `scratch/card-test-elastic-kv.sh` on mortar plus the
named unit test.

## 1. HIGH - address-equality probe skipped on attribute-less cards, so mortar never got pieces mode -> FIXED `1002b5c78`

`host_elastic_on()` (`layer.cpp`): the probe loop was seeded `bool proved = same;`, so a card that fails
`cudaDevAttrCanUseHostPointerForRegisteredMem` (HIP) short-circuited the loop and forced `g_host_prefix = true`
- the probe could only ever *demote* attribute-true cards, never *promote* an attribute-less flat-VA card. Now
`bool proved = true;` and the loop runs the real `probe_registered_page_address()` per device whatever the
attribute says (`if (!probe_registered_page_address()) proved = false;`), keeping the two stderr variants keyed
on `same`. This is what makes the report's headline ("HIP fails the attribute but passes the probe, so mortar
gets pieces mode") true instead of aspirational; the acceptance table now measures the intended mode.

## 2. HIGH - prefix mode unpinned but never released the pages -> FIXED `2b8b49c60`

`host_prefix_reregister_locked` (`layer.cpp`) unregistered with `cudaHostUnregister` only; the range is
`MAP_NORESERVE`, so touched pages stay populated and a park reported RAM it kept holding (the pieces path
pairs every unregister with `MADV_DONTNEED` per piece). Now, when the call is giving memory back
(`pages < old_backed` - the trim, or the `pages <= 0` unback), each array's given-back run
`[prefix_bytes(pages), prefix_bytes(old_backed))` is `madvise(..., MADV_DONTNEED)`-ed after the unregister and
before any re-registration, whole registered run on the `pages <= 0` path. Grows (`pages >= old_backed`) madvise
nothing, and the `!whole` grow fallback runs no madvise either - so live data is never zeroed by this change;
only bytes a trim is discarding are. `qsa_host_trim`/`solo_host_trim` now report RAM the RSS actually drops.

## 3. MEDIUM - give-way still raced `complement_offsets_` writers (grace covered readers only) -> FIXED `398ced7ff`

Took the review's first option (a source-level mutex), the cheaper of the two it offered. Added
`complement_mu_` (`expert_source.hpp`); `relieve_ram`'s snapshot and every phase-1 offset trim hold it, and the
three writers - `commit_exchanges`, `commit_flip`, `install_resident` (the `ram_install`/prefetch path) - now
hold it from reading a slot's place through `exchange_cache_complement` flipping it. A slot relief trimmed
answers `kNoComplement` to a writer that read it earlier, so the incoming expert just stays file-backed instead
of naming a place phase 2 discards (the silent zeroed-weights class the round-1 finding 2 fixed for readers).
The grace `sleep` and the `madvise` stay OUTSIDE the lock, so relief never hands a sleeping thread to the decode
path's offset updates. Lock order: relief takes `g_host_mu` (grow) then `complement_mu_`; no offset writer
takes `g_host_mu`, so there is no cycle. (Declined the alternative "refuse relief while `!staged_.empty()`" - a
prefetch that passed its `!= kNoComplement` check tens of ms earlier is not covered by a `staged_`/in-flight
test, which is exactly the interleaving.)

## 4. MEDIUM - give-way's page arithmetic assumed a 4 KiB-aligned arena -> FIXED `8bd4b4744`

`relieve_ram` (`expert_source.cpp`): `lo`/`hi` are rounded relative to `complement_host_`, which is page-aligned
only on the `posix_memalign` arenas. On the pageable fallback (cudaHostAlloc refused, `std::malloc`,
`complement_pinned_ == complement_registered_ == false`, so neither early-out fired) every `madvise` returned
`EINVAL` (swallowed), the `munlock` rounding stopped matching `complement_locked_`, and `gave` was still returned
and printed. Added a bail: `if (((uintptr_t) complement_host_ & (page - 1)) != 0) return 0;` (the review's first
option - bail rather than round-and-skip partial pages). Page-aligned arenas stay eligible.

## 5. MEDIUM - prefix mode installed the device base where code reads host memory -> FIXED `9d621f1bb`

In prefix mode `cudaHostGetDevicePointer` answers with something other than the host address (that is the mode's
reason to exist) and `host_apply` writes that value into `QsaState::host`, but `host_both` in the park/restore
fast path (`conversation_snapshot.cpp:128-135`) counts an unrecognized pointer as host and would `std::memcpy`
the whole conversation through the GPU alias (harmless on flat-VA HIP where `d == a`, wrong on the WDDM/dxg
class). Took the review's safe rule via a narrow predicate instead of rewriting `host_both`: a new
`qsa_host_elastic_prefix()` (`layer.cpp`/`.hpp`) reports the mode; `conversation_snapshot` adds
`host_end_is_cpu(st)` and threads a `cpu_ends` flag through `transfer`/`transfer_jobs`, keeping a prefix-mode
state's pool ends on the driver copy (`copy_nonblocking`, which maps those bytes from both sides) while plain
host buffers - the parked image's own memory - keep the threaded fast path (the 11-19 park-stall fix is not
regressed). Applied at the four pool-copy sites (save, restore, restore_prefix, kv_source) and in `verify`.
`keep_host0`/prompt-cache keep-records already re-stage on a remap (address-keyed, degrades to re-staging).
The REPORT open question 1 now requires the real-hardware run to include a stream park/restore, not just a grow.

## 6. LOW - growth line printed once per state; pieces line had no timing -> FIXED `9d452f5c4` + `761adcaa7`

Took the cheap half. Added the elapsed ms to the pieces-mode "backs N cells now" line (`layer.cpp`) so it matches
the prefix line and the script's `(…, N ms)` criteria. Did NOT aggregate one line per grow *call* across a
session's states: a grow call is inherently per state (one `HostPool` per QSA layer), so aggregating is a plumbing
change across `session.cpp`/`generate.cpp` the review itself only "suggests". Instead the report (steps 2+3 item
3 and open question 7) and the card script are restated: the raw count is one line per QSA layer per boundary
crossing, and the script's growth block now collapses the lines to one per backed size.

## 7. LOW - give-way held `g_host_mu` across all-device syncs + the 250 ms grace -> FIXED (named) `761adcaa7`

The review offers "mention or avoid"; taking "mention" (avoiding means phase 2 outside the lock, a redesign
it rates unnecessary - "Relief is a rare already-slow path"). REPORT step 2+3 item 4 now states plainly that one
give-way holds layer's `g_host_mu` across that sync AND the grace window and that other threads' grows/trims and
the `HMEM` status read wait it out - named in the plan's "no mid-read stalls" rather than hidden.

## 8. LOW - pieces pinning failure lost its CUDA error string -> FIXED `c1f547f8d`

Kept the `SAGUARO_KV_HOST_MIB reached` cap wording and appended `cudaGetErrorString`: the pieces path captures
the refusing `cudaError_t` (`why`) and the prefix path gets an `err` out-param from
`host_prefix_reregister_locked` (set to whichever call refused, `cudaHostRegister` or `cudaHostGetDevicePointer`),
so a mortar log can tell a driver limit from an unregistered range from plain OOM.

## 9. LOW - report accuracy (doubling/gave wording not both modes) -> FIXED `ca9fe3e5f` + `761adcaa7`

* `relieve_ram`'s `gave` summed each trimmed slot's whole blob (`s.len`) even when its page-run was empty, so
  the give-way reported relief that freed nothing and the grow retried to a second failure - `gave` now counts
  only the discarded run bytes (`hi - lo`), matching the function header's "the bytes `gave` counts are the bytes
  discarded". (`ca9fe3e5f`)
* REPORT corrected (`761adcaa7`): step 2+3 item 3 - the doubling (`kv_host_grow_pages`) is a **prefix-mode**
  behaviour, pieces mode pins the asked step; the "handful of lines" wording is gone; the pieces line carries ms.
  The one-step park floor is left as-is (already report-documented open question 3, Ken's call, both modes agree).

## Not taken from the review

* Nothing declined. Finding 6's "log once per grow call (aggregate the session's states)" and finding 7's
  "phase 2 outside the lock" were the two changes the review itself offered as alternatives to the cheaper ones
  taken (restate the report / name the stall); each is a cross-file plumbing change or a redesign the review
  explicitly rates not-required for a rare, already-slow path, so the honest-cheap fix (report + script wording,
  adding the missing ms) was taken instead.

## Verification

* `ninja -C scratch/build-p124e` -> clean incremental rebuild from `318f5da92`, every edited TU
  (`layer.cpp`, `expert_source.cpp`, `conversation_snapshot.cpp`) compiles, `saguaro-qfn` links; the warning set
  is only the pre-existing HEAD ones (three `-Wmisleading-indentation`/two unused locals in `layer.cpp`, none in
  the other two files) - the `t0 set but used nowhere` warning the review's finding 6 implies even got fixed.
* `ctest --test-dir scratch/build-p124e -R '^kv_host_growth_test$'` -> 100% passed, **run alone** (the only test
  module the plan names). No other test module was run; no GPU here.
* `serve/qfn_engine.py` / `engines/qfn/serve/server.py` untouched (no flag plumbing added, so no
  `extract_qfn_engine.py` re-run needed); `scratch/` artifacts (the card script edits) stay uncommitted.
