# REVIEW-RESULT - P124e elastic host K/V (`git diff 88809c4e..397d1f852`, third round)

MERGE-WITH-NITS

All nine round-2 findings are genuinely fixed and verified below (the probe now decides pieces mode whatever the
attribute says, the prefix trim releases what it unpins, `complement_mu_` excludes the `complement_offsets_` writers
from the give-way, a non-page-aligned arena is declined, prefix-mode device bases stay off the threaded copy, the
pieces line carries ms, both pinning-failure lines name the CUDA refusal, `gave` counts discarded bytes, and the
stall is named in the report). Nothing new blocks: one MEDIUM is a lock the give-way holds longer than the fix
report claims (a 250 ms decode-commit stall per relief, correctness unaffected), three LOWs are rare-path issues in
the prefix-mode registration fallback. Ken's bar, the 68 GiB budget, the speed parity and the 300K conversation
across the YaRN line remain mortar's to settle with `scratch/card-test-elastic-kv.sh`.

Verified on this box: `ninja -C scratch/build-p124e` = "no work to do" against HEAD `397d1f852` (so HEAD compiles
and links, CUDA sm_75, ggml from `scratch/ggml-pinned`); `ctest --test-dir scratch/build-p124e -R
'^kv_host_growth_test$'` run alone -> 100% passed (binary prints "22 checks OK"). No other test module was run; no
GPU here.

## Findings

### 1. MEDIUM - the give-way holds `complement_mu_` across the grace sleep and the madvise, stalling every decode-path
###     offset writer - the comment and the FIX-REPORT say the opposite (expert_source.cpp:3499 vs 3551-3554; the
###     comment at 3494-3498 and FIX-REPORT section 3 both claim "the grace wait and the madvise stay OUTSIDE the
###     lock")
The `std::lock_guard<std::mutex> cmu_lk(complement_mu_)` at expert_source.cpp:3499 is function-scoped: the
`SAGUARO_KV_RELIEF_GRACE_MS` sleep (3551), the `MADV_DONTNEED` loop (3552-3554) and the give-way's log line (3556)
all run inside it. `commit_exchanges` (4417), `commit_flip` (4623) and `install_resident` (4668) - the decode
path's exchange commits and prefetch installs - block on that same mutex for the whole grace window (default
250 ms) plus the discard. Correctness is fine (a *longer* hold keeps the round-2 writer-race fix intact, and the
lock-order edge is one-way: `install_resident` takes `complement_mu_` -> `ForesightSwap::hold`, which takes only
`smu_`, expert_source.cpp:6623-6634, and nothing under `smu_` takes `complement_mu_`). But a give-way happens
exactly when RAM is short and a conversation is growing, and it then pauses the expert tier's own commits for a
quarter second - on top of the `g_host_mu` hold the report names (REPORT step 2+3 item 4). Fix: close the lock
scope after the phase-1 loop (line 3539) and sleep/madvise/log after it; `runs` and `gave` are already locals.

### 2. LOW - prefix re-registration leaks the registration of an array whose device alias was refused, and the
###     step-down fallback then re-registers the same address (layer.cpp:879-894 + 896-900; fallback 906-913 +
###     923-927)
In the primary loop `cudaHostRegister` may succeed (`e == cudaSuccess`) while `cudaHostGetDevicePointer` fails or
answers null (884-888); the `!whole` rollback at 896-900 unregisters only the arrays whose `nd[i]` was set, so the
refused array's run stays registered, uncounted in `g_kv_host_bytes`. The fallback's cleanup (923-927) has the same
hole, and the next `cudaHostRegister` at that address must fail - so the walk steps down to nothing, `p.backed = 0`
(929) reports the copy unbacked while the driver still holds pages, and every later grow of that state fails the
same way. The pieces path handles exactly this case (`if (e == cudaSuccess) cudaHostUnregister(a)`, layer.cpp:1007).
Realistically unreachable on drivers where a Mapped registration always answers an alias - which is precisely the
class of card that ends up in prefix mode - hence LOW, not MEDIUM. Fix: unregister `at` when `e == cudaSuccess` but
the combined test fails, in both loops.

### 3. LOW - a prefix TRIM whose re-registration fails re-registers the run it just discarded: `pg = kept = p.backed`
###     re-covers the `MADV_DONTNEED`-ed tail (layer.cpp:901-902 vs 857-866)
On the trim path (`pages < old_backed`) each array's given-back run is advised away at 866; if registering the
smaller prefix then fails, the fallback starts at `p.backed` = `old_backed` and re-registers that run - re-faulting
zero pages into RAM (so it also re-pins what it just freed) while the bytes past `keep` are now zeros. No caller
depends on those bytes (the trim contract: `solo_host_trim(solo_pin)` runs only after the outgoing conversation was
retained, generate.cpp:19987, and the live range `[0, keep)` is never madvised), and the trim then correctly
returns 0 with the warn line - but the fallback's first step should be `pages`, not `old_backed`, on the trim path
(`announce == false` already marks which path this is). Today the "room did not come back" log is honest by
accident; the re-pin round-trip and the zeroed tail are silent surprises.

### 4. LOW - a prefix grow that fails outright can leave a previously-backed copy with null `host` pointers
###     (layer.cpp:929-932, vs the pieces path's rollback-to-`had` at 1015-1024)
If every step-down re-registration fails - possible only when registering the *resident, already-populated* old run
fails, i.e. after relief already failed - `p.backed = 0` and `host_apply(p, p.d.data())` null every `st.host`
pointer of a running conversation; its next park then fails the snapshot's null check ("missing state buffer") even
though the pages themselves were never madvised (grow releases nothing) and a later grow would re-register them
fine. Pieces mode never drops below the old pinned prefix on a failed step. A one-time `ok = false` return keeping
the last full prefix (retrying the old size before stepping down) would make the failure state as benign; the
surviving engine behaviour (grow fails -> `ram_reclaim` parks) hides this today.

## Round-2 findings, fixed - verified here
1. `host_elastic_on()` probes whatever the attribute says: `bool proved = true;` and the per-device
   `probe_registered_page_address()` loop (layer.cpp:721-724), the two stderr variants still keyed on `same`,
   `g_host_prefix` only on a probe failure (734-735) - the report's "HIP passes the probe, gets pieces mode" claim
   is now true; feature OFF still returns at 704 before any device call, probe or new line.
2. The prefix trim madvises what it unpins: `[prefix(pages), prefix(old_backed))` per array after the unregister
   and before any re-registration, whole run on `pages <= 0` (layer.cpp:857-866, 871-875); grows madvise nothing.
3. `relieve_ram`'s snapshot and every phase-1 trim hold `complement_mu_` (expert_source.cpp:3499), which
   `commit_exchanges` (4417), `commit_flip` (4623) and `install_resident` (4668) now hold across their
   read-place/copy/flip; a trimmed slot answers `kNoComplement` to a writer and the exchange is skipped. (The
   grace sleep should be outside - finding 1.)
4. Non-page-aligned arena declined: `if (((uintptr_t) complement_host_ & (page - 1)) != 0) return 0;`
   (expert_source.cpp:3484).
5. Prefix-mode device bases stay off the threaded copy: `qsa_host_elastic_prefix()` (layer.cpp:1106-1112) feeds
   `host_end_is_cpu` (conversation_snapshot.cpp:142-144), threaded through `transfer`/`transfer_jobs` (`cpu_ends`,
   203-227) and applied in save/restore/restore_prefix/kv_source/verify (280, 383, 409, 443, 453); the parked
   image's own plain buffers keep the threaded fast path.
6. The pieces growth line carries the grow call's ms (layer.cpp:1019-1023) like the prefix line (940-944); the
   per-QSA-layer line count is restated in REPORT (step 2+3 item 3, open question 7) and the card script collapses
   them (scratch/card-test-elastic-kv.sh:130, 149-151 - scratch, uncommitted, as the plan words it).
7. The `g_host_mu` hold across the relief sync+grace is named in REPORT step 2+3 item 4 (finding 1 adds the
   `complement_mu_` hold the same sentence should carry).
8. Both pinning-failure lines append `cudaGetErrorString`: `why` in pieces (layer.cpp:1041), the `err` out-param of
   `host_prefix_reregister_locked` in prefix (847, 890-891, 981), cap wording kept.
9. `gave += hi - lo` - the discarded run bytes, not the blob (expert_source.cpp:3534-3537); REPORT's doubling/step
   wording restated as prefix-mode-only (REPORT lines 80-83, 96).

## Checked and clean
- Feature OFF byte-identical: `SAGUARO_KV_HOST_ELASTIC=0` returns at layer.cpp:704 before the attribute loop, the
  probe, the new lines and any pool; the pinned-whole `kv_host_alloc` path is untouched (HIP's "pinned whole"
  fallback therefore stays reachable via the env, and on Windows); both load lines print exactly the old text
  (empty `grow_note` since `qsa_kv_host_reserved_bytes()` stays 0; the moved "every session" block keeps `now ==
  all` on the pinned-whole path, so `solo_now` stays `kv_host_solo` and the elastic clause stays skipped,
  generate.cpp:6222-6237). The relief hook is installed at load (6728-6741) but can only fire from an elastic
  state's grow, and OFF states have `host_elastic < 0`.
- Memory follows USE at load, in code: one 32768-cell step (default `host_step_pages`, layer.cpp:752) per
  conversation - solo session + stages + its drafter (`host_follow`, session.cpp:189) pinned by `solo_host_fit(1)`
  before the manager's load line (generate.cpp:6217-6221), each batch slot at 5985, each slot drafter at 6187 -
  all before `pin_cache_complement` sizes the expert tier from the free RAM at 9021, so `--resident-budget-gib 68`
  is filled to the ceiling again; growth past the backed end doubles in prefix mode / pins the asked step in
  pieces mode, never per token (admit fits prompt + max_new; trims only at park/reset/end).
- No nested acquire of the non-reentrant `g_host_mu` (the relief handler at generate.cpp:6728-6738 syncs devices
  and calls `relieve_ram` without touching any layer lock; `host_relief` copies the `std::function` out under
  `g_host_relief_mu` and calls it unlocked); `g_kv_host_bytes` balances on every prefix path (`- had` then
  `+ got`/`+ rb`/nothing, and the trim's `before - after` return matches); `g_host_pools` stays push_back-only so
  `st.host_elastic` indices and `HostPool::owner` (states live in fixed arenas, session.cpp:117/145, mtp.cpp:396 -
  never moved) stay valid.
- Stdout/stderr: every new line is one `fprintf` ending in `\n`; the multi-part "every session" line is load-time
  and pre-existing in shape; the give-way line is a single `fprintf`.
- K/V stays INT8: no dtype or `kv_per_slot` change; `host_apply`'s field order matches the `qsa_state_init` carve
  (layer.cpp:823-831 vs 2003-2011); elastic YaRN untouched.
- Generated file respected: `serve/qfn_engine.py` and `engines/qfn/serve/server.py` are absent from `git diff
  --name-only 88809c4e..HEAD` (no flag plumbing, so no `extract_qfn_engine.py` re-run needed).
- Tests exercise the change: both modes run through `kv_host_growth.hpp` (`host_pieces` -> `kv_host_pieces`,
  grow -> `kv_host_pages_for`/`kv_host_grow_pages`, trim -> `kv_host_trim_pages`, prefix bytes ->
  `kv_host_prefix_bytes`); the 22-check test includes the acceptance line (start identical at 262144/1048576) and
  is registered beside `park_policy_test` under `STRATA_BUILD_CONVERSATION_TESTS` (engines/qfn/CMakeLists.txt:173-176).
- No absolute paths, no secrets in any changed file; the card script carries no host paths; `scratch/` is
  gitignored (.gitignore:20) and the report's scratch artifacts stay uncommitted.

## Not verifiable here (mortar, `scratch/card-test-elastic-kv.sh`)
Load-time pinned K/V equal at 262144 and 1048576 and near one step per conversation; the 68 GiB expert budget held
at load (and the read gains from it); 30K/200K read and decode equal between contexts and against the pre-change
binary; one conversation grown to 300K staying coherent across the YaRN switch while memory grows only as it goes;
what a prefix remap costs per step on a card that truly needs prefix mode - that run must include a streamed
park/restore (REPORT open question 1); and a `SAGUARO_KV_HOST_ELASTIC=0` load to confirm the pinned-whole fallback
still matches today byte for byte.
