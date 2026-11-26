# FIX-REPORT - REVIEW-RESULT findings, one commit each (`ab3817c5..2d658bf6f`)

All seven findings fixed (two FIX-FIRST, five nits - all were cheap enough to take). Nothing here was testable
beyond compilation on this box: the touched paths need a card + resident tier, so the proof remains
`scratch/card-test-elastic-kv.sh` on mortar, plus the named unit test below.

## 1. HIGH - give-way never unlocked the mlock'ed tail next to a registered prefix -> FIXED `f1bf7480b`

`relieve_ram` (expert_source.cpp): the unlock condition dropped `!complement_registered_ &&` - the lock is
undone whenever `complement_locked_ > 0` and the slot's page-run overlaps `[lock_off, lock_off + locked)`, in
every backing mode. The overlap math already clamped to the lock range, and `keep_below = pin_limit = lock_off`
in the registered case means trimmed slots sit exactly on the mlock'ed tail; the prefix stays protected because
its slots are skipped, not because the tail was never unlocked. The comment records why: the kernel's zap skips
locked pages and `madvise` still answers 0, so without the unlock the give-way claimed bytes it never freed.

## 2. HIGH/MEDIUM - give-way raced concurrent CPU readers -> FIXED `93072d283`

Took the review's second option (defer the advise past a quiet window), made synchronous so the grow's retry
still sees freed RAM (a deferred queue drained "later" would return 0 or a lie - the grow retries once and
would park the conversation). `relieve_ram` is two-phase now:

* phase 1: `complement_offsets_[i]` flips to `kNoComplement` via a **release** `std::atomic_ref` store (each
  lock-free reader races it and sees the old offset or the trim, never a torn middle) and the working-set lock
  is undone - unlocking does not drop bytes, so readers that latched a blob address keep reading whole data;
* the grace window: `SAGUARO_KV_RELIEF_GRACE_MS`, default 250 ms, one wait per give-way. The relief handler
  synced every device, so the card is quiet; what remains is CPU-lane use of a latched blob (a copy into
  staging, or the compute of the tokens routed through it - milliseconds), which finishes inside the window;
* phase 2: `MADV_DONTNEED` on the queued page-runs. `gave` now counts exactly the bytes discarded.

`complement_locked_`'s decrement goes through an `atomic_ref` store with a note: relief calls are serialized
under layer's `g_host_mu`, the other accesses are load-time (`locked_bytes()` in the resident log line) and
`close()` at teardown, so there is no writer-writer race - the atomic is for the formal read-write race.
Function header, the `ExpertSource::relieve_ram` doc in the hpp, and REPORT step 2+3 item 4 updated; REPORT
open question 4 rewritten - it no longer claims the old "never wrong data" reasoning, it states the two-phase
protocol (a leftover prefetch of bytes just given back stays wasted work, never wrong data).

Not done: a per-complement rw/epoch. It would need reader cooperation around every site that returns a blob
pointer across the source boundary (`blob`, `blob_stable`, `blob_parts`, `resident_blob` into generate.cpp and
the kernels) - not a review-fix-sized change; the grace window is the documented, bounded alternative.

## 3. MEDIUM - prefix remap rewrites live `host` pointers, rule unenforced -> FIXED (comments) `305148753`

As the review asked: `host_prefix_reregister_locked`'s doc now states the ownership rule (the caller is the
thread that launches this state's copies - its session's admit thread; the device sync drains GPU queues but
cannot stop a CPU thread that latched an old base, and the mistake is silent in exactly this mode), and both
call sites (`qsa_host_grow`, `qsa_host_trim`, prefix branches) say it at the spot. No code change - adding a
per-state lock that covers grow + every copy launch is a redesign the review did not ask for.

## 4. LOW - stale `p.backed` on the `pages <= 0` early-out -> FIXED `146ae7e85`

The early-out now sets `p.backed = 0` before nulling the pointers, with the booby-trap named in the comment.

## 5. LOW - trim ignored a failed re-registration -> FIXED `1d980758f`

`qsa_host_trim` (prefix branch) checks `ok` and prints one stderr line when the smaller prefix was refused and
the fallback left the copy on more pages (or none) than the park needs, with the pinned total. The returned
byte count was already honest (`before - after`).

## 6. LOW - "backs N cells now" printed on trims too -> FIXED `ebee8f2c7`

`host_prefix_reregister_locked` gained `bool announce = true`; the grow calls keep it, the trim call passes
`false`. The line's wording is unchanged, so the card script's growth-step grep still matches grows only.

## 7. LOW - probe checked only the current device; silent mode switch -> FIXED `2d658bf6f`

The proof-probe now runs per device under `OnDevice` (restores the current device), so a second card answering
the probe differently gets prefix mode instead of pieces mode that fails every grow on it. And the
attribute-passes-but-probe-fails path prints its own stderr line (keeping the "grow as one registered prefix
per array (P124e)" wording the card script greps for; the attribute-failure line is unchanged).

## Not taken from the review

* Nothing declined. Finding 3's "nothing enforces it" is documented, not enforced - enforcement (per-state
  capture lock across grow and all copy launches) exceeds the finding's ask ("worth a comment at the call
  sites"); the review itself rates the window closed in the current serve code.

## Verification

* `ninja -C scratch/build-p124e` -> all targets rebuilt OK (30 dirty), `saguaro-qfn` links; no CUDA changes,
  both edited sources compile in `strata_engine`.
* `ctest --test-dir scratch/build-p124e -R '^kv_host_growth_test$'` -> 100% passed, run alone (binary:
  "22 checks OK"). No other test module exists for these paths; the relief/remap code needs a card, which the
  plan confines to `scratch/card-test-elastic-kv.sh` on mortar.
* `serve/qfn_engine.py` / `engines/qfn/serve/server.py` untouched (no flag plumbing added).
