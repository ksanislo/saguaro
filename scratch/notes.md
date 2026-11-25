# F1 step 1 - what the resident low-RAM (complement) layout assumes about VRAM slots

All paths are inside engines/qfn/. Line numbers from the tree at e6d840db2.

## The layout

`--resident-cpu-experts` builds the complement (`pin_cache_complement`, generate.cpp ~8923;
expert_source.cpp 3466-4124). `complement_offsets_[index]` is the source's ONLY per-expert placement
fact: an offset in the pinned RAM copy, or `kNoCacheComplement` ("not in RAM: the miss path reads it
from the GGUF/experts.bin"). The planner fills RAM for every expert the GPU cache does NOT hold at
startup, PLUS (dual-held, RAM and GPU) the prompt path's lend-region experts (`lend_from_slot` /
`stage_lend_regions`, counted by `complement_lent_slots_`). With `--resident-budget-gib` (CS-T,
1e663cc70) only the hottest ranked experts get a RAM place; the rest stay file-tier from the start -
"the rest stay on the mapped files (the SSD tier)" (expert_source.hpp ~739). `resident_blob()` /
`has_resident()` answer exactly `complement_offsets_` (`cache_complement_blob_or_fallback`, 237);
the source NEVER stores which experts are in VRAM slots.

## Structures that hold slot numbers / "in VRAM" facts

1. `host_res` (generate.cpp 8096): residency index -> cache slot / kNotResident, the host copy of the
   device's `d_res` (token-graph hit path, `thits.d_res`, per-stage `st->d_res`, `drive.d.host_res`
   for the pool's "leave the resident expert's row at zero"). `res_put`/`res_upload` (8100, 11384)
   upload it. The VRAM command already maintains it: shrink -> `vram_evicted` list + kNotResident +
   `res_upload()` (12817-12820); grow -> refills and `host_res[pick] = slot` (12825-12850).
2. The adaptive tier's swaps: blocking `adapt()` (11564) and async `a_pick` (11831) choose `out` from
   `host_res >= 0` (slot from `host_res[out]`) and `in` from `host_res < 0` with usage >= 2 - they
   never assume WHERE `in` lives: `resident_stage_swaps` (453) stages an exchange only when
   `has_resident(in) && !has_resident(out)`; `a_from` (11886) reads a non-RAM `in` through a bounce
   buffer with `copy_blob`. So a file-tier (or freshly evicted) expert is already a normal `in`.
3. `exchange_storage_` (expert_source.hpp 841, SAGUARO_EXCHANGE_ROTATE): the rotation ownership
   table standing in for the offset moves; `commit/commit_flip` call
   `detail::exchange_cache_complement` (301) / `exchange_storage_.commit`, which move RAM places
   between `in` and `out` and require `offsets[in] != kNoCacheComplement && offsets[out] == kNo`.
   They are keyed by RAM placement only - unaffected by a slot losing its expert.
4. `resident_lent_slots()` -> `complement_lent_slots_` (set at 4122): the number of lend-region
   slots whose experts the copy holds. Read at STARTUP only (prompt plan 8504, warnings 8954,
   9519). Nothing re-derives it at runtime; `relend()` (12760) re-lays the loan against
   `xcache.slots()` after a resize.
5. The miss path's residency query (expert_source.cpp 2107, 2140, 2166; also 1096, 1248, 2974,
   3301): every read/skip decision in `prefetch`/`claim_todo` is per expert,
   `complement_ready_ && resident_blob(index) != nullptr` -> RAM, otherwise the file (staged /
   mapped / io). There is no code that computes a non-VRAM expert "from RAM" without looking it up;
   `blob()` (4560-4601) is the same rule and counts `file_reads_` when it falls through. So an
   expert in NEITHER tier is already a well-defined state (the CS-T budget file tier), and there is
   nothing in the source to "tell" about an eviction: `complement_offsets_[evicted]` is already
   `kNoCacheComplement`, i.e. file-tier, the moment the expert leaves the cache.

## Why the refusal exists

`git log -S 'not with the resident low-RAM'`: introduced by 89bad9a4e (2026-10-03) - the #533 feature
commit ITSELF ("Hot VRAM resize, opt-in: ... VRAM <reserve>, POST /v1/vram"), vendored here by
f5ec61040 (prune engines/qfn). The commit message scopes it: "One NVIDIA GPU only (no layer split,
helper caches, --peer-device or the resident low-RAM mode); HIP refuses the flag." The layer-split /
helper-cache / peer exclusions became startup/`vram_command` guards (6953-6957, 12786). The
complement line is a scope guard for the untested combination: the mode's promise at the time was
"no decode or prompt step reads the file" (generate.cpp ~8795), and a shrink moves cache experts -
which the RAM copy deliberately does NOT hold - into nobody-holds-it, a state the mode was meant to
keep empty at steady state. CS-T had landed three days earlier (1e663cc70, `--resident-budget-gib`,
and `--resident-soft` #467 keeps "what fits", so a steady file tier inside the complement layout was
already a shipped, tested configuration), so the reads themselves are safe; what a shrink/grow adds
on top is only the question of WHICH readable tier the evicted expert falls to - file, or its
lend-region RAM copy - both of which `blob()` already answers per expert.

## Other facts checked for the change (step 2)

- `vram_command` (12773) already quiesces: `ver.wait_commit` + `adapt_quiesce()` (12123: device_sync,
  `apply_pending(true)` lands the blocking tier's `pending` swaps + `commit_exchanges`,
  `adapt_tick(true)` drains the async round), and refuses while a prompt's loan is out or batch
  slots decode. NOT drained: `lazy_fly` (SAGUARO_SHORT_READS lazy refills of lent slots, 11399) -
  entries set `host_res[i] = slot` in `lazy_land` (11417) whenever the copies land; a shrink between
  the request and the next land could resurrect kNotResident->stale slot >= live. Request end
  drains it (22495 "between requests the cache is whole again"), but the command thread must not
  depend on that: land it explicitly before shrinking.
- Freed slots are unreachable: a slot the eviction vacated is referenced by NO host_res entry, and
  both tiers only overwrite slots owned by a currently-resident expert, so the adaptive tier can
  never move into a shrunk-away slot; the grow path's "back already (the adaptive tier)" test
  (`host_res[i] >= 0`, meaning the evicted expert itself was swapped back into a live slot) keeps
  that true. `vram_evicted` carries (i, slot) across grows; entries whose segment is not remapped
  yet stay in `keep_out`.
- The grow refill reads `srcp->blob_stable(layer, pick)` (12837). In the complement layout that is:
  the RAM copy (exact file bytes) for evicted lend-region experts; otherwise the mapped/staged file
  read (exact file bytes), staged buffers protected by `note_async_read` + `sync_queued`. Where the
  source cannot hand out a stable pointer (Windows closed mapping / unbuffered reads, Linux
  SAGUARO_TEST_UNMAP_GGUF), `blob_stable` answers null and the grow must fall back to
  `srcp->copy_blob` into a bounce buffer instead of erroring half-refilled.
  NOTE: the existing loop updates `host_res[pick] = slot` entry-by-entry and `return false`s on the
  first failure BEFORE `res_upload()`, leaving host_res claiming slots the device table does not -
  fix by uploading what was queued before returning the error.
- `--kv-grow` (kvg) cannot coexist with `--vram-elastic` (5669), so the K/V slot dance cannot race
  the VRAM command. `--vram-elastic` itself refuses at startup with a layer split or helper caches
  (6953-6957); the command refuses `--peer-device` and open loans.
- Bit-identity: refill bytes are the pack's file bytes (RAM copy = file bytes copied at startup;
  swaps only move bytes between places, never rewrite them), the experts are unchanged, only WHERE
  a miss reads from -> answers stay bit-identical; same argument as the profile-driven cache fill.
