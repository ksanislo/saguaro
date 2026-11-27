# REPORT-p125b-ik2-vram - why does Inkling 2 Mini keep NO expert in VRAM at a larger --max-context?

Investigation only: NO code was changed. Engine = saguaro-ik2 (`src/ik2/ik2_main.cu` and its
includes `ik2_kernels.cu/.cuh`, `ik2_off.cuh`, `ik2_draft.cu`); the PC-side sizing model is
`tools/ik2_fit.py`; the context arrives through `serve/`.

ref/mortar-run-20261126.txt, RX 9070 XT (16 GB), Inkling 2 Mini T4-NVFP4-IQ3_XXS (3072 experts,
11.45 GiB of them - `tools/ik2_fit.py:52`):

| --max-context | path chosen at load                | GPU used |
|---------------|------------------------------------|----------|
| 32768         | resident: every expert in VRAM     | 15716.8 MiB |
| 262144        | offload: **0 of 3072** experts in VRAM (4.5 GiB pool) | 15406.6 MiB |

Same card, same weights, 8x context ceiling -> the expert tier goes from "all" to "none
preloaded" *before a single token is generated*, even though the conversation at load time uses
0 tokens. Both rows fill the card to ~15.4-15.7 GB: the K/V reservation that evicts the experts
then takes their space itself.

## 1. How the engine decides resident vs offload, and what the context reserves

### 1.1 How the configured context reaches the engine

- The model config names it directly, e.g. `configs/Inkling-2-Mini-UD-Q4_K_XL.json`
  (`"--max-context", "131072"`; and `SAGUARO_IK2_WARM=data/expert-profile-ik2mini.bin`, see 2.2).
- `serve/models.py:408-409` (ik2's `--ctx` is its `--max-context`) and `serve/models.py:540-541`
  keep the flag's name when rewriting configs.
- `serve/server.py:1452` (`StrataEngine`) starts `bin/saguaro-ik2` with those args verbatim
  (the exe check for `saguaro-ik2`/`maya-ik2` at `serve/server.py:1470`), reads the engine's
  `READY <ctx>` back at `serve/server.py:1515-1517`, and the dashboard's context slider rewrites
  `--max-context` in the spawn line and restarts the engine (`serve/server.py:1607-1616`).
  The ik2 start - context, cache, drafter - is deliberately identical with pictures on or off
  (`serve/server.py:2802-2809` comment in `vision_plan`).
- The engine parses it at `src/ik2/ik2_main.cu:7432` (`--ctx`/`--max-context` -> `ctx`);
  `READY %d` prints the ctx (or `ctx_s` per slot under --parallel) at `src/ik2/ik2_main.cu:10165-10166`.

### 1.2 The load-time plan (src/ik2/ik2_main.cu:7857-8071)

For one GPU (`if ((n_gpu == 1 || g_tpn > 1) && !g_pool)`, :7857) the engine sizes, **before
allocating anything**, four numbers:

- `exp_b` - the routed experts' bytes (every `*_exps.weight` tensor, :7858-7864; a Strata pack's
  packed size at :7876).
- `dense_b` - every other weight (:7863).
- `kv_b` - the K / V cache (:7865-7874): for a full-attention layer `cap = ctx`, for a sliding
  layer `cap = min(ctx, window + max(chunk, kBlock))` (:7866); a pass-multiplying core counts
  `c.n_pass` visits (:7873). **`cap` is the full --max-context** - the cache is planned as one
  buffer of `ctx` positions per full visit.
- `reserve` - 1.5 GiB for work buffers + headroom (`SAGUARO_IK2_RESERVE_MB` to shrink,
  :7902-7905).

The decision (:7906):

```
fits = exp_b + dense_b + kv_b + reserve <= fr0 && exp_b <= cache_gb     (:7906)
```

with `fr0` the card's free memory measured at that moment (:7879-7880). Where it does not fit,
auto KV streaming is *considered* (:7917-8001) but on NVIDIA/AMD desktop cards it is declined in
favour of the offload path: "a token reads the WHOLE cache but only the ~8 experts a layer it
picks: where the model and the cache do not fit together, the experts leave VRAM first"
(:7938-7943). Concretely: `off_ok` (:7944, true on the RX 9070 XT; APUs excepted :7945-7947)
forces `all_exp = false` (:7954-7955), keeps every full visit in VRAM if `rest + n_full*vis_b`
fits (:7957-7958), rows go to pinned RAM only where the cache alone would crowd out even a
half-GiB tier (:7964-7988; `fits = all_exp` at :7980 is reached only when streaming is taken).

Then the path flags (:8021-8034):

- `g_fast_want = fits && ...` - the resident path (:8022).
- `g_off_want = ... !fits ...` - the offload path: a VRAM tier + experts in pinned host RAM
  fetched by the GPU (:8024; the path itself: `src/ik2/ik2_off.cuh:1-7`, device-side staging
  caches with clock-hand eviction, "a token stays one CUDA graph").
- `g_off_dynamic` - the dynamic GPU expert cache, on by default on one GPU (:8031-8034).

The engine prints the plan at :8067-8071 ("every expert stays in VRAM" / "a tier of them in
VRAM, the rest fetched by the GPU"). After everything is loaded, the expert-cache budget is the
**left-over** VRAM minus a room (:9166-9173): 256 MiB room on the resident path, 1 GiB on the
offload path (`SAGUARO_IK2_ROOM_MB`), else 2 GiB:

```
budget = min(cache_gb, fr - room - g_lookup_room)                        (:9172-9173)
```

and `EC.init` turns it into pool slots (:9188-9190 -> `ExpertCache::init` :1180-1333: slot sizes
grouped per layer, uniform split by binary search :1262-1272, or profile-priced by an
`SAGUARO_IK2_WARM` profile :1284-1306; one `cudaMalloc` per pool at :1322).

### 1.3 What the context actually reserves, and how it scales

| item | allocated at | scales with --max-context | scales with --parallel |
|---|---|---|---|
| K/V of each full-attention visit | `:8411` (`kv_cap = ctx`), `dalloc` at `:8475-8481` | linear, one buffer up front | none: slots share it, slice `s*ctx_s*KD` (`:4871-4873`) |
| sliding layers' rings | `:8411` (`min(ctx, window+chunk)`), `:8475-8481` | grows only until `window + chunk`, then flat | one ring per slot (`:4866-4869`) -> xN |
| per-visit conv histories | `:8503-8504` | constant | xN (`:4875-4881`) |
| streamed visits' rows (when taken) | pinned host, `:8446-8452` | linear, host RAM | n/a (parallel off, `:10117-10126`) |
| tokens' copy ring + swing staging (streaming) | `:8466-8473`, sized `kKvRing*2*rows*KD*eb` at `:7932-7937` (`kKvRing=2` :542, `g_ring_tgt=16384` :545) | ~constant (ring target) | n/a |
| prompt path scratch (~256 KB a chunk token) | `Work`/`BigScratch`; a big pass is **lent slots of the expert cache** and given back (`:6816-6828`, `ExpertCache::lend` `:1404-1422`) | per chunk, not per ctx; auto chunk counts +640 KiB a token (`:8042-8066`) | sliding rings xN counted only in the chunk-upgrade path (`:8056`) |
| attention logits/probs scratch | counted `48*H*ctx` in the streaming plan (`:7926-7928`); `tools/ik2_fit.py:32,112` adds 1152 B a token past 262144 | linear past 262144 | - |
| drafter (when given room) | `draft_room(...ctx...)` `:8136`, measured `:9055-9163` | linear (`tools/ik2_fit.py:200-208`: 4096 B a token of FP16 context rows) | - |
| snapshots (serve) | `Snap` struct `:9591-9599`, 8 by default (`:9608`) | host RAM only (windowed rows, histories) - **no VRAM** | per conversation's slot (`Snap.slot` `:9593`) |
| flat work reserve | `:7902-7905` | constant 1.5 GiB | - |

`--parallel N` is only taken on the resident path with >= 4096 positions per conversation
(`:10117-10126`); full-visit KV stays one ctx-sized buffer split into N slices, so
--parallel's real cost is the per-slot sliding rings + conv state, reported after the fact as
`slot_extra` (`slots_init` `:4854-4887`, printed `:10130`). Note the plan at `:7873` does not
multiply those extra rings by N on the f16/fp8 KV path (only the q4 path multiplies sliding at
`:7871` and the chunk-upgrade check at `:8056` does) - they eat the post-cache free memory
unplanned.

The PC-side model the setup/Models page use mirrors this: `tools/ik2_fit.py:88-90`
`need_gib = dense + experts + ctx * 32768 B` (f16 KV; 8 full visits, `:27,:31`), and `plan()`
`:93-127` decides resident/offload the same way: bigger ctx -> offload, cache stays in VRAM.

### 1.4 The two mortar rows, mapped to the code

- **ctx 32768**: `fits` true -> `g_fast_want`; every expert copied into per-expert slots at load
  (`:9520-9528`) and the log line "resident: every expert in VRAM (3072 copied in 7.2 s)"
  (`:9545-9546`).
- **ctx 262144**: full-visit KV needs ~8x (f16, 8 visits: 1 GiB -> 8 GiB,
  `tools/ik2_fit.py:27`); `fits` false (`:7906`); auto streaming declined because the offload
  path runs (`:7954-7958`); KV stays fully in VRAM, experts become host RAM: `g_off_want`
  (`:8024`), the whole non-tier expert set pinned at load (`:9250-9361`), pool budget = free
  after weights+KV minus 1 GiB = 4.5 GiB (`:9172-9173`). "offload: 0 of 3072 experts in VRAM
  (4.5 GiB)" is printed at `:9479-9486` - `n_tier` counts experts *placed* at that moment, the
  4.5 GiB is the *allocated* pool.

## 2. Why the decision is all-or-nothing (0 of 3072) rather than "as many as fit"

Four separate reasons, in order of effect:

1. **The path choice is one boolean over the whole expert set.** `fits` (:7906) tests
   `exp_b + dense_b + kv_b + reserve <= fr0` with `kv_b` computed at the **full** --max-context
   (:7865-7874, committed at `:8475-8481`). There is no third mode between "every expert
   resident" and "offload": `g_fast_want` (:8022) requires *all* of them (the resident kernels
   index experts through a per-layer VRAM pointer table `Ly.tab_blob`, filled only when every
   expert has a slot - `:9533-9541`, `init_packed`'s "every expert has a slot - the resident
   check" :1159; the host-side expert is only reachable through the offload path's pinned rows +
   staging rings, `ik2_off.cuh:1-7`, `g_offs.host` :9364-9372). So a context *ceiling* alone
   flips the card from all-experts to no-experts.
2. **The tier is loaded from a popularity profile or not at all.** At load the tier gets experts
   only from `SAGUARO_IK2_WARM` (a picks profile: pools priced by picks-per-byte :1284-1306,
   cache filled most-picked-first :9211-9248) or `SAGUARO_IK2_EP` (:9199, :9208). With neither -
   the mortar run's T4-NVFP4-IQ3_XXS config carries no warm file - nothing is copied: "warm: the
   cache starts empty" path (:9198-9210) and `n_tier = 0` (:9479-9480). The 4.5 GiB tier then
   fills itself **at runtime**: the dynamic device cache (`g_off_dynamic`, :8031-8034) carries
   `loc/owner/used/freq/evict_seq/hand` per slot with max_freq=7 and a 2x ghost window
   (:9391-9427), and each token's graph computes on-device which picks are tier hits and which
   need a staging fetch (:2273-2310, `cls_dyn` :2304, :4194-4195). So "0 of 3072" is the load-time
   count of a self-learning tier, not "nothing fits" - but every cold expert still costs a host
   fetch until the cache learns, and the whole run uses the offload path's launches.
3. **The tier is only ever leftover VRAM, and context is reserved first.** Budget =
   free-after-everything-minus-room (:9166-9173): raising --max-context adds ~8 GiB of
   never-committed K/V (:8475-8481) before `EC.init` even sees the card, shrinking the tier from
   "everything" to 4.5 GiB. The order of preferences - cache in VRAM, experts out first - is an
   explicit design rule ("the experts leave VRAM first", :7938-7943), sound for a conversation
   that *uses* the context, wrong for one that merely *allows* it (Ken's rule).
4. **Why not "keep as many as fit" on the fast path** (rather than offload with an empty tier):
   the resident kernels have no notion of a host-resident expert (reason 1), and the offload
   path measures slower than the resident one on this very card - 71 vs 83 tok/s decode,
   `tools/ik2_fit.py:51` (`MEASURED_AMD`). Nothing in the engine can express "2600 experts
   resident, 472 fetched": residency is either universal (`tab_blob`) or per-slot cache state
   (`dcache`) reached only through the offload launches. (The fixed-tier offload mode
   `SAGUARO_IK2_OFF_DYNAMIC=0` with a warm profile - the UD-Q4_K_XL config's
   `data/expert-profile-ik2mini.bin` - is the closest existing "as many as fit", but it needs a
   shipped profile and still pays the offload path's per-token fetch bookkeeping.)

## 3. What it would take

Goal per Ken's rule: the context ceiling must cost ~0 until used; and a card that *almost* fits
all experts should keep the many that fit and demote the coldest only when a conversation
actually grows. Reusable machinery already in the tree: the streaming split (`--kv-streaming`,
`--kv-resident` :7450-7457, host rows + ring + swing :524-545, :7917-8001, :8446-8474), the
q4 KV's "partly in VRAM, part in pinned RAM" single cache (:8424-8427 - the existing precedent
for a growable K/V), the device-side frequency/ghost eviction (:9391-9427, `alloc_slot`
:1102-1136, freq decay :1048-1051), the warm fill loop (:9211-9230), `init_packed`'s per-expert
slots (:1140-1178), and the prompt path's slot-lending (:1404-1422, :6816-6828).

Parts, smallest first. (h) = focused engineering hours incl. its own test; shared VMM plumbing
(a2+b3) is counted once.

1. **(b1) Preload the offload tier at load - 2-3 h.** Where `g_off_want` and no warm profile
   exists, copy experts into the pool until its slots are full (the resident loop :9520-9528
   already does exactly this copy; interleave layers within a pool so every layer keeps its
   share instead of the first layers eating it all). Turns the 262144 row into "~1200 of 3072
   in VRAM" at no extra risk; also a 1-line answer for configs that ship no profile.
2. **(a1) Split "ceiling" from "committed" in the plan - 4-6 h.** `fits` (:7906) and `plan()`
   (`tools/ik2_fit.py:93-127`) stop counting full `ctx` and count a committed floor
   (`min(ctx, SAGUARO_IK2_KV_START)`, default 32768 - the ceiling-only 262144 row then plans
   resident). Positions above the commitment stay unwritten (attention reads `[0, pos)`, and
   "positions rewrite cleanly", :9590), so nothing else has to know yet.
3. **(b2) Make the path decision a tier fraction, not a boolean - 6-8 h.** With b1+a1: when the
   tier covers >= ~85% of the experts, run the offload-dynamic path pre-filled to the brim
   (near-resident hit rate with offload launches); keep the resident path only when all fit.
   Mostly measurement work: KL/speed of the hybrid vs both ends (the draft defaults at
   :8072-8111 keyed on the boolean would need a third arm).
4. **(a3) Grow the committed K/V on demand - 6-8 h.** Before writing position `p`, commit the
   chunk holding it (host-side check per request; the buffer must first be VMM-backed - a2).
   On allocation failure: demote experts first (b4), else fail the request as `cu_fail` does
   today (:4804-4818) with a NOTE. CUDA graphs are unaffected (K/V base pointers and `kv_cap`
   are bound once and VA-stable under a2; kernels only touch pages below `pos`).
5. **(a4) Give committed K/V back - 6-8 h.** On session end / `reset()` (:10107) / snapshot
   rewind, unmap every chunk above the live high-water mark across slots (attention never
   reads above `pos`, and snapshots hold the rewound state in host RAM :9591-9599). Return
   chunks to a free-chunk list; stats line beside the `VRAM dev=...` map (:10159-10161).
6. **(a2) VMM-backed K/V - 12-16 h.** Per full visit: `cuMemAddressReserve` the full `kv_cap`
   VA, `cuMemCreate`+`cuMemMap` committed chunks (e.g. 4096 positions; fresh physical pages map
   zeroed, which is exactly the DEPTH test's semantics :8482-8501). Touch points: alloc
   :8475-8481, streaming/swing exclusion :8446-8474, `bind_slot` slices (:4871-4873 - stay
   valid because VA is contiguous), `slots_init` rings (:4866-4869), the q4 split (:8429-8443)
   and HIP support (open question). Same chunk granularity serves a4/b3/a3.
7. **(b3) Chunked expert pools - 12-16 h.** `Pool.data` is one `cudaMalloc` (:1166, :1322), so a
   slot's VRAM is not returnable while any slot of the pool is wanted. Back pools with the same
   VMM chunks (VA = `n_slots*slot_bytes` reserved up front) so `EC.where`/`get`'s flat
   `base + s*slot_bytes` (:1362, :1381) and the device `OffCache` base (:9403) stay untouched;
   track which chunks are physically mapped.
8. **(b4) Demote the coldest experts when the context grows - 12-16 h.** When a3 must commit and
   free VRAM is short: pick pool chunks holding no slot with `freq == max_freq` and no in-flight
   lazy copy (`P.sseq` vs `g_seq_done`, the same tests as `evict_from_main` :1057-1083), sync,
   `cuMemUnmap`, mark their experts `loc = -1` so the next pick is a miss. A demoted expert needs
   a host copy: on the resident path the files are mmap'd by default (`host_mmap` :8564), so
   either page-register the needed runs (the offload path's `cudaHostRegister` machinery
   :9278-9334) or pin them up front like offload does (:9250-9361) and skip the
   `MADV_DONTNEED` release (:9500-9517, off by default except Pascal, :8566-8573). Victim order:
   host-side walk of `owner/freq/used` (mirrors `alloc_slot` :1102-1136); after demotion the
   expert is reachable through the existing staging fetch - the token stays one CUDA graph
   (`ik2_off.cuh:1-7`).
9. **(e2e) Tests - 6-8 h.** The mortar `mem_vs_ctx` check becomes the acceptance test (same GPU
   used at 32768 and 262144 ceilings until a long prompt runs); parity of a grown/shrunk session
   against full-ctx from load; grow-under-pressure and demote/re-fetch soak.

Totals: (a) alone ~34-44 h (~5 dev-days), (b) alone ~34-45 h; done together ~55-70 h since a2
carries b3/b4 and a1/b2 share the planner. The smallest useful step is b1 + a1 (~1 day): the
262144 row becomes "resident, ~everything fits, a self-learning tier for the rest" without
touching any kernel.

## 4. Open questions

1. **HIP VMM on gfx1201** (ROCm `hipMemCreate/hipMemMap/hipMemSetAccess`): assumed for a2/b3.
   If unreliable on the target stack, the fallback is per-chunk `cudaMalloc` + a device-side
   chunk->base table, which needs the expert/attention kernels' flat slot and row addressing
   changed (`OffCache`/`OffStage` in `ik2_off.cuh`, the `tab_blob` reads) - add ~1-2 days.
2. **Demoted experts' host copies**: register-then-pin on first miss (cheap RAM, first fetch
   slower) vs pin-everything-up-front (the offload path's ~11.5 GiB RSS, mortar row shows RSS
   13.6 GB) - which does Ken want as the default? A Strata pack has the same question answered
   differently (`pin_later = false` :8597, file-backed sources).
3. **The dashboard's context slider** (`serve/server.py:1607-1616`) restarts the engine today.
   Once ctx is a VA ceiling, should it become live (raise the ceiling without a reload, engine
   protocol needs a new line next to `READY %d` :10165-10166), or stay a restart?
4. **Streaming interaction**: a4's give-back and the existing auto-streaming split (:7917-8001)
   both decide where rows live, but the split is fixed at load (`g_kv_host`). Should a
   conversation that grew into streamed territory migrate rows to host mid-session (the swing
   machinery at :1867, :2078-2086 already moves a visit's rows between RAM and VRAM around
   prompts - a possible precedent), or is demoting experts (b4) always the cheaper first answer?
5. **Shipping a warm profile** for T4-NVFP4-IQ3_XXS (the UD file has
   `data/expert-profile-ik2mini.bin`) - fixes only symptom 2.2, but is a zero-code mitigation
   today (`SAGUARO_IK2_WARM`).
6. **--parallel accounting**: the f16 plan (:7873) does not count the per-slot sliding rings that
   `slots_init` (:4866-4869) allocates; harmless today (they fit in the reserve), but with a4's
   chunk accounting it should move into the same bookkeeping.
