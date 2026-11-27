# Code review round 8 (local agent, read-only) - `88809c4ea..0bc528f51` vs PLAN/REPORT + the round-7 card

VERDICT: MERGE-WITH-NITS

Round 7 (`e758a5b07`..`12dc848fb` + report `0bc528f51`) fixes the card's YaRN-switch failure correctly as far as
this GPU-less box can prove: the give-way-before-map order now runs for BOTH rope allocations, the rows follow the
positions in use, the tests assert the order per room path, and the serve wording matches the generator. No
correctness bug, lock inversion or OFF-path break found in the diff; the three findings below are LOW. The card's
own proof of criterion G (the 546K request answering, the rows climbing step by step) still belongs to mortar -
this box has no GPU and the plan forbids GPU runs here.

## Findings (severity order)

1. LOW - a rope grow uses its captured `RopeGrow*` after releasing `g_rope_mu` (measure at layer.cpp:2151/2230,
   map at 2174/2254 under a re-taken lock). `qsa_rope_release` (layer.cpp:2195-2211, via `session_release`,
   session.cpp:347-351) erases the `g_rope_grow` entry under the same lock, so a concurrent release would leave
   the grow's `r` dangling at 2177/2257. Not reachable as the tree stands: `session_release` has no caller
   anywhere in engines/qfn. A concurrent grow-vs-grow or grow-vs-`qsa_rope_yarn_trim` is benign (rope_map is
   monotonic and no-ops when covered; trim can only cause a harmless re-back). Suggest a comment stating the
   lifetime rule (tables outlive every grow) rather than code.

2. LOW - `g_dev_failed` (layer.cpp:692) is a process-wide last-writer-wins flag and round 7 widened its writers
   (rope/YaRN refusals and SUCCESSES now store, layer.cpp:2170/2180/2184, 2250/2260/2264 - a successful rope
   grow stores false). Its readers choose the RAM-vs-VRAM wording seconds later (generate.cpp:19529, 22496,
   14363, 14410), so with two conversations growing at once a VRAM refusal can still be answered with the RAM
   message. Pre-existing class (round 6's follow grow wrote it too); wording only, data unaffected. A per-thread
   flag would close it.

3. NIT - when the rope map consumes fewer chunks than measured (a concurrent grow mapped first), the leftover
   chunks are `vmm_chunk_free`d unconditionally (layer.cpp:2183, 2263), including budget-sourced ones when the
   map took fewer than `from_budget` - permanently shrinking the slot budget (it only refills through `give()`).
   The ROOM-failure path gets this right (give back [0,from_budget), free the rest - 2168-2169, 2248-2249);
   the map-short path should split the same way.

4. INFO (behaviour note, not a defect) - the yarn-fit change alters a rig no card covers: with
   `SAGUARO_KV_VRAM_ELASTIC=0` + `SAGUARO_YARN_ELASTIC=1` + `--batch`, a budget-short YaRN grow now fills from
   the driver (and asks a relief that is not installed - correctly a no-op, layer.cpp:1450-1457) instead of
   failing as the round-6 build did. That is the card's requested routing ("EVERY growth allocation"), but the
   "byte-identical when OFF" promise for this round holds exactly with the elastic-KV features off, not for the
   elastic-YaRN rows under a VRAM-elastic=0 rig. `SAGUARO_KV_HOST_ELASTIC=0` (pinned whole, layer.cpp:2648-2693)
   and `SAGUARO_ROPE_GROW=0` (whole carve, 2624-2626) are untouched.

5. INFO - pre-existing, unchanged this round: `slots_low_water` (generate.cpp:14073-14079) reads `bs[b].active/p`
   unsynchronized while slot/stage threads run - it is reached from the relief on any thread. Worth a one-line
   note or atomics someday, not a round-7 regression.

## Verified (file:line per claim)

- Round-7 scope, fix 1: `vmm_grow_room_pooled` takes the budget whole, runs the held-chunk order for the
  shortfall, asks the relief ONCE for `need - chunks made THIS call` - the zero-byte-ask bug the test caught is
  fixed and commented (vmm_grow.hpp:45-59, 70-76); the rope grow/yarn fit keep measure-room-map apart so NO lock
  is held across `vram_relief` (layer.cpp:2146-2174, 2225-2254), matching the follow pools (1726-1764). The
  budget helpers are leaf-lock (g_slot_mu, 1411-1478); relief -> `free_to_driver` -> `slots_low_water` ->
  `session_slot_need` -> `qsa_rope_need` takes g_rope_mu only with nothing else held
  (generate.cpp:14088-14094, 18722-18732; session.cpp:331). The only rope->slot ordering is
  `qsa_rope_yarn_trim` (g_rope_mu -> budget give, layer.cpp:2276-2291) and nothing takes g_rope_mu under
  g_slot_mu; no bs[b].mu exists at any rope call site (slot_yarn callers generate.cpp:14643, 15683).
  `qsa_slots_grow` remaps budget chunks only, all-or-nothing (layer.cpp:1548-1596) - the FIX-REPORT's table is
  accurate.
- Fix 2: `solo_yarn_rows` sizes `min(request end, at + kSoloPinAhead) + 64` over the positions reached
  (generate.cpp:21218-21219); request start caps the rows at `solo_pin` while the K/V state keeps the whole
  request (19525, 19541; session.cpp:297-317 caps `rope_cells <= cells`); piped requests pin the whole request,
  so the pipeline path is unchanged; the decode step keeps rows >= p+T+64 before the window runs
  (generate.cpp:22491-22503 -> session.cpp:254-273, rope step at 270); the crossing backs [0, p+32768) before
  the reply's windows (21238, 21263); slots trail the step-rounded target (14030) or the next window
  (14578-14587, 16809) with rope_map's +4096 margin. A prefix (not a window) is the right shape - a re-read
  re-rotates any cell below the conversation's length.
- Wording chain: refusals record `qsa_dev_pin_failed` (layer.cpp:2170/2180/2250/2260, false on success
  2184/2264), read by the solo admit (19529), the decode's end (22496) and the slot's BDONE (14363, 14410);
  `engine_refusal` maps "no VRAM for..." to EngineOutOfVram (engines/qfn/serve/server.py:457) -
  serve/qfn_engine.py re-extracted with tools/saguaro/extract_qfn_engine.py reproduces the committed file
  byte-for-byte (re-run in this review: zero diff; never hand-edited).
- Manager items: load start is ONE 32768-cell step at any --max-context (layer.cpp:766-774,
  kv_host_growth.hpp:44-51; the load's fits at generate.cpp:6244/6260 back cells=1 -> one step); the rope/YaRN
  tables back 0 beyond the trained 258048 at load (layer.cpp:2604-2623, 2711-2725) - identical at 262144 and
  1048576, matching the card's 1.30/1.30 GiB and 68/68 GiB; growth is real and doubling
  (kv_host_growth.hpp:44-51, card log: 50 backs-lines to 546148 cells) and given back on park/reset/end
  (session.cpp:275-287, yarn trim 2275-2295); K/V stays INT8 - kv_stream.cu only gained the reset bound and the
  ring fill, no dtype/layout line changed; the HIP pinned-whole fallback (SAGUARO_KV_HOST_ELASTIC=0) and the
  attribute-vs-probe gate are untouched this round (712-745, 2648-2693); elastic YaRN's LEVEL logic (yarn_move,
  yarn_thr crossing, tables) is unchanged - only the rows' backing moved; growth stays off the per-token path
  (watermark steps, fits between windows, relief's `may_relief` no-op at load).
- Tests: `ctest -R '^kv_host_growth_test$'` run ALONE -> passed (22 checks); `ctest -R '^vmm_grow_test$'` run
  ALONE -> passed (44 checks) in `scratch/build-p124e-r7` (ninja: no work to do - current HEAD); nothing else
  run. vmm_grow_test genuinely exercises round 7: the pooled cases assert budget->give->map event by event, the
  SHORTFALL ask after the budget, map-never-tried-without-room with `from_pool` for the give-back, and the
  zero-byte regression (vmm_grow_test.cpp:154-224); registered beside park_policy_test (CMakeLists.txt:178-180).
- Hygiene: no paths outside the repo and no secrets in the diff (only an interpreter path inside report prose);
  the growth/failure lines are single fprintf calls on stderr, the stdout protocol lines (ERR/DONE) stay on the
  serve thread; scratch/ (ggml export, build dirs, card script incl. criterion G) is gitignored.
