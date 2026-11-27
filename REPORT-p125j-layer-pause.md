# P125j REPORT - a QFN prompt read can pause between LAYERS of a chunk (Saguaro 1.2.5)

Redone from scratch: the original branch's work was destroyed (REVIEW-RESULT.md BLOCKER 2 - a reset wiped
it, nothing recoverable). All line numbers below are of this tree at the P125j base (d789228f0) unless a
commit says otherwise; they were re-read from the files, not carried over.

## Step 1 - the prompt chunk loop, where STOP is checked today, and what a chunk holds between two layers

### The chunk loop and the between-chunk STOP

- The batched prompt read is `strata::prefill::Prefill` (engines/qfn/include/strata/prefill/prefill.hpp:109).
  Its worker is `Prefill::run_impl` (engines/qfn/src/prefill/prefill.cpp:2900), entered through
  `Prefill::run` (prefill.cpp:6068).
- The prompt chunk loop: `for (int64_t c0 = 0; c0 < n; c0 += C)` - engines/qfn/src/prefill/prefill.cpp:3122,
  where `C` is the chunk the buffers hold (`m.T`, or the --prefill-pipe plan's chunk, prefill.cpp:2992-2995)
  and `T = min(C, n - c0)`, `p0 = pos0 + c0` (prefill.cpp:3133).
- Where a STOP is checked today: `if (stop_cb && stop_cb()) { err = "cancelled"; return false; }` -
  prefill.cpp:3124, i.e. only BEFORE every chunk, never inside one. `stop_cb` is `PromptJob::should_stop`
  for a queued job or `Prefill::should_stop` otherwise (prefill.cpp:2937); the member is documented
  "Checked before every chunk: true stops the prompt early" (prefill.hpp:340-341). The batched-slot paths
  use their own `should_stop` (prefill.cpp:2937, 6028) - unchanged here.
- The STOP itself: stdin is read on its own thread and a `STOP` line sets `stop_req`
  (engines/qfn/src/program/generate.cpp:12611-12613, `StdinShared` at 12560-12575); the flag is wired as
  `sp.should_stop = [&] { return stop_req.load(); };` (generate.cpp:12648). The comment at
  generate.cpp:12555-12556 says it as plainly as the plan does: "the flag is checked between prompt chunks
  and between verify windows".
- The server side of a cut read: `Service._sliced` (serve/server.py:5654) gives a low prompt read's place up
  on a read tick where `step_aside` says so and the engine keeps a stopped read (serve/server.py:5744-5758:
  `cut = True` -> `gen.close()` sends STOP and drains to the DONE), logs
  "prompt read paused at X of Y prompt tokens" (serve/server.py:5873-5877) and, on the resumed call, the
  engine's own reused figure (serve/server.py:5707-5715, P124b). The engine keeps the cut read at the
  #1620 keep point (generate.cpp:22732-22742) or lands it on its deepest checkpoint, P124b's
  `conv_cache::cut_read_landing` (generate.cpp:22743-22784, conv_cache.hpp:232).
- The solo read loop calls `read_part(at, to)` per part (generate.cpp:21489), and `read_part` runs the part
  through `sp.run(ids + q, r - q, q, e)` (the `run` lambda at generate.cpp:20494-20501; one run for the
  whole part when unplanned, generate.cpp:20928 `return run(a0, b0)`). A part's layers are all read in one
  run, so "chunk c" inside a run means `c0 = C * c` tokens into it.

### What a chunk holds between two layers (the state a pause must keep)

Inside a chunk the layers run `for (int64_t l = LB; l < LE; ++l)` (prefill.cpp:3543), two HC halves each
(`for (int half = 0; half < 2; ++half)`, prefill.cpp:3611). At the end of a layer pass - the host point of
the U5 heartbeat, prefill.cpp:5431-5440 - the chunk's state is:

1. The hidden states: the chunk's residual rows, `m.R`, T x D floats on the device with
   `D = N * HC = 2560 * 4` (constants at prefill.cpp:138, allocation at prefill.cpp:1736
   `m.R = o.take<float>(T * D, ok)`). Layer l's pass writes the rows layer l + 1 reads (gr_write /
   gr_write_norm_rs / gr_write_cvec_norm_rs, prefill.cpp:5390-5422). This is THE thing to save: 40 KiB a
   token, so 320 MiB for an 8192-token chunk ("tens of MB" for a smaller one).
   The normed scratch `m.grs`/`m.xn16` (the F-2 norm written with the rows) is derived from m.R and is
   recomputed (`normed` starts false at prefill.cpp:3528); the PLE row upload `m.ple_emb` is rebuilt from
   the tokens; the expert staging ring / KV staging slots are pure scratch refilled per run.
2. Per-layer K/V already written for the layers done: every layer of the chunk that has run wrote its
   QSA K/V cells for positions [p0, p0 + T) (the layer body writes cells then attends; the indexer's
   pooled/index rows and the block state the same way). Layers not yet run hold nothing of these tokens.
   This is exactly "K/V of layers < l for this chunk already written" - it is in the SESSION state, not
   in the prompt path's buffers, so a pause does not copy it anywhere.
3. The GDN / recurrent states for this chunk: every GDN layer's recurrence + conv history
   (per-state size `gdn_floats`, prefill.cpp:2977) - the layers that ran advanced theirs through the
   chunk; the layers that did not still hold the state at the chunk's start. Same asymmetry as the K/V,
   same reason it parks for free: it is session state. (ConvCheckpoint copies exactly this set:
   "the 36 GDN recurrences and conv histories, each QSA layer's indexer tail ... and the PLE's
   normalized history", generate.cpp:2131-2132.)
4. The PLE: the normalized history is advanced token by token in the layer-1 block only
   (prefill.cpp:3549-3610, `ple_history_advance` at 3607). Pausing at any layer >= 1 means it has already
   advanced through the chunk; a run going on at layer >= 2 must not - and structurally does not - touch
   it. Advancing it again over the SAME tokens is idempotent (the ring holds the last conv-depth rows),
   which is what makes a full chunk re-run (the fallback when a resume cannot happen) safe.
   The chunk's PLE rows themselves are a pure function of the tokens (`ple_gather`, prefill.cpp:3059-3082)
   - recomputed on resume, not saved.
5. MTP / drafter state: none in flight. The draft layer's K/V for a chunk is built from the chunk's
   FINAL rows in `on_chunk` after the whole layer loop (prefill.hpp:315-318, called at the chunk end,
   prefill.cpp ~5500), so a chunk paused mid-layers has contributed nothing to the drafter, and the
   resumed run calls `on_chunk` at its chunk end as any run does.
6. Not state, refills per run: the expert staging ring and its issuing thread (prefill.cpp:3436-3507 -
   the RAII `IssuerJoin`/`StagerDone` at 3432-3476 unwind it on any early return), the K/V prefix
   staging side stream and its keep records (`KvDrain` prefill.cpp:2969-2972; the keep's records go stale
   when the loan is refilled after a cancelled read, prefill.hpp:149-151), the next chunk's PLE gather
   thread (its `std::future` waits on any early return, prefill.cpp:3084).

### Why a layer pause is park-able with the existing machinery

After pausing at (chunk from token P, layer L) the session is "layers < L through P + T, layers >= L
through P" and the conversation's token boundary is P + T (the paused chunk counts as read for reuse:
its lower layers were). Parking copies the session whole (running state + K/V up to live.size() = P + T,
conversation_snapshot_save), so every partial-layer cell the model will read later is in the snapshot;
the restored conversation either goes on at layer L (saved rows re-uploaded) - layers >= L rewrite their
cells before anything reads them - or re-reads the chunk from layer 0, which rewrites every cell with the
same values (and re-advances the PLE history idempotently). The only thing the park format cannot carry
by itself is the saved rows - P125j adds exactly that to `SavedConversation` and moves them out of the
prompt path on a park and back on a restore. A conversation whose read is paused never goes to the
session-dir disk (its file could not hold the rows), and dropping the pause is always safe: the fallback
is P124b's landing or a full re-read.

## Step 2 - what was built (engine)

- `include/strata/program/layer_pause.hpp` (new, pure, unit-tested by `src/program/layer_pause_test.cpp`):
  `Point`/`pause_point()` - where a STOP taken inside a chunk pauses it (the chunk's first token, the first
  layer not yet run, the chunk's token count; {} after the stage's last layer, where the plain
  chunk-boundary stop already ends the chunk); `resumes()` - the saved pause goes on with a run whose
  shape matches (same grid, the chunk whole in it); `state_bytes()` - the saved rows' size (T rows of
  D FP32 floats, saturating), the figure the park estimate and the log line show.
- `Prefill` (prefill.hpp / prefill.cpp): `stop_between_layers` (checked at each layer pass of a SOLO run -
  no queued job, no batch, no layer split), `SAGUARO_LAYER_PAUSE` (default on; OFF takes the old path
  exactly), and the saved-pause state (`paused_chunk()/paused_layer()/paused_tokens()`,
  `take_pause_rows/give_pause/clear_pause`). On a pause `run_impl` finishes the layer, syncs the compute
  stream, copies m.R (T x D floats) into a host vector, ends the run with err "layer paused" and says so on
  stderr ("the prompt read paused at chunk c (token P), layer L of N: X MiB of chunk rows kept"). On the
  next run whose chunk grid matches, the chunk's rows go back up into m.R instead of the embeddings, the
  embedding/PLE-row gather for it is skipped (PLE rows rebuild from the tokens as usual), and the layer
  loop starts at the saved layer. `normed` stays false, so the first half recomputes its norms from m.R.
- `generate.cpp` (solo path): `sp.stop_between_layers` is wired to the same stop_req; a read that ends
  "layer paused" keeps the conversation LIVE through the paused chunk (its own keep branch, beside #1620's
  and P124b's - those stay untouched and still cover every other cancel), records
  `read_pause = {pos, layer, tokens, ids}` at engine scope with the rows in the Prefill, prints
  "saguaro serve: the read paused at chunk c, layer l" and adds `paused_layer=/paused_chunk=/paused_at=`
  to that request's DONE so the server can say WHERE the read stopped. The next request of the same
  conversation at exactly its end with the same chunk tokens goes on at the saved layer (REUSED covers the
  paused chunk: reused + read = prompt, nothing read twice); any other continuation drops the rows and
  falls back to the existing paths (checkpoint restore, zero, P124b landing). Park/restore: the rows and
  the small record ride in `SavedConversation` (`pause_pos/pause_layer/pause_tokens/pause_ids/pause_rows`,
  counted in `bytes()`), are attached when the paused conversation is parked, and come back with it; a
  paused conversation is never written to the session dir (turn save, park file write and exit save all
  refuse while a pause is live), so a disk-restored conversation never looks complete while half-read.
- Re-running the whole paused chunk (whenever the saved pause does not match - different cuts, an edited
  prompt prefix) stays CORRECT by the argument in "Why a layer pause is park-able" above: identical cell
  rewrites and an idempotent PLE history; only the layer-skip optimization is lost.

## Step 3 - server

- The paused read's pause line names the layer and chunk when the engine reports them: "a low prompt read
  paused at X of Y prompt tokens (the low prompt read paused at layer l of chunk c)" (serve/server.py).
- `SAGUARO_PRIORITY_IDLE_S` default 5 -> 0 (serve/server.py priority_idle_seconds, docs/SETTINGS.md, the
  test in serve/test_priority.py): with a read that can stop at a layer boundary, a person's follow-up no
  longer needs a 5 s head start - the read steps aside within one layer pass instead of up to ~10 s.
- engines/qfn/serve/server.py `_parse_done` takes the new key=value fields; serve/qfn_engine.py was
  regenerated with tools/saguaro/extract_qfn_engine.py (never hand-edited).

## Step 4 - tests

- `engines/qfn/src/program/layer_pause_test.cpp` (+ CMake entry `add_test(NAME layer_pause_test ...)`) walks
  the pure helpers: pause points (not after the last layer, not after layer LB-1, chunk arithmetic at the
  run's end), resume matching (chunk grid, short last chunk, layer range), and `state_bytes` (D=10240:
  40 KiB a token; the saturating path).
- CUDA sm_75 build (the P40 box's arch): built in `scratch/build-p125j-cuda75/` with
  `cmake -B scratch/build-p125j-cuda75 -DCMAKE_CUDA_ARCHITECTURES=75 ...` - build exit 0, no GPU run.
- `tools/card_tests/layer_pause.sh` (port 8092, own dirs, no api key, per PLAN): a fresh ~60K low read; a
  short normal request at a random moment; PASS requires the normal request's first token within 1 s, the
  engine's "the read paused at chunk ... layer" line, the resumed read reusing the cut position
  ("prompt read continued: M of N tokens reused" with M > 0), every leg's `reused + new = prompt` (the
  engine's own RESUME/RDONE figures - nothing read twice), and the interrupted read's answer matching an
  uninterrupted control read's first 64 tokens.

## Notes / limits (honest scope)

- The pause fires only on the solo prompt path (no batch slots, no layer split, no piped plan - the shape
  the priority cut actually uses; serve/server.py only ever cuts a solo QFN read, keeps_stopped_read and
  not batch). Everywhere else the STOP keeps landing at the chunk end, exactly as before.
- SAGUARO_LAYER_PAUSE=0 restores the base behaviour exactly: the layer-end check, the row copies and the
  resume all sit behind it, and the keep branch only runs when a pause actually happened.
- A parked conversation carrying a paused read is counted in the park budget with its rows (image.bytes())
  and is refused disk writes while it carries them; if it is evicted, the pause goes with it and the
  prompt re-reads (or lands on a checkpoint), as before P125j.
- Stats wording: the RESUME/reused figure counts the paused chunk as reused (its lower layers were read);
  its unread layers show up as neither reused nor new, which is what makes "reused + read = the prompt"
  hold across the legs. The PP progress lines count the paused chunk as read on the leg that paused it.
