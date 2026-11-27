# P124w REPORT - what was running before each of the three mangonel crashes

Inputs: ref/crash-around-line-351460.log (crash 1, 2026-11-25 01:28), ref/crash-around-line-356061.log
(crash 2, 2026-11-25 01:41), ref/crash-around-line-775696.log (crash 3, 2026-11-26 20:13),
ref/mangonel-q4-config.json. Machine: 4x T4 (sm_75), layer_split 11,24,36, `--batch 8` (parallel 8),
`--spec 3 --spec-min-p 0.7 --mtp Qwen3.8-Flash-Next-MTP-Lamina`, `--kv int8 --kv-resident 32768`,
`--max-context 1048576`, YaRN 4 against a trained 262144 (`SAGUARO_YARN_ELASTIC=1`), and the live env that
matters: `SAGUARO_ELASTIC_SLOTS=1 SAGUARO_ELASTIC_SLOTS_GIVE=1 SAGUARO_ELASTIC_IDLE=1` (batch-slot K/V
memory is VMM-backed and given back to the expert cache), `SAGUARO_PIPED_MTP=1 SAGUARO_PIPED_MTP_DRAFTS=2`,
`SAGUARO_SLOT_ADMIT_MULTI=1 SAGUARO_BATCH_READS=1 SAGUARO_FAST_LANE=1 SAGUARO_READ_YIELD=1`
(admission reads run as batch-window rows beside the decoding slots, and can be held at piece boundaries).

## Crash 1 - 2026-11-25 01:28:55, "dequant launch" + "ERR verify: batch launch", exit -11

What was running: four agent conversations decoding (the `[saguaro]` lines: 1361, 992, 3015, 2486+ tokens
into their turns), two re-admissions of the just-saved conversations beside the decoding slots, and a third
prompt being read the queued way at the moment of the crash.

    :2026-11-25 01:28:40.146 saguaro batch: slot 2 admitted beside the decoding slots: 156473 prompt tokens (156452 reused) read in 558 ms after 29 ms queued; the other slots decoded 21 rows during the read; first token 597 ms after its BGEN; fast lane: 3 windows, 21 rows, set up in 10 ms
    :2026-11-25 01:28:46.551 saguaro batch: 1 admission read beside the slots in 3618 ms (queued, SAGUARO_SLOT_ADMIT_MULTI); the groups stopped 599 ms for the loan (lend, refill), waited 33 ms for lines, 2795 ms in the pump; decoded 178 rows (stage 1: 99 windows ...; stage 4: 102 windows ...)
    :2026-11-25 01:28:46.633 saguaro batch: slot 3 admitted beside the decoding slots: 10838 prompt tokens (10580 reused) ... KV staged from RAM: 0 ms in 24 layer passes (0.25 GB)
    :2026-11-25 01:28:50.230 saguaro serve: session dir: turn save (slot2): 156582 tokens, 123 MiB written, 2384 MiB already in its file ...
    :2026-11-25 01:28:51.361 saguaro batch: slot 2: SAGUARO_FAST_LANE: 156650 prompt tokens (156582 held) read through its group's windows; set up in 9 ms
    :2026-11-25 01:28:52.645 saguaro verify: captured the batch window over slots 2,2,2,2,2,4
    :2026-11-25 01:28:52.673 saguaro verify: captured the batch window over slots 2,2,2,2,2,4
    :2026-11-25 01:28:52.706 saguaro verify: captured the batch window over slots 2,2,2,2,2,4
    :2026-11-25 01:28:52.734 saguaro verify: captured the batch window over slots 2,2,2,2,2,4
    :2026-11-25 01:28:52.761 saguaro batch: slot 2 admitted beside the decoding slots: 156650 prompt tokens (156582 reused) read in 1398 ms after 20 ms queued; the other slots decoded 70 rows during the read; ... fast lane: 10 windows, 68 rows, set up in 9 ms
    :2026-11-25 01:28:54.058 saguaro batch: elastic slots: CUDA1's expert cache takes 42 MiB back, 14 slots refilled from the profile; 3783 of 4385 slots hold experts
    :2026-11-25 01:28:54.288 saguaro batch: slot 3: fast-lane decision: 3313 rows x 24.1 ms/row vs 2174 ms prompt-path fixed cost measured at 0K context: the queued way
    :2026-11-25 01:28:54.502 saguaro prefill: 3309 tokens in 1792-token chunks over 4 stages (--prefill-pipe 1536: b/a)
    :2026-11-25 01:28:55.026 dequant launch: an illegal memory access was encountered
    :2026-11-25 01:28:55.094 [saguaro] engine said: ERR verify: batch launch: an illegal memory access was encountered

Notes. The dying window's shape is `2,2,2,2,2,4`: slot 2 (the re-admitted fast-lane read at ~156650 cells)
owns five rows and slot 4 one - a read sharing a verify window with a decoding slot, captured *during* the
read (the read only completed 27 ms after the last capture). The slot was being re-admitted from its turn
save (156582 reused, so its K/V was being restored/staged around it), `SAGUARO_ELASTIC_SLOTS_GIVE` was
re-shuffling the same cards' VMM a second before ("takes 42 MiB back, 14 slots refilled"), and the admission
path had needed a loan re-lend earlier in the same minute ("stopped 599 ms for the loan (lend, refill)").
The illegal access first *surfaced* at a dequant launch (a later kernel noticing the sticky context error)
and the engine then died in the verify batch path (exit -11 during the crash unwinding).

## Crash 2 - 2026-11-25 01:41:15, 60 s stall of an admission read, server stops the engine, exit -6

What was running: the same recipe one hour after crash 1's restart - long conversations decoding, two
prompts being read into batch slots - with the cards almost empty of free VRAM.

    :2026-11-25 01:40:09.877 saguaro prefill: KV staged this read: 0 ms on stream, 0.05 GiB copied host->VRAM in 2 layer passes
    :2026-11-25 01:40:13.649 [saguaro] reading the prompt: 17,079 of 17,084 tokens, 10 s so far
    :2026-11-25 01:40:14.605 [saguaro] reading the prompt: 26,994 tokens, 10 s so far
    :2026-11-25 01:41:15.016 saguaro serve: no progress for 60 s during a request (reading a prompt into a batch slot, done up to token 17083) - stopping the engine so the server starts it again (issue #29)
    :2026-11-25 01:41:15.016 saguaro serve: stall report (engine 0.1.49.2): stage "reading a prompt into a batch slot, done up to token 17083" for 60 s; 0 layers served since the last finished step (0 = stopped, more = slow)
    :2026-11-25 01:41:15.016   VRAM free when this request began: 383 MiB - LOW: a card this full can page GPU memory to system RAM ...
    :2026-11-25 01:41:15.017   verify window (last window, not the current stage): 2 tokens at position 249827, host at layer step 11; the GPU rang 11; flags: served 11, plan (A) 11, copies (B) 11
    :2026-11-25 01:41:15.017   threads waiting on the disk (state D): 0 of 91
    :2026-11-25 01:41:17.021 saguaro: released the verify window's GPU waits (#267): the GPU finished in 0 ms
    :2026-11-25 01:41:26.987 [saguaro] the engine had stopped (exit code -6); starting it again (a minute or two) ...

Notes. No illegal access appears here: a 17084-token prompt stalled one token short of its end with 383 MiB
free - the paging stall of #2065/#29 - and the last verify window (2 rows, position 249827, a conversation
one step from the 262144 YaRN boundary) had rung all its layers. Same ingredients as the two hard crashes
(admission reads beside slots on an almost-full set of cards), but the symptom is a 60 s freeze rather than
an OOB. The stall report names the stage and the last window yet not one slot's cell range or backed window.

## Crash 3 - 2026-11-26 20:13:47, "dequant launch" + "ERR verify batch: layer 3 never rang", exit -6

What was running: 260K/150K/117K-token conversations; slot 2 re-admitted at 117548 (117211 reused); slot 5's
262312-token conversation crossing the YaRN boundary mid-read while its read was held at a piece boundary;
the prompt-path K/V staging loan re-lent under that held read; slot 1 re-admitted at 149903 (148256 reused)
finishing 0.26 s before the crash.

    :2026-11-26 20:13:20.429 saguaro serve: session dir: turn save (slot2): 117211 tokens ...
    :2026-11-26 20:13:25.583 saguaro batch: 1 admission read beside the slots in 5279 ms (queued, SAGUARO_SLOT_ADMIT_MULTI); the groups stopped 657 ms for the loan (lend, refill) ...
    :2026-11-26 20:13:25.690 saguaro batch: slot 2 admitted beside the decoding slots: 117548 prompt tokens (117211 reused) ... KV staged from RAM: 0 ms in 36 layer passes (4.15 GB)
    :2026-11-26 20:13:32.322 saguaro serve: session dir: turn save (slot5): 260477 tokens ...
    :2026-11-26 20:13:34.358 saguaro batch: SAGUARO_YARN_ELASTIC: slot 5's prompt is 262312 tokens, resumes 260477 at level 0 -> 1
    :2026-11-26 20:13:34.746 saguaro serve: session dir: turn save (slot1): 148256 tokens ...
    :2026-11-26 20:13:43.017 saguaro serve: SAGUARO_YARN_ELASTIC: slot 5 reaches past 262144 tokens: its 260477 cells (13 states) and 11 checkpoints moved to YaRN in 5191 ms
    :2026-11-26 20:13:43.017 saguaro batch: a read reaches past the loan's K/V staging: the reads stop at their next piece boundary and the loan is lent again for it
    :2026-11-26 20:13:45.776 saguaro batch: SAGUARO_READ_YIELD: 1 held read at a piece boundary (loan kept): slot 5 at 260477/262312
    :2026-11-26 20:13:46.142 saguaro serve: CUDA0: the prompt path's K/V staging holds 393216 cells (a loan of 1190 slots, chunk 4096)
    :2026-11-26 20:13:47.121 saguaro batch: SAGUARO_READ_YIELD: the loan was lent again under held reads
    :2026-11-26 20:13:47.121 saguaro batch: SAGUARO_READ_YIELD: 1 held read goes on
    :2026-11-26 20:13:47.121 saguaro prefill: 1667 tokens in 1024-token chunks over 4 stages (--prefill-pipe 1536: b/a)
    :2026-11-26 20:13:47.188 saguaro batch: slot 1 admitted beside the decoding slots: 149903 prompt tokens (148256 reused) read in 7859 ms after 27 ms queued; ... KV staged from RAM: 0 ms in 24 layer passes (3.52 GB)
    :2026-11-26 20:13:47.451 dequant launch: an illegal memory access was encountered
    :2026-11-26 20:13:47.451 [saguaro] engine said: ERR verify batch: layer 3 never rang (an illegal memory access was encountered)
    :2026-11-26 20:13:47.959 double free or corruption (out)

Notes. "layer 3 never rang" pins the dead window's stage to CUDA0's layers [0,11); the window was still
being staged/launched 0.3 s after the loan was re-lent *under a held read* and 0.26 s after slot 1's
re-admission finished staging 3.52 GB of reused K/V. The dequant line again is the noticing, not the
doing: the process died with a corrupted context (exit -6, "double free or corruption" in the unwinding).

## What the three have in common

1. All three: `--batch 8` conversations of 110K-262K tokens whose **admission / re-admission reads run
   beside the decoding slots** (SAGUARO_SLOT_ADMIT_MULTI + SAGUARO_BATCH_READS + SAGUARO_FAST_LANE): every
   crash line is preceded (seconds to sub-second) by a `slot N admitted beside the decoding slots` of a
   huge reused prompt (156650 / 26994-read / 149903 tokens) with its "KV staged from RAM" staging.
2. Both hard crashes die **in the verify batch** ("ERR verify: batch launch", "ERR verify batch: layer 3
   never rang"), and in both the last observable verify activity is a **multi-slot window whose rows include
   the slot being read/re-read** (crash 1: "captured the batch window over slots 2,2,2,2,2,4" - five rows of
   the re-admitted slot 2 plus a decoding slot - captured while its read was still running).
3. Both hard crashes happen **within ~2.5 s of a VMM reshuffle of the very memory the slots' K/V lives in**:
   crash 1 "elastic slots: CUDA1's expert cache takes 42 MiB back, 14 slots refilled" (elastic slot budget
   given back to the expert cache, SAGUARO_ELASTIC_SLOTS_GIVE/IDLE); crash 3 "the loan was lent again under
   held reads" (the prompt-path K/V staging window re-allocated, 393216 cells / ~1200 slots / chunk 4096,
   remapped on all four cards) plus slot 5's cells and checkpoints *moved to YaRN* mid-read. Crash 2 shows
   the same machinery jammed with 383 MiB free: the read into a batch slot never finished, so the window
   never came back.
4. The illegal access is **never observed where it happens**: it first prints at a `dequant launch` check
   (the sticky CUDA error noticed by whichever kernel launches next), the verify batch then reports only
   its own launch failure / never-rang layer, and no line ever names the failing window's slot, cell range
   or the backed window at the failing launch. That exact gap is what step 3 instruments.

## Step 2 - the out-of-bounds candidates, ranked (paths relative to `engines/qfn/`)

Ranked by how well each fits the three timelines above.

1. **A verify-batch row reads past the slot's *backed* K/V (elastic slot memory given back while a window
   touches it).** The launch-time bound is only `pos[t] < 0 || pos[t] + 1 > max_cells` (the whole
   `stage_batch` check at src/core/verify.cpp:3875-3886) - `max_cells` is the reservation ceiling
   (include/strata/core/session.hpp:32), while an elastic `--batch` slot's cells are backed per
   `SessionState::slot_cells` (session.hpp:63-64) by `session_slot_fit` (src/core/session.cpp:161-183: grow
   maps VMM chunks up, `trim` gives them back) - cells past it are unmapped addresses, so a read there is
   exactly an "illegal memory access". The guards that keep `slot_cells` ahead of the rows live in the
   driver, not at the launch: the per-window grow `slot_grow_or_end` (src/program/generate.cpp:14250-14255),
   the admission backing `slot_admit_fit` (:14191-14226), the idle trim `elastic_idle` (:14156-14168) -
   each keyed on `BSlot.active/reading` flags around a *different* thread's window launches
   (`run_slot_rows` :14564, `batch_launch_rows` :15620), and read rows advance piece by piece while
   backing is a single up-front `upto + 1` fit. Fit: crashes 1 and 3 both have a re-admission in flight in
   the dying window's rows; a missed grow on any stage's session (`bslot_ss[k]`, four cards) unmaps cells
   the very next window reads.
2. **A drafter state of a slot that was parked or re-admitted.** The per-slot MTP drafters
   (`slot_mtp`, src/program/generate.cpp:6094, 6185; rows fed at :14517-14525 through the `batch_mtp` path,
   which this machine reaches because SAGUARO_PIPED_MTP forces `batch_mtp = true` (generate.cpp:5837;
   ref/mangonel-q4-config.json:69) - not because of `--spec` itself) keep their *own* `QsaState` (bound at
   :10240-10258; mirrored per slot session via `host_follow`, session.hpp:70-71) - it is not in `bslot_ss`:
   the elastic fit/trim grows and trims only its *host* copy through `host_follow` (session.cpp:172, 180),
   so its VRAM rows never follow the slot's fit/trim, and a park/re-admit rebinds the slot around it (the
   "restored draft KV" verify of the solo path, :20000-20013, is debug-only). Fit: crash 1's dying window was
   a drafting window (five rows of slot 2) seconds after slot 2's re-read; crash 3's config runs
   SAGUARO_PIPED_MTP with two drafts per slot.
3. **A dequant / attention read of a cell outside the prompt-path K/V staging window (the "loan").**
   Reused prompts stream through a loan of expert-cache chunks ("the prompt path's K/V staging holds
   393216 cells", :16076), a read past it **re-lends** the loan (":17950 the loan is lent again for it",
   ":17432 the loan was lent again under held reads") - windows reading staged pieces through the old
   mapping while the re-lend remaps those chunks read freed/unmapped memory, and the noticing kernel is a
   dequant ("dequant launch: ...", src/kernels/cuda/dequant_bf16.cu:359,375,406, which `exit(1)`s - matching
   both crashes dying during unwinding with -11/-6). Fit: crash 3 names the re-lend 0.33 s before death;
   crash 1 had a loan (re)lend/refill in the minute before.
4. **A buffer sized for fewer rows than the verify batch.** Rows, group contiguity and the hand-off span
   are checked at stage time (`batch_rows_error`, include/strata/core/batch_rows.hpp:16-27, called at
   verify.cpp:3875) and the row-indexed buffers are sized `kVerifyMaxT` (verify.cpp:3911-3914, 3945); the
   observed shapes (6 rows) fit. Least likely; kept only because the same stage-time site is where the
   cheap checked bounds below go.

Common to 1-3: none of the three can be named *the* cause from these logs - the crash lines never say
which slot, which cell range, or which window was being read when the access became illegal. So step 3
takes the plan's second branch: a checked bound at the batch staging (where row positions are staged for
every window) plus a log line naming the slot, the stage's layer range, the rows' cell range and the
backed window - printed both when a row is caught past its slot's backed cells and when a batch launch or
a never-rang layer reports the illegal access.
