# Grid checkpoints for batch slots (STRATA_SLOT_GRID)

Opt-in, off by default. `STRATA_SLOT_GRID=6` turns it on with 6 distance buckets.

## Why

A batch slot keeps its turn checkpoint (and, with `STRATA_CACHE_MESSAGE_BOUNDARY`, a message checkpoint), so the
next turn of its conversation continues where it left off. A request that keeps only a **prefix** of the
conversation finds nothing near its cut. Examples are an agent's compaction summary over its oldest messages (the
hermes-lcm chain summary), or a client that rewrites an older turn. That request then reads everything again after
the root. The solo path keeps a checkpoint every `--prompt-cache-every` tokens, but the hand-off to a slot used to
copy only the deepest one. Measured 2026-11-07 on the live Q4 config with two ~424K conversations in slots: a 103K
prefix request reused 0 and read for 60 s, where the same request on a solo conversation reused 512,964 of 513,246.

## What it does

- **Thinning** (`include/strata/program/slot_grid.hpp`, unit test `slot_grid_test`). Grid checkpoints sit at
  multiples of `--prompt-cache-every` (16384). A slot keeps them by distance from the conversation's end: one bucket
  for the last 16K, then one per doubling, and the last bucket takes everything further. Each bucket keeps its
  oldest and its newest, so a checkpoint ages outward from bucket to bucket instead of being replaced.
- **Simulated coverage.** Growth steps of 2K-150K up to 1M, cuts 40K-250K back. With 6 buckets at most 10
  checkpoints are held, and the re-read below a cut is at most 0.71x (average ~0.22x) of the cut's distance.
- **Memory.** Each checkpoint holds the GDN state (~118 MB with Qwen3.8-Flash-Next), so a long conversation in a
  slot holds up to ~1.2 GB more host RAM, and a parked image grows by the same.
- **Saving.** An admission's read saves only the multiples that the thinning keeps at the prompt's end, beside what
  the slot already holds. A fresh 424K read saves 10, an incremental 23K turn 1-2.
  - Path A (whole read on its own thread) stops at each one, as it does for the message and turn checkpoints.
  - Path B (queued pieces), `STRATA_BATCH_READS` and `STRATA_READ_YIELD` end a piece at each position, and each
    stage saves its part through `on_stage_chunk` when it reaches it.
  - An incomplete set of parts makes no checkpoint. `rope_level` comes from `checkpoint_save`.
- **Carry-forward.** An admission keeps the slot's earlier checkpoints that the new prompt still starts with, at or
  below where it resumes, and thins them together with the new ones. An old message checkpoint loses its `msg_tail`
  mark (the new one is the tail; `adopt_msg_tail` is unchanged). Before this, an admission dropped all of them.
- **Hand-off.** When the solo path hands its conversation to a slot, the chain's checkpoints of that conversation
  that the thinning keeps are copied into the slot (copies: the solo session still holds it). The slot's
  "takes N tokens (copied in X ms)" line says how many, so the copy time is visible.
- Parking, session files, hand-back and `STRATA_YARN_ELASTIC` moves take the slot's checkpoints as they took the
  turn checkpoint: `sl.checks` is the same list.

## Test

`/data/llm/bench/strata-yarn-rope-level/run_par.sh` (two conversations at ~424K in slots). Pass: the compaction-shaped
103K prefix request reuses a grid checkpoint near its cut (cached in the tens of thousands, not 0), recall 60/60, and
no level-0 read past 262K except first crossings.
