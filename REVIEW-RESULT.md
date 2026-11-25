# REVIEW: 1.2.1-B fit policy, re-review of `git diff 2d0a71bba..HEAD` (4 step commits + report + review-fix rounds + CHANGELOG/docs)

**MERGE-WITH-NITS** - the policy is implemented as planned, both standing mediums of the last round are
fixed and pinned by tests that really exercise them (reproduced independently, not only through the
tests), the comfortable layouts write byte-identical configs, LiveT4Parity is untouched, and every
Ken-policy item checks out; what remains is three lows around judgement calls the report already
records and three wording/doc nits - none blocks the merge.

Tests run (one module at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest <module>`, exactly
the plan's list): `tools.test_qfn_fit` 38 OK, `tools.test_setup_qfn` 69 OK (1 skip: the
worktree-local `ref/live-ud-q4.json`, pre-existing), `tools.test_setup_amd` 33 OK,
`tools.test_setup_ik2` 54 OK. The AMD/ik2 modules and their code are untouched by the branch
(`test_setup_amd`/`test_setup_ik2` do not appear in the diff) - they pass as a regression check.

## The last round's findings - all seven verified fixed

1. **MEDIUM (picker/guard refusal) - fixed.** `best_gpus` now drops an equal group standing below its
   own `card_min_gib` when the largest card alone clears its single-card minimum
   (tools/qfn_fit.py:326-330, 342-350), and `check_pc`'s pre-guard applies the cc filter before
   pricing (saguaro.py:1670-1678). Reproduced directly: `best_gpus(A4000 + 2x GT 710)` -> `[0]`,
   pre-guard line `None`; the real refusal that must stay does - two 4 GB cards alone price their
   split ("its 4 GB cannot hold the 4.1 GB ... (50% of the layers)", tools/test_setup_qfn.py:909-924).
2. **MEDIUM (the 7.24x K/V contradiction) - fixed.** `KV_RES_GIB` is gone and `kv_resident_gib` IS
   `kv_vram_gib` (tools/qfn_fit.py:171-175); verified both read 0.3867 GiB for 32768 cells; the
   refusal charges the 4K window's 0.05 GiB (tools/qfn_fit.py:254-258); tied by
   `Sizes.test_the_two_kv_readings_are_one_number`. Settled against the engine's own INFO line,
   generate.cpp:12401, as directed.
3. **LOW (tier vs. RAM minimum) - fixed.** The slow zone scales the pinned tier to what the RAM
   minimum leaves and writes none where it leaves nothing (tools/qfn_fit.py:410-415,
   saguaro.py:7531-7535). Verified through the CLI: `--ram 16 8` exits 0, slow,
   "resident budget -", "no room for a pinned tier"; 32/64 GB rows keep 12/44 GiB.
4. **LOW (mixed pre-Turing arch list) - fixed.** `check_pc` warns plainly on a chosen list mixing
   archs < 75 with >= 75 (saguaro.py:1710-1720), matching CMakeLists.txt:1798
   (`if(_qfn_old AND NOT _qfn_new)` - the pre-Turing path needs every arch old); a warning with the
   `--gpu N` way out, not a refusal, engine C++ untouched. Tested both ways.
5. **LOW (flag claimed without a drafter) - fixed.** `turned_off` says "drafting off" flag-free
   (tools/qfn_fit.py:440-445); `--mtp-max-t 1` exists only in `write_config_qfn`, gated on
   `mtp is not None` and `--drafting != on` (saguaro.py:7590-7594); `--drafting on` overrides both
   the flag and the step-2 line (saguaro.py:7748-7751). Tested with `_nomtp` and with `drafting=on`.
6. **NIT (stale "needs" wording) - fixed.** Models page and docs say the measured box, not a
   requirement (serve/model_pull.py:95-113, docs/AMD_SAGUARO.md:10); the quant note says "cannot run
   on this PC" (saguaro.py:7297).
7. **NIT (CHANGELOG) - fixed.** The 1.2.1-B section at CHANGELOG.md:22-39 matches the shipped
   behaviour.

## The brief's checklist - clean

* **Refuse only what cannot run / minimum computed, not a GB constant** - refusals are the computed
  minimums (`card_min_gib`/`ram_min_gib`, tools/qfn_fit.py:185-197); `MIN_VRAM_GB`/`MIN_RAM_GB`
  demoted to the comfortable level (tools/qfn_fit.py:30-32, used only by `comfortable()`/`slow_why`);
  the old 96 GB refusal, the quant-shortfall refusal and the 15 GB filters are gone; the
  quant menu prices the chosen cards over the real split (saguaro.py:7264-7283). The
  `16 GB or more` / `about 96 GB are needed` refusal texts appear nowhere in the QFN paths anymore
  (remaining hits at saguaro.py:1628/7255/7731/7754 all say "measured on", not "needs").
* **Confirm skipped by `--yes`, `--check` only reports** - `if not a.yes and ask(...)` at
  saguaro.py:7752; "n" stops in one plain line (7753-7755); the `--check` line reports the same
  cause + scale-down without asking (saguaro.py:7688-7690, 9149-9155). The scale-down is said
  plainly: warn with the cause, ok with what it turned off (saguaro.py:7743-7751), `said` carrying
  "expect slow decode" (tools/qfn_fit.py:439).
* **Pascal with the CUDA 12 rule** - `MIN_CC = 61` (tools/qfn_fit.py:33), "can be used (Pascal:
  experimental)" + the standing experimental warning (saguaro.py:1681-1683, 1706-1709) as the
  ik2/GLM path has always done; the toolkit rule is the shared one - `nvcc_range` hi=(13,0) for
  archs < 75 (saguaro.py:1441), `pick_nvcc` prefers a fitting 12.x (1466-1471), and with only
  CUDA 13 installed the plainly said "cannot build ... CUDA 12.0 or newer but older than 13.0
  (CUDA 13 dropped Volta and Pascal)" + 12.x install hint (saguaro.py:1731-1734, 1479-1482); no
  T4 env on a P40 layout (`sm75_only` gate, tested).
* **Byte-identical where the policy does not bite** - comfortable layouts: `write_config_qfn`'s
  `min(ctx, lay["kv_resident"])` (saguaro.py:7596) equals the old `min(ctx, 32768)` for every
  comfortable row; the `Layouts1To4T4` 96/256 rows assert the old values unchanged; verified
  `plan(4x T4, 256)` -> not slow, 262144/32768/no tier.
* **LiveT4Parity unchanged** - zero changed lines inside the class (the one diff hunk near it is
  `Layouts1To4T4`'s docstring); its pinned live config and `test_four_t4s_256_gb_ud_q4_...` pass.
* **Generated file** - `serve/qfn_engine.py` and `engines/qfn/` have zero lines in the diff
  (nothing there needed changing, the scale-down writes existing engine flags), so
  `tools/saguaro/extract_qfn_engine.py` correctly never ran.
* **Locks / threads / stdout** - no `threading`/`Lock` line added anywhere in the diff; the setup
  path is sequential say/warn/ok + one stdin prompt; `serve/model_pull.py` changes are string
  literals in the lock-free pure `qfn_fit()`.
* **No paths outside the repo, no secrets** - the only absolute paths added are inert strings in
  patched test doubles ("/usr/local/cuda-12.8/bin/nvcc", "/usr/bin/g++"); nothing reads outside
  the repo, no key-like content.

## Findings

**LOW**

1. **A 4x T4 row below the comfortable RAM keeps the "measured" badge while being called slow** -
   the 15.0 GB RAM-64 rows are `live`/`measured` (tools/qfn_fit.py:418, 429) and get the live
   split and pipe rate on top of the warn+confirm (saguaro.py:7736-7737 then 7743), and a 14.x GB
   T4 is in the same boat (`t4_sized` 14.0-17.0, tools/qfn_fit.py:58 vs. `comfortable` >= 15).
   Deliberate ("fits today stays fits", report:128-129) and harmless to LiveT4Parity (256 GB is
   comfortable), but the matrix never rows a 14.x card - the report's own open question.
2. **A comfortable PC whose experts are over VRAM+RAM gets no note** - e.g. 16 GB + 96 GB with
   Q8_0 (the shape of the measured AMD box): comfortable -> no warn, `slow_why`'s quant clause
   (tools/qfn_fit.py:287-290) only speaks inside the slow zone. The plan's "Same for quants..."
   sentence is met exactly where the old code refused (all such rows have thin RAM and ARE slow),
   so this is the report's documented follow-up call (report:130-133), not a plan break - but
   worth Ken's yes/no before merge-after-nits turns into silence.
3. **With `--yes`, a runnable small-card group now beats a comfortable single card** -
   1x 16 GB + 2x 8 GB: `best_gpus` takes the 8 GB split (count over size, tools/qfn_fit.py:347-350)
   where the base code filtered the pair out and took the 16 GB card; interactively
   `choose_qfn_gpus` asks (saguaro.py:1600-1607) so only `--yes`/`--check` pick the split
   silently. Consistent with the pre-existing "equal group wins" rule and newly possible by
   design (that is the policy), but it is a real `--yes` behaviour change on mixed PCs the
   plan's matrix has no row for.

**NIT**

4. **The step report keeps pre-fix figures** - REPORT-v121b-fit-policy.md:13, 20, 25, 50 still
   read `KV_RES_GIB = 2.8`, "6.3 GiB ... 9.0 at Q8_0", "0.35 GiB minimum resident K/V" and
   "Q8_0 = 9.0 needed" for what the fix round moved to 0.05 GiB / 6.0 / 8.7 (d7a7fcd1b);
   FIX-REPORT.md:33-47 maps the move, but a REPORT-only reader gets the current minimums wrong.
5. **Comment overstates its own effect** - saguaro.py:7675 ("so a start does not measure drafting
   on a PC that cannot use it"): a QFN start never measures drafting anyway
   (needs_speed_calibration returns False for the family, saguaro.py:4375-4376); what the record
   actually suppresses is the setup-time question (saguaro.py:7779) - which the same comment's
   second half says correctly.
6. **A below-cc PC with a tiny card gets blamed on VRAM first** - when no card passes the cc
   filter, the pre-guard prices `found` (saguaro.py:1672), so a cc-5.2 4 GB card refuses with
   "its 4 GB cannot hold the 6.0 GB" although the cause is compute capability; `choose_gpus` says
   the arch line moments later (saguaro.py:1522-1524) and the base code had the same shape, so no
   regression - naming the arch in that corner would just be kinder.

## Verdict

MERGE-WITH-NITS. Nothing refuses a machine that can run; nothing byte-visible moved for a
comfortable or live layout; the confirm, `--yes`, `--check`, Pascal/CUDA-12 and the scale-down
wording all behave as the plan and the Ken policy require; four test modules green on their own.
The lows are recorded judgement calls, the nits are wording/doc - none of them asks for a code
change before merge.
