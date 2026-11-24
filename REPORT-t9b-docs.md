# T9b - QFN user docs report

Docs only: Qwen3.8-Flash-Next on `saguaro-qfn` gets its user-facing pages, written from the T1-T6 and T9a reports
(the facts source) and nothing else. No code, no `docs/SETTINGS.md`, no `serve/qfn_engine.py`, no `REPORT-*.md` touched.

## What changed, per step (one commit each)

| Step | Commit | What changed |
| --- | --- | --- |
| 1 | `7d807b4ba` | `docs/SETUP.md`: a **Qwen3.8-Flash-Next** bullet in the setup's model list (the style of the Hy4-Flash / Inkling 2 Mini / MiniMax ones, which I read first): `--model qwen3.8-flash-next` / `-q8_0` (and `qfn`, `qfn-q8_0`), the two quants and their sizes (UD-Q4_K_XL 111 GB, Q8_0 177 GB), NVIDIA from the V100 on - one 16 GB+ card or up to four equal ones with the 48 layers split, four "16 GB" sm_75 cards getting the live 4x Tesla T4 server's own split - AMD one RDNA 3 / 4 GPU of 16 GB+ (an APU not yet), Linux only with Windows refused, the ~96 GB RAM and the `--resident-budget-gib` tier of RAM less 20 GB, the contexts (65536 one card / 262144 a split), `--gguf-dir` and `--import-model`, the picture tower (0.9 GB mmproj) and its reader (`vision-qfn`, `vision-qfn-hip`, `--no-vision`), the Lamina drafter with `--drafting off` / `on`, `--parallel` (off on one 16 GB card, 2 on several GPUs), and `--calibrate` / `--speed-calibration` / `--bench`; links to SETTINGS.md, MODELS.md and AMD_SAGUARO.md. |
| 2 | `aa8532c1d` | `docs/MODELS.md`: the **## Qwen3.8-Flash-Next (saguaro-qfn)** section (what the model is - 125 B, 10 of 512 experts, Qwen Community License 1.0, unsloth's GGUFs, the engine, not CYBER-FROST), the files table (111 / 177 GB and what they run on), what it needs (16 GB card + ~96 GB RAM, the RAM tier, the contexts, no Windows), the four-card split and what `--check` says, drafter / pictures / `--parallel`, and only the speeds already in the repo: the RX 9070 XT config's 27.5-40.9 tok/s answers and 524-1,079 tok/s 8K-16K reads with the 45 s load, and Q8_0's 14-15 / 73-315 tok/s, pointing at AMD_SAGUARO.md for the full table. Its TOC line added; the intro's now-false "on NVIDIA it is set up by hand for now" replaced with what T2 does. |
| 3 | `04b34d268` | `docs/AMD_SAGUARO.md`'s "## Qwen3.8-Flash-Next on the saguaro-qfn engine": it says the same model now sets up on NVIDIA too (with the SETUP.md link) and points at the SETTINGS.md section; a new paragraph for T3-T5: the mmproj and the `vision-qfn-hip` reader (`--no-vision`), the Lamina layer with the drafting question, `--calibrate` / `--speed-calibration` / `--bench`, `--parallel` supported but off by default on this one 16 GB card because parallel decoding measured slower there (2 on several GPUs), and the Models page's own fields incl. the live context change and the Turbo button; the closing "Not checked on AMD yet" now names only what is still unmeasured (pictures, two AMD GPUs) instead of the three it listed. `README.md` lines 82-86: `saguaro-qfn` is said to run **on NVIDIA and AMD** under Linux (one 16 GB+ card with ~96 GB RAM either vendor, up to four equal cards on NVIDIA; the four-card split is the live 4x Tesla T4 server's own). |
| 4 | `26b6cce3a` | `CHANGELOG.md`: one entry at the top, "### QFN as a first-class engine (2026-11-24)", in the existing entries' style (bold lead-ins, one line each): the own family and the corrected label, `--bench` no longer falling through to the GLM binary, the NVIDIA setup with `qfn_fit` and `--check` and the two refusals, the picture tower / drafting choice / `--parallel` / `--import-model`, the three caliper commands, the pages' own settings and the three UI fixes, the Monitor's `mtp=` and solo `STAT` / `promo=`, and the docs. |

No number appears that the repo did not already state: the sizes come from `tools/qfn/download.json` and
`tools/qfn_fit.py` (111 / 177 GB, the mmproj's 907,542,944 B, the Lamina layer's 0.8 GB), the needs from the manifest's
`needs` (a 16 GB card, 88 GB of RAM) and `qfn_fit` (RAM less 20 GB, 48 layers, four cards, contexts 32K-256K, Volta as
the floor), the speeds from `docs/AMD_SAGUARO.md`'s measured table and its Q8_0 line.

## Files

- `docs/SETUP.md` - the QFN bullet in "What the first run does" (24 lines added).
- `docs/MODELS.md` - the QFN section (36 lines), its TOC line, the intro's NVIDIA sentence.
- `docs/AMD_SAGUARO.md` - the QFN section's intro, its new features paragraph, its "not checked" line.
- `README.md` - the models paragraph, lines 82-86.
- `CHANGELOG.md` - the new entry (31 lines).
- `REPORT-t9b-docs.md` - this.
- Untracked (gitignored): `scratch/check_tables.py`, `scratch/check_anchors.py` (the two checks below).

`git diff --stat` over the four commits: 5 files, +115 / -7, nothing else.

## Tests

The plan names none (docs only), so no test module was run. The two checks it asks for were run with
`/opt/llm/saguaro/.venv/bin/python`:

| check | scope | result |
| --- | --- | --- |
| table column counts (`scratch/check_tables.py`: every row of a table has as many unescaped `\|` as its own header row) | `docs/SETUP.md`, `docs/MODELS.md`, `docs/AMD_SAGUARO.md`, `README.md`, `CHANGELOG.md` | bad rows: 0 |
| links point at headings that exist (`scratch/check_anchors.py`: slug of every `file.md#anchor` link, GitHub's rules, checked against the target's headings) | `docs/SETUP.md`, `docs/MODELS.md`, `docs/AMD_SAGUARO.md`, `README.md`, `CHANGELOG.md` | bad links: 0 |

The anchor check's one complaint is `CHANGELOG.md:1285`, an entry from an earlier release whose
`docs/AMD_MAYA.md#experimental-expert-helper` link predates this task's rename and was left as history.

## Open questions

1. **README's hardware paragraph (lines 74-80)** still lists only GLM / DeepSeek / MiniMax for NVIDIA and Windows and
   Inkling for "one AMD GPU"; the plan limited step 3 to lines 82-86, so it was left alone. Say the word and QFN's
   "NVIDIA and AMD, Linux" line can join it.
2. **No NVIDIA speed figures exist.** The repo has no `--bench` result for the 4x Tesla T4 server (no QFN rows under
   `bench/`), so README, SETUP.md and MODELS.md state the four-card layout's provenance and that `--check` says when a
   layout is unmeasured, without tokens/s. If the T4 box or another NVIDIA PC is benched, `docs/MODELS.md` is where the
   numbers belong.
3. **`docs/AMD_SAGUARO.md`'s "What runs on AMD" table** (the QFN row, line 11) was left as it is: it stays true and the
   plan did not name it. It still says "the setup offers it ... for one RDNA GPU", which reads AMD-only next to the
   NVIDIA line added below it.
4. **No version bump.** `VERSION` stays 1.1.50 and the CHANGELOG entry has no version number, only the date, like the
   entries above it; a release commit that moves the number is someone else's.
5. **The setup's model list order** in `docs/SETUP.md`: the QFN bullet went last, after the DeepSeek "Or orcarouter's"
   paragraph (which is itself indented under the last bullet in the file today). It reads fine; moving it under the
   other models means fixing that pre-existing indentation, which is beyond a docs step.
6. **`docs/SETUP.md`'s "Options" table** (its `--context` row, line 234) still lists the contexts for GLM / DeepSeek /
   MiniMax only, though its `--no-vision` row already names Qwen-3.8-Flash-Next; adding QFN's 32K-256K there was not in
   the plan's steps.
