# FIX-REPORT: T2 review findings (REVIEW-RESULT.md) - what each one got

Branch `saguaro/q1-t2-setup-nvidia`, on top of the reviewed HEAD `5cd08fe27`. One commit per finding, in review
order. Tests run at the final HEAD with `/opt/llm/saguaro/.venv/bin/python -m unittest ...`, the four plan-named
modules one at a time:

| module | before these fixes | after |
| --- | --- | --- |
| `tools.test_qfn_fit` | 18 OK | 20 OK (+2) |
| `tools.test_setup_qfn` | 17 OK | 20 OK (+3) |
| `tools.test_setup_amd` | 29 OK | 29 OK (untouched) |
| `tools.test_setup_ik2` | 54 OK | 54 OK (untouched) |

`serve/`, `engines/` and `serve/qfn_engine.py` are still untouched (nothing here needed an engine change, so
`tools/saguaro/extract_qfn_engine.py` did not run). Commits carry no Co-Authored-By / Claude-Session / Assisted-by.

## 1. HIGH - crash when every card is under 16 GB (no `--gpus`) -> FIXED (cc9f90ccd)

Three changes, following ik2's exact pattern in `check_pc`:

- `saguaro.py:1648-1650` - before the cards are chosen, qfn now refuses on its own line when no card of a layout
  exists: `if family == "qfn" and not QF.best_gpus(found): fail(QF.unsupported(found, mem_gb()[0]) or ...)`.
  Gated on `best_gpus` being empty rather than on `unsupported(found, ...)` alone, because `qfn_fit.unsupported`
  refuses a PC where *any* card is under 16 GB - a 24 GB + 8 GB PC is healthy for qfn (the picker ignores the
  small card) and must not be refused here.
- `saguaro.py:1578` - `choose_qfn_gpus` takes `QF.best_gpus(usable) or big[:1]`: an all-small card set yields the
  largest card, never `[]`, so the row logic and `min(archs)` downstream can't see an empty list.
- `saguaro.py:1661-1664` - after the cards are picked, `QF.unsupported(chosen, mem_gb()[0])` fails in one line
  (ik2's post-check mirrored): catches `--gpus` naming a small card, a small-only RAM-poor PC, and the edge of a
  big card whose compute capability qfn cannot use.

Test: `tools/test_setup_qfn.py` `QfnGpus.test_cards_under_16_gb_are_refused_in_one_line` - the review's RTX 4060
repro through `check_pc` (no `--gpus`; `--gpus` naming a small card; both small cards named; a healthy card with
60 GB of RAM), each a one-line refusal, no traceback. `scratch/repro_smallcards.py` now gets the card back from
`choose_gpus` instead of `[]`.

## 2. MEDIUM - `--check` calls a measured one-card AMD layout the T4 server's split -> FIXED (6fd6c8647)

The sentence is now gated on the layout actually being a split: `lay["layer_split"] and lay["measured"]`. The line
moved out of `main()` into `qfn_check_line(quant, lay)` (`saguaro.py:7566-7573`) so it is testable. A measured
one-card layout (the RX 9070 XT / 96 GB PC) prints its `said` with neither the T4 sentence nor "not measured";
4x T4 keeps "the live 4x Tesla T4 server's own split"; an unmeasured card keeps "(not measured on these cards
yet)"; a non-fitting layout keeps the plain reason. The `set_up_qfn` NVIDIA branch was left as it is - the review
confirmed its `measured` can only be the T4 split.

Test: `tools/test_setup_qfn.py` `CheckLine` (the four prints; the review's `scratch/repro_check_amd.py` re-run
through the helper prints no Tesla T4 for the AMD card).

## 3. LOW - the q8_0 config wrote the UD-Q4_K_XL box's T4 env -> FIXED (db29fa750)

Quant-keyed like `MEASURED_SPLITS`/`MEASURED_PREFILL_PIPE`: `T4_ENV` keeps the values both live boxes share, and
two per-quant dicts carry the ones they differ on (`tools/qfn_fit.py:42-50`) - `T4_BF16_TC_SKIP`
`{"ud-q4_k_xl": "none", "q8_0": "indexer,router,ssm"}` and `T4_SELECT_SM75` `{"ud-q4_k_xl": "1"}` - matching
`./ref/Qwen3.8-Flash-Next-Q8_0.json` / `...-UD-Q4_K_XL.json` exactly. New `t4_env(quant)`
(`tools/qfn_fit.py:93-101`) composes them; `plan` uses `t4_env(quant)`. The env *gate* (every card sm_75) is
unchanged. The note that admitted the UD value was applied to q8_0 is replaced by the per-quant comment, so no
open question remains on this point.

Test: `tools/test_qfn_fit.py` `FourT4.test_the_t4_env_follows_the_quant`.

## 4. NIT - dead constants and the duplicated context table -> FIXED (aac0f1437)

`QSA_LAYERS`, `KV_RESIDENT_MIN` and `CONTEXTS` deleted from `tools/qfn_fit.py` (nothing referenced them). The
remaining duplication (`DEFAULT_CONTEXT`/`SPLIT_CONTEXT` vs `saguaro.py`'s `QFN_CONTEXTS`) cannot be imported
away - `qfn_fit` is hardware-free and standalone - so a comment points at it and a test guards it:
`tools/test_setup_qfn.py` `Offered.test_qfn_fit_contexts_stay_offered_by_the_setup` asserts both values stay
among `QFN_CONTEXTS`, so `lay["context"]` cannot drift out of what `choose_context` offers.

## 5. NIT - `plan([])` claimed it fits -> FIXED (ea631ddcc)

`plan` with no cards returns `fits=False` with the reason "Qwen3.8-Flash-Next on saguaro-qfn runs on one GPU of
16 GB or more, or up to 4 equal ones: no cards were named" (`tools/qfn_fit.py:165-166`); `env`/`layer_split` stay
empty/None and the CLI exits 1 instead of printing "fits: no cards named". No caller depended on the old True -
with fix 1 every `plan()` call site gets at least one card.

Test: `tools/test_qfn_fit.py` `TooSmall.test_no_cards_named_is_not_a_fit` and the CLI test's `F.main([])`.

## 6. INFO - stale docs/MODELS.md and serve/model_pull.py lines -> NOT CHANGED, on purpose

Both confirmed still stale ("on NVIDIA it is set up by hand for now" at `docs/MODELS.md:7`; `qfn_fit` in
`serve/model_pull.py` looking at the first AMD GPU only). The plan's Don't list keeps them out of this branch
("don't touch ... the web UI", and no doc step), and the review itself marks them "out of this plan's scope ...
they want a follow-up commit" - so they stay a follow-up, as does the unmirrored generic engine args question
(report open #3), which the review already graded "fine".

## The review's "checked and good" list

Untouched by these fixes: the AMD config output (finding fixes 1/2 are NVIDIA-gated `check_pc`/`--check` lines,
finding 3 only changes the sm_75-gated env, and `test_setup_amd` is unchanged at 29 OK), the generated
`serve/qfn_engine.py`, and the engine sources.
