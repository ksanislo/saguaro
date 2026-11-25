# FIX-REPORT: REVIEW-RESULT.md findings -> what was changed

Branch `saguaro/q1-d1-readme`; base of the reviewed diff `b2bd0ecd`. One commit per finding/nit group;
verification after both HIGH fixes and again after the nit commit:
`python -m unittest tools.test_docs_new_engines` - **7 tests, OK** (the plan says "none to run"; the review's
fix instruction names this module and it is exactly the one that pins README content, so it was run, alone).

## FIX-FIRST

### 1. HIGH - README.md:68 broke `test_setup_ids_are_the_setups` (shortened Hy4 ids) - FIXED (`b1f5efb8b`)
The Hy4 `--model id` cell now spells out all four setup ids again:
`` `hy4-flash-maya-s` / `hy4-flash-maya-xs` / `hy4-flash-maya-m` / `hy4-flash-maya-l` `` (table row, so the
~120-column prose rule is not involved). `hy4-flash-maya-xs/-m/-l` are substrings of README.md once more;
the test's `Q.resolve` assertions hold (ids unchanged, only spelled out). The `--model (hy4|inkling)...`
regex assertion is untouched - no new `--model hy4...` string was added.

### 2. HIGH - README.md:69 broke `test_inkling_on_amd_is_said_alike_everywhere` (reworded AMD cell) - FIXED (`9dd59f5a1`)
The Inkling AMD cell now reads "builds and runs on one AMD RDNA 3 / 4 GPU of 12 GB or more (measured on an
RX 9070 XT)" - the exact asserted phrase, with the measurement note kept and no "experimental" label.
The whole module passes: 7/7 (the "16-23 GB AMD RDNA 3 / 4 GPU" sibling, the negative assertions and the
speed regexes were verified by the same run). No change to tools/ - the test keeps pinning the README.

## Nits

### REPORT-d1-readme.md:55-56 - line-length claim omitted the HF badge lines - FIXED (`b576fd27c`)
Item 4 now lists the four HF badge `<a href><img>` lines (README.md:9-12) alongside table rows and
`<a name>` markup as the exceptions, and says they are unchanged from base. README.md itself untouched here.

### README.md:68 - "(the measured size)" vs the "(default marked)" legend - FIXED (`b576fd27c`)
Now "**Maya-S** (default; measured size)": re-checked `saguaro.py:748` (`HY4_DEFAULT_QUANT = "maya-s"`)
before changing. The cell is otherwise unchanged.

### README.md:70 - "no AMD picture run measured yet" - NOT CHANGED
The review itself calls it "consistent" (docs build `vision-qfn-hip` but publish no AMD picture measurement)
and only asks to re-check "when one lands". Nothing to fix today; the claim matches the repo as it stands.

### README.md:69 - "NVIDIA from 8 GB cards (V100 / T4 / RTX 20 and newer)" vs ik2_fit CC 60 - NOT CHANGED
The review states "no contradiction - noting for completeness": the table gives the validated reach and the
What-you-need line (README.md:240) covers Pascal second-class builds correctly. Changing it would risk the
re-checked wording for no stated error.

## Scope note

Commits: `b1f5efb8b`, `9dd59f5a1`, `b576fd27c`, plus this report. Only README.md and REPORT-d1-readme.md
changed; `serve/`, `engines/`, `tools/`, `docs/`, `saguaro.py` and the generated `serve/qfn_engine.py` were
not touched (no extraction was needed - no engine source changed). No Co-Authored-By / Claude-Session /
Assisted-by lines; each commit stages only the files it edits.
