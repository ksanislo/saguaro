# T9a - QFN settings docs report

## What changed

- **Step 1 - inventory** (`3f87253f8`): generated `scratch/qfn-settings.txt` with
  `scratch/inventory.py` (run with `/opt/llm/saguaro/.venv/bin/python`). It lists every setting the
  QFN engine and server can take: **649** `SAGUARO_*` env vars read in the `engines/qfn` C++
  (9 of them only in test/bench tools), **57** read in the server's Python (`serve/`,
  `strata_control/`), **185** engine argv flags from `generate.cpp`'s parser with their `usage()`
  text and defaults, and **82** `config.json` keys read by `serve/server.py` (the `vision.*` keys
  by its `class Vision`). `scratch/` is gitignored, so the commit is empty and only names the
  inventory in its message; the file itself is on disk, untracked.
- **Step 2 - docs** (`e7db6128f`): appended `## Qwen3.8-Flash-Next (saguaro-qfn)` to
  `docs/SETTINGS.md` (217 lines, append-only - the diff touches nothing above it, so the
  CYBER-FROST section is untouched). It has:
  - an intro saying where the exhaustive list lives (the `engines/qfn` source comments carry the
    ~600 remaining internal kernel switches; they are pointed to, not transcribed);
  - an **env vars** table (~63 rows) covering everything the server reads, everything the two
    live `ref/` configs pass, and the launch/acquisition vars the setup sets;
  - a **Turing (sm_75/T4) tuning** sub-section: `SAGUARO_SM75_PERF` plus its member switches
    (`SM75_STACK`, `SM75_Q8G`, `SM75_KQL`, `SM75_BF16_TC(_SKIP)`, `SM75_MMQ_KQ/DIRECT/KTAIL/SWIGLU_Q8`,
    `SM75_SELECT`, `GDN_KEYHEAD`, `ARGMAX_ROWS_SM70`), each row saying whether it gives bitwise the
    same tokens or changes the sums; the live configs set these through `"env"`;
  - an **engine arguments** table (~44 rows) for what the server actually passes, from
    `generate.cpp`'s `usage()`;
  - a **config keys** table covering every key `server.py` reads of `config.json`.
- **Step 3 - settings read nowhere**: `SAGUARO_BATCH_READS_OVERTAKE` and
  `SAGUARO_PARK_STAGING_MIB` appear in `ref/Qwen3.8-Flash-Next-Q8_0.json` and
  `ref/Qwen3.8-Flash-Next-UD-Q4_K_XL.json` (and in this plan), and in **no** code anywhere in the
  repo (repo-wide grep, both spellings and as substrings). They are not documented as working:
  the new section ends with a note that they are set in those configs but read nowhere and do
  nothing.

## Files touched

- `docs/SETTINGS.md` - new section appended (step 2 commit).
- `REPORT-t9-docs.md` - this file.
- Untracked, gitignored: `scratch/qfn-settings.txt` (the inventory) and `scratch/inventory.py`
  (its generator), plus scratch grep dumps used while writing the rows.
- No code was changed; no other docs touched.

## Tests

- None to run (docs only, per plan).
- Verified the new section's markdown: scripted check that every table row has the same column
  count (4 unescaped pipes) as its table's header - all 4 tables pass, with `|` inside option
  lists escaped as `\|`. The step-2 diff is insertions only, so the pre-existing file content,
  including the CYBER-FROST section, is byte-identical.

## Open questions

- Step 1 had nothing committable (`scratch/` is gitignored); it was marked with an empty commit
  whose message carries the inventory summary. If the plan intended the inventory itself tracked,
  it needs an exception to the "scratch only in ./scratch" rule.
- The two dead switches are documented as inert at the end of the new section (rather than left
  fully undocumented) so a reader copying the ref configs learns they do nothing. If the plan
  wants them absent from SETTINGS.md entirely, delete that closing note.
