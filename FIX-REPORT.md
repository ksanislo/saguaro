# FIX-REPORT - P125s tiers panel, response to REVIEW-RESULT.md (round 2, the review of f2e39c87d..2b5d46b0c)

## FIX-FIRST
- Finding 1, HIGH (the sibling pin of the old STAT format, serve/test_new_engines_ui.py:465-466): FIXED.
  Commit 147a0bf85. test_monitor_qfn_stats_and_context_facts asserted the pre-P125s literal
  `"STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=%.3f disk=0.000 promo=%.3f "` is in generate.cpp -
  817ed91c1 stopped printing it (disk= is a read field), so the test was deterministically red while
  serve/test_dashboard.py:566-569 asserted the opposite. The pin now mirrors test_dashboard.py:566-567's two
  new literals against the same QFN_SRC: the printf's first chunk `"STAT tok_s=%.2f ... disk=%.3f promo=%.3f "`
  and its second chunk `"vram_used=%lld ... disk_gb=%.1f "` (generate.cpp:13867-13868). The module is not a
  plan-named one and the instruction runs only the plan's two modules, so it was not run; the asserts are pure
  substring checks and the same checks were made directly: both new literals present, the old one absent.
- Finding 2, LOW (the false sentence one section above the fix, generate.cpp:13245-13247): FIXED as the cheap
  nit. Commit e2f782c93. The comment above the INFO line still said "QFN has no SSD tier: ... the STAT lines
  report disk=0 / disk_gb=0" - the wording deleted at the batch_stat_line comment in 817ed91c1 but not this
  copy. It now says what the STAT lines do: the resident complement's (or --mmap-experts' page cache's) experts
  as the RAM tier and the rest of the files as the SSD tier, disk/disk_gb staying 0 only without a copy.
  The INFO line itself is untouched - the plan scoped bug 1 to the STAT line and the finding calls the
  behaviour compliant.

## Findings 3-4 (review: recorded / note only, no action asked)
- Finding 3, LOW (resident_expert_stats pre-tests complement_ready_ lock-free; install reallocates under no
  mutex): NOT CHANGED - the finding itself records it as established style with no action ("pinned() and
  complement_ready() have exactly this shape", install runs before serving, worst case one line one state
  behind), and round-1 LOW 5 already left it for the same consistency call (7a129ca62's message).
- Finding 4, LOW (adaptive-tier exchange storage freezes the STAT line's RAM-tier count/bytes at activation):
  NOT CHANGED - the finding is "note only"; the freeze predates the change ("the old figures ... were no
  truer"), and making the line follow commit_exchanges is a behavior change the plan does not ask for.

## Verification
The plan's two modules, one at a time, after both commits: file_expert_source_test PASS (ninja reports "no
work to do" - the target does not compile generate.cpp, and the C++ edit is comment-only), and
serve.test_dashboard 91 OK. The comment-only generate.cpp edit was syntax-checked with the build's own flags
(-fsyntax-only over the ninja -t commands line; OK, the single unused-variable warning at 16236 is
pre-existing). serve/qfn_engine.py untouched, the extractor had nothing to run (engines/qfn/serve/server.py
unchanged this round). No GPU runs.
