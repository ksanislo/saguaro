# FIX-REPORT - P125s tiers panel, response to REVIEW-RESULT.md

## FIX-FIRST
- HIGH 1 (nothing committed; the review range f2e39c8..HEAD empty): FIXED. The work is committed as the plan
  orders, one commit per bug: 817ed91c1 (bug 1, the SSD tier) and aac993d9d (bug 2, the pipelined-solo STAT
  line). generate.cpp held both bugs' hunks; the split used `git apply --cached` of a patch with the four
  pre-22000-line hunks staged for bug 1, the rest committed as bug 2. The new header
  engines/qfn/include/strata/program/tier_stats.hpp is `git add`ed in the bug-1 commit (the review's build
  hazard), and each commit verifies standalone: HEAD is the green `tools/qfn/build.sh` build ("QFN BUILD OK
  aac993d9d", scratch/p125s-fix-build), and commit 1's only delta from HEAD is aac993d9d's three additive
  generate.cpp hunks - its generate.cpp and expert_source.cpp compile standalone with the build's own flags
  (-fsyntax-only over `git show 817ed91c1:` copies): both OK.
- HIGH 2 (REPORT-p125s-tiers-panel.md missing): FIXED. Written at the repo root with what was actually run in
  this fix session (the plan's build recipe -> QFN BUILD OK, file_expert_source_test PASS, serve.test_dashboard
  91 OK, re-runs after the nit, the standalone-commit checks). Not committed by design: `REPORT-*.md` is in
  .git/info/exclude and commit 81a6d581c drops task reports from the tree - the file exists, which is what the
  review contract needs.

## Findings 3-7 (review: notes/follow-ups, not blockers)
- MEDIUM 3 (plain --mmap-experts ram_gb: 0.0 -> the mapped experts' bytes): NOT CHANGED - deliberate.
  PLAN:9-10 mandates "the mmap page cache when --mmap-experts" as the RAM tier, and the old pairing
  (ram_used=9728 with ram_gb=0.0) was the inconsistency the plan calls a bug. The change is recorded in
  817ed91c1's message, as the review suggests for the commit messages; arena and resident-copy values are
  byte-identical to the old ones and batch paths keep field names/order.
- LOW 4 (count/bytes can straddle one relief trim): FIXED - the cheap nit. resident_experts() +
  resident_expert_bytes() become one resident_expert_stats(count, bytes) reading both under a single
  complement_mu_ pass (expert_source.cpp:4322; generate.cpp:13858 is the only caller). Commit 7a129ca62;
  rebuilt and both plan-named tests re-run green.
- LOW 5 (complement_ready_ read lock-free): NOT CHANGED, per the finding itself ("consistency says leave
  it") - it copies the established style of the existing readers (pinned(), expert_source.cpp:4778) and no
  figure's correctness depends on it. Noted in 7a129ca62's message.
- LOW 6 (lent-region experts counted in both tiers at the edge): NOT CHANGED - the finding states no
  negative or nonsensical share reaches the bar (tier_stats.hpp clamps, tierShares recomputes) and the plan's
  "the experts the resident complement actually holds" wording sanctions the overlap. Changing the lend
  region's accounting is a behavior change the plan does not ask for.
- LOW 7 (no automated exercise of the new pipelined-solo STAT call): NOT CHANGED - the path needs a two-GPU
  layer split and the plan allows no GPU runs, so nothing CI-runnable was available; the review itself says
  "a gap to note, not to block on". Noted in aac993d9d's message.

## Verification
Fresh build in scratch/p125s-fix-build + both plan-named test modules run one at a time, before the nit and
again after it: file_expert_source_test PASS, serve.test_dashboard 91 OK. serve/qfn_engine.py untouched and
not needed (extractor had nothing to do - engines/qfn/serve/server.py unchanged). Details:
REPORT-p125s-tiers-panel.md.
