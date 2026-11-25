# Staying based on upstream: how Saguaro takes Project Maya and Strata changes

Goal: Saguaro stays upstream Project Maya + our changes on top, with our own names (SAGUARO_*, saguaro-*, Saguaro
branding). Every upstream release is taken in full. Intakes must stay routine: a few hours, not a rewrite.

## 1. Branches
- `upstream/main` + upstream tags: a read-only mirror (git remote `origin` = mw00/project-maya).
- `saguaro/qfn-vram` = our main (GitHub ksanislo/saguaro main). Releases are tags on it (v1.2.0, ...).
- Our work: one topic branch per change, merged into main (fork-branch-layout rule). The integration branch for a
  release (today saguaro/qfn-first) collects them before main.

## 2. The rename layer (the main source of pointless conflicts)
- Upstream code arrives with upstream names; ours are SAGUARO_* settings, saguaro-* programs, Saguaro branding.
- Before an intake merge, the upstream release is renamed on a throwaway branch by ONE tool run
  (tools/saguaro/rename_*.py: settings MAYA_X -> SAGUARO_X by pattern, so new upstream settings are caught automatically;
  program/brand names; an explicit exceptions list). That renamed copy is what gets merged, so the merge only shows
  REAL conflicts. The tool is idempotent and tested; a missed name is a tool bug, fixed in the tool, not by hand.
- Optional later: a compatibility map so a user's MAYA_* setting still works (read as SAGUARO_*).

## 3. Divergence register (why we differ)
- docs/DIVERGENCE.md: per engine, each of our modifications - what, why, where (files/functions), its switch, its
  measured gain, and whether upstream could take it. Plus what we DROP from upstream on purpose (the Maya Qwen path:
  its hunks are dropped automatically by path). A conflict is resolved by reading the register, not by guessing.
- Every new topic branch adds its line when merged.

## 4. Keeping our changes easy to carry
- Prefer switch-gated additions in our own files/functions with a small hook in the shared code, over edits inside
  upstream's hot functions. Where we must edit upstream code, keep the edit small and marked (`// saguaro:` comment).
- Changes upstream might want stay as clean topic branches (PR-ready by construction), even though we don't send PRs.

## 5. Cadence
- Each upstream release: intake within a day or two (rename run -> merge -> gates below -> into the next Saguaro release).
- Open upstream PRs: a triage pass about weekly (take anything supportable for ANY hardware; inexact ones opt-in or
  KL-checked). Each pass resumes from the last one recorded here: new PRs above its newest number, plus older open
  PRs updated since its snapshot.
- QFN (engines/qfn, our own port of Strata): a sweep of upstream Strata about every two weeks; `docs/DIVERGENCE.md`
  records the last swept Strata commit. Exact gains taken, inexact ones opt-in.

## 6. Gates for every intake
- Build all engines (CUDA sm_75 + the HIP syntax check for gfx1201 / gfx1100); tools + serve test suites = main's pass
  set + new tests.
- Where upstream says "same tokens": T4 SASS / output identity spot check. Where upstream changes answers: KL vs the
  previous release on our T4s (opt-in if it fails our bar). Q8_0 stays exact.
- Smoke every live profile (mangonel llama-swap, mortar :8080); then it ships in the next Saguaro release.

## 7. Triage log (open upstream PRs)
| Date | Newest PR checked | Snapshot | PRs | Report |
|---|---|---|---|---|
| 2026-11-24 | #1236 | open PRs as of 2026-11-24 ~23:00 UTC | 143 open | `docs/triage/TRIAGE-upstream-prs-20261124.md` |
