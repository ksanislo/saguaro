# FIX-REPORT - P124v2 review (REVIEW-RESULT.md): every finding -> what changed, or why it did not

Fixed against REVIEW-RESULT.md (review of b18e7002e). One commit per numbered finding, then this report
(force-added past the checkout's FIX-REPORT.md exclude, as the sibling branches' reports). All work in this
worktree; scratch probes under ./scratch/, never committed. The three plan-named modules ran one at a time
(/opt/llm/saguaro/.venv/bin/python -m unittest) after every Python-side fix: serve.test_read_rate_true 10 OK,
serve.test_read_rate 34 OK, serve.test_read_heartbeat 17 OK - green at every commit, and green at this tip.

## 1. Blocker: the swap-aside vector kept the OLD 5-argument on_layer_pass type - FIXED (f9377b2a2)

engines/qfn/src/program/generate.cpp:16942: added the sixth `int64_t`, matching
`Prefill::on_layer_pass` (engines/qfn/include/strata/prefill/prefill.hpp:341), so the stash
(16989) and put-back (16968) both compile again. Verified exactly as the review did, on the two move
sites with the header's own types (g++ -std=c++17): the old shape reproduces both
"no match for operator=" errors (scratch/rhb_funcrepro_old.cpp), the fixed shape compiles clean
(scratch/rhb_funcrepro_fixed.cpp). No full C++ build here (no nvcc on this box, as the REPORT said) - the
card build stays owed either way, but the type error the review proved is gone. A grep for the 5-argument
pattern across engines/qfn, include/ and src/ finds no other site. serve/qfn_engine.py needed no
re-extraction (these fixes touch no line of engines/qfn/serve/server.py; re-running
tools/saguaro/extract_qfn_engine.py regenerates serve/qfn_engine.py byte-identically - working tree clean).

## 2. Should-fix: `_own_read_rate` could compute `gap = now - None` - FIXED (3b8803563)

serve/server.py, the ended-branch fallback now reads exactly as the review prescribed:
`ended = r[2] if r is not None and r[2] is not None else st.read_t1 or now`. The case it guards is the
one the entry condition at 4522 routes here anyway - an OPEN leg whose first_token landed before
_prefill_tok_s's closer ran (that closer is skipped while not busy, and a token can land from the engine
thread between its row-builder calls): the readout now ages from the last read step instead of raising
TypeError inside metrics(). The dropped `elif r is None or r[2] is not None:` guard's behaviour is
restored; test_read_rate's retry test never enters this branch (its engine_t0 is None), and the three
modules stay green.

## 3. Nit: read-restart left `engine_t0` stale, double-counting the span - FIXED (750b626c6)

serve/server.py, the restart branch in _note_read (the engine's counter goes backwards mid-admission):
the `read_leg_ms = 0.0` line is followed by re-baselining `engine_t0` to now (when one exists - the None
case keeps the documented no-start-hook shape, where the engine's own pair is the figure). Probe
(scratch/nit3_check.py, the review's worked example: admission hook, 10 s of beats, then the counter
restarts without a new hook): span 11.0 s with the old code (the previous read's whole time inside the
open leg), 0.0 with the fix, the new read's span starting at its restart. Why unconditional (rather than
only when no hook re-baselined this leg): at the restart branch both shapes look identical - a fresh
admission's leg and a mid-admission restart both stand at r[1] == engine_t0 - so the stale-previous-read
case (the 11 s / mangonel-17 s error) and the fresh-hook case cannot be told apart. The cost on the fresh
side is bounded at the admission-to-first-beat delay (one beat, ~0.25 s, and the pair's max() floors
the span anyway); the bug fixed was whole previous reads.

## 4. Nit: `pos0=` mislabeled on the piece-boundary line - FIXED (b206488e6)

engines/qfn/src/program/generate.cpp, rhb_solo_piece: the rhb_print argument changed from the piece's END
`pos` to `-1` (with the comment saying so): chunk=0 on that line means no unit is in flight, so no unit's
first token exists to print - the protocol comments (8395-8396, 11302-11308, 16630-16640) now hold for
every printed line. Safe for every reader: the driver's _rhb parses pos0 as an int with no range
assumption (engines/qfn/serve/server.py:1777, mirrored byte-identically into serve/qfn_engine.py, which
needed no re-extraction - only generate.cpp changed), and serve/server.py:4606 requires chunk > 0 before
pos0 is used at all, so no figure moves. test_read_heartbeat's EngineSideContract pins the printf FORMAT
(the same literal), not this value - untouched, 17 green.

## 5. Nit: the "recorded lines" replay is a synthesized shape - NOT CHANGED, deliberately

There is nothing to change: ref/mangonel-engine-after-p124v.txt holds ONLY the summary line (its line 1),
no RHB beats, so "replay the recorded file through the computation" cannot literally be done - the
committed MangonelTrueReadout builds the uniform 4,608 x 6 / 230 ms shape from that summary's own numbers
and the REPORT discloses exactly this (REPORT 7, 86-91, 107-109). Hand-writing a per-chunk-cold beat
sequence instead would invent "recorded" lines the box never printed - a fake record is worse than a
disclosed synthetic one - and the manager's acceptance gates are all asserted on the committed test and
pass: never 0 after the read's first RHB (test_read_rate_true:334-337), time-average within 5% of 1,567.8
(342-346), peak <= 1.25x (347-352), closing at exactly 27,254 (355). If real per-chunk-cold RHB lines are
ever recorded on a card run, replaying them is the follow-up that would close this.

## 6. Nit: fractional counts where the annotation promises ints - PARTLY PRESENT, FIXED the rest (004f75697)

At the reviewed tip the booking ALREADY rounds: the line the finding cites (b18e7002e:4608, the `n = int(n
+ min(chunk, new - n) * pass_i/pass_n)` at the head of _note_hb's share) wraps the product, and a
mangonel-shape probe showed stat.read_pos, the r[4] window steps and _read_counted all carrying int
values - so the finding's premise did not hold at this tip (it holds for any future booking that skips
the rounding, which is the useful part of the observation). Took the review's other half: _read_counted's
return is now wrapped in int(), so the `-> int` annotation stands whatever a booking books (probe: a
deliberately fractional step leaves read_pos 233.5 while _read_counted returns 233). No committed figure
moved: the exact BRA close already restored integers and the test's 27,254 stands.

## Runs at this tip (one module at a time)

- serve.test_read_rate_true: 10 OK
- serve.test_read_rate: 34 OK
- serve.test_read_heartbeat: 17 OK
- py_compile clean on every touched file; tools/saguaro/extract_qfn_engine.py regenerates
  serve/qfn_engine.py byte-identically (no hand edits, none needed); C++ verified only by the two
  compile probes named under finding 1 (no nvcc here - the card build remains owed, as in the REPORT).
