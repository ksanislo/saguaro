# FIX-REPORT - P125l review (REVIEW-RESULT.md): FIX-FIRST - the resume re-read the prompt from 0

Manager's card test (mangonel 4xT4, candidate saguaro/rc124-cand, tools/card_tests/lost_read.sh on the live
Inkling gpu01 config, log scratch/card-r1-engine.txt). This first commit is fix step 1: the trace of the paused
read's K/V from the pause to the resume, with file:line, and where it is lost. Steps 2 and 3 (keep the read; the
fake test and the honest continue line) follow in their own commits; the finding -> what-changed table is added
with the last commit.

## 1. Where the K/V of the paused read goes (finding 1: the trace)

The read is a solo saguaro-ik2 `GEN`; the card's prompt is a NEW conversation's FIRST prompt (~86,237 tokens in
one user message). Pause = the server's read preemption, resume = the same prompt sent again.

1. **The pause (server).** A PP tick finds a waiting normal request: serve/server.py:5849-5863 (the
   `keeps_stopped_read and not batch` gate, `step_aside` yes) -> `cut = True; break` -> `finally: gen.close()`
   (serve/server.py:5920-5921) writes STOP (serve/server.py:2140-2145) and drains to the DONE; the pause line
   prints "paused at 32,768 of 86,237" (serve/server.py:5978-5982). Card log line 39.

2. **The engine stops between chunk passes.** STOP sets `g_cancel` (src/ik2/ik2_main.cu:10181-10182); the read
   loop looks at it only BEFORE a chunk pass (src/ik2/ik2_main.cu:10817, :10838) - at that moment the passes up
   to 32,768 have run and their K/V rows are in the cache: the sliding visits' ring rows and the four full
   visits' rows [0, 32,768).

3. **What is lost at the stop - the cancelled branch (src/ik2/ik2_main.cu:10870-10892).** It rolls the
   conversation back to the latest SNAPSHOT it kept: snapshots exist only at a finished turn's end
   (:10893-10901, taken after a WHOLE prompt read) and at the last user message's start `p_ms`
   (:10799-10809, taken only after that message's leading segment completed, :10860-10866). For this request
   there is no snapshot at all: it is the first prompt (no turn ever finished) and `p_ms` is 0 - its `user_token`
   sits at ~20, below the `i >= L + 64` bar (:10807), so the message snapshot is never armed - and even a
   finished one would be at the MESSAGE START (~20 tokens), not at 32,768: the passes had read 32,748 tokens
   INSIDE the single message, and no snapshot of them is ever taken. `best < 0` -> `reset()` (:10884): every
   snapshot cleared, `hist` cleared. The rows stay in the cache for the moment, but nothing references them and
   no snapshot can restore the recurrent states at 32,768. **The read is lost here; the resume could not
   continue from 32,768 even with no request in between.**

4. **The normal request runs (card log lines 40-48).** `hist` is empty (the reset), so `reach()` gives
   `r_best = 0` (:10676-10702) and the branch at :10703-10704 has nothing to park; the short conversation reads
   from 0 and its passes WRITE ROWS from position 0 - overwriting the first rows of the discarded read; its
   snapshots land at ~20/~38.

5. **The resume (card log lines 51-54).** The same 86,237-prompt arrives: `reach()` matches the short
   conversation only at the template prefix and no snapshot reaches past it -> `r_best = 0` -> the parked path
   :10703-10704 parks the short conversation first - "a conversation of 38 tokens set aside (36 MiB, 20 ms)"
   (log 51/52, park_active at :9949-9973) - and `reset()`s. `cached = L = 0` (:10787): the first PP line reports
   "8,192 of 86,237 ... reused 0" (log 54; the reuse field is `(int) L`, :10829-10830/:10850-10851) - a whole
   second read of the prompt, ~26 s thrown away per cut.

6. **What the server said.** The line driver reports no `reused` attribute, so the continue line fell to the
   numberless "the low prompt read continues from where the engine kept it" (serve/server.py:5817-5819) -
   printed whether the engine kept anything or not (log 53). The engine DID report its figure over the pipe
   (the PP reuse field, and DONE's `cached` at :10886 / :11096 -> `last["reused"]`, serve/server.py:1866) - the
   server's continue line never looked at it for this driver.

**In one line:** the K/V is lost inside the engine at the STOP, at src/ik2/ik2_main.cu:10870-10892 - a cut read
rolls back to the last snapshot, and a first prompt has no snapshot past ~token 0, so `reset()` drops everything
the passes read; the short request then overwrites the rows; and the server's numberless "continues" line
(serve/server.py:5817-5819) claimed a continuation while the engine's own PP reuse said 0.

## 2. The keep (fix step 2, next commit)

The engine has everything needed to keep a cut read: the rows a stopped read wrote are still in the cache when
it stops (nothing has run), and `snap_copy` (:9612-9678) can snapshot the recurrent states and the sliding
visits' window rows at ANY position (that is what park_active does at a conversation's end, :9952-9960); the
full visits' rows are exactly what `full_rows` (:9882-9999) carries to and from the kept conversations. So:

- the cancelled branch keeps the conversation AT the chunk boundary the STOP landed on (a new `read_pos`), with
  a snapshot there, instead of rolling back to a (nonexistent) older snapshot;
- a DIFFERENT conversation arriving (its `reach()` reaches less than the kept read's end) parks that read with
  everything it read - `park_active()` + `reset()`, the same way a new conversation already parks a finished
  one (:10703-10704) - instead of rolling back over its rows;
- the read's own request coming back matches its kept end exactly (`r_best == hist.size()`, live in the cache,
  or the parked copy via `unpark` whole) and its PP/DONE report `cached = 32,768` - the reused figure the
  review asks the resume to show.

## 3. The honest line (fix step 3, next commit)

serve/server.py: the PP parse keeps the reuse field separately (`progress_reuse`, None when the engine prints
no field, so a GLM line is untouched); the continue line prefers `reused` (QFN's RESUME), then
`progress_reuse` (saguaro-ik2's PP field): a figure of 0 prints "continued: 0 of N tokens reused" (the P124b
rule), never "continues", and the numberless wording is left to engines that report no figure at all (and waits
for the read's first report before claiming anything). serve/test_priority.py: the ik2 fake models the fixed
engine (a cut read stays at the boundary it stopped at, the resume continues there and reports the figure), a
kept cut is never re-read, and the card's lost shape (engine re-reads from 0, reuse 0) prints the 0 and not the
claim.
