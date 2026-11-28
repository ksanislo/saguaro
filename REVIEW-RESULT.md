# MANAGER: FIX-FIRST (card, mangonel, 2026-11-27 17:23 PST, candidate saguaro/rc124-cand2 = dev + this branch + others)
A fresh 27,299-token read on the live Q4 config (test copy), the Monitor's /metrics polled 4x/s
(scratch/card-rc2-timeline.txt): the client got the answer in 17.8 s (7 tokens) -> the TRUE read rate is ~1,580 tok/s.
The readout while reading: mean 2,393 tok/s (+51 %), MAX 9,881 tok/s (6x), first non-zero at 0.78 s, 2 zero samples
during the read. The peak the Monitor keeps is therefore far too high again, and the running value ~50 % high -
the replay test (the recorded engine lines) passes but the live readout does not match it.
Fix (commit each):
1. FIX-REPORT: replay scratch/card-rc2-timeline.txt's own fields (prompt_read / prompt_total / the time column) and
   show where the shown value departs from (tokens finished) / (now - the read's start): which input makes it 2,393
   and 9,881 (e.g. prompt_read jumping a whole chunk at once divided by a short interval; the start time taken at
   the first progress line, not the request's start; a per-interval rate instead of a cumulative one).
2. The readout = cumulative tokens read / time since THIS read started (the request's arrival at the engine), never an
   interval rate; the peak = the max of that readout over windows >= 5 s; within 10 % of the client's figure on this
   timeline. Add the timeline as a test fixture: mean within 10 % of 1,580, max <= 1.25 x 1,580.
Run only the read-rate test modules.
