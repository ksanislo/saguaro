# P124i - Hy4-Flash keeps a stopped prompt read (request priority, audit item 2)

## 1. Verified: a STOP during a Hy4 prompt read keeps what was read, and the next GEN reuses it

(src/hy4/hy4_main.cpp, the `--serve` GEN path; the audit's §5.2 line refs hold up.)

- A `STOP` line on stdin sets the engine's stop flag: the reader thread, :430-432 (`stop.store(true)`).
- The prompt is read in chunks and every chunk fed to the caches is appended to `state` ("the tokens in the
  caches, positions 0 ..", :969): read loop :1079-1096, `state.insert(...)` per chunk at :1086.
- The STOP is honored **between chunks** and ends the read: :1092-1095 (`if (in.stop.load() && !last) {
  cancel = true; break; }` — a STOP inside the last chunk is not taken, the `&& !last`).
- The cancel path keeps the read: :1106-1110 prints `DONE ... cancel ...` and continues the request loop **with no
  `state.clear()`** — the clears are only the engine-error path (:1101) and a picture error (:1060). So the
  conversation stays live in the caches at the chunk boundary reached.
- The next GEN of the same prompt reuses it: `from` = the shared start of `state` and the prompt's ids, :1024-1025,
  and the read loop starts at `a = from` (:1079) — after a kept read `from` is the chunk boundary reached, so only
  the rest is read (unless the sliding windows are gone: `m.resumable(from)` check :1035).
- The DONE's reused figure: the protocol is `DONE ... <reused> 0 0` (:60-61); the finished request's DONE reports
  `from` there (:1273-1274), so the resumed GEN's DONE carries the chunk boundary as its reused figure (the PP
  lines report `from` the same way, :1089). The **cancelled** read's own DONE reports the old base as reused
  (:1107) — as the audit's §10.2 noted, not the partial end.

So it does keep the read: steps 2 and 3 proceed. (No C++ change needed; the keep already exists, it is just
never exercised by the server — `serve/server.py:5726-5729` gates read preemption to `can_pause and multi` or
`keeps_stopped_read and not batch`, and the hy4 StrataEngine reports neither.)
