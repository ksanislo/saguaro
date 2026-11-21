#!/usr/bin/env python3
"""serve/fake_solo_parity_engine.py - a stand-in for one-conversation `saguaro-glm --serve` that answers with the lines
the parity contract (REPORT-parity-output.md) has the GLM engine print: a `RESUME <reused>` line before the read, PP
lines whose ms is the read's own time, and the full 10-field DONE.  For serve/test_parity_lines.py: the StrataEngine
reader must take them as before (the reuse base rides the PP line; RESUME is not its base).

    serve/fake_solo_parity_engine.py --serve
"""
from __future__ import annotations

import sys


def out(s: str):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


out("INFO context=8192 engine_kind=glm-fast experts=74 expert_slots=64 vram_slots=48 ram_slots=96")
out("READY 8192 stop pause")
for line in sys.stdin:
    f = line.split()
    if not f:
        continue
    if f[0] == "QUIT":
        break
    if f[0] not in ("GEN", "GENI"):
        out("ERR unsupported command")
        continue
    max_new = int(f[1])
    ids = [int(x) for x in f[-1].split(",")]
    reuse = min(4, max(0, len(ids) - 1))          # (the state holds this much: the read goes on from here)
    out(f"RESUME {reuse}")
    pos = reuse
    while pos < len(ids):
        pos = min(len(ids), pos + 8)
        out(f"PP {pos} {len(ids)} {(pos - reuse) * 2.0} 500.0 {reuse}")   # (ms: the read's own time)
    for i in range(min(max_new, 3)):
        out(f"T {1000 + i}")
    out(f"DONE {min(max_new, 3)} {len(ids)} {(len(ids) - reuse) * 2.0:.1f} 5.0 stop 2 4 {reuse} 10 12")
