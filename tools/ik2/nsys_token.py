#!/usr/bin/env python3
"""tools/ik2/nsys_token.py - one decode token's GPU time by kind, per card, from an nsys sqlite export of a
saguaro-ik2 CLI run (nsys profile --cuda-graph-trace=node; nsys export --type sqlite).

    nsys_token.py RUN.sqlite [--tokens N]

The decode tokens are the CUDA graphs' launches (graphId != 0); a token's window is from its first graph kernel to
the last before the next token. Kinds by kernel name; copies by kind (H2D / D2H / D2D / peer). Idle: a card's window
time with no kernel or copy of its own running.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict

KINDS = [
    ("attention", r"attn|prep_qkv|rel_|sink|kv_|flash"),
    ("experts", r"expert|swiglu|moe_|off_|fetch|grp_|two_take|stage|combine|scatter"),
    ("router", r"route|topk|router|probs_b"),
    ("dense", r"mv_|gemv|q8p|gemm|mix_rms|rms|conv4|conv_|norm|res_|argmax|logit|embed|gate_apply|cand"),
]


def kind_of(name: str) -> str:
    for k, rx in KINDS:
        if re.search(rx, name, re.I):
            return k
    return "other"


def cols(db, table):
    return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--skip", type=int, default=8, help="tokens skipped at the start (graph capture, warm-up)")
    ap.add_argument("--out", help="the run's stdout: its 'generation: N tokens in S s' line sets the window (the last S s)")
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    kc = cols(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    gcol = "graphId" if "graphId" in kc else None
    nm = "demangledName" if "demangledName" in kc else "shortName"
    q = f"SELECT start, end, deviceId, {nm}, {gcol or 0}, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ker = [(s, e, d, names.get(n, str(n)), g) for s, e, d, n, g, _ in db.execute(q)]
    mc = cols(db, "CUPTI_ACTIVITY_KIND_MEMCPY")
    mg = "graphId" if "graphId" in mc else "0"
    src = "srcDeviceId" if "srcDeviceId" in mc else "deviceId"
    dst = "dstDeviceId" if "dstDeviceId" in mc else "deviceId"
    mem = list(db.execute(f"SELECT start, end, deviceId, bytes, copyKind, {mg}, {src}, {dst} FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"))
    if a.out:
        import re as _re
        m = _re.search(r"generation: (\d+) tokens in ([\d.]+) s", open(a.out).read())
        n_gen, gen_s = int(m.group(1)), float(m.group(2))
        t_end = max(max(k[1] for k in ker), max((x[1] for x in mem), default=0))
        t1 = t_end
        t0 = int(t_end - gen_s * 1e9 * 0.95)   # (the last 95 %: past the first token's own launches)
        n_tok = n_gen * 0.95
    else:
        graph = [k for k in ker if k[4]]
        if not graph:
            print("no graph kernels: decode not found")
            return 1
        first = graph[0][3]
        starts = [k[0] for k in graph if k[3] == first]
        if len(starts) < a.skip + 4:
            a.skip = 0
        starts = starts[a.skip:]
        t0, t1 = starts[0], starts[-1]
        n_tok = len(starts) - 1
    devs = sorted({k[2] for k in ker})
    busy = {d: defaultdict(float) for d in devs}
    iv = {d: [] for d in devs}
    for s, e, d, n, g in ker:
        if s < t0 or s >= t1:
            continue
        busy[d][kind_of(n)] += (e - s)
        iv[d].append((s, e))
    copies = defaultdict(lambda: [0, 0, 0.0])   # kind -> count, bytes, ns
    ck = {1: "H2D", 2: "D2H", 8: "D2D", 10: "peer"}
    for s, e, d, b, k, g, sd, dd in mem:
        if s < t0 or s >= t1:
            continue
        key = ck.get(k, str(k)) + (f" {sd}->{dd}" if k == 10 else f" dev{d}")
        c = copies[key]
        c[0] += 1
        c[1] += b
        c[2] += e - s
        iv[d].append((s, e))
    span = (t1 - t0) / n_tok
    print(f"{n_tok} tokens, {span / 1e6:.3f} ms a token ({1e9 / span:.1f} tok/s); per token, ms:")
    for d in devs:
        ivs = sorted(iv[d])
        u, cs, ce = 0.0, None, None
        for s, e in ivs:
            if cs is None or s > ce:
                if cs is not None:
                    u += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        if cs is not None:
            u += ce - cs
        row = "  ".join(f"{k} {busy[d][k] / n_tok / 1e6:.3f}" for k in [x[0] for x in KINDS] + ["other"])
        print(f"  GPU {d}: {row}  | busy {u / n_tok / 1e6:.3f}  idle {(span - u / n_tok) / 1e6:.3f}")
    for k, (c, b, ns) in sorted(copies.items()):
        print(f"  copies {k}: {c / n_tok:.1f} a token, {b / n_tok / 1024:.1f} KiB, {ns / n_tok / 1e6:.3f} ms "
              f"({ns / max(c, 1) / 1e3:.1f} us each)")
    # the heaviest kernels
    tot = defaultdict(float)
    for s, e, d, n, g in ker:
        if t0 <= s < t1:
            tot[(d, n[:70])] += e - s
    print("  top kernels (ms a token):")
    for (d, n), ns in sorted(tot.items(), key=lambda x: -x[1])[:18]:
        print(f"    GPU {d} {ns / n_tok / 1e6:7.3f}  [{kind_of(n)}] {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
