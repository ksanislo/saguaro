#!/usr/bin/env python3
"""tools/ik2/par_check.py - saguaro-ik2 --parallel N: two greedy answers at once through the multi protocol (GEN ...
rid=<id>, tagged T / DONE lines), each held to its solo answer, with each chat's and the pair's tok/s.

    tools/ik2/par_check.py --gpus 2,3 [--build build-ik2up] [--parallel 2] [--ctx 131072] [--gen 256] [--arg=--chunk --arg=8192]

The cards are the ones named (CUDA_VISIBLE_DEVICES); IK2_NO_CLAIM=1 for a runner that holds them.  When READY has no
multi= the engine did not take --parallel: the line says "not taken" with the engine's NOTE lines (exit 0).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ik2"))
sys.path.insert(0, str(ROOT / "tools"))
import bench as B       # noqa: E402
import t4_check as TC   # noqa: E402

QS = ("Explain in a few paragraphs how a refrigerator moves heat out of its inside.",
      "Write a short story about a lighthouse keeper who finds a message in a bottle.")


def run(eng, reqs: dict, gen: int, timeout: float) -> dict:
    """Send every request (rid -> ids) at once; rid -> (tokens, decode tok/s from DONE, first T time, DONE time)."""
    t0 = time.time()
    for rid, ids in reqs.items():
        eng.p.stdin.write("GEN %d temperature=0 rid=%d %s\n" % (gen, rid, ",".join(map(str, ids))))
    eng.p.stdin.flush()
    out = {rid: {"toks": [], "first": None, "done": None, "tps": 0.0} for rid in reqs}
    left = set(reqs)
    end = time.time() + timeout
    while left and time.time() < end:
        got = eng.until(lambda l: l.startswith(("T ", "DONE ", "ERR ")), max(1.0, end - time.time()))
        if got is None:
            raise RuntimeError("engine ended: " + " | ".join(eng.lines[-8:]))
        f = got[-1].split()
        if len(f) < 2 or not f[1].isdigit() or int(f[1]) not in out:
            continue
        r = out[int(f[1])]
        if f[0] == "T":
            r["toks"].append(int(f[2]))
            r["first"] = r["first"] or time.time() - t0
        elif f[0] == "ERR":
            raise RuntimeError("ERR: " + " ".join(f[2:]))
        else:   # DONE rid generated prompt prompt_ms gen_ms ...
            r["done"] = time.time() - t0
            n, ms = int(f[2]), float(f[5])
            r["tps"] = 1000.0 * n / ms if ms > 0 else 0.0
            left.discard(int(f[1]))
    if left:
        raise RuntimeError("no DONE for %s" % sorted(left))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build-ik2up")
    ap.add_argument("--model", default=TC.MODEL)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--arg", action="append", default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    out = a.out or a.build / "par_check"
    out.mkdir(parents=True, exist_ok=True)
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_gguf(a.model)
    ids = [tok.encode(TC.chat_prompt(q), parse_special=True) for q in QS]
    B.wait_free(gpus)
    line = B.claim("par-check", gpus)
    eng = None
    try:
        args = ["--model", a.model, "--serve", "--max-context", str(a.ctx), "--gpus", str(len(gpus)),
                "--parallel", str(a.parallel)] + a.arg
        eng = TC.Engine(a.build / "saguaro-ik2", args, TC.env_for(gpus), out / "engine.log")
        got = eng.until(lambda l: l.startswith("READY "), 1800)
        if got is None:
            print("PAR CHECK FAILED: the engine did not come up: " + " | ".join(eng.lines[-8:]))
            return 1
        notes = [l for l in got if l.startswith("NOTE ") or l.startswith("parallel:")]
        multi = next((x for x in got[-1].split() if x.startswith("multi=")), None)
        print("READY: %s" % got[-1])
        for n in notes:
            print("  " + n)
        if not multi:
            print("PAR CHECK: --parallel %d not taken (READY without multi=)" % a.parallel)
            return 0
        solo = {}
        for i, x in enumerate(ids):
            solo[i] = run(eng, {10 + i: x}, a.gen, a.timeout)[10 + i]
            print("solo %d: %d tokens, %.1f tok/s, first token %.2f s" % (i, len(solo[i]["toks"]), solo[i]["tps"],
                                                                         solo[i]["first"] or -1))
        both = run(eng, {20 + i: x for i, x in enumerate(ids)}, a.gen, a.timeout)
        ok = True
        for i in range(len(ids)):
            r = both[20 + i]
            same = r["toks"] == solo[i]["toks"]
            k = next((j for j, (p, q) in enumerate(zip(r["toks"], solo[i]["toks"])) if p != q),
                     min(len(r["toks"]), len(solo[i]["toks"])))
            ok = ok and same
            print("together %d: %d tokens, %.1f tok/s; the solo answer's tokens: %s" %
                  (i, len(r["toks"]), r["tps"], "alike" if same else "differ from token %d" % k))
        n_all = sum(len(r["toks"]) for r in both.values())
        span = max(r["done"] for r in both.values()) - min(r["first"] for r in both.values())
        print("together: %d tokens in %.2f s = %.1f tok/s for the pair (solo %.1f / %.1f)" %
              (n_all, span, n_all / max(span, 1e-6), solo[0]["tps"], solo[1]["tps"]))
        print("PAR CHECK %s" % ("PASSED" if ok else "FAILED (answers differ from solo)"))
        return 0 if ok else 1
    finally:
        if eng is not None:
            eng.close()
        B.release(line)


if __name__ == "__main__":
    sys.exit(main())
