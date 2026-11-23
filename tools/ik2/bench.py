#!/usr/bin/env python3
"""tools/ik2/bench.py - saguaro-ik2 speeds on one load: decode over a long answer at several context depths, the
read of each prompt, the time to the first token.  Greedy, through the engine protocol; one CLAIM line in
/tmp/gpu-status for the run.

    tools/ik2/bench.py --gpus 2 --ctx 32768 --depths 0,8000,30000 [--think] [--gen 256] [--env K=V ...]
                       [--arg=--chunk --arg=1024] [--repeat 2] [--label NAME] [--out FILE.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ik2"))
sys.path.insert(0, str(ROOT / "tools"))
import t4_check as TC   # noqa: E402
CLOCKS = (os.environ.get("SAGUARO_CLOCK_CMD") or "sudo -n llm-gpu-clocks").split()   # a command taking lock / unlock

MODEL = TC.MODEL
STATUS = Path("/tmp/gpu-status")
TASK = ("\n\nWrite a detailed, flowing essay of at least 500 words about what the text above describes and why "
        "it matters. Do not use lists.")


NO_CLAIM = bool(os.environ.get("IK2_NO_CLAIM"))   # (a runner that holds the cards itself: no waiting, no line)


def claim(tag: str, gpus) -> str:
    if NO_CLAIM:
        return ""
    line = "CLAIM ik2-agent: %s cards %s %s" % (tag, ",".join(map(str, gpus)), time.strftime("%H:%M:%S %Z"))
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    return line


def release(line: str):
    if NO_CLAIM:
        return
    try:
        keep = [l for l in STATUS.read_text().splitlines() if l != line]
        STATUS.write_text("\n".join(keep) + ("\n" if keep else ""))
    except FileNotFoundError:
        pass


def wait_free(gpus=(2, 3)):
    """Until no other agent's line is in /tmp/gpu-status (the standing llama-swap "normal use" note aside) and the
    cards are empty."""
    if NO_CLAIM:
        return
    while True:
        txt = STATUS.read_text() if STATUS.exists() else ""
        busy = [l for l in txt.splitlines() if l.startswith(("CLAIM", "BUSY", "WAIT")) and "normal use" not in l]
        used = [g for g, (u, _) in TC.smi(list(gpus)).items() if u > 500] if not busy else []
        if not busy and not used:
            return
        print("waiting on:", busy[0] if busy else "cards %s not empty" % used, flush=True)
        time.sleep(60)


def corpus() -> str:
    parts = [p.read_text(errors="replace") for p in sorted((ROOT / "docs").glob("*.md"))]
    return "\n\n".join(parts)


def prompt_ids(tok, body_ids: list[int], think: bool, salt: str) -> list[int]:
    head = "<|message_system|><|content_text|>Thinking effort level: %s<|end_message|>" % ("0.9" if think else "0")
    ids = tok.encode(head + "<|message_user|><|content_text|>" + salt + "\n", parse_special=True) + body_ids
    ids += tok.encode(TASK + "<|end_message|>" + ("" if think else "<|message_model|><|content_text|>"),
                      parse_special=True)
    return ids


def timed_gen(eng, ids, max_new, timeout, keys=""):
    """One greedy request: tokens, new prompt tokens, read ms, decode ms, seconds to the first token, stat lines."""
    t0 = time.time()
    eng.p.stdin.write("GEN %d temperature=0 %s%s\n" % (max_new, keys + " " if keys else "", ",".join(map(str, ids))))
    eng.p.stdin.flush()
    first = [None]

    def pred(l):
        if first[0] is None and l.startswith("T "):
            first[0] = time.time() - t0
        return l.startswith("DONE ")
    got = eng.until(pred, timeout)
    if got is None:
        return None
    d = got[-1].split()
    toks = [int(l.split()[1]) for l in got if l.startswith("T ")]
    stats = [l for l in got if "stats" in l or "sources" in l]
    return dict(toks=toks, new=int(d[2]) - int(d[8]), read_ms=float(d[3]), dec_ms=float(d[4]), ttft=first[0],
                reason=d[5], stats=stats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build-ik2")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--depths", default="0,8000,30000")
    ap.add_argument("--think", action="store_true", help="also a thinking-on answer at each depth")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1, help="answers per depth after its read (the same request again)")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--arg", action="append", default=[])
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--ab", type=int, default=0, help="at each depth, after its read: N pairs of follow-up turns, the "
                    "token path off (fast=0) and on in turn - and one regenerated answer each way, compared token by token")
    ap.add_argument("--lock", action="store_true", help="GPU clocks pinned for the run (llm-gpu-clocks), unpinned after")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_gguf(MODEL)
    text_ids = tok.encode(corpus(), parse_special=False)
    wait_free()
    line = claim("bench " + a.label, gpus)
    rows = []
    eng = None
    try:
        if a.lock:
            subprocess.run(CLOCKS + ["lock"], capture_output=True)
        before = TC.smi(gpus)
        if any(u > 500 for u, _ in before.values()):
            print("cards not empty:", before)
            return 2
        env = TC.env_for(gpus)
        env["SAGUARO_IK2_STATS"] = "1"
        for kv in a.env:
            k, v = kv.split("=", 1)
            env[k] = v
        args = ["--model", MODEL, "--serve", "--max-context", str(a.ctx), "--gpus", str(len(gpus))] + a.arg
        t0 = time.time()
        eng = TC.Engine(a.build / "saguaro-ik2", args, env, Path("/tmp/ik2-bench-%s.log" % a.label))
        got = eng.until(lambda l: l.startswith("READY "), 1800)
        if got is None:
            print("engine did not come up:", *eng.lines[-10:], sep="\n  ")
            return 1
        load_s = time.time() - t0
        vram = {g: u for g, (u, _) in TC.smi(gpus).items()}
        print("%s: loaded in %.0f s; VRAM MiB %s" % (a.label, load_s, vram), flush=True)
        for l in got:
            if any(k in l for k in ("expert cache", "resident", "warm:", "pool ", "hot set")):
                print("  ", l)
        n = 0
        for depth in [int(x) for x in a.depths.split(",")]:
            for think in ([False, True] if a.think else [False]):
                n += 1
                ids = prompt_ids(tok, text_ids[n * 1000:n * 1000 + depth], think, "[%s %d]" % (a.label, n))
                for rep in range(a.repeat):
                    r = timed_gen(eng, ids, a.gen, a.timeout)
                    if r is None:
                        print("engine ended:", *eng.lines[-10:], sep="\n  ")
                        return 1
                    nt = len(r["toks"])
                    row = dict(label=a.label, gpus=len(gpus), depth=len(ids), think=think, rep=rep, new=r["new"],
                               read_s=r["read_ms"] / 1e3, read_tps=r["new"] / max(r["read_ms"], 1e-3) * 1e3 if r["new"] else 0,
                               ttft=r["ttft"], n=nt, dec_tps=nt / max(r["dec_ms"], 1e-3) * 1e3, reason=r["reason"],
                               load_s=load_s, vram=vram, stats=r["stats"], toks=r["toks"])
                    rows.append(row)
                    print("  depth %6d %s rep %d: read %6d tok %7.2f s %6.0f tok/s | first token %6.2f s | %3d tok "
                          "%5.1f tok/s (%s)" % (len(ids), "think" if think else "plain", rep, r["new"], row["read_s"],
                                                 row["read_tps"], r["ttft"] or -1, nt, row["dec_tps"], r["reason"]),
                          flush=True)
                    for s in r["stats"]:
                        if s.startswith("decode stats") or s.startswith("prefill stats"):
                            print("       ", s[:150])
                if not a.ab:
                    continue
                follow = tok.encode("<|message_user|><|content_text|>Continue with another long paragraph on a new "
                                    "aspect.<|end_message|><|message_model|><|content_text|>", parse_special=True)
                cur = list(ids)
                sp = {0: [], 1: []}
                for i in range(2 * a.ab):
                    fast = (i + (i // 2)) % 2   # (0 1 1 0 0 1 1 0: each arm first as often)
                    cur = cur + r["toks"] + follow if r["reason"] != "length" else cur + r["toks"] + \
                        tok.encode("<|end_message|><|content_model_end_sampling|>", parse_special=True) + follow
                    r = timed_gen(eng, cur, a.gen, a.timeout, "fast=%d" % fast)
                    if r is None:
                        print("engine ended:", *eng.lines[-10:], sep="\n  ")
                        return 1
                    nt = len(r["toks"])
                    tps = nt / max(r["dec_ms"], 1e-3) * 1e3
                    if nt >= 100:
                        sp[fast].append(tps)
                    st = [x for x in r["stats"] if x.startswith("decode stats")]
                    print("    ab depth %6d fast=%d: read %4d tok %5.2f s | %3d tok %5.1f tok/s | %s" %
                          (len(cur), fast, r["new"], r["read_ms"] / 1e3, nt, tps, st[0][14:90] if st else ""), flush=True)
                    rows.append(dict(label=a.label, gpus=len(gpus), depth=len(cur), ab=fast, n=nt, dec_tps=tps,
                                     new=r["new"], read_s=r["read_ms"] / 1e3))
                if sp[0] and sp[1]:
                    m0, m1 = sum(sp[0]) / len(sp[0]), sum(sp[1]) / len(sp[1])
                    print("    ab depth ~%d: fast=0 %.1f tok/s (n=%d), fast=1 %.1f tok/s (n=%d): %+.1f%%" %
                          (len(cur), m0, len(sp[0]), m1, len(sp[1]), (m1 / m0 - 1) * 100), flush=True)
                if True:   # the same prompt both ways: greedy tokens alike?
                    ra = timed_gen(eng, ids, a.gen, a.timeout, "fast=1")
                    rb = timed_gen(eng, ids, a.gen, a.timeout, "fast=0")
                    rc = timed_gen(eng, ids, a.gen, a.timeout, "fast=0")
                    if ra and rb and rc:
                        def same(x, y):
                            n = 0
                            while n < min(len(x), len(y)) and x[n] == y[n]:
                                n += 1
                            return n
                        print("    same prompt, depth %d: fast=1 vs fast=0 alike for %d of %d tokens; fast=0 twice: %d of %d"
                              % (len(ids), same(ra["toks"], rb["toks"]), len(rb["toks"]), same(rb["toks"], rc["toks"]),
                                 len(rc["toks"])), flush=True)
        vram2 = {g: u for g, (u, _) in TC.smi(gpus).items()}
        print("  VRAM MiB after: %s" % vram2)
    finally:
        if eng:
            eng.close()
        time.sleep(1)
        print("  cards after exit:", {g: u for g, (u, _) in TC.smi(gpus).items()})
        if a.lock:
            subprocess.run(CLOCKS + ["unlock"], capture_output=True)
        release(line)
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
