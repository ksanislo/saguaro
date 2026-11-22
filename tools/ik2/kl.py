#!/usr/bin/env python3
"""tools/ik2/kl.py - saguaro-ik2's token paths against each other over forced real text: each arm (a set of
settings) runs the same tokens one at a time (SAGUARO_IK2_FORCE) and writes every position's logits
(SAGUARO_IK2_LOGITS_OUT); the arms are then held to the first: KL(first || arm) mean / median / p99 / max, the same
top token.

    tools/ik2/kl.py --gpus 2 --n 2000 --arm base:SAGUARO_IK2_FAST=0 --arm base2:SAGUARO_IK2_FAST=0 --arm fast: \\
                    --arm fast-scalar-experts:SAGUARO_IK2_EXPERT_MMA=0 [--arg=--chunk --arg=1024] [--keep]
    tools/ik2/kl.py --compare A.f32 B.f32     (two logits files already written)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ik2"))
sys.path.insert(0, str(ROOT / "tools"))
import bench as B       # noqa: E402
import t4_check as TC   # noqa: E402

V = 200058


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def compare(fa: Path, fb: Path, v: int = V, block: int = 64) -> dict:
    a = np.memmap(fa, dtype=np.float32, mode="r").reshape(-1, v)
    b = np.memmap(fb, dtype=np.float32, mode="r").reshape(-1, v)
    n = min(len(a), len(b))
    kl, same, dmax = [], 0, 0.0
    for i in range(0, n, block):
        la, lb = log_softmax(a[i:i + block][:n - i]), log_softmax(b[i:i + block][:n - i])
        kl.append((np.exp(la) * (la - lb)).sum(axis=1))
        same += int((la.argmax(axis=1) == lb.argmax(axis=1)).sum())
        dmax = max(dmax, float(np.abs(la - lb).max()))
    kl = np.concatenate(kl)
    return dict(n=n, mean=float(kl.mean()), median=float(np.median(kl)), p99=float(np.percentile(kl, 99)),
                max=float(kl.max()), top1=same, bit_equal=bool(dmax == 0.0), dmax=dmax)


def show(name: str, r: dict):
    print("%-28s n %5d  KL mean %.3e  median %.3e  p99 %.3e  max %.3e  top-1 same %d of %d (%.2f%%)%s" %
          (name, r["n"], r["mean"], r["median"], r["p99"], r["max"], r["top1"], r["n"], 100.0 * r["top1"] / r["n"],
           "  BIT EQUAL" if r["bit_equal"] else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, type=Path)
    ap.add_argument("--gpus", default="2")
    ap.add_argument("--build", type=Path, default=ROOT / "build-ik2")
    ap.add_argument("--model", default=TC.MODEL)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--start", type=int, default=20000, help="where in the corpus's tokens the text starts")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--arm", action="append", default=[], help="NAME:K=V,K=V (the first is the reference; MODEL=path "
                    "runs that arm on another model file)")
    ap.add_argument("--arg", action="append", default=[])
    ap.add_argument("--out", type=Path, default=ROOT / "build-ik2" / "kl")
    ap.add_argument("--keep", action="store_true", help="keep the logits files")
    a = ap.parse_args()
    if a.compare:
        show("%s vs %s" % (a.compare[0].name, a.compare[1].name), compare(*a.compare))
        return 0
    gpus = [int(x) for x in a.gpus.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_gguf(a.model)
    ids = tok.encode(B.corpus(), parse_special=False)[a.start:a.start + 32 + a.n]
    prompt, force = ids[:32], ids[32:]
    (a.out / "force.txt").write_text(",".join(map(str, force)))
    files = []
    for arm in a.arm:
        name, _, kv = arm.partition(":")
        f = a.out / ("%s-%dgpu.f32" % (name, len(gpus)))
        B.wait_free(gpus)
        line = B.claim("kl " + name, gpus)
        try:
            env = TC.env_for(gpus)
            env.update(SAGUARO_IK2_FORCE=str(a.out / "force.txt"), SAGUARO_IK2_LOGITS_OUT=str(f))
            model = a.model   # (MODEL=path in an arm: that arm's model - a Strata pack against a GGUF)
            for x in filter(None, kv.split(",")):
                k, v = x.split("=", 1)
                if k == "MODEL":
                    model = v
                else:
                    env[k] = v
            t0 = time.time()
            r = subprocess.run([str(a.build / "saguaro-ik2"), "--model", model, "--max-context", str(a.ctx), "--gpus",
                                str(len(gpus)), "--tokens", ",".join(map(str, prompt)), "--gen", str(a.n)] + a.arg,
                               env=env, capture_output=True, text=True)
            tail = [l for l in r.stdout.splitlines() if l.startswith(("forced:", "generation:", "token path", "resident"))]
            print("%s: %.0f s, exit %d; %s" % (name, time.time() - t0, r.returncode, " | ".join(tail)), flush=True)
            if r.returncode:
                print(r.stdout[-1500:], r.stderr[-1500:])
                return 1
        finally:
            time.sleep(1)
            B.release(line)
        files.append((name, f))
    for name, f in files[1:]:
        show("%s vs %s" % (files[0][0], name), compare(files[0][1], f))
    if not a.keep:
        for _, f in files:
            f.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
