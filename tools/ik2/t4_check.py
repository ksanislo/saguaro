#!/usr/bin/env python3
"""tools/ik2/t4_check.py - saguaro-ik2's card check (Inkling 2 Mini on Tesla T4s, sm_75), unattended.

    python3 tools/ik2/t4_check.py --gpus 2,3 [--build build-ik2] [--model FILE.gguf] [--ctx 32768]
                                  [--skip-kernels] [--skip-1card] [--skip-2card] [--read-tokens 2048]

  1. ik2_kernels_test on the first card: every kernel against a double reference on the CPU (before sm_80 that is the
     m16n8k8 pairs and the in-place tiles of the attention and expert kernels).
  2. The model on one card (--gpus' first), then on two (its first two): a greedy question whose answer is checked, a
     long read and a short write for the speeds, the VRAM each card holds.

It uses the cards named and nothing else, refuses a card with less than --need-free-gib free (something else lives
there), and ends with one line: T4 CHECK PASSED / FAILED.  Exit status 0 only when every step ran and passed.
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("SAGUARO_IK2_MODEL") or str(ROOT / "models/Inkling-2-Mini-UD-Q4_K_XL/Inkling-2-Mini-UD-Q4_K_XL.gguf")
QUESTION = "What is the capital of France? Answer with the city's name only."
EXPECT = "paris"
PASSAGE = ("The saguaro grows slowly: a plant ten years old may be no taller than a thumb, and its first arm comes "
           "after fifty to seventy years. It stores the water of the summer rains in its pleated trunk, which swells "
           "as it drinks. Birds nest in holes cut into it, and its fruit ripens in June. ")


def smi(gpus: list[int]) -> dict[int, tuple[int, int]]:
    """{card: (used MiB, total MiB)} of the cards named (PCI order, as CUDA_DEVICE_ORDER=PCI_BUS_ID numbers them)."""
    if shutil.which("nvidia-smi") is None:   # (AMD: rocm-smi's numbering, HIP_VISIBLE_DEVICES')
        import json
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True,
                             check=True).stdout
        js = json.loads(out)
        rows = {int(k[4:]): (int(v["VRAM Total Used Memory (B)"]) >> 20, int(v["VRAM Total Memory (B)"]) >> 20)
                for k, v in js.items() if k.startswith("card")}
        return {g: rows[g] for g in gpus}
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = {int(a): (int(b), int(c)) for a, b, c in (l.split(",") for l in out.strip().splitlines())}
    return {g: rows[g] for g in gpus}


def env_for(gpus: list[int]) -> dict:
    ids = ",".join(map(str, gpus))
    return dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=ids, HIP_VISIBLE_DEVICES=ids)


def chat_prompt(question: str) -> str:
    # tools/ik2/chat_template.jinja's render of one user turn at effort "none", the answer's text segment opened
    return ("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            "<|message_user|><|content_text|>" + question + "<|end_message|><|message_model|><|content_text|>")


class Engine:
    """saguaro-ik2 --serve: its lines read on a thread, requests by the engine protocol (GEN / T / DONE)."""

    def __init__(self, exe: Path, args: list[str], env: dict, log: Path):
        self.log = open(log, "w")
        self.p = subprocess.Popen([str(exe)] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        self.lines: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.log.write(line)
            self.log.flush()
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def until(self, pred, timeout: float):
        """Lines up to the first that pred takes; None if the engine ended or the time ran out."""
        end = time.time() + timeout
        got = []
        while time.time() < end:
            try:
                line = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if line is None:
                return None
            got.append(line)
            self.lines.append(line)
            if pred(line):
                return got
        return None

    def gen(self, ids: list[int], max_new: int, timeout: float):
        """(token ids, prompt tokens read, prompt ms, decode ms, reason) of one greedy request, or None."""
        self.p.stdin.write("GEN %d temperature=0 %s\n" % (max_new, ",".join(map(str, ids))))
        self.p.stdin.flush()
        got = self.until(lambda l: l.startswith("DONE "), timeout)
        if got is None:
            return None
        toks = [int(l.split()[1]) for l in got if l.startswith("T ")]
        d = got[-1].split()
        return toks, int(d[2]) - int(d[8]), float(d[3]), float(d[4]), d[5]   # (less the tokens kept from before)

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
            self.p.wait()
        self.log.close()


def run_kernels(build: Path, gpu: int, out: Path) -> bool:
    print(f"== kernels test on card {gpu}", flush=True)
    r = subprocess.run([str(build / "ik2_kernels_test")], env=env_for([gpu]), capture_output=True, text=True,
                       timeout=1800)
    (out / "kernels_test.out").write_text(r.stdout + r.stderr)
    lines = r.stdout.splitlines()
    for l in lines[:2] + [l for l in lines if l.startswith("FAIL")] + lines[-1:]:
        print("   ", l)
    n_ok = sum(l.startswith("ok") for l in lines)
    ok = r.returncode == 0 and bool(lines) and lines[-1].strip() == "all passed"
    print(f"   {n_ok} checks ok, exit {r.returncode}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_model(a, tok, gpus: list[int], out: Path) -> bool:
    n = len(gpus)
    name = f"{n}card"
    print(f"== model on {n} card{'s' if n > 1 else ''} ({','.join(map(str, gpus))}), context {a.ctx}", flush=True)
    before = smi(gpus)
    for g, (used, total) in before.items():
        if (total - used) / 1024 < a.need_free_gib:
            print(f"   card {g} has {(total - used) / 1024:.1f} GiB free, under --need-free-gib {a.need_free_gib}: "
                  f"not started (something else lives there)")
            return False
    args = ["--model", a.model, "--serve", "--max-context", str(a.ctx), "--gpus", str(n), "--chunk", str(a.chunk),
            "--threads", str(a.threads)]
    env = env_for(gpus)
    if n == 1 and a.room_mb:
        env["SAGUARO_IK2_ROOM_MB"] = str(a.room_mb)
    t0 = time.time()
    eng = Engine(a.build / "saguaro-ik2", args, env, out / f"engine_{name}.log")
    ok = True
    try:
        got = eng.until(lambda l: l.startswith("READY "), a.load_timeout)
        if got is None:
            print("   the engine did not come up; its last lines:")
            for l in eng.lines[-12:]:
                print("     ", l)
            return False
        print(f"   loaded in {time.time() - t0:.0f} s")
        for l in got:   # how the load placed the weights
            if any(k in l for k in ("experts ", "expert cache", "pool ", "resident", "expert parallel", "dense weights",
                                    "kv cache", "warm:", "NOTE", "looped")):
                print("     ", l)
        for g, (used, total) in smi(gpus).items():
            print(f"   card {g}: {used} MiB of {total} MiB used ({used - before[g][0]} MiB by this load)")

        ids = tok.encode(chat_prompt(QUESTION), parse_special=True)
        r = eng.gen(ids, a.gen, a.req_timeout)
        if r is None:
            print("   no answer to the question (the engine ended or timed out)")
            return False
        toks, n_prompt, p_ms, d_ms, reason = r
        text = tok.decode(toks)
        good = EXPECT in text.lower()
        ok = ok and good
        print(f"   question: {n_prompt} new prompt tokens in {p_ms:.0f} ms; {len(toks)} tokens in {d_ms:.0f} ms "
              f"({len(toks) / max(d_ms, 1e-3) * 1e3:.1f} tok/s), ended by {reason}")
        print(f"   answer: {text!r}")
        print(f"   the answer names Paris: {'PASS' if good else 'FAIL'}")
        r2 = eng.gen(ids, a.gen, a.req_timeout)   # the same request again: greedy repeats itself
        if r2 is not None:
            same = r2[0] == toks
            print(f"   the same request again gives the same tokens: {'yes' if same else 'NO (near ties on this path?)'}")

        passage = PASSAGE * 400
        body = tok.encode(passage, parse_special=False)[:a.read_tokens]
        long_ids = tok.encode("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                              "<|message_user|><|content_text|>", parse_special=True) + body + \
            tok.encode(" In one sentence, what plant is this about?<|end_message|><|message_model|><|content_text|>",
                       parse_special=True)
        r = eng.gen(long_ids, a.gen, a.req_timeout)
        if r is None:
            print("   no answer to the long read")
            return False
        toks, n_prompt, p_ms, d_ms, reason = r
        text = tok.decode(toks)
        good = "saguaro" in text.lower() or "cactus" in text.lower()
        ok = ok and good
        print(f"   read: {n_prompt} tokens in {p_ms / 1e3:.2f} s = {n_prompt / max(p_ms, 1e-3) * 1e3:.0f} tok/s")
        print(f"   decode after it: {len(toks)} tokens in {d_ms / 1e3:.2f} s = "
              f"{len(toks) / max(d_ms, 1e-3) * 1e3:.1f} tok/s, ended by {reason}")
        print(f"   answer: {text!r}")
        print(f"   the answer names the plant: {'PASS' if good else 'FAIL'}")
        for g, (used, total) in smi(gpus).items():
            print(f"   card {g} after the requests: {used} MiB of {total} MiB")
    finally:
        eng.close()
    print(f"   engine log: {out / f'engine_{name}.log'}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpus", required=True, help="cards to use, PCI order: one for the kernels test and the 1-card "
                                                  "load, two for the 2-card load (e.g. 2,3)")
    ap.add_argument("--build", type=Path, default=ROOT / "build-ik2")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", type=Path, default=None, help="logs go here (default <build>/t4_check)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--read-tokens", type=int, default=2048)
    ap.add_argument("--room-mb", type=int, default=0, help="SAGUARO_IK2_ROOM_MB of the 1-card load (0: the default)")
    ap.add_argument("--need-free-gib", type=float, default=14.5)
    ap.add_argument("--load-timeout", type=float, default=1800)
    ap.add_argument("--req-timeout", type=float, default=1200)
    ap.add_argument("--skip-kernels", action="store_true")
    ap.add_argument("--skip-1card", action="store_true")
    ap.add_argument("--skip-2card", action="store_true")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",") if x.strip()]
    for exe in ("saguaro-ik2", "ik2_kernels_test"):
        if not (a.build / exe).exists():
            print(f"no {a.build / exe}: build it first (ninja -C {a.build} saguaro-ik2 ik2_kernels_test)")
            return 2
    out = a.out or a.build / "t4_check"
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    if not a.skip_kernels:
        results["kernels test"] = run_kernels(a.build, gpus[0], out)
    if not (a.skip_1card and a.skip_2card):
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer as ST
        tok = ST.Tokenizer.from_gguf(a.model)   # (the file's header alone)
        if not a.skip_1card:
            results["1 card"] = run_model(a, tok, gpus[:1], out)
        if not a.skip_2card:
            if len(gpus) < 2:
                print("== model on 2 cards: skipped, --gpus names one card")
            else:
                results["2 cards"] = run_model(a, tok, gpus[:2], out)
    print()
    for k, v in results.items():
        print(f"{k:14s} {'PASS' if v else 'FAIL'}")
    ok = bool(results) and all(results.values())
    print("T4 CHECK PASSED" if ok else "T4 CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
