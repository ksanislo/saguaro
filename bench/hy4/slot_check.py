#!/usr/bin/env python3
"""bench/hy4/slot_check.py [--exe E] [--pack P] [--dir D] -- <engine args>: saved conversations, checked greedy.

Load 1: A (the 10k prompt) 120 tokens = the uninterrupted answer; then again A 60 tokens, B (short) 60, A 60 (must be
read back, the same 60), A + its first 60 tokens as the prompt, 60 more (read back; must equal the uninterrupted
answer's tokens 60..119), QUIT (saves).  Load 2 (a restart): A 60 (read back from the folder, the same 60) and the
continuation again.  Prints each step's reused tokens, wall time and PASS / FAIL."""
import argparse, os, pathlib, shutil, subprocess, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run

ROOT = pathlib.Path(__file__).resolve().parents[2]


class Engine:
    def __init__(self, a, env, log):
        self.p = subprocess.Popen([a.exe, "--serve", "--pack", a.pack, *a.rest], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        self.notes = []
        for line in self.p.stdout:
            if line.startswith("NOTE "):
                self.notes.append(line.strip())
            if line.startswith("READY"):
                return
        raise SystemExit("the engine ended before READY")

    def gen(self, ids, n):
        t0 = time.time()
        self.p.stdin.write(f"GEN {n} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        toks, notes = [], []
        for line in self.p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("NOTE "):
                notes.append(line.strip()[5:])
            elif line.startswith("ERR "):
                notes.append(line.strip())
            elif line.startswith("DONE "):
                d = line.split()
                return toks, int(d[8]), time.time() - t0, notes
        raise SystemExit("the engine ended mid-request")

    def quit(self):
        self.p.stdin.write("QUIT\n")
        self.p.stdin.flush()
        self.p.wait(timeout=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(ROOT / "build-hy4" / "saguaro-hy4"))
    ap.add_argument("--pack", default=str(ROOT / "build-hy4" / "pack-check"))
    ap.add_argument("--ids-dir", default=str(ROOT / "build-hy4" / "bench"))
    ap.add_argument("--dir", default=str(ROOT / "build-hy4" / "bench" / "slots-check"))
    ap.add_argument("--log", default="/tmp/hy4-slot-check.log")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    shutil.rmtree(a.dir, ignore_errors=True)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", HY4_Q4X="0", HY4_MEASURE="0", HY4_SLOT_DIR=a.dir)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    A = [int(x) for x in (pathlib.Path(a.ids_dir) / "10k.ids").read_text().split(",")]
    B = [int(x) for x in (pathlib.Path(a.ids_dir) / "short.ids").read_text().split(",")]
    log = open(a.log, "w")
    ok = True

    def step(name, got, want, reused, need_reused, dt, notes):
        nonlocal ok
        good = (want is None or got == want) and reused >= need_reused
        ok = ok and good
        print(f"{'PASS' if good else 'FAIL'} {name}: {len(got)} tokens, reused {reused} (need >= {need_reused}), {dt:.1f} s"
              f"{'; ' + ' | '.join(notes) if notes else ''}", flush=True)

    run.claim("slot-check")
    try:
        e = Engine(a, env, log)
        full, r, dt, nt = e.gen(A, 120)
        step("A uninterrupted 120", full, None, r, 0, dt, nt)
        t, r, dt, nt = e.gen(A, 60)
        step("A repeat 60 (in the caches)", t, full[:60], r, len(A) - 1, dt, nt)
        t, r, dt, nt = e.gen(B, 60)
        step("B 60 (A is set aside)", t, None, r, 0, dt, nt)
        t, r, dt, nt = e.gen(A, 60)
        step("A read back 60", t, full[:60], r, len(A) - 1, dt, nt)
        t, r, dt, nt = e.gen(B, 30)
        step("B again", t, None, r, 0, dt, nt)
        t, r, dt, nt = e.gen(A + full[:60], 60)
        step("A + 60 continued (read back)", t, full[60:120], r, len(A) + 59, dt, nt)
        e.quit()
        e = Engine(a, env, log)
        print("restart:", " | ".join(n[5:] for n in e.notes if "sessions" in n), flush=True)
        t, r, dt, nt = e.gen(A, 60)
        step("after a restart: A read back 60", t, full[:60], r, len(A) - 1, dt, nt)
        t, r, dt, nt = e.gen(A + full[:60], 60)
        step("after a restart: A + 60 continued", t, full[60:120], r, len(A) + 59, dt, nt)
        e.quit()
    finally:
        run.release()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
