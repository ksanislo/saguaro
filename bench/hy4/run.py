#!/usr/bin/env python3
"""bench/hy4/run.py - one load of saguaro-hy4 --serve, a sequence of greedy requests, their times.

    run.py LABEL --seq short:160,10k:160 [--ids-dir D] [--exe E] [--pack P] [--log L] [--env K=V ...] -- <engine args>

Each request: the prompt's ids file (bench/hy4/make_prompts.py) and the tokens to generate.  Prints a line a request:
the prompt's read (tokens, tok/s), the decode (tokens, tok/s, the CPU lane's experts a token), the first tokens.
The engine's stderr goes to --log.  The cards are claimed in /tmp/gpu-status for the run (waits for others' claims)."""
import argparse, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATUS = pathlib.Path("/tmp/gpu-status")
ME = "CLAIM hy4-agent:"


_lock = None


def claim(what):
    global _lock
    import fcntl
    _lock = open("/tmp/hy4-agent.lock", "w")   # (this agent's own runs: one at a time)
    fcntl.flock(_lock, fcntl.LOCK_EX)
    while True:
        lines = STATUS.read_text().splitlines() if STATUS.exists() else []
        other = [l for l in lines if l.startswith(("CLAIM", "BUSY")) and not l.startswith(ME)]
        if not other:
            break
        print("waiting for:", other[0], flush=True)
        time.sleep(4)
    with STATUS.open("a") as f:
        f.write(f"{ME} {what} {time.strftime('%Y-%m-%d %H:%M %Z')}\n")


def release():
    if STATUS.exists():
        keep = [l for l in STATUS.read_text().splitlines() if not l.startswith(ME)]
        STATUS.write_text("\n".join(keep) + ("\n" if keep else ""))
    if _lock is not None:
        _lock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--ids-dir", default=str(ROOT / "build-hy4" / "bench"))
    ap.add_argument("--exe", default=str(ROOT / "build-hy4" / "saguaro-hy4"))
    ap.add_argument("--pack", default=str(ROOT / "build-hy4" / "pack-check"))
    ap.add_argument("--log")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--json")
    ap.add_argument("--clock", choices=("lock", "unlock"), help="llm-gpu-clocks, once the cards are claimed")
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", HY4_Q4X="0", HY4_MEASURE="0")
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log = open(a.log or f"/tmp/hy4-{a.label}.log", "w")
    claim(a.label)
    if a.clock:
        subprocess.run(["sudo", "-n", "/usr/local/sbin/llm-gpu-clocks", a.clock], stdout=subprocess.DEVNULL)
    out = []
    p = None
    try:
        t0 = time.time()
        p = subprocess.Popen([a.exe, "--serve", "--pack", a.pack, *a.rest], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        info = {}
        for line in p.stdout:
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    info[k] = v
            if line.startswith("READY"):
                break
        else:
            raise SystemExit(f"{a.label}: the engine ended before READY (see {log.name})")
        load = time.time() - t0
        print(f"UP {a.label} load={load:.0f}s vram_gb={info.get('vram_gb')} experts_vram={info.get('vram_slots')}/"
              f"{info.get('experts')} mtp={info.get('mtp')} dfly2={info.get('dfly2')}", flush=True)
        for item in a.seq.split(","):
            name, _, n = item.partition(":")
            ids = (pathlib.Path(a.ids_dir) / f"{name.split('@')[0]}.ids").read_text().strip()
            t1 = time.time()
            p.stdin.write(f"GEN {int(n)} {ids}\n")
            p.stdin.flush()
            toks, stat, done, err = [], "", None, None
            for line in p.stdout:
                if line.startswith("T "):
                    toks.append(int(line.split()[1]))
                elif line.startswith("STAT "):
                    stat = line.strip()
                elif line.startswith("ERR "):
                    err = line.strip()
                elif line.startswith("DONE "):
                    done = line.split()
                    break
            if done is None:
                raise SystemExit(f"{a.label} {name}: the engine ended mid-request (see {log.name})")
            gen, prompt, pms, dms, why, reused = int(done[1]), int(done[2]), float(done[3]), float(done[4]), done[5], int(done[8])
            st = dict(kv.split("=") for kv in stat.split()[1:]) if stat else {}
            rec = {"label": a.label, "req": name, "prompt": prompt, "reused": reused, "read_tok_s": round(1000.0 * (prompt - reused) / pms, 1) if pms > 0 else 0,
                   "gen": gen, "dec_tok_s": round(1000.0 * gen / dms, 2) if dms > 0 else 0, "why": why, "cpu_exp_tok": st.get("cpu"),
                   "drafts": st.get("drafts"), "engram_ms": st.get("engram_ms"), "wall_s": round(time.time() - t1, 1), "first": toks[:12], "tok_md5": __import__("hashlib").md5(",".join(map(str, toks)).encode()).hexdigest()[:12], "err": err}
            out.append(rec)
            print(f"{a.label:14s} {name:8s} prompt {prompt:6d} (reused {reused:6d}) read {rec['read_tok_s']:7.1f} tok/s | gen {gen:4d} "
                  f"{rec['dec_tok_s']:6.2f} tok/s {why} cpu_exp/tok {st.get('cpu')} drafts {st.get('drafts')} engram_ms {st.get('engram_ms')}"
                  f" md5 {rec['tok_md5']}{' ' + err if err else ''}", flush=True)
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=120)
    finally:
        if p is not None and p.poll() is None:
            p.kill()
            p.wait()
        release()
        if a.json:
            with open(a.json, "a") as f:
                for r in out:
                    f.write(json.dumps({**r, "load_s": round(load, 1) if 'load' in dir() else None, "info": info if 'info' in dir() else {}}) + "\n")


if __name__ == "__main__":
    main()
