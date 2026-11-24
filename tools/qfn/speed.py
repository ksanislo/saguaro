#!/usr/bin/env python3
"""tools/qfn/speed.py - one honest speed sweep of the QFN engine (saguaro-qfn) from a server config: load time, then for each
context depth one fresh prompt (real text from kl-set.json, a distinct first token so no cached prefix is reused) read and
answered greedily - prompt read tok/s, time to the first token, decode tok/s, the decode's expert lookups by tier (GPU
cache hits / RAM / the model files) and the bytes the engine read from the drive per answered token (/proc/<pid>/io),
and the memory split (VRAM in use, the engine's RSS, its RAM-tier and GPU-cache lines from the log).

    tools/qfn/speed.py --config CFG.json --set tools/qfn/kl-set.json --out sweep.json [--depths 1024,8192,32768]
                       [--decode 256] [--repeat 2]
"""
import argparse, json, os, pathlib, re, subprocess, sys, threading, queue, time

IM_START = 248045


def vram_used_mib(cfg):
    try:
        if cfg.get("backend") == "hip":
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True).stdout
            d = json.loads(out)
            g = d.get(f"card{(cfg.get('gpu') or [0])[0]}") or next(iter(d.values()))
            return int(g["VRAM Total Used Memory (B)"]) / 2**20
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        return sum(float(x) for i, x in enumerate(out) if i in (cfg.get("gpu") or [0]))
    except Exception:
        return float("nan")


def proc_field(pid, path, key):
    try:
        for l in open(f"/proc/{pid}/{path}"):
            if l.startswith(key):
                return int(l.split()[1])
    except OSError:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="1024,4096,8192,16384,32768")
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1, help="answers per depth (fresh prompts each)")
    ap.add_argument("--arg", action="append", default=[], help="extra engine argument (repeatable)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    st = json.load(open(a.set))
    text = [t for p in st["prompts"] for t in p["ids"]]
    env = dict(os.environ)
    if cfg.get("gpu"):
        key = "HIP_VISIBLE_DEVICES" if cfg.get("backend") == "hip" else "CUDA_VISIBLE_DEVICES"
        env[key] = ",".join(map(str, cfg["gpu"]))
        if key == "CUDA_VISIBLE_DEVICES":
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    cmd = [cfg["exe"], "--serve", *cfg["args"], *a.arg]
    out = pathlib.Path(a.out)
    log = open(out.with_suffix(".engine.log"), "w")
    vram0 = vram_used_mib(cfg)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env,
                         cwd=cfg.get("cwd"), bufsize=1)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put((time.time(), l)) for l in p.stdout] + [q.put((time.time(), None))],
                     daemon=True).start()

    def line(timeout=3600):
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return time.time(), None

    while True:
        t, l = line()
        if l is None:
            raise SystemExit(f"the engine stopped before READY; see {log.name}")
        if l.startswith("READY"):
            break
    res = {"config": a.config, "cmd": cmd, "load_s": round(time.time() - t0, 1), "vram_before_mib": vram0,
           "runs": []}
    time.sleep(2)
    res["vram_loaded_mib"] = vram_used_mib(cfg)
    res["rss_loaded_mib"] = proc_field(p.pid, "status", "VmRSS:") / 1024
    res["locked_mib"] = proc_field(p.pid, "status", "VmLck:") / 1024
    print(f"READY after {res['load_s']} s; VRAM {res['vram_loaded_mib']:.0f} MiB (before {vram0:.0f}), RSS "
          f"{res['rss_loaded_mib']:.0f} MiB, locked {res['locked_mib']:.0f} MiB", flush=True)

    def ask(ids, n):
        rb0 = proc_field(p.pid, "io", "read_bytes:")
        ts = time.time()
        p.stdin.write(f"GEN {n} " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        first, toks, done = None, 0, None
        while True:
            t, l = line()
            if l is None:
                raise SystemExit(f"the engine stopped; see {log.name}")
            if l.startswith("T "):
                toks += 1
                first = first or t
            elif l.startswith(("DONE", "ERR")):
                done = l.strip()
                te = t
                break
        rb1 = proc_field(p.pid, "io", "read_bytes:")
        f = done.split()
        r = {"prompt": len(ids), "generated": toks, "ttft_s": round((first or te) - ts, 2), "wall_s": round(te - ts, 2),
             "done": done, "disk_read_mb": round((rb1 - rb0) / 1e6, 1)}
        if f[0] == "DONE" and len(f) >= 14:
            pm, dm = float(f[3]), float(f[4])
            r.update(prompt_ms=pm, decode_ms=dm, finish=f[5], drafts=(int(f[6]), int(f[7])), reused=int(f[8]),
                     hits=int(f[9]), lookups=int(f[10]), ram_blobs=int(f[11]), file_blobs=int(f[12]),
                     file_mb=float(f[13]))
            r["prompt_tok_s"] = round((len(ids) - int(f[8])) / (pm / 1e3), 1) if pm > 0 else None
            r["decode_tok_s"] = round((toks - 1) / (dm / 1e3), 2) if dm > 0 and toks > 1 else None
        return r

    ask([IM_START] + text[:64], 16)   # warm-up (the first launches, the routers' first reads)
    k = 0
    for d in [int(x) for x in a.depths.split(",")]:
        for _ in range(a.repeat):
            k += 1
            off = (k * 7919) % max(1, len(text) - d)
            ids = [IM_START, 1000 + k] + (text[off:] + text)[:d - 2]
            r = ask(ids, a.decode)
            r["depth"] = d
            res["runs"].append(r)
            dpt = r["file_mb"] / max(1, r["generated"]) if "file_mb" in r else float("nan")
            print(f"depth {d:6d}: prompt {r.get('prompt_tok_s')} tok/s (TTFT {r['ttft_s']} s), decode "
                  f"{r.get('decode_tok_s')} tok/s, lookups {r.get('lookups')}: GPU hits {r.get('hits')}, RAM "
                  f"{r.get('ram_blobs')}, files {r.get('file_blobs')} ({r.get('file_mb')} MB, {dpt:.1f} MB/token), "
                  f"drive reads {r['disk_read_mb']} MB, drafts {r.get('drafts')}", flush=True)
            out.write_text(json.dumps(res, indent=1))
    res["vram_end_mib"] = vram_used_mib(cfg)
    res["rss_end_mib"] = proc_field(p.pid, "status", "VmRSS:") / 1024
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
    log.close()
    res["engine_lines"] = [l.rstrip() for l in open(out.with_suffix(".engine.log"), errors="replace")
                           if re.search(r"expert cache|resident RAM|file tier|RAM budget|VRAM|prompt chunk|GiB", l)][:40]
    out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
