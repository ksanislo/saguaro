"""tools/mem_vs_ctx.py <config.json> [--contexts 32768,262144] [--port 8093] - P124n: memory at load must not
depend on --max-context.  Context not in use costs nothing (Ken's rule): starting the same model at two context
sizes must hold the same memory.  QFN broke it - it pinned host K/V for the whole max at load - and a 96 GB box
hid it, so this check belongs on the smallest-RAM machine there is.  It builds the CHECK for every engine; it
does not fix engines (QFN fails it until the elastic K/V work of P124e lands).

For each context it starts the server from a COPY of the config (its own port, its own session dir and engine
log, no api_key, --max-context set to that value), waits for /health to answer ok, and records what the run
holds.  The pinned RAM is measured TWO ways (P124t): the process tree's VmLck + VmPin summed over EVERY
process of the server's tree (children of children, from /proc/<pid>/status), and host-wide - the delta of
/proc/meminfo Mlocked + Unevictable + Shmem between before the start and after /health ok.  ROCm/CUDA
page-locked host memory lands in neither process's VmLck (the 1.2.4 ref run: every VmLck read 0.0 while the
engines logged "68.00 GiB page-locked" and "the latent cache in 2.95 GB of pinned RAM"), but the check owns
the moment - nothing else loads between the two reads - so the host-wide delta is the run's.  Both are shown;
the rule judges on the larger.  Plus the run's RSS, the GPU memory used (nvidia-smi or rocm-smi, whichever
this PC has, summed over the config's cards), and the engine's own log lines that state a K/V or expert
budget (lines containing "pinned RAM", "resident" or "KV") read into PLACEMENT FIGURES (P124t): the experts
in the VRAM cache ("1729 in the GPU cache", "every expert in VRAM (3072 copied", "N of M experts in VRAM",
"(N of M in VRAM)"), the experts in RAM / pinned ("resident RAM mode: X GiB of experts in RAM", "experts in
pinned memory: X GiB", "X GB pinned pool"), the GB left for the expert pool ("VRAM before the expert pool:
A of T GB in use" -> T - A, summed over the cards), and the GB of pinned RAM the engine declares ("the latent
cache in 2.95 GB of pinned RAM"; GLM since P125g says what is pinned NOW at the load instead - "0 MiB of pinned
RAM at start, up to 2.95 GB" - and it is the ceiling the run declares, so that is what gets read).  Then it
stops the run - kill by the port's owner - and starts the next.

A table of every figure per context, the placement figures and the budget lines, and the verdict: PASS when
every memory figure differs by less than max(2%, 256 MiB) between the contexts AND no placement figure
(experts in VRAM, experts in RAM, the expert pool, the declared pinned RAM) differs by more than 2% - the
pinned row judges on the larger of its two measures, and the placement rule is what catches Inkling's
"every expert in VRAM (3072)" at 32768 becoming "offload: 0 of 3072 experts in VRAM" at 262144 behind equal
GPU totals.  FAIL names the difference furthest past its limit.  Exit 0 PASS, 1 FAIL, 2 nothing
could be measured: a port already in use, or no /proc here.  A PC with no GPU reader prints "-" for the GPU row
and leaves it out of the rule; a box without /proc/meminfo loses the host-wide pinned row and judges on the
tree alone.

    python3 tools/mem_vs_ctx.py saguaro-inkling-2-mini.json
    python3 tools/mem_vs_ctx.py saguaro-qfn.json --contexts 262144,1048576 --port 8093

The run copies live under scratch/mem-vs-ctx/<config>/ (one folder per context: the config copy, the engine log,
the server's own output).  Needs Linux (/proc/self/status is checked before anything starts: a box without /proc
would read zeros for every figure and PASS on nothing) and a GPU and the model to load - it is a release check,
not a unit test; tools/test_mem_vs_ctx.py covers the parsing, the table and the verdict, and the start/stop cycle
on the mock.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEALTH_S = 3600.0        # a real model on a cold box takes minutes to load; a failed load answers early (503 failed)
STOP_S = 180.0           # a graceful stop wraps the request in progress (SAGUARO_ENGINE_WRAP_S) and ends the engine
SETTLE_S = 10.0          # read /proc this long after /health ok: a load that just finished still settles (warm-up)
TOL_PCT, TOL_MIB = 0.02, 256.0          # the rule: every figure within max(2%, 256 MiB) between the contexts
BUDGET_MARKS = ("pinned ram", "resident", "KV")   # the engine's own lines that state a K/V or expert budget
MEMINFO_KEYS = ("Mlocked", "Unevictable", "Shmem")  # host-wide: where pinned/unevictable/shared RAM shows up

# the engines' expert-placement lines, grepped from their start-up prints (P124t):
#   QFN      "resident RAM mode: 68.00 GiB of experts in RAM (page-locked), 1729 in the GPU cache" (generate.cpp)
#   QFN      "every expert is in the GPU cache (N of M)" (generate.cpp)
#   GLM      "glm fast: CUDA0 VRAM before the expert pool: A of T GB in use - ... state/KV (N ctx) X" (glm_fast_path.cu)
#   GLM      "glm pack: CUDA0 KV streaming: ..., the latent cache in X GB of pinned RAM" (glm_model.cu, pre-P125g)
#   GLM      "pinned latent cache grows with use: N MiB of pinned RAM at start, up to M GB" - the lazy arena, and
#            "pinned latent cache: X GB of pinned RAM at the load, the whole cache" - the eager fallback (glm_model.cu)
#   Inkling  "resident: every expert in VRAM (N copied in ...)" / "offload: N of M experts in VRAM" (ik2_main.cu)
#   MM3      "saguaro mm3: loaded in ..., N of M experts in VRAM, X GB pinned pool" (mm3_main.cpp)
#   Hy4      "INFO CPU lane: N of M routed experts in VRAM (..%)" (hy4_model.cu)
#   DS4      "... experts X GB (N of M in VRAM), caches ..." / "N routed experts on the CPU lane (X GB ...)" (ds4_model.cu)
EXPERT_VRAM_RES = (                          # counts: summed over the matches (DS4 prints one line a card)
    re.compile(r"(\d[\d,]*) of \d[\d,]* (?:routed )?experts in VRAM"),
    re.compile(r"every expert in VRAM \((\d[\d,]*) copied"),
    re.compile(r"\((\d[\d,]*) of \d[\d,]* in VRAM\)"),
    re.compile(r"(\d[\d,]*) in the GPU cache\b"),
    re.compile(r"every expert is in the GPU cache \((\d[\d,]*) of"),
)
EXPERT_RAM_RES = (                           # the RAM/pinned expert tier, GiB/GB as the engine says it: the largest
    re.compile(r"resident RAM mode: ([\d.]+) GiB of experts in RAM"),
    re.compile(r"resident RAM: ([\d.]+) GiB of experts in RAM"),
    re.compile(r"experts in pinned (?:\([^)]*\) )?memory: ([\d.]+) GiB"),
    re.compile(r"([\d.]+) GB pinned pool"),
    re.compile(r"routed experts on the CPU lane \(([\d.]+) GB(?: of RAM)?"),
)
EXPERT_POOL_RES = re.compile(r"VRAM before the expert pool: ([\d.]+) of ([\d.]+) GB in use")
PINNED_DECLARED_RES = (re.compile(r"([\d.]+) Gi?B of pinned RAM"),
                       # P125g: GLM's lazy-arena line states what is pinned NOW ("N MiB of pinned RAM at start" -
                       # 0 for a fresh arena, which no Gi?B pattern sees) and the ceiling after "up to" - that is
                       # the declared pin, what the pre-P125g line gave as one cudaHostAlloc's size.  The eager
                       # fallback's whole-cache figure sits before "of pinned RAM" and matches the pattern above.
                       re.compile(r"pinned latent cache grows with use: .*up to ([\d.]+) GB"))
PLACEMENT_FIGURES = (("experts_vram", "experts in VRAM"), ("experts_ram_gib", "experts in RAM"),
                     ("expert_pool_gb", "expert pool (GB)"), ("pinned_declared_gib", "pinned RAM declared (GB)"))

ROCM_USED = re.compile(r"GPU\[(\d+)\]\s*:\s*VRAM Total Used Memory \(B\):\s*(\d+)")


# ---- the process tree: /proc parsing ---------------------------------------------------------------

def parse_proc_status(text: str) -> dict:
    """VmLck, VmPin and VmRSS of one /proc/<pid>/status, in MiB (the file speaks KiB; absent keys count 0)."""
    out = {"VmLck": 0.0, "VmPin": 0.0, "VmRSS": 0.0}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        name = name.strip()
        if name in out:
            parts = rest.split()
            if parts:
                v = float(parts[0])
                if len(parts) > 1 and parts[1].lower() == "kb":
                    v /= 1024.0
                out[name] = v
    return out


def read_proc_status(pid: int, proc="/proc") -> dict:
    try:
        with open(os.path.join(proc, str(pid), "status"), encoding="ascii", errors="replace") as f:
            return parse_proc_status(f.read())
    except (OSError, ValueError):
        return {"VmLck": 0.0, "VmPin": 0.0, "VmRSS": 0.0}


def has_proc(proc: str = "/proc") -> bool:
    """Whether /proc is really there: every reader above swallows its absence into zeros, so without this
    check a box with no /proc (a Mac, some containers) would print 0.0 for everything and PASS on nothing."""
    try:
        with open(os.path.join(proc, "self", "status"), encoding="ascii", errors="replace"):
            return True
    except OSError:
        return False


def ppid_of(pid: int, proc="/proc") -> int | None:
    try:
        with open(os.path.join(proc, str(pid), "stat"), encoding="ascii", errors="replace") as f:
            stat = f.read()
    except OSError:
        return None
    try:                                        # the comm in parens can hold spaces and parens: count from its end
        return int(stat.rpartition(")")[2].split()[1])
    except (IndexError, ValueError):
        return None


def all_pids(proc="/proc") -> list[int]:
    try:
        return sorted(int(p) for p in os.listdir(proc) if p.isdigit())
    except OSError:
        return []


def process_tree(pid: int, proc="/proc") -> list[int]:
    """The process and every descendant of it (the server and its engine child), lowest pid first."""
    ppid = {p: ppid_of(p, proc) for p in all_pids(proc)}
    out, queue = [pid], [pid]
    while queue:
        head = queue.pop(0)
        kids = [p for p, par in ppid.items() if par == head and p not in out]
        out += sorted(kids)
        queue += kids
    return out


def tree_mibs(pids: list[int], proc="/proc") -> dict:
    """What the tree holds: the pinned/locked RAM (VmLck + VmPin) and RSS, all MiB. process_tree already
    reaches the children of children, so nothing of the server's tree is left out (P124t)."""
    total = {"VmLck": 0.0, "VmPin": 0.0, "VmRSS": 0.0}
    for pid in pids:
        for k, v in read_proc_status(pid, proc).items():
            total[k] += v
    return total


# ---- the pinned RAM, host-wide -----------------------------------------------------------------------

def parse_meminfo(text: str) -> dict:
    """The host-wide Mlocked, Unevictable and Shmem of a /proc/meminfo, in MiB (the file speaks kB; an
    absent key counts 0.0). ShmemHugePages and ShmemPmdMapped are other keys and stay out."""
    out = {k: 0.0 for k in MEMINFO_KEYS}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        name = name.strip()
        if name in out:
            parts = rest.split()
            if parts:
                v = float(parts[0])
                if len(parts) > 1 and parts[1].lower() == "kb":
                    v /= 1024.0
                out[name] = v
    return out


def read_meminfo(path: str = "/proc/meminfo") -> dict | None:
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            return parse_meminfo(f.read())
    except (OSError, ValueError):
        return None


def host_pinned_delta(before: dict | None, after: dict | None) -> float | None:
    """What the run pinned host-wide: the rise of Mlocked + Unevictable + Shmem between the two reads. The
    check owns the moment (nothing else loads while a context runs), so the rise is the run's - and it
    catches what no process's VmLck sees (ROCm/CUDA page-locked memory). A key that FELL counts 0 (another
    process gave shared pages back); the figure is None when either read failed."""
    if before is None or after is None:
        return None
    return sum(max(0.0, after[k] - before[k]) for k in MEMINFO_KEYS)


def listening_inodes(port: int, proc="/proc") -> set:
    """The inode of every socket LISTENing on this port (/proc/net/tcp{,6})."""
    want = set()
    for name in ("tcp", "tcp6"):
        try:
            with open(os.path.join(proc, "net", name), encoding="ascii", errors="replace") as f:
                lines = f.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) > 9 and f[3] == "0A":                       # 0A = TCP_LISTEN
                try:
                    if int(f[1].rsplit(":", 1)[1], 16) == port:
                        want.add(f[9])
                except ValueError:
                    continue
    return want


def port_owner(port: int, proc="/proc") -> int | None:
    """The pid holding the LISTEN socket of this port, or None when nothing listens on it."""
    want = listening_inodes(port, proc)
    if not want:
        return None
    for pid in all_pids(proc):
        fds = os.path.join(proc, str(pid), "fd")
        try:
            names = os.listdir(fds)
        except OSError:                                           # gone, or not ours to read
            continue
        for n in names:
            try:
                link = os.readlink(os.path.join(fds, n))
            except OSError:
                continue
            if link.startswith("socket:[") and link[8:-1] in want:
                return pid
    return None


# ---- the GPU's used memory: nvidia-smi or rocm-smi, whichever exists --------------------------------

def parse_nvidia_used(text: str) -> dict:
    """nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -> {card: MiB}."""
    out = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = float(parts[1])                  # (already MiB: nounits)
    return out


def parse_rocm_used(text: str) -> dict:
    """rocm-smi --showmeminfo vram -> {card: MiB} ("GPU[0] : VRAM Total Used Memory (B): ...")."""
    return {int(m.group(1)): int(m.group(2)) / 2**20 for m in ROCM_USED.finditer(text)}


GPU_READERS = (
    ("nvidia-smi", ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
     parse_nvidia_used),
    ("rocm-smi", ["rocm-smi", "--showmeminfo", "vram"], parse_rocm_used),
)


def gpu_used_mib(cards: list[int], run=subprocess.run, which=shutil.which) -> float | None:
    """The GPU memory used now on the engine's cards (all of them when the config names none), MiB; None when
    this PC has no reader, or the reader cannot say. The first reader that answers wins: nvidia-smi, else
    rocm-smi."""
    for _name, cmd, parse in GPU_READERS:
        if not which(cmd[0]):
            continue
        try:
            r = run(cmd, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            continue
        used = parse(r.stdout) if r.returncode == 0 else {}
        if not used:
            continue
        if cards:
            if not all(c in used for c in cards):                 # a card the reader does not list: it cannot say
                continue
            return float(sum(used[c] for c in cards))
        return float(sum(used.values()))
    return None


# ---- the engine's own budget lines -------------------------------------------------------------------

def budget_lines(text: str) -> list[str]:
    """The engine log lines that state a K/V or expert budget: those naming pinned RAM, a resident budget,
    or KV (kept as written, table included)."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if s and any(m in (low if m != "KV" else s) for m in BUDGET_MARKS):
            out.append(s)
    return out


def parse_placement(text: str) -> dict:
    """The engine's own expert-placement lines read into figures: the experts in the VRAM cache (count, the
    matches summed - DS4 prints one line a card), the experts in RAM / pinned (the largest GiB/GB stated),
    the GB left for the expert pool (per "VRAM before the expert pool: A of T GB in use", summed over the
    cards - GLM states no count, its expert pool IS its experts in VRAM), and the largest GB/GiB of pinned
    RAM the engine declares. A figure the log said nothing about is None: compared only where two contexts
    both state it."""
    def nums(pats):
        return [float(m.group(1).replace(",", "")) for p in pats for m in p.finditer(text)]
    out = {"experts_vram": None, "experts_ram_gib": None, "expert_pool_gb": None, "pinned_declared_gib": None}
    vram = nums(EXPERT_VRAM_RES)
    if vram:
        out["experts_vram"] = sum(vram)
    ram = nums(EXPERT_RAM_RES)
    if ram:
        out["experts_ram_gib"] = max(ram)
    pool = [float(t) - float(used) for used, t in EXPERT_POOL_RES.findall(text)]
    if pool:
        out["expert_pool_gb"] = sum(pool)
    declared = nums(PINNED_DECLARED_RES)
    if declared:
        out["pinned_declared_gib"] = max(declared)
    return out


# ---- the config copy and the run ---------------------------------------------------------------------

def set_context(args: list, context: int) -> list:
    """The context flag in the engine's args, set to this context (as the server's set_context rewrites it).
    serve/models.py's ctx_flag: a config that names its context --ctx (an ik2 shape) keeps the ONE flag -
    appending --max-context beside the old --ctx would leave which value wins to the engine's mood."""
    args = list(args)
    ctx = "--max-context" if "--max-context" in args[:-1] or "--ctx" not in args[:-1] else "--ctx"
    if ctx in args[:-1]:
        args[args.index(ctx) + 1] = str(int(context))
    else:
        args += ["--max-context", str(int(context))]         # (neither flag: the canonical name is added)
    return args


def gpu_cards(cfg: dict) -> list[int]:
    """The config's "gpu": one card (2), several for a layer split ([0, 2] or "0,2"), or none."""
    g = cfg.get("gpu")
    if g is None:
        return []
    if isinstance(g, str):
        return [int(x) for x in g.split(",") if x.strip()]
    return [int(x) for x in (g if isinstance(g, list) else [g])]


def run_config(cfg: dict, run_dir: Path, context: int) -> dict:
    """The config copy for one run of the check: its own session dir and engine log (under run_dir, so two
    runs of one model never share slots), no api_key, host pinned to loopback, never idle-unloading, and
    --max-context set to this context."""
    out = json.loads(json.dumps(cfg))                     # (a deep copy: the caller's dict stays as it is)
    out.pop("api_key", None)
    args = list(out.get("args") or [])
    if "--session-dir" in args[:-1]:                      # (QFN keeps its session dir in the args)
        args[args.index("--session-dir") + 1] = str(run_dir / "sessions")
    env = dict(out.get("env") or {})
    for k in env:                                         # (the others name it in the env: *_SLOT_DIR)
        if k.endswith("_SLOT_DIR"):
            env[k] = str(run_dir / "sessions")
    out["env"] = env
    out["args"] = set_context(args, context)
    out["log"] = str(run_dir / "engine.log")              # (the engine's stderr and the server's own lines)
    out["host"] = "127.0.0.1"
    out["idle_unload_s"] = 0
    return out                                       # (no api_key, and the start itself gets --api-key "")


def server_cmd(python, port: int, config=None) -> list:
    """The server's command line: this config on this port with no API key, or the mock engine without a config."""
    cmd = [str(python), "-m", "serve.server", "--port", str(port), "--api-key", ""]
    cmd += ["--config", str(config)] if config else ["--engine", "mock"]
    return cmd


def start_server(cmd: list, run_dir, cwd=ROOT) -> subprocess.Popen:
    """Start the server (its session, so a Ctrl+C here does not reach it); its output goes to run_dir/server.log."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("SAGUARO_API_KEY", None)                      # (the key lives in the config copy: "" = none)
    with open(run_dir / "server.log", "wb") as log:       # (the child keeps the fd; our copy closes right after)
        return subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)


def wait_health(port: int, timeout: float = HEALTH_S, host: str = "127.0.0.1",
                proc: subprocess.Popen | None = None) -> dict:
    """The /health body once the server answers 200 ok; the failed page (503 status failed) raises at once,
    and so does a server process that ended before it ever bound the port (a bad config, a used port)."""
    url = f"http://{host}:{port}/health"
    t0, last = time.time(), "nothing yet"
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"port {port}: the server ended (exit {proc.returncode}) before /health was ok")
        body = {}
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                try:
                    body = json.loads(r.read().decode("utf-8", "replace"))
                except ValueError:
                    body = {}
                if r.status == 200 and body.get("status") == "ok":
                    return body
                last = body.get("status") or f"HTTP {r.status}"
        except urllib.error.HTTPError as e:               # (while it loads: 503 and the loading page's body)
            try:
                body = json.loads(e.read().decode("utf-8", "replace"))
            except (OSError, ValueError):
                body = {}
            if body.get("status") == "failed":
                raise RuntimeError(f"port {port}: the model failed to load: {body.get('error') or body}")
            last = body.get("status") or f"HTTP {e.code}"
        except OSError:
            last = "no answer yet"
        time.sleep(0.5)
    raise TimeoutError(f"port {port}: /health did not answer ok within {timeout:.0f} s (last: {last})")


def stop_server(port: int, proc: subprocess.Popen | None = None, wait_s: float = STOP_S,
                proc_dir="/proc") -> bool:
    """Stop OUR run: SIGTERM the server - the port's owner when that owner is part of our own tree (the
    server ends the engine on its way out), our own server when nothing of ours holds the port yet; a
    stranger that took the port after our server died is never signalled. Wait for the port to fall free,
    SIGKILL whatever of our tree is still alive. True when the port is free again."""
    pid = port_owner(port, proc_dir)
    tree = process_tree(proc.pid, proc_dir) if proc is not None else []
    if proc is not None and proc.pid not in tree:
        tree.append(proc.pid)
    if proc is not None and (pid is None or pid not in tree):
        # (nothing of ours holds the port: the server had not bound it yet, or ours died and someone else
        #  took it - in both cases the target is our own server, and a stranger never gets a signal)
        pid = proc.pid if proc.poll() is None else None
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    t0 = time.time()
    while time.time() - t0 < wait_s and port_owner(port, proc_dir) is not None:
        time.sleep(0.1)
    for p in tree:
        if p != pid:                                      # (the owner got the TERM; leftovers get the KILL)
            try:
                os.kill(p, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    if proc is not None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    return port_owner(port, proc_dir) is None


# ---- the record, the table, the verdict ---------------------------------------------------------------

def measure_context(cfg: dict, context: int, port: int, run_dir, python=sys.executable,
                    health_timeout: float = HEALTH_S, settle: float = SETTLE_S,
                    meminfo: str = "/proc/meminfo") -> dict:
    """One run of the check: read the host's memory, start at this context, wait for /health ok, read what
    it holds, stop it. A stop that leaves the port held raises, so the next context never dies at the bind
    and gets its FAIL mis-attributed to "the server ended before /health was ok" when the truth is a stale
    port."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = run_config(cfg, run_dir, context)
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(run_cfg, indent=1), encoding="utf-8")
    mi0 = read_meminfo(meminfo)
    proc = start_server(server_cmd(python, port, cfg_path), run_dir)
    try:
        wait_health(port, health_timeout, proc=proc)
        time.sleep(settle)
        m = tree_mibs(process_tree(proc.pid))
        gpu = gpu_used_mib(gpu_cards(cfg))
        mi1 = read_meminfo(meminfo)
        try:
            log_text = Path(run_cfg["log"]).read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
    finally:
        freed = stop_server(port, proc)
    if not freed:
        raise RuntimeError(f"port {port} is still held after the stop - free it before the next context "
                           "starts (our server is stopped, so whoever holds the port is not ours)")
    return {"context": context, "pinned_tree_mib": m["VmLck"] + m["VmPin"],
            "pinned_host_mib": host_pinned_delta(mi0, mi1), "rss_mib": m["VmRSS"],
            "gpu_mib": gpu, "budget": budget_lines(log_text), "placement": parse_placement(log_text)}


def format_table(rows: list[dict]) -> str:
    """One row per context: the pinned RAM both ways - host-wide (the meminfo delta) and the tree's
    VmLck+VmPin - the RSS, and the GPU memory used (MiB; "-" when the PC has no GPU reader or no
    /proc/meminfo)."""
    def f(v):
        return "-" if v is None else f"{v:.1f}"
    lines = [f"{'context':>10}  {'pinned host':>11}  {'pinned tree':>11}  {'RSS MiB':>11}  {'GPU used MiB':>13}"]
    for r in rows:
        lines.append(f"{r['context']:>10}  {f(r.get('pinned_host_mib')):>11}  {f(r.get('pinned_tree_mib')):>11}"
                     f"  {f(r['rss_mib']):>11}  {f(r['gpu_mib']):>13}")
    return "\n".join(lines)


def pinned_of(row: dict) -> float | None:
    """The pinned figure to judge: the LARGER of the two measures (the tree can miss ROCm/CUDA pinning, the
    host-wide delta can be the only reader on a box whose VmLck stays 0.0)."""
    vals = [v for v in (row.get("pinned_host_mib"), row.get("pinned_tree_mib")) if v is not None]
    return max(vals) if vals else None


RAM_FIGURES = (("pinned RAM", pinned_of), ("RSS", lambda r: r.get("rss_mib")), ("GPU used", lambda r: r.get("gpu_mib")))


def compare_rows(rows: list[dict], tol_pct: float = TOL_PCT, tol_mib: float = TOL_MIB):
    """Every RAM figure over at least two contexts: (name, biggest - smallest, 2% of the biggest)."""
    out = []
    for name, get in RAM_FIGURES:
        vals = [v for r in rows if (v := get(r)) is not None]
        if len(vals) >= 2:
            out.append((name, max(vals) - min(vals), max(vals) * tol_pct))
    return out


def compare_placement(rows: list[dict], tol_pct: float = TOL_PCT):
    """Every placement figure over at least two contexts that state it:
    (name, biggest - smallest, 2% of the biggest)."""
    out = []
    for key, name in PLACEMENT_FIGURES:
        vals = [v for r in rows if (v := (r.get("placement") or {}).get(key)) is not None]
        if len(vals) >= 2:
            out.append((name, max(vals) - min(vals), max(vals) * tol_pct))
    return out


def verdict(rows: list[dict], tol_pct: float = TOL_PCT, tol_mib: float = TOL_MIB):
    """The rules: every memory figure (pinned, RSS, GPU) differs by less than max(2%, 256 MiB) between the
    contexts (2% of the bigger one), and no placement figure differs by MORE than 2% - a placement figure
    that moved with the context is the engine placing the model differently because of it, which equal RSS
    and GPU totals hide (Inkling: "every expert in VRAM (3072)" at 32768, "offload: 0 of 3072" at 262144).
    Returns (ok, line); the FAIL line names the difference furthest past its own limit, the PASS line the
    biggest memory difference there was."""
    checked = compare_rows(rows, tol_pct, tol_mib)
    pchecked = compare_placement(rows, tol_pct)
    if not checked and not pchecked:
        return False, "FAIL: nothing was measured at two contexts to compare"
    failed = [(n, d, max(pct, tol_mib), d / max(pct, tol_mib), False)
              for n, d, pct in checked if d >= max(pct, tol_mib)]
    pfailed = [(n, d, pct, d / pct if pct > 0 else float("inf"), True)
               for n, d, pct in pchecked if d > pct]
    if failed or pfailed:
        n, d, tol, _breach, is_placement = max(failed + pfailed, key=lambda c: c[3])
        if is_placement:
            return False, f"FAIL: {n} differs by {d:g} between the contexts (limit {tol_pct:.0%}: {tol:g})"
        return False, f"FAIL: {n} differs by {d:.1f} MiB between the contexts (limit {tol:.1f} MiB)"
    if checked:
        n, d, pct = max(checked, key=lambda c: c[1])
        return True, (f"PASS: every figure differs by less than max({tol_pct:.0%}, {tol_mib:.0f} MiB) between the "
                      f"contexts (biggest: {n} {d:.1f} MiB of the {max(pct, tol_mib):.1f} MiB limit)")
    n, d, _pct = max(pchecked, key=lambda c: c[1])
    return True, (f"PASS: every placement figure differs by at most {tol_pct:.0%} between the contexts "
                  f"(biggest: {n} {d:g})")


def placement_summary(row: dict) -> str:
    """One context's parsed placement figures as a line: "experts in VRAM 3072 | ..." ("-" for a figure
    the engine's log said nothing about)."""
    p = row.get("placement") or {}
    return "  " + " | ".join(f"{name} " + ("-" if (v := p.get(key)) is None else f"{v:g}")
                             for key, name in PLACEMENT_FIGURES)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="the engine's run config (the one setup wrote)")
    ap.add_argument("--contexts", default="32768,262144", help="the context sizes to compare (default "
                      "32768,262144; any two or more, e.g. 262144,1048576)")
    ap.add_argument("--port", type=int, default=8093, help="the first port to start on (one per context)")
    a = ap.parse_args(argv)
    if not has_proc():
        print("this check reads the process tree through /proc (Linux): /proc/self/status cannot be read "
              "here, so every figure would be 0.0 and the verdict a PASS on nothing - not run", file=sys.stderr)
        return 2
    contexts = [int(x) for x in a.contexts.replace(" ", "").split(",") if x.strip()]
    if len(contexts) < 2:
        ap.error("--contexts needs at least two sizes to compare")
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    taken = [(ctx, p) for ctx, p in zip(contexts, range(a.port, a.port + len(contexts))) if port_owner(p)]
    if taken:
        print(f"port(s) already in use: {', '.join(f'{p} (context {c})' for c, p in taken)} - free them or "
              "pass another --port", file=sys.stderr)
        return 2
    base = Path("scratch") / "mem-vs-ctx" / Path(a.config).stem
    rows = []
    for ctx, port in zip(contexts, range(a.port, a.port + len(contexts))):
        print(f"starting {Path(a.config).name} at context {ctx} on port {port} ...", flush=True)
        try:
            rows.append(measure_context(cfg, ctx, port, base / str(ctx)))
        except (TimeoutError, RuntimeError) as e:
            print(f"FAIL: {e} (the run's own log: {base / str(ctx)})", file=sys.stderr)
            return 1
        print(f"  stopped, port {port} free again", flush=True)
    print()
    print(format_table(rows))
    for r in rows:
        print(f"\nExpert placement read from the engine's log at context {r['context']}:\n{placement_summary(r)}")
        print(f"\nK/V and expert budget lines from the engine's log at context {r['context']}:")
        for line in r["budget"]:
            print("  " + line)
        if not r["budget"]:
            print("  (none)")
    ok, why = verdict(rows)
    print("\n" + why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
