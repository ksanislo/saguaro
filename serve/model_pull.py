"""Downloading models from the Models page and the API (/api/pull).  Nothing here downloads or verifies a file
itself: the download is the setup's own (`saguaro.py --setup --model <id> --download-model --yes --plain --no-start`: curl
with resume, sha256 check, the pack, the config), run as a child process whose output is read for its progress line.
One download at a time."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import maya_quants as Q  # noqa: E402  (tools/) the quants the setup offers

from serve import models  # noqa: E402

# the setup's progress line: "12.3 / 96.5 GB (13%) · 40 MB/s · 4 files at once"
PROGRESS = re.compile(r"([\d.]+) / ([\d.]+) GB(?: \((\d+)%\))?(?: · ([\d.]+) MB/s)?")


def catalog(root: Path, glm_sizes: dict | None = None, cards: list | None = None, ram_gb: float | None = None) -> list:
    """The models the setup can download: [{"id" (the setup's --model id), "family", "name" (as users see it),
    "installed" (a config here runs it), "size_gb" (None when the setup tells it at its start)}].  Hy4-Flash's sizes
    and Inkling 2 Mini also say what they need and how they fit this PC's cards ("needs", "fit", "recommended":
    fit_notes; cards / ram_gb: this PC's, read here when not given)."""
    have = set()
    for p in models.config_paths(root):
        cfg = models.read(p)
        if isinstance(cfg.get("installer"), dict):
            have.add((Q.family_of(cfg), str(Q.config_key(cfg))))
    sizes = glm_sizes if glm_sizes is not None else _glm_sizes()
    out = []
    new = new_engine_sizes() if glm_sizes is None else {}   # (sizes given: a caller that reads no table)
    notes = fit_notes(*((cards, ram_gb or 0.0) if cards is not None else this_pc() if glm_sizes is None else ([], 0.0)))
    for fam, key, _tier, ident in Q.QUANTS:
        if fam == "hy4" and new and (fam, key) not in new:   # (a size the table does not publish: not offered)
            continue
        out.append({"id": ident, "family": fam, "name": Q.shown(fam, key), "installed": (fam, key) in have,
                    "size_gb": new.get((fam, key)) if fam == "hy4" else sizes.get(key),
                    **(notes.get((fam, key)) or notes.get((fam, None)) or {})})
    out.append({"id": "cyber-frost", "family": "qwen", "name": Q.shown("qwen", "q5_k_m"),
                "installed": ("qwen", "q5_k_m") in have, "size_gb": None})
    for key in Q.IK2_QUANTS:                           # (unsloth's UD-Q4_K_XL, then Strata's packs)
        if key != Q.IK2_QUANT and new and ("ik2", key) not in new:   # (a pack the table does not publish)
            continue
        out.append({"id": Q.ik2_setup_id(key), "family": "ik2", "name": Q.shown("ik2", key),
                    "installed": ("ik2", key) in have, "size_gb": new.get(("ik2", key)),
                    **(notes.get(("ik2", None)) or {}), **(notes.get(("ik2", key)) or {})})
    return out


def new_engine_sizes() -> dict:
    """{(family, key): GB to download} for Hy4-Flash's published sizes (trunk, n-gram table, mmproj) and Inkling 2
    Mini (its GGUF and mmproj), from the setup's own tables."""
    try:
        sys.path.insert(0, str(ROOT))
        import saguaro as maya                                    # (nothing runs on import)
        out = {("hy4", k): round(maya.hy4_entry(k)["download_gb"], 1) for k in maya.HY4_QUANTS if maya.hy4_published(k)}
        vision = maya.ik2_vision_model()
        for k in Q.IK2_QUANTS:                                    # (each with the pictures' mmproj)
            if maya.ik2_published(k):
                out[("ik2", k)] = round(maya.ik2_entry(k)["download_gb"] + (vision["download_gb"] if vision else 0), 1)
        return out
    except (Exception, SystemExit):  # noqa: BLE001 - a damaged table: the setup says so when the model is picked
        return {}


_pc: dict = {}


def this_pc() -> tuple:
    """(this PC's NVIDIA cards as the setup reads them, its RAM in GB); ([], 0.0) where they cannot be read.  Asked
    once a minute."""
    if _pc and time.time() - _pc["at"] < 60:
        return _pc["cards"], _pc["ram"]
    cards, ram = _read_pc()
    _pc.update(at=time.time(), cards=cards, ram=ram)
    return cards, ram


def _read_pc() -> tuple:
    try:
        import setup as SU
        cards = [g for g in SU.gpus() or [] if g.get("vram_gb")]
    except Exception:  # noqa: BLE001 - no NVIDIA tools here
        cards = []
    if not cards:                                  # (AMD: the KFD topology's GPUs, "vendor": "amd")
        try:
            import setup as SU
            cards = [g for g in SU.amd_gpus() or [] if g.get("vram_gb")]
        except Exception:  # noqa: BLE001 - no amdgpu driver here
            cards = []
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30   # (as the setup counts it)
    except (ValueError, OSError, AttributeError):
        ram = 0.0
    return cards, ram


def fit_notes(cards: list, ram_gb: float) -> dict:
    """What Hy4-Flash and Inkling 2 Mini need and how they fit these cards (tools/hy4_fit.py, tools/ik2_fit.py - the
    setup's own rules): {(family, key or None): {"needs", "fit", "recommended"}}.  "fit" is "" where no card was read."""
    import hy4_fit as HY4F
    import ik2_fit as IK2F
    out = {("hy4", None): {"needs": "Made for four 16 GB NVIDIA cards (Linux); Maya-S takes about 68 GB of VRAM and "
                                    "RAM together. One conversation at a time.", "fit": "", "recommended": False},
           ("ik2", None): {"needs": "Linux: one or two NVIDIA cards with 8 GB or more (V100 / Tesla T4 / RTX 20 and "
                                    "newer), or one AMD RDNA 3 / 4 GPU with 12 GB or more; the experts a card does not "
                                    "hold stay in RAM (the offload path). Made for one or two 16 GB cards.", "fit": "",
                           "recommended": False}}
    if not cards:
        return out
    amd = any(g.get("vendor") == "amd" for g in cards)
    if amd:                                        # (the setup's AMD choice: one RDNA 3 / 4 GPU of 12 GB or more)
        cards = [g for g in cards if str(g.get("arch") or "").startswith(IK2F.AMD_ARCHS)] or cards
        cards = sorted(cards, key=lambda g: -float(g.get("vram_gb") or 0))[:1]
    vram = sum(float(g.get("vram_gb") or 0) for g in cards)
    small = all(float(g.get("vram_gb") or 0) < 24 for g in cards)
    ik2 = out[("ik2", None)]
    if amd:
        why = IK2F.unsupported("hip", sys.platform == "win32", 0, cards, ram_gb)
        lay = None if why else IK2F.amd_layout(cards[0])
    else:
        lay = IK2F.layout(cards)
        why = IK2F.unsupported("cuda", sys.platform == "win32", 0, cards, ram_gb)
    if lay is None or why:
        ik2["fit"] = "This PC: " + (why or "no card runs it")
    else:
        ik2["fit"] = f"This PC: {lay['said']}" + (f" - {IK2F.speeds(lay)}" if IK2F.speeds(lay) else "")
        ik2["recommended"] = small and len(cards) <= 2
        try:   # (each file set: how it fits; the one the setup would pick recommended)
            sys.path.insert(0, str(ROOT))
            import saguaro as maya
            rec = maya.ik2_recommend(cards, amd)
            for k in Q.IK2_QUANTS:
                if not maya.ik2_published(k):
                    continue
                ex = maya.ik2_entry(k)["experts_gib"]
                lk = IK2F.amd_layout(cards[0], ex) if amd else IK2F.layout(cards, experts_gib=ex)
                out[("ik2", k)] = {"recommended": ik2["recommended"] and k == rec}
                if k != Q.IK2_QUANT and lk:   # (UD-Q4_K_XL keeps the general line: its speeds were measured with it)
                    out[("ik2", k)]["fit"] = f"This PC: {lk['said']}"
        except (Exception, SystemExit):  # noqa: BLE001 - the general line stays
            pass
    group = HY4F.equal_group(cards) if len(cards) >= HY4F.T4_GROUP and not amd else []
    hy4 = out[("hy4", None)]
    if amd:
        hy4["fit"] = "This PC: Hy4-Flash runs on NVIDIA GPUs only for now" + (
            "; Inkling 2 Mini is the model for this PC" if ik2["recommended"] else "")
    elif group and small:
        said = "four equal cards: tensor parallel over them"
        hy4["fit"] = "This PC: " + said + (" (not enough VRAM and RAM for Maya-S)" if vram + ram_gb < 68 else "")
        out[("hy4", "maya-s")] = {**hy4, "fit": f"This PC: {said} - about 32 tok/s writing, 590 tok/s reading a "
                                                "prompt (measured on 4x Tesla T4, GPU clocks locked)",
                                  "recommended": vram + ram_gb >= 68}
    else:
        hy4["fit"] = (f"This PC: {len(cards)} card{'s' if len(cards) != 1 else ''} - it runs with most experts on the "
                      "CPU (not measured)" + ("; Inkling 2 Mini is the model for this PC" if ik2["recommended"] else ""))
    return out


def _glm_sizes() -> dict:
    try:
        sys.path.insert(0, str(ROOT))
        import saguaro as maya                                    # (the setup's own table; nothing runs on import)
        return {k: m.get("download_gb") for k, m in maya.MODELS.items()}
    except Exception:  # noqa: BLE001 - a size is a nicety
        return {}


def setup_command(model_id: str, cfg: dict | None) -> list:
    """The setup run that downloads `model_id` the way this install was set up (its models folder and backend)."""
    inst = (cfg or {}).get("installer") or {}
    cmd = [sys.executable, str(ROOT / "saguaro.py"), "--setup", "--model", model_id, "--download-model", "--yes",
           "--plain", "--no-start"]
    if inst.get("models_dir"):
        cmd += ["--models-dir", str(inst["models_dir"])]
    if (cfg or {}).get("backend") in ("hip", "metal"):
        cmd += ["--backend", cfg["backend"]]
    return cmd


class Pull:
    """One download: its child process, the last progress and the last lines of the setup's output."""

    def __init__(self, model_id: str, command: list, cwd: Path = ROOT):
        self.id = model_id
        self.started = time.time()
        self.state = "starting"                        # starting, downloading, success, error, cancelled
        self.completed = self.total = 0
        self.speed = None
        self.error = None
        self.log: list = []
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._read, daemon=True).start()

    def _line(self, text: str):
        text = text.strip()
        if not text:
            return
        m = PROGRESS.search(text)
        with self._lock:
            if m:
                self.state = "downloading"
                self.completed, self.total = int(float(m.group(1)) * 1e9), int(float(m.group(2)) * 1e9)
                self.speed = float(m.group(4)) * 1e6 if m.group(4) else None
            else:
                self.log = (self.log + [text])[-40:]

    def _read(self):
        buf = b""
        while True:
            b = self.proc.stdout.read(1)               # (the setup rewrites its progress line with \r)
            if not b:
                break
            if b in (b"\r", b"\n"):
                self._line(buf.decode("utf-8", "replace"))
                buf = b""
            else:
                buf += b
        self._line(buf.decode("utf-8", "replace"))
        rc = self.proc.wait()
        with self._lock:
            if self.state == "cancelled":
                return
            if rc == 0:
                self.state = "success"
            else:
                self.state = "error"
                self.error = next((ln for ln in reversed(self.log) if ln.strip()), f"the setup stopped (code {rc})")

    def cancel(self):
        with self._lock:
            if self.state in ("success", "error", "cancelled"):
                return
            self.state = "cancelled"
        self.proc.terminate()

    def snapshot(self) -> dict:
        with self._lock:
            pct = int(100 * self.completed / self.total) if self.total else None
            return {"model": self.id, "state": self.state, "completed": self.completed, "total": self.total,
                    "percent": pct, "bytes_per_s": self.speed, "error": self.error, "started": self.started,
                    "message": self.log[-1] if self.log else "", "log": list(self.log[-12:])}

    def active(self) -> bool:
        return self.state in ("starting", "downloading")


class Puller:
    """The server's one download slot."""

    def __init__(self, root: Path, command=None):
        self.root = Path(root)
        self.command = command                         # tests: a function (model_id, cfg) -> argv
        self.current: Pull | None = None
        self._lock = threading.Lock()

    def start(self, ident: str, cfg: dict | None) -> Pull:
        known = {c["id"] for c in catalog(self.root, {})}
        ident = str(ident or "").strip().lower()
        ident = ident.removesuffix(":latest")
        if ident not in known:
            raise ValueError(f"{ident!r} is not in Saguaro's catalog (GET /api/catalog lists the ids)")
        with self._lock:
            if self.current is not None and self.current.active():
                raise ValueError(f"{self.current.id} is downloading; wait for it or cancel it")
            self.current = Pull(ident, (self.command or setup_command)(ident, cfg))
            return self.current

    def snapshot(self) -> dict | None:
        return self.current.snapshot() if self.current is not None else None

    def busy(self) -> bool:
        return self.current is not None and self.current.active()

    def cancel(self) -> bool:
        if self.busy():
            self.current.cancel()
            return True
        return False


def ndjson_events(pull: Pull, poll: float = 1.0):
    """The /api/pull stream for `pull`: {"status": ...} lines, with total / completed while it downloads, then
    "success" (or an {"error": ...} line)."""
    yield {"status": "pulling " + pull.id}
    last = None
    while True:
        s = pull.snapshot()
        if s["state"] == "downloading" and s["total"]:
            ev = {"status": f"downloading {pull.id}", "digest": pull.id, "total": s["total"], "completed": s["completed"]}
        elif s["state"] in ("starting", "downloading"):
            ev = {"status": s["message"] or "preparing"}
        elif s["state"] == "success":
            if s["total"]:                              # (a fast one finishes between two polls: the full bar anyway)
                yield {"status": f"downloading {pull.id}", "digest": pull.id, "total": s["total"], "completed": s["total"]}
            yield {"status": "verifying sha256 digest"}
            yield {"status": "success"}
            return
        else:
            yield {"error": s["error"] or s["state"]}
            return
        if ev != last:
            yield ev
            last = ev
        time.sleep(poll)
