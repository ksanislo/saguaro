"""tools/ik2_fit.py - Inkling 2 Mini on saguaro-ik2: the layout for the cards a PC has, for the setup (saguaro.py)
and the Models page (serve/model_pull.py) alike.

saguaro-ik2 runs unsloth's UD-Q4_K_XL (17.6 GB) on one or two NVIDIA cards from Turing (sm_75) on, Linux.  The two
layouts measured on Tesla T4s (16 GB, NEXT-ik2.md) are the setup's:

  one card    --gpus 1 --chunk 2048 SAGUARO_IK2_PIN=0, context 32768: about a quarter of the experts are streamed from RAM
  two cards   --gpus 2 --chunk 8192, context 131072: every expert in VRAM

A single card of 22 GB or more holds every expert (upstream Maya's layout, measured there on an RTX 3090, not here):
the context is then the largest whose cache fits beside the model's 18.3 GiB.  Nothing here touches a GPU."""
from __future__ import annotations

GIB = 1 << 30
MIN_CC = 75                                         # the engine's kernels: Turing's tensor cores on
MIN_VRAM_GB = 14.5                                  # a "16 GB" card (a T4 reports 15-16); less does not hold the dense parts
BIG_VRAM_GB = 22.0                                  # a "24 GB" card: the whole model on it
BASE_GIB = 18.3                                     # one big card: the weights, the windowed layers' cache, the scratch
KV_BYTES = {"fp8": 16384, "f16": 32768}             # the cache of a context token (SAGUARO_IK2_KV; f16 the engine's own)
MARGIN_GIB = 0.5
CONTEXTS = [32768, 65536, 131072, 262144]
ROOM_MB = 1024                                      # SAGUARO_IK2_ROOM_MB: VRAM left free on a T4-sized card
THREADS = 8
LAYOUT_ENV = {"SAGUARO_IK2_ROOM_MB", "SAGUARO_IK2_PIN"}   # (the env keys a layout owns)
# (decode tok/s, read tok/s) measured on Tesla T4s, GPU clocks not pinned, ~1K tokens deep, read 8-16K (NEXT-ik2.md, 2026-11-22)
MEASURED = {1: (42, 640), 2: (52, 1130)}


def usable(cards: list) -> list:
    """The cards saguaro-ik2 can use ({"arch", "vram_gb"}, nvidia-smi's), the largest first."""
    ok = [g for g in cards if int(g.get("arch") or 0) >= MIN_CC and float(g.get("vram_gb") or 0) >= MIN_VRAM_GB]
    return sorted(ok, key=lambda g: (-round(float(g["vram_gb"])), g.get("index", 0)))


def big_context(vram_gb: float, kv: str = "f16") -> int:
    """One big card: the largest of CONTEXTS whose cache leaves every expert in VRAM, else the smallest."""
    fits = [c for c in CONTEXTS if BASE_GIB + c * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB <= vram_gb - MARGIN_GIB]
    return max(fits) if fits else CONTEXTS[0]


def layout(cards: list, want: int = 0, kv: str = "f16") -> dict | None:
    """The layout for these cards: {"gpus": the cards it runs on, "args": saguaro-ik2's flags beside --model and
    --max-context, "context", "env", "resident": every expert in VRAM, "measured": on T4s here, "said": one plain
    line}.  want: 1 or 2 cards asked for (0: the best this PC has).  None: no card can run it (unsupported says why)."""
    have = usable(cards)
    if not have:
        return None
    big = float(have[0]["vram_gb"]) >= BIG_VRAM_GB
    n = want if want in (1, 2) else (1 if big or len(have) < 2 else 2)
    n = min(n, len(have))
    if n == 2:
        return {"gpus": have[:2], "args": ["--gpus", "2", "--chunk", "8192", "--threads", str(THREADS)],
                "context": 131072, "env": {"SAGUARO_IK2_ROOM_MB": str(ROOM_MB)}, "resident": True,
                "measured": all(float(g["vram_gb"]) < BIG_VRAM_GB for g in have[:2]),
                "said": "two cards: every expert in VRAM, a 131K context"}
    if big:
        ctx = big_context(float(have[0]["vram_gb"]), kv)
        return {"gpus": have[:1], "args": ["--gpus", "1", "--chunk", "2048", "--threads", str(THREADS)],
                "context": ctx, "env": {}, "resident": True, "measured": False,
                "said": "one card of 24 GB or more: every expert in VRAM"}
    return {"gpus": have[:1], "args": ["--gpus", "1", "--chunk", "2048", "--threads", str(THREADS)],
            "context": 32768, "env": {"SAGUARO_IK2_ROOM_MB": str(ROOM_MB), "SAGUARO_IK2_PIN": "0"}, "resident": False, "measured": True,
            "said": "one 16 GB card: about a quarter of the experts streamed from RAM, a 32K context"}


def speeds(lay: dict) -> str:
    """What a layout measured on Tesla T4s, as one phrase ("" where it was not measured)."""
    if not lay.get("measured"):
        return ""
    dec, read = MEASURED[len(lay["gpus"])]
    return f"about {dec} tok/s writing, {read} tok/s reading a prompt (measured on {len(lay['gpus'])}x Tesla T4)"


def unsupported(backend: str, windows: bool, cards: list) -> str | None:
    """Why this PC cannot run Inkling 2 Mini, in one plain line, or None.  cards: the NVIDIA GPUs it would pick from -
    empty when they are not read yet (nothing said about them then)."""
    if backend in ("hip", "metal"):
        return "Inkling 2 Mini runs on NVIDIA GPUs only for now (no AMD or Mac engine yet)"
    if windows:
        return "Inkling 2 Mini runs on Linux only for now (not on Windows yet)"
    if cards and not usable(cards):
        if not [g for g in cards if int(g.get("arch") or 0) >= MIN_CC]:
            return ("Inkling 2 Mini needs an NVIDIA GPU of compute capability 7.5 or newer (Tesla T4 / RTX 20 and "
                    "newer)")
        return "Inkling 2 Mini needs an NVIDIA GPU with 16 GB of VRAM or more (two of them hold the whole model)"
    return None
