"""tools/hy4_fit.py - Hy4-Flash on saguaro-hy4: the layout for the cards a PC has, for the setup (saguaro.py) and the
Models page (serve/model_pull.py) alike.

saguaro-hy4 keeps everything that is not a routed expert in VRAM (no RAM fallback: a size's "plan" in
tools/hy4/download.json says how much, by context), fills the rest with experts and computes the others on the CPU.
The one layout measured on Tesla T4s (bench/hy4/RESULTS.md) is the setup's for four equal 16 GB cards:

  four cards  --tp 4 --chunk 2048 --reserve-gb 0.5, HY4_HOT=1 (the experts placed by the learned scores), context 131072

Any other set of cards gets the engine's own choices (--tp auto, --chunk auto) at a 32K context: it runs, nothing of it
was measured here.  Nothing in this file touches a GPU."""
from __future__ import annotations

MIN_CC = 70                                         # the engine's kernels: V100 on (sm_75 is what runs here)
T4_GROUP = 4
T4_ARGS = ["--tp", "4", "--chunk", "2048", "--reserve-gb", "0.5"]
T4_CONTEXT = 131072
PLAIN_CONTEXT = 32768                               # (the largest context each size's plan lists)
CONTEXTS = [8192, 32768, 65536, 131072]
MARGIN_GIB = 0.5
# (decode tok/s, read tok/s): Maya-S on 4x Tesla T4, GPU clocks locked (bench/hy4/RESULTS.md)
MEASURED = {"maya-s": (32, 590)}


def need_gib(plan: dict, ctx: int) -> float:
    """GiB of VRAM a size takes for all that is not a routed expert at this context: the plan's figure of the nearest
    context listed at or below it, plus the cache's growth from there (16.6 KB a position)."""
    table = {int(k): float(v) for k, v in (plan.get("need_gib") or {}).items()}
    if not table:
        return 0.0
    below = [c for c in table if c <= ctx]
    base = max(below) if below else min(table)
    return table[base] + max(0, ctx - base) * 16640 / 2**30


def equal_group(cards: list, n: int = T4_GROUP) -> list:
    """n cards of one kind (the same name and VRAM) among these, or []."""
    kinds: dict = {}
    for g in cards:
        kinds.setdefault((g.get("name"), round(float(g.get("vram_gb") or 0))), []).append(g)
    best = max(kinds.values(), key=len, default=[])
    return best[:n] if len(best) >= n else []


def layout(cards: list, plan: dict, quant: str = "maya-s") -> dict:
    """{"gpus", "args": saguaro-hy4's flags beside --pack and --max-context, "env", "context", "measured", "fits":
    the non-expert parts fit in these cards' VRAM, "need_gib", "said"} for the cards chosen ({"arch", "vram_gb",
    "name"})."""
    cards = [g for g in cards if int(g.get("arch") or 0) >= MIN_CC]
    vram = sum(float(g.get("vram_gb") or 0) for g in cards)
    group = equal_group(cards) if len(cards) == T4_GROUP else []
    small = group and float(group[0].get("vram_gb") or 0) < 22
    if small:
        ctx, args, env = T4_CONTEXT, list(T4_ARGS), {"HY4_HOT": "1"}
        said = "four equal cards: tensor parallel over them, the experts placed by the learned scores, a 131K context"
    else:
        ctx, args, env = PLAIN_CONTEXT, [], {}
        said = "the engine's own split of these cards (--tp auto, --chunk auto), a 32K context"
    need = need_gib(plan, ctx)
    return {"gpus": cards, "args": args, "env": env, "context": ctx, "measured": bool(small) and quant in MEASURED,
            "fits": bool(cards) and vram - MARGIN_GIB * len(cards) >= need, "need_gib": need, "said": said}


def speeds(lay: dict, quant: str) -> str:
    """What the layout measured on Tesla T4s, as one phrase ("" where it was not measured)."""
    if not lay.get("measured") or quant not in MEASURED:
        return ""
    dec, read = MEASURED[quant]
    return f"about {dec} tok/s writing, {read} tok/s reading a prompt (measured on 4x Tesla T4, GPU clocks locked)"


def recommended(sizes: dict, vram_gb: float, ram_gb: float, default: str = "maya-s") -> str | None:
    """The size the setup recommends: the largest whose files' VRAM + RAM ("memory_gb") this PC has - but never past
    Maya-S on cards that were only measured with it; None when none fits.  sizes: {key: memory_gb}, smallest first."""
    fits = [k for k, gb in sizes.items() if gb and gb <= vram_gb + ram_gb]
    if not fits:
        return None
    return default if default in fits else fits[-1]
