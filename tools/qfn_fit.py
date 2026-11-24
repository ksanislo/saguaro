"""tools/qfn_fit.py - Qwen3.8-Flash-Next on saguaro-qfn: what fits on a PC's cards and the layout for them, for
the setup (saguaro.py) and its --check alike.  Nothing here touches a GPU: the caller passes the facts
(cards {"name", "vram_gb", "arch"}, the PC's RAM, the quant), so it is unit-testable.

saguaro-qfn reads unsloth's GGUFs in place (its pack indexes them): a card holds the most-routed experts of the
layers it serves in its VRAM, the next ones in a pinned RAM tier (--resident-budget-gib: the RAM less RAM_SPARE_GB),
and reads the rest from the SSD as tokens route them.  With --kv int8 a context token costs 12,672 bytes of K/V
(measured: the 12 QSA layers x 1,056 B - 1,024 int8 codes and their 32 B of fp16 scales); with --kv-resident only
the newest N cells of each layer stay in VRAM and the whole K/V sits in pinned RAM, so a long context costs the
cards a fixed slice and a bigger context is nearly free (engines/qfn/src/program/generate.cpp).

One GPU is the layout measured on AMD (an RX 9070 XT with 96 GB of RAM: 27-41 tok/s answers) and what the setup
writes for one NVIDIA card.  Several cards take the 48 layers in turn (the config's "layer_split": the FIRST LAYER
of each later card) the way the live 4x Tesla T4 configs do - UD-Q4_K_XL "11,24,36", Q8_0 "10,22,35", each stage
loading only its own layers' dense weights (--trim-stage-weights) and its prompt chunks sized for the pipeline
(--prefill-pipe, the rate its --calibrate measured).  The files' sizes come from tools/qfn/download.json."""
from __future__ import annotations

import json
import os

GIB = 1 << 30
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qfn", "download.json")
KV_BYTES = 12672                                      # K/V bytes a context token, --kv int8 (measured)
LAYERS = 48                                           # the model's transformer layers (the layer split's reach)
KV_RESIDENT = 32768                                   # the newest cells of each QSA layer kept in VRAM (the configs)
RAM_SPARE_GB = 20                                     # RAM left beside the tier: the OS, the server, the page cache
MIN_VRAM_GB = 15.0                                    # every card of a layout (the manifest's "needs": a 16 GB card)
MIN_RAM_GB = 88.0                                     # the RAM the tier and the system need (the manifest's "needs")
MIN_CC = 70                                           # NVIDIA: the engine builds for Volta and newer
MAX_GROUP = 4                                         # a layer split runs on up to four cards (the live configs)
DEFAULT_CONTEXT = 65536                               # one card (measured there: prompts up to 60K tokens)
SPLIT_CONTEXT = 262144                                # a split: the K/V's RAM is cheap, the cards do not notice
# (both must stay among the contexts saguaro.py offers - QFN_CONTEXTS; test_setup_qfn guards it)
T4_LOW_GB, T4_HIGH_GB = 14.0, 17.0                    # a "16 GB" card as the driver reports it (Tesla T4: 15.0)
SM75 = 75                                             # compute capability 7.5 (Tesla T4, Quadro RTX 3000)
# the live 4x Tesla T4 configs' splits (./ref/Qwen3.8-Flash-Next-*.json): first layer of GPU 2, 3 and 4
MEASURED_SPLITS = {"ud-q4_k_xl": "11,24,36", "q8_0": "10,22,35"}
# ... and their pipeline rates (--prefill-pipe: this PC's b/a in tokens, its --calibrate measures it; 1 only evens
# out the stages' chunk counts, which is what an unmeasured PC gets)
MEASURED_PREFILL_PIPE = {"ud-q4_k_xl": "1536", "q8_0": "6144"}
# the live configs' T4 tuning env (SAGUARO_SM75_*, KQL, BF16_TC*, MMQ_*): write it only where every chosen card is
# compute capability 7.5 - it is what makes mangonel's T4s fast, and untested elsewhere.  SAGUARO_MMQ_KQ is not
# part of it: the setup writes it everywhere, AMD included.  Two values differ between the live boxes' quants:
# the Q8_0 box skips the BF16 tensor-core projections of indexer, router and ssm (UD-Q4_K_XL skips nothing) and
# runs no SAGUARO_SELECT_SM75 (./ref/Qwen3.8-Flash-Next-*.json).
T4_ENV = {"SAGUARO_SM75_PERF": "1", "SAGUARO_SM75_STACK": "1", "SAGUARO_SM75_Q8G": "1", "SAGUARO_KQL": "3",
          "SAGUARO_BF16_TC": "1", "SAGUARO_MMQ_DIRECT": "1", "SAGUARO_MMQ_KTAIL": "1", "SAGUARO_MMQ_SWIGLU_Q8": "1"}
T4_BF16_TC_SKIP = {"ud-q4_k_xl": "none", "q8_0": "indexer,router,ssm"}
T4_SELECT_SM75 = {"ud-q4_k_xl": "1"}                # (the live Q8_0 config carries no such key)
SIZES: dict = {}                                      # quant -> GiB, from the manifest (read once)


def quant_gib(quant: str) -> float:
    """GiB of a quant's GGUF files, summed from tools/qfn/download.json (UD-Q4_K_XL 103.7, Q8_0 175.3);
    0.0 for a quant or a manifest that does not read."""
    if not SIZES:
        try:
            with open(MANIFEST, encoding="utf-8") as fh:
                man = json.load(fh)
            for k, q in (man.get("quants") or {}).items():
                SIZES[k] = sum(v[0] for v in (q.get("files") or {}).values()) / GIB
        except (OSError, ValueError, KeyError, TypeError):
            return 0.0
    return SIZES.get(quant, 0.0)


def resident_budget_gib(ram_gb: float) -> int:
    """--resident-budget-gib for a PC with `ram_gb` GiB of RAM: all but RAM_SPARE_GB (measured on 91.7 GiB: 71
    answered 32% faster than 64 with 10.8 GiB still free; the engine takes no more than the experts)."""
    return max(8, int(ram_gb - RAM_SPARE_GB))


def kv_vram_gib(kv_resident: int = KV_RESIDENT) -> float:
    """GiB the resident K/V window of the whole model takes: --kv-resident cells x QSA layers x 1,056 B."""
    return kv_resident * KV_BYTES / GIB


def _cc(g: dict) -> int:
    """A card's compute capability as an int (an AMD card's "gfx..." arch reads as 0)."""
    try:
        return int(str(g.get("arch") or "0").split(".")[0])
    except ValueError:
        return 0


def sm75_only(cards: list) -> bool:
    """True where every card is compute capability 7.5 (Tesla T4 / Quadro RTX 3000): the gate for the live
    configs' T4 tuning env (t4_env)."""
    return bool(cards) and all(_cc(g) == SM75 for g in cards)


def t4_env(quant: str) -> dict:
    """The live configs' T4 tuning env for `quant`: T4_ENV plus its own BF16_TC_SKIP / SELECT_SM75 (a quant the
    live boxes never ran gets the shared part alone)."""
    env = dict(T4_ENV)
    if quant in T4_BF16_TC_SKIP:
        env["SAGUARO_BF16_TC_SKIP"] = T4_BF16_TC_SKIP[quant]
    if quant in T4_SELECT_SM75:
        env["SAGUARO_SELECT_SM75"] = T4_SELECT_SM75[quant]
    return env


def t4_sized(g: dict) -> bool:
    """A "16 GB" card as the driver reports it (a Tesla T4 says 15.0)."""
    return T4_LOW_GB <= float(g.get("vram_gb") or 0) < T4_HIGH_GB


def equal_group(cards: list, n: int = MAX_GROUP) -> list:
    """Up to n cards of one size among these (a split wants equal stages), or []."""
    kinds: dict = {}
    for g in cards:
        kinds.setdefault(round(float(g.get("vram_gb") or 0)), []).append(g)
    best = max(kinds.values(), key=len, default=[])
    return best[:n] if len(best) >= 2 else []


def unsupported(cards: list, ram_gb: float) -> str | None:
    """Why these cards ({"name", "vram_gb"}) and this RAM cannot run Qwen3.8-Flash-Next on saguaro-qfn, in one
    plain line, or None.  A card whose VRAM is not known (0) is not a refusal (the engine measures at its start);
    ram_gb 0: not read yet."""
    for g in cards:
        v = float(g.get("vram_gb") or 0)
        if 0 < v < MIN_VRAM_GB:
            return (f"Qwen3.8-Flash-Next on saguaro-qfn needs GPUs of 16 GB or more: every card of a layout caches "
                    f"the experts of its own layers (this PC's {g.get('name') or 'GPU'}: {v:.0f} GB)")
    if ram_gb and ram_gb < MIN_RAM_GB:
        return (f"Qwen3.8-Flash-Next needs about 96 GB of RAM (this PC: {ram_gb:.0f} GB): the experts the cards do "
                "not hold must fit in RAM beside the system, or nearly every token waits on the SSD")
    return None


def layer_split(cards: list, quant: str = "ud-q4_k_xl") -> str | None:
    """The config's "layer_split" for these cards (the FIRST LAYER of each later card; None for one card): the live
    4x Tesla T4 configs' own values on four "16 GB" cards of compute capability 7.5, else the 48 layers shared out
    evenly."""
    n = len(cards)
    if n <= 1:
        return None
    if n == 4 and quant in MEASURED_SPLITS and all(t4_sized(g) for g in cards) and sm75_only(cards):
        return MEASURED_SPLITS[quant]
    return ",".join(str(round(LAYERS * i / n)) for i in range(1, n))


def best_gpus(cards: list) -> list:
    """The cards the setup takes without being told: one big card alone (it caches its experts without a split
    over the bus), else up to MAX_GROUP equal cards for a split - more cards, more experts in VRAM - else simply
    the largest one (one 16 GB card runs it).  [] for no cards."""
    have = sorted((g for g in cards if not float(g.get("vram_gb") or 0) or float(g["vram_gb"]) >= MIN_VRAM_GB),
                  key=lambda g: (-round(float(g["vram_gb"])), g.get("index", 0)))
    if not have:
        return []
    if float(have[0].get("vram_gb") or 0) >= 20.0 or len(have) == 1:
        return have[:1]
    group = equal_group(have)
    return group or have[:1]


def plan(cards: list, ram_gb: float, quant: str = "ud-q4_k_xl") -> dict:
    """The layout for these cards: {"fits", "reason", "gpus", "n_gpus", "layer_split" (None for one card),
    "context" (to recommend), "kv_resident", "resident_budget_gib", "args" (the engine's flags beside the generic
    ones: --trim-stage-weights and --prefill-pipe on a split), "env" (the T4 tuning env where every card is sm_75,
    else {}), "t4", "measured", "said" (one plain line)}.  The cards it lists are the ones handed in - best_gpus()
    picks those, plan() lays them out; nothing is read from the hardware."""
    why = (f"Qwen3.8-Flash-Next on saguaro-qfn runs on one GPU of 16 GB or more, or up to {MAX_GROUP} equal ones: "
           "no cards were named") if not cards else unsupported(cards, ram_gb)
    n = len(cards)
    split = layer_split(cards, quant) if why is None else None
    ctx = DEFAULT_CONTEXT if n <= 1 else SPLIT_CONTEXT
    t4 = sm75_only(cards)
    budget = resident_budget_gib(ram_gb) if ram_gb else None
    live = bool(split) and split in set(MEASURED_SPLITS.values())        # a live 4x Tesla T4 config's split
    args = ([] if n <= 1 else
            ["--trim-stage-weights", "--prefill-pipe", MEASURED_PREFILL_PIPE.get(quant, "1") if live else "1"])
    amd = str((cards[0] if cards else {}).get("arch") or "").startswith("gfx")
    measured = why is None and ((live and sm75_only(cards)) or
                                (n == 1 and amd and t4_sized(cards[0]) and ram_gb >= 90.0))
    said = ("no cards named" if not cards else
            f"one card: the experts it holds in VRAM, the next ones in a {budget if budget is not None else 'RAM - 20'} "
            f"GiB RAM tier, the rest from the SSD, a {ctx // 1024}K context" if n <= 1 else
            f"{n} cards: the {LAYERS} layers split across them ({split}), each caching its own layers' experts, "
            f"a {ctx // 1024}K context")
    return {"fits": why is None, "reason": why, "gpus": list(cards), "n_gpus": n, "layer_split": split,
            "context": ctx, "kv_resident": min(ctx, KV_RESIDENT), "resident_budget_gib": budget,
            "args": args, "env": t4_env(quant) if (t4 and why is None) else {}, "t4": t4,
            "measured": measured, "said": said}


def main(argv=None) -> int:
    """The same from a shell, for checking a layout without a PC: cards as "VRAM[:CC]" GiB figures.
    tools/qfn_fit.py --ram 220 --quant ud-q4_k_xl 16 16:75 16 16"""
    import argparse
    ap = argparse.ArgumentParser(description="Qwen3.8-Flash-Next on saguaro-qfn: the fit and layout for these cards")
    ap.add_argument("cards", nargs="*", help="one card per GPU: VRAM GiB, optionally :compute-capability (16:75)")
    ap.add_argument("--ram", type=float, default=0.0, help="the PC's RAM in GiB (0: not known)")
    ap.add_argument("--quant", default="ud-q4_k_xl", choices=sorted(MEASURED_SPLITS), help="the quant to lay out")
    a = ap.parse_args(argv)
    cards = []
    for i, spec in enumerate(a.cards):
        v, _, cc = spec.partition(":")
        cards.append({"index": i, "name": f"GPU {i}", "vram_gb": float(v), "arch": cc or 0})
    p = plan(cards, a.ram, a.quant)
    print(f"{a.quant}: {quant_gib(a.quant):.0f} GiB of files")
    print(("fits: " + p["said"]) if p["fits"] else "does not fit: " + p["reason"])
    if p["fits"]:
        b = p["resident_budget_gib"]
        print(f"context {p['context']}, --kv-resident {p['kv_resident']} ({kv_vram_gib(p['kv_resident']):.1f} GiB "
              f"of K/V in VRAM), resident budget {b if b is not None else '-'} GiB")
        print(f"layer_split: {p['layer_split'] if p['layer_split'] else '-'}   args: {' '.join(p['args']) or '-'}")
        print(f"T4 env: {'yes' if p['t4'] else 'no'}" + ("   (measured on this layout)" if p["measured"] else ""))
    return 0 if p["fits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
