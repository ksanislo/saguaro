"""tools/ik2_fit.py - Inkling 2 Mini on saguaro-ik2: what fits on a PC's cards, for the setup (saguaro.py) and the
Models page (serve/models.py, serve/model_pull.py) alike.  Nothing here touches a GPU.

saguaro-ik2 (upstream Maya v1.1.74's engine) keeps every expert in VRAM where they fit (the resident token path);
where they do not - a card under 24 GB with unsloth's UD-Q4_K_XL, or a context past what one holds - it takes the
offload path: a tier of the most-used experts in VRAM, the others in pinned RAM, fetched by the GPU (or computed by
the CPU lane) as tokens pick them.  Two NVIDIA cards: the first alone where it holds every expert, else the split -
the first on the offload path (dense weights, cache, prompts, its tier), the second with the next experts.  AMD (HIP,
Linux, one GPU of 12 GB or more): the same plan - every expert in the GPU's memory where the file fits, else the offload
path; a card under 20 GB keeps 1 GiB beside the experts (SAGUARO_IK2_RESERVE_MB), which plan() counts (AMD_SMALL_CREDIT_GIB).

The model's sizes are upstream's (an RTX 3090 measured: 18.3 GiB beside the cache at 262,144 with an FP8 cache, 14.7
GiB of it UD-Q4_K_XL's experts); another file's experts (experts_gib_of: our one-T4 quant holds 11.45 GiB) take their
place.  The layouts' speeds were measured on Tesla T4s (16 GB, NEXT-ik2.md) and an RX 9070 XT (docs/INKLING2-MINI.md)."""
from __future__ import annotations

import os

GIB = 1 << 30
BASE_GIB = 18.3                                     # the model's weights, the windowed layers' cache, the scratch
EXPERTS_GIB = 14.7                                  # the routed experts (UD-Q4_K_XL) - of BASE_GIB
DENSE_GIB = BASE_GIB - EXPERTS_GIB                  # the rest, whatever file's experts beside it
OFF_ROOM_GIB = 1.2                                  # the offload path's room for prompts and its staging slots
HOST_GIB = 1.5                                      # the engine's RAM beside the experts (embedding rows, buffers)
OFF_HOST_GIB = 0.5                                  # the offload path's RAM beside the experts it pins
OS_GIB = 1.5                                        # the system's and the server's RAM beside the engine ("tight" under)
KV_BYTES = {"fp8": 16384, "f16": 32768}             # the cache of a context token (SAGUARO_IK2_KV)
MARGIN_GIB = 0.5                                    # the driver's and the desktop's share of the card
CONTEXTS = [32768, 65536, 131072, 262144, 524288, 1048576]   # up to the trained 1,048,576
RECOMMEND_MAX = 262144                              # the largest context the setup recommends by itself
FULL_VISITS = 8                                     # the visits of the layers without a window: KV_BYTES / 8 each
SCRATCH_TOKEN_BYTES = 1152                          # the attention's scratch a context token past 262,144
RING_OF_VISIT = 0.1                                 # the tokens' copy ring beside a full visit (KV streaming)
STREAM_HOST_GIB = 4.0                               # the RAM the engine leaves the system beside the streamed rows
OFF_TIER_MIN_GIB = 0.5                              # an offload layout streams only where its cache leaves the tier less
DEFAULT_KV = "f16"                                  # the engine's own (the T4 layouts were measured with it)
MIN_VRAM_GB = 7.0                                   # the engine's own floor (an "8 GB" card)
MIN_TOTAL_GB = 24.0                                 # RAM + VRAM that hold the model, as the refusal says it
MIN_CC = 60                                         # the engine's kernels: Pascal on (second-class: scalar kernels, no tensor cores; Volta: m8n8k4)
AMD_ARCHS = ("gfx11", "gfx12")                      # AMD: RDNA 3 / 3.5 / 4 - the matrix-core prompt kernels (HIP)
TWO_ROOM_GIB = 0.3                                  # the second card's room beside its experts
ROOM_MB = 1024                                      # SAGUARO_IK2_ROOM_MB: the offload path's room (the engine's default)
THREADS = 8
TWO_CONTEXT = 131072                                # two 16 GB cards: every expert in VRAM at this context (measured)
AMD_MIN_GB = 11.0                                   # (a 12 GB AMD card as the driver reports it)
LAYOUT_ENV = {"SAGUARO_IK2_ROOM_MB", "SAGUARO_IK2_PIN", "SAGUARO_IK2_PIN_KEEP", "SAGUARO_IK2_WARM", "SAGUARO_IK2_RESERVE_MB"}   # (a layout's keys;
                                                    # PIN / PIN_KEEP: the previous engine's, dropped from old configs)
# (decode tok/s, read tok/s) on Tesla T4s, GPU clocks not pinned, ~50 tokens deep, read 8K / 16K (2026-11-22)
MEASURED = {1: (40, 840), 2: (51, 1100)}
# ... and on an RX 9070 XT (gfx1201, 16 GB), 32K: UD-Q4_K_XL on the offload path, the one-T4 quant resident (2026-11-22)
MEASURED_AMD = {"offload": (71, 1670), "resident": (83, 1150)}
ONE_T4_EXPERTS_GIB = 11.45                          # our Inkling-2-Mini-T4-NVFP4-IQ3_XXS's experts
AMD_RESERVE_UNDER_GB = 20.0                         # an AMD card under this keeps SAGUARO_IK2_RESERVE_MB=1024 (amd_layout)
# What that 1 GiB reserve (the engine's default: 1.5) and the 32K scratch (BASE_GIB counts 262K's) leave the plan
# beside every expert: the one-T4 quant at 32K is resident on an RX 9070 XT, 15.5 of its 15.9 GiB used (measured)
AMD_SMALL_CREDIT_GIB = 1.0


def subprocess_error():
    import subprocess
    return subprocess.SubprocessError


def experts_gib_of(path) -> float | None:
    """GiB of a GGUF's routed experts (its *_exps tensors, from the header's offsets), or None where it does not
    read.  UD-Q4_K_XL: 14.71; our Inkling-2-Mini-T4-NVFP4-IQ3_XXS: 11.45.  A Strata pack: its experts as saguaro-ik2
    holds them (tools/strata/strata_pack.py: SQ-20 14.95, SQ-15 13.03, SQ-10 10.45)."""
    try:
        with open(path, "rb") as fh:
            if fh.read(8) == b"STRATA\x00\x01":
                import sys as _sys
                _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "strata"))
                import strata_pack as SP
                return SP.inkling_expert_bytes(SP.Pack(path)) / GIB
    except (OSError, ValueError, KeyError, ImportError, subprocess_error()):
        return None
    try:
        import gguf_reader as G
        f = G.GGUFFile(path)
        ts = sorted(f.tensors, key=lambda t: t.offset)
        end = os.path.getsize(path) - f.data_start
        return sum((ts[i + 1].offset if i + 1 < len(ts) else end) - t.offset
                   for i, t in enumerate(ts) if "_exps" in t.name) / GIB
    except (ImportError, OSError, ValueError, KeyError, IndexError):
        return None


def need_gib(ctx: int, kv: str = DEFAULT_KV, experts_gib: float = EXPERTS_GIB) -> float:
    """GiB of the card the engine takes at this context with every expert in VRAM."""
    return DENSE_GIB + experts_gib + ctx * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB


def plan(vram_gb: float, ctx: int, kv: str = DEFAULT_KV, hip: bool = False, experts_gib: float = EXPERTS_GIB) -> dict:
    """What this card holds at `ctx`: {"resident": every expert in VRAM, "need_gib", "layout": "resident" or
    "offload", "tier_gib": the experts' GiB the card keeps, "ram_gib": the engine's RAM, "kv_streamed": the cache of
    some full-attention visits in RAM, "kv_ram_gib"}.  Where the model and the cache do not fit together the experts
    leave the card first (the offload path, the cache in VRAM); the cache streams only where it alone does not fit
    beside the dense weights and a tier of OFF_TIER_MIN_GIB - on AMD as on NVIDIA (hip: an AMD card under
    AMD_RESERVE_UNDER_GB has AMD_SMALL_CREDIT_GIB more room, its layout's smaller reserve).  A card whose VRAM is not
    known (0): resident - the engine measures at its start."""
    ex = float(experts_gib)
    if not vram_gb:
        return {"resident": True, "need_gib": None, "layout": "resident", "tier_gib": ex, "ram_gib": HOST_GIB,
                "kv_streamed": False, "kv_ram_gib": 0.0}
    room = float(vram_gb) - MARGIN_GIB + (AMD_SMALL_CREDIT_GIB if hip and float(vram_gb) < AMD_RESERVE_UNDER_GB else 0.0)
    need = need_gib(ctx, kv, ex)
    if room >= need:
        return {"resident": True, "need_gib": need, "layout": "resident", "tier_gib": ex, "ram_gib": HOST_GIB,
                "kv_streamed": False, "kv_ram_gib": 0.0}
    kv_gib = ctx * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB
    vis = kv_gib / FULL_VISITS
    base = DENSE_GIB + ex + max(0, ctx - 262144) * SCRATCH_TOKEN_BYTES / GIB
    ring = vis * RING_OF_VISIT
    n_res = min(FULL_VISITS, int((room - base - ring) // vis)) if room >= base + ring + vis else -1
    dense = base - ex
    tier = room - dense - kv_gib - OFF_ROOM_GIB
    kv_ram = 0.0
    if tier < OFF_TIER_MIN_GIB:
        n_res = max(1, min(FULL_VISITS, int((room - dense - OFF_ROOM_GIB - OFF_TIER_MIN_GIB - ring) // vis)))
        kv_ram = (FULL_VISITS - n_res + 1) * vis
        tier = room - dense - n_res * vis - ring - OFF_ROOM_GIB
    tier = round(max(0.0, min(ex, tier)), 1)
    return {"resident": False, "need_gib": need, "layout": "offload", "tier_gib": tier,
            "ram_gib": round(ex - tier + OFF_HOST_GIB + (kv_ram + STREAM_HOST_GIB if kv_ram else 0.0), 1),
            "kv_streamed": kv_ram > 0, "kv_ram_gib": round(kv_ram, 1)}


def plan_two(vram0_gb: float, vram1_gb: float, ctx: int, kv: str = DEFAULT_KV, experts_gib: float = EXPERTS_GIB) -> dict:
    """Two NVIDIA cards (saguaro-ik2 --gpus 2, the larger first): the first alone where it holds every expert, else
    the split.  plan()'s keys plus "gpus" (1 or 2), "layout" "resident" / "split" (every expert on the two cards) /
    "split+ram", "gpu1_gib" (the second card's experts)."""
    p0 = plan(vram0_gb, ctx, kv, experts_gib=experts_gib)
    if p0["resident"]:
        return dict(p0, gpus=1, gpu1_gib=0.0)
    ex = float(experts_gib)
    g1 = round(max(0.0, min(ex - p0["tier_gib"], float(vram1_gb or 0) - MARGIN_GIB - TWO_ROOM_GIB)), 1)
    return dict(p0, gpus=2, gpu1_gib=g1, layout="split" if p0["tier_gib"] + g1 >= ex - 0.05 else "split+ram")


def too_small_line(ram_gb: float, vram_gb: float, need: float | None = None) -> str:
    need = plan(vram_gb, CONTEXTS[0])["ram_gib"] if need is None else need
    return (f"Inkling 2 Mini needs about {MIN_TOTAL_GB:.0f} GB of RAM and VRAM together to hold the model: about "
            f"{need:.0f} GB of RAM with this {vram_gb:.0f} GB card (this PC has {ram_gb:.0f} GB of RAM)")


def verdict(vram_gb: float, ram_gb: float, ctx: int, kv: str = DEFAULT_KV, hip: bool = False,
            experts_gib: float = EXPERTS_GIB) -> tuple:
    """(word, line) for the setup and the Models page: "fast" (every expert in VRAM), "streamed" (every expert in
    VRAM, part of the cache in RAM), "runs" (the offload path), "tight" (the offload path, with less RAM beside what it
    takes than the system needs), "does not fit".  ram_gb 0: not known."""
    ex = float(experts_gib)
    p = plan(vram_gb, ctx, kv, hip, ex)
    if ram_gb and ram_gb < p["ram_gib"] and p["kv_streamed"]:
        return "does not fit", (f"at this context the cache keeps about {p['kv_ram_gib']:.0f} GB in RAM (KV "
                                f"streaming): about {p['ram_gib']:.0f} GB of RAM in all, this PC has {ram_gb:.0f} GB - "
                                "a smaller context")
    if p["resident"] and p["kv_streamed"]:
        return "streamed", (f"every expert in VRAM; about {p['kv_ram_gib']:.1f} GB of the cache in RAM (KV streaming), "
                            "read over the bus at every token - slower the deeper a chat goes")
    if p["resident"]:
        return "fast", "every expert in VRAM"
    if ram_gb and ram_gb < p["ram_gib"]:
        return "does not fit", too_small_line(ram_gb, vram_gb, p["ram_gib"])
    share = 100.0 * p["tier_gib"] / ex
    line = (f"about {p['tier_gib']:.1f} of the experts' {ex:.1f} GB on the card ({share:.0f}%), the other "
            f"{ex - p['tier_gib']:.1f} GB pinned in RAM - slower than with every expert in VRAM")
    line += (f"; about {p['kv_ram_gib']:.0f} GB of the cache in RAM too (KV streaming)" if p["kv_streamed"]
             else "; the whole cache stays on the card")
    if ram_gb and ram_gb < p["ram_gib"] + OS_GIB:
        return "tight", line + f"; it takes about {p['ram_gib']:.0f} GB of RAM, this PC has {ram_gb:.0f} GB"
    return "runs", line


def verdict_two(vram0_gb: float, vram1_gb: float, ram_gb: float, ctx: int, kv: str = DEFAULT_KV,
                experts_gib: float = EXPERTS_GIB) -> tuple:
    """verdict() for two cards (plan_two)."""
    ex = float(experts_gib)
    p = plan_two(vram0_gb, vram1_gb, ctx, kv, ex)
    if p["gpus"] == 1:
        return verdict(vram0_gb, ram_gb, ctx, kv, experts_gib=ex)
    if ram_gb and ram_gb < p["ram_gib"]:
        return "does not fit", too_small_line(ram_gb, vram0_gb, p["ram_gib"])
    held = p["tier_gib"] + p["gpu1_gib"]
    if p["layout"] == "split":
        word, line = "fast", (f"every expert in VRAM across the two cards (about {p['tier_gib']:.1f} GB on the first, "
                              f"{p['gpu1_gib']:.1f} GB on the second)")
    else:
        word, line = "runs", (f"about {held:.1f} of the experts' {ex:.1f} GB on the two cards ({100.0 * held / ex:.0f}%"
                              f"), the other {ex - held:.1f} GB pinned in RAM - slower than with every expert in VRAM")
    if ram_gb and ram_gb < p["ram_gib"] + OS_GIB:
        return "tight", line + f"; it takes about {p['ram_gib']:.0f} GB of RAM, this PC has {ram_gb:.0f} GB"
    return word, line


# The drafter's room as saguaro-ik2's draft_room plans it: the file, its fold through the head, its windowed layers'
# rings, the window's records and buffers, the target's taps; plus its layer without a window's context, FP16 K and V
# 4 KiB a position (FP8: half).  The GGUF (Q8_0, 0.37 GiB, what the setup fetches): the engine planned 0.64 GiB at 8K
# (chunk 1024) and 0.77 at 32K (chunk 2048) on a T4 (REPORT-ik2-draft-gain.md runs t6/t7; it used 0.48 / 0.57) -
# 0.61 GiB beside the context rows at chunk 1024, 0.65 at 2048.  mtp.safetensors (BF16, 0.59 GiB: a fallback) 0.25 more.
DRAFT_BASE_GIB = 0.65                               # the GGUF drafter, a 2048-token chunk's taps
DRAFT_BF16_EXTRA_GIB = 0.25                         # mtp.safetensors instead (its BF16 matrices)
DRAFT_TOKEN_BYTES = 4096                            # its layer without a window's context, FP16 rows a token


def draft_gib(ctx: int, gguf: bool = True) -> float:
    """GiB the drafter takes beside the model at `ctx` (saguaro-ik2's draft_room, FP16 context rows): the GGUF, or
    mtp.safetensors (gguf=False)."""
    return DRAFT_BASE_GIB + (0.0 if gguf else DRAFT_BF16_EXTRA_GIB) + ctx * DRAFT_TOKEN_BYTES / GIB


PLAN_DENSE_GIB = 1.1                                # the engine's own sum for the drafter's room (saguaro-ik2's fit
PLAN_RESERVE_GIB = 1.5                              # block): the experts, the dense weights as blocks, the cache and
                                                    # its reserve - less than BASE_GIB's measured use
OFF_WIN_STAGE_GIB = 0.7                             # the offload path's verify window: 64 staging slots a slot size
DRAFT_MIN_ROWS = 65536                              # the fewest context rows the engine plans for the drafter there


def draft_fits(vram_gb: float, ctx: int, kv: str = DEFAULT_KV, experts_gib: float = EXPERTS_GIB,
               hip: bool = False, gguf: bool = True) -> bool:
    """Whether the drafter's room fits beside the model at `ctx` (else the engine leaves it out at its start; the
    prompt lookup still drafts).  Every expert in VRAM: the engine's own check - its plan's sum (the experts, the dense
    weights, the cache, its reserve) and the drafter (gguf: the GGUF the setup fetches, else mtp.safetensors) with
    FP16 context rows, else FP8 ones (a 24 GB card drafts at 262,144 as the engine does).
    The offload path with the whole cache in VRAM: the tier gives the drafter's room, at most half of its own - the
    verify window's staging and at least DRAFT_MIN_ROWS of FP8 context rows (the last positions) with it (upstream
    v1.1.75).  VRAM not known: True (the engine measures)."""
    if not vram_gb:
        return True
    p = plan(vram_gb, ctx, kv, hip, experts_gib)
    if p["resident"]:    # (the engine's own sum; the drafter's context rows FP16, else FP8: half of them)
        small = hip and float(vram_gb) < AMD_RESERVE_UNDER_GB   # (amd_layout's SAGUARO_IK2_RESERVE_MB=1024)
        used = float(experts_gib) + PLAN_DENSE_GIB + ctx * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB + \
            (1.0 if small else PLAN_RESERVE_GIB)
        return float(vram_gb) - MARGIN_GIB >= used + draft_gib(ctx, gguf) - ctx * DRAFT_TOKEN_BYTES / GIB / 2
    least = draft_gib(0, gguf) + OFF_WIN_STAGE_GIB + min(ctx, DRAFT_MIN_ROWS) * DRAFT_TOKEN_BYTES / GIB / 2
    return not p["kv_streamed"] and (p["tier_gib"] + OFF_ROOM_GIB + MARGIN_GIB) / 2 >= least


OFF_CACHE = ("drafting off: at this context part of the cache streams from RAM, where drafting does not run - a "
             "smaller context or a bigger card brings it back")
OFF_ROOM = ("drafting off: at this context the experts' tier has no room for the drafter - a smaller context or a "
            "bigger card brings it back")


def drafting_off_reason(arch, vram_gb: float, ctx: int, kv: str = DEFAULT_KV, hip: bool = False,
                        experts_gib: float = EXPERTS_GIB, gpus2: bool = False) -> str | None:
    """Why drafting is off on this card (one line), or None.  Every NVIDIA class from Volta drafts - Volta and Turing
    have their own verify-window kernels since upstream v1.1.75 (on a T4: Ken 11-23, measured +4..+9 %) - Pascal
    (6.x, scalar kernels) does not.  The offload path (the card does not hold every expert: a small card, or a long
    context) drafts where its whole cache stays in VRAM and the experts' tier has the drafter's room, on one NVIDIA
    card; not on AMD's (measured slower there - upstream's auto measures it, we keep it off).  Two GPUs (gpus2: the
    split): off - our engine leaves the drafter out there (not checked against plain decoding yet).  With every
    expert in VRAM it is kept.  A VRAM not known is not a reason."""
    try:
        cc = int(arch)
    except (TypeError, ValueError):
        cc = 0
    if not hip and 0 < cc < 70:
        return ("drafting off: on this GPU (Pascal) the engine runs its scalar kernels, where a verify window costs "
                "more than it keeps - every token comes from a full model step")
    if gpus2 and not hip:
        return ("drafting off: on two GPUs (the split) the engine does not draft yet - it is not checked against plain "
                "decoding there")
    if vram_gb and not plan(vram_gb, ctx, kv, hip, experts_gib)["resident"]:
        if plan(vram_gb, ctx, kv, hip, experts_gib)["kv_streamed"]:
            return OFF_CACHE
        if not draft_fits(vram_gb, ctx, kv, experts_gib, hip):
            return OFF_ROOM
        if hip:
            return ("drafting off: on an AMD card's offload path drafting is slower than plain decoding (an RX 9070 XT "
                    "measured: -1..-12 %) - a context or a file whose experts all fit brings it back")
    return None


def default_drafting(arch, hip: bool = False, resident: bool = False, gpus2: bool = False) -> str:
    """--drafting as the setup writes it where drafting runs (drafting_off_reason None): "on" where Ken's defaults
    (11-23, REPORT-ik2-draft-gain.md) say drafting pays whatever one start measures - a Tesla T4 / RTX 20 (Turing,
    one card: the expert cache or every expert resident) and an AMD GPU holding every expert; else "auto" (upstream's:
    measured once on the PC - Volta, Ampere and newer)."""
    try:
        cc = int(arch)
    except (TypeError, ValueError):
        cc = 0
    if hip:
        return "on" if resident else "auto"
    return "on" if cc == 75 and not gpus2 else "auto"


def recommended_context(vram_gb: float, kv: str = DEFAULT_KV, hip: bool = False,
                        experts_gib: float = EXPERTS_GIB) -> int:
    """The largest of CONTEXTS (up to RECOMMEND_MAX) whose cache leaves every expert in VRAM, else the smallest."""
    fits = [c for c in CONTEXTS if c <= RECOMMEND_MAX and plan(vram_gb, c, kv, hip, experts_gib)["resident"]
            and not plan(vram_gb, c, kv, hip, experts_gib)["kv_streamed"]]
    return max(fits) if fits else CONTEXTS[0]


def amd_unsupported(cards: list, experts_gib: float = EXPERTS_GIB, ram_gb: float = 0.0) -> str | None:
    """The AMD half of unsupported(): saguaro-ik2's HIP build (Linux) runs on RDNA 3 / 3.5 / 4 (gfx11 / gfx12) with
    12 GB or more - every expert in VRAM where the file fits, else the offload path as on NVIDIA (checked on an RX 9070
    XT, gfx1201; 12 GB cards since upstream v1.1.75).  An APU's memory is the RAM ("unified_memory": its "memory_gb").
    ram_gb: the PC's RAM (0: not read yet)."""
    if not cards:
        return None
    rdna = [g for g in cards if str(g.get("arch") or "").startswith(AMD_ARCHS)]
    if not rdna:
        return ("Inkling 2 Mini on AMD needs a Radeon RX 7000 / 9000, a Radeon PRO W7000 / AI PRO R9700 or a Ryzen AI "
                "Max (Strix / Gorgon Halo) GPU - RDNA 3, 3.5 or 4 (gfx11 / gfx12); older AMD GPUs are not supported "
                "by its engine")
    room = max(float(g.get("memory_gb") if g.get("unified_memory") else g.get("vram_gb") or 0) for g in rdna)
    if room < AMD_MIN_GB:
        return "Inkling 2 Mini on AMD needs one GPU with 12 GB of VRAM or more"
    if ram_gb and room - MARGIN_GIB < need_gib(CONTEXTS[0], experts_gib=experts_gib) and \
            ram_gb < plan(room, CONTEXTS[0], hip=True, experts_gib=experts_gib)["ram_gib"]:
        return too_small_line(ram_gb, room)            # (the experts a card does not hold need the RAM)
    return None


def unsupported(backend: str, windows: bool, gpus_asked: int, cards: list, ram_gb: float = 0.0,
                experts_gib: float = EXPERTS_GIB) -> str | None:
    """Why this PC cannot run Inkling 2 Mini, in one plain line, or None (said before anything is built or
    downloaded).  backend: cuda / hip / metal; gpus_asked: how many GPUs --gpu / --gpus named (0: none named); cards:
    the GPUs the model would pick from (NVIDIA: {"arch", "vram_gb"}; AMD: amd_unsupported's) - empty when not read
    yet; ram_gb: the PC's RAM (0: not read yet; the offload path pins what the card does not keep)."""
    if backend == "metal":
        return "Inkling 2 Mini runs on NVIDIA and AMD GPUs only for now (no Mac engine yet)"
    if windows:
        return "Inkling 2 Mini runs on Linux only for now (its engine is not built on Windows here yet)"
    if backend == "hip":
        if gpus_asked > 1:
            return "Inkling 2 Mini on AMD runs on one GPU for now: pick one (--gpu N)"
        return amd_unsupported(cards, experts_gib, ram_gb)
    if gpus_asked > 2:
        return ("Inkling 2 Mini runs on one or two NVIDIA GPUs for now (three or more are not in it yet): pick one or "
                "two (--gpu N, --gpus N,M)")
    if cards:
        new = [g for g in cards if int(g.get("arch") or 0) >= MIN_CC]
        if not new:
            return ("Inkling 2 Mini needs an NVIDIA Tesla P40 / GTX 10 (Pascal, second class: measured on a "
                    "Tesla P40) or newer GPU (compute "
                    "capability 6.0 or newer) - older cards are not supported by its engine")
        vram = max(float(g.get("vram_gb") or 0) for g in new)
        if vram < MIN_VRAM_GB:
            return "Inkling 2 Mini needs one NVIDIA GPU with 8 GB of VRAM or more"
        if gpus_asked == 2 and min(float(g.get("vram_gb") or 0) for g in cards) < MIN_VRAM_GB:
            return "Inkling 2 Mini on two GPUs needs 8 GB of VRAM or more on each of them (or one GPU: --gpu N)"
        if ram_gb and vram - MARGIN_GIB < need_gib(CONTEXTS[0], experts_gib=experts_gib) and \
                ram_gb < plan(vram, CONTEXTS[0], experts_gib=experts_gib)["ram_gib"]:
            return too_small_line(ram_gb, vram)
    return None


# ------------------------------------------------------------------ the setup's layouts (NVIDIA)
def usable(cards: list) -> list:
    """The NVIDIA cards saguaro-ik2 can use ({"arch", "vram_gb"}, nvidia-smi's), the largest first."""
    ok = [g for g in cards if int(g.get("arch") or 0) >= MIN_CC and float(g.get("vram_gb") or 0) >= MIN_VRAM_GB]
    return sorted(ok, key=lambda g: (-round(float(g["vram_gb"])), g.get("index", 0)))


def big_context(vram_gb: float, kv: str = DEFAULT_KV) -> int:
    """One card that holds every expert: the largest of CONTEXTS (up to RECOMMEND_MAX) whose cache leaves them all in
    VRAM, else the smallest."""
    return recommended_context(vram_gb, kv)


def pascal(g: dict) -> bool:
    """A compute capability 6.x card (Tesla P40 / P100, GTX 10)."""
    return 60 <= int(g.get("arch") or 0) < 70


def _t4_sized(g: dict) -> bool:
    return 14.5 <= float(g.get("vram_gb") or 0) < 22.0


def layout(cards: list, want: int = 0, kv: str = DEFAULT_KV, experts_gib: float = EXPERTS_GIB) -> dict | None:
    """The layout for these NVIDIA cards: {"gpus": the cards it runs on, "args": saguaro-ik2's flags beside --model
    and --max-context, "context", "env", "resident": every expert in VRAM, "path": "resident" / "offload" / "split" /
    "split+ram", "measured": on T4s here, "said": one plain line}.  want: 1 or 2 cards asked for (0: the best this PC
    has - one where the largest holds every expert, else two).  None: no card can run it (unsupported says why)."""
    have = usable(cards)
    if not have:
        return None
    v0 = float(have[0]["vram_gb"])
    p1 = plan(v0, CONTEXTS[0], kv, experts_gib=experts_gib)
    n = want if want in (1, 2) else (1 if p1["resident"] or len(have) < 2 else 2)
    n = min(n, len(have))
    if n == 2:
        v1 = float(have[1]["vram_gb"])
        fits = [c for c in CONTEXTS if c <= TWO_CONTEXT and plan_two(v0, v1, c, kv, experts_gib)["layout"] == "split"]
        ctx = max(fits) if fits else CONTEXTS[0]
        p = plan_two(v0, v1, ctx, kv, experts_gib)
        if p["gpus"] == 1:   # (the first card holds every expert there: the second is not used)
            n = 1
        else:
            t4s = all(_t4_sized(g) for g in have[:2])
            return {"gpus": have[:2], "args": ["--gpus", "2", "--chunk", "8192", "--threads", str(THREADS)],
                    "context": ctx, "env": {"SAGUARO_IK2_ROOM_MB": str(ROOM_MB)} if t4s else {},
                    "resident": p["layout"] == "split", "path": p["layout"],
                    "measured": t4s and p["layout"] == "split" and ctx == TWO_CONTEXT,
                    "said": ("two cards: every expert in VRAM across them" if p["layout"] == "split" else
                             "two cards: most experts in VRAM across them, the rest in RAM") +
                            f", a {round(ctx / 1000) if ctx >= 100000 else ctx // 1024}K context"}
    if p1["resident"]:
        ctx = big_context(v0, kv) if experts_gib == EXPERTS_GIB else recommended_context(v0, kv, False, experts_gib)
        # (Pascal: no --chunk - the engine's default there is auto, 8192 / 4096 where every expert stays in VRAM; a P40
        #  8K read 384 -> 458 tok/s, REPORT-ik2-p40-tuning.md)
        chunk = [] if pascal(have[0]) else ["--chunk", "2048"]
        return {"gpus": have[:1], "args": ["--gpus", "1"] + chunk + ["--threads", str(THREADS)],
                "context": ctx, "env": {}, "resident": True, "path": "resident", "measured": False,
                "said": "one card: every expert in VRAM"}
    t4 = _t4_sized(have[0])
    return {"gpus": have[:1], "args": ["--gpus", "1", "--chunk", "2048", "--threads", str(THREADS)],
            "context": CONTEXTS[0], "env": {"SAGUARO_IK2_ROOM_MB": str(ROOM_MB)} if t4 else {}, "resident": False,
            "path": "offload", "measured": t4,
            "said": f"one {'16 GB ' if t4 else ''}card: the most-used experts in VRAM, the others in RAM (the offload "
                    "path), a 32K context"}


def speeds(lay: dict) -> str:
    """What a layout measured on Tesla T4s or an RX 9070 XT, as one phrase ("" where it was not measured)."""
    if not lay.get("measured"):
        return ""
    if lay.get("amd"):
        dec, read = MEASURED_AMD[lay["path"]]
        return f"about {dec} tok/s writing, {read} tok/s reading a prompt (measured on an RX 9070 XT)"
    dec, read = MEASURED[len(lay["gpus"])]
    return f"about {dec} tok/s writing, {read} tok/s reading a prompt (measured on {len(lay['gpus'])}x Tesla T4)"


def chunk_max(vram_gb: float, ctx: int, kv: str = DEFAULT_KV, hip: bool = False,
              experts_gib: float = EXPERTS_GIB) -> int | None:
    """The largest --chunk this card runs at `ctx` without a long prompt running it out of GPU memory, or None (no
    cap: the offload path plans its tier on what the chunk leaves).  Every expert resident on an AMD card under
    AMD_RESERVE_UNDER_GB (1 GiB kept beside them): 1024 - on an RX 9070 XT the one-T4 quant at --chunk 2048 kept
    0.19 GiB free and a 10K prompt ran out of memory, at 1024 a 27K prompt read (2026-11-22)."""
    if hip and vram_gb and float(vram_gb) < AMD_RESERVE_UNDER_GB and plan(vram_gb, ctx, kv, True, experts_gib)["resident"]:
        return 1024
    return None


def amd_layout(card: dict, experts_gib: float = EXPERTS_GIB) -> dict:
    """One AMD GPU (HIP, Linux; card: its "vram_gb" = the memory it gives the engine): plan()'s - every expert in its
    memory where the file fits, else the offload path (a tier of them, the rest in pinned RAM), 32K and the F16 cache
    as measured on an RX 9070 XT; a card under AMD_RESERVE_UNDER_GB keeps 1 GiB beside the experts
    (SAGUARO_IK2_RESERVE_MB), which lets our one-T4 quant stay resident there with --chunk 1024.  The engine plans on
    the card's real free memory either way."""
    vram = float(card.get("vram_gb") or 0)
    ex = float(experts_gib)
    p = plan(vram, CONTEXTS[0], DEFAULT_KV, True, ex)
    small = bool(vram) and vram < AMD_RESERVE_UNDER_GB
    env = {"SAGUARO_IK2_RESERVE_MB": "1024"} if small else {}
    res = p["resident"]
    xt = str(card.get("arch") or "").startswith("gfx1201") and 15.0 <= vram < 17.0   # (the RX 9070 XT measured)
    ctx = recommended_context(vram, DEFAULT_KV, True, ex) if res else CONTEXTS[0]
    return {"gpus": [card], "args": ["--gpus", "1", "--chunk", "1024" if res else "2048", "--threads", str(THREADS)],
            "context": ctx, "chunk_max": chunk_max(vram, ctx, DEFAULT_KV, True, ex),
            "env": env, "resident": res, "path": "resident" if res else "offload", "amd": True,
            "measured": xt and (abs(ex - ONE_T4_EXPERTS_GIB) < 0.1 if res else abs(ex - EXPERTS_GIB) < 0.1),
            "said": "one AMD GPU: every expert in its memory" if res else
                    "one AMD GPU: the most-used experts in VRAM, the others in RAM (the offload path), a 32K context"}
