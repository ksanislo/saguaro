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
KV_BYTES = 12672                                      # K/V bytes a context token, --kv int8 (measured: the engine's
#                                                     #  own INFO kv_bytes_token= line - qsa_kv_bytes_per_cell x its
#                                                     #  12 QSA layers, engines/qfn/src/program/generate.cpp:12401)
LAYERS = 48                                           # the model's transformer layers (the layer split's reach)
KV_RESIDENT = 32768                                   # the newest cells of each QSA layer kept in VRAM (the configs)
RAM_SPARE_GB = 20                                     # RAM left beside the tier: the OS, the server, the page cache
MIN_VRAM_GB = 15.0                                    # the COMFORTABLE card (the manifest's "needs": a 16 GB card);
MIN_RAM_GB = 88.0                                     # ... and RAM: the layout as measured, not the floor (v1.2.1-B:
#                                                     #  below them the plan is "slow", refused only under its minimum;
#                                                     #  v1.2.1-D: it is also the expert-size line of the waiver in
#                                                     #  comfortable(): a quant with fewer experts than this is not
#                                                     #  called slow by thin RAM alone)
MIN_CC = 61                                           # NVIDIA, v1.2.1-B: Pascal and newer (a Tesla P40 ran QFN by
#                                                     #  hand: sm_61, 24 GB, 32 GB RAM, UD-Q4 ~4-5 tok/s) - sm_6x /
#                                                     #  sm_70 build through engines/qfn's experimental pre-Turing
#                                                     #  path (STRATA_EXPERIMENTAL_SM60), which needs a CUDA 12.x
#                                                     #  toolkit: CUDA 13 dropped Pascal (saguaro.py's nvcc_range)
# v1.2.1-B: the TRUE minimum, measured on one card (mortar Monitor, UD-Q4_K_XL): dense weights 3.9 GB (a layer
# split shares them by the split), the resident K/V window of --kv-resident N (its own bytes: the engine's INFO
# kv_bytes_token above - 0.39 GiB at 32768 cells, 0.05 GiB at the minimum's 4096, kv_resident_gib; v1.2.1-B
# review finding 2 settled this one constant against generate.cpp:12401 - the plan's "2.8 GB" Monitor line
# counted more than the resident window), activations 1.4 GB, and the reserve the setup keeps free beside the
# expert cache (--vram-reserve-mib 700).
# The MTP drafter (0.92 GB there) is NOT in the minimum: below the comfortable layout the setup scales down and
# turns drafting off.
DENSE_GIB = 3.9                                       # the whole model's dense weights, UD-Q4_K_XL (measured)
ACTIVATIONS_GIB = 1.4                                 # the engine's activation buffers (measured there)
RESERVE_GIB = 700 / 1024                              # kept free beside the expert cache (--vram-reserve-mib)
RAM_BUFFER_GIB = 4.0                                  # a 32K context's K/V pinned in RAM (0.4 GiB at --kv int8)
#                                                     #  and read staging
RAM_SYSTEM_GIB = 8.0                                   # the OS, the server and a working page cache
KV_RESIDENT_MIN = 4096                                 # the smallest resident K/V window a layout is laid out with
SCALED_CONTEXT = 32768                                 # the context a scaled-down layout gets (QFN_CONTEXTS' smallest)
MAX_GROUP = 4                                         # a layer split runs on up to four cards (the live configs)
DEFAULT_CONTEXT = 65536                               # one card (measured there: prompts up to 60K tokens)
SPLIT_CONTEXT = 262144                                # a split: the K/V's RAM is cheap, the cards do not notice
# (both must stay among the contexts saguaro.py offers - QFN_CONTEXTS; test_setup_qfn guards it)
# C18 - the YaRN extension past the trained context: the engine reads plain RoPE to YARN_ORIG_CTX and the YaRN
# table past it (SAGUARO_YARN_ELASTIC=1, elastic: short chats never feel it), with --rope-scale reaching
# CONTEXT_MAX = orig x factor (the live mangonel box runs 1M this way: ./ref, and CONTEXT_TESTED qfn).  The
# engine's --rope-scale parses a double (src/program/generate.cpp: std::atof into "double rope_scale", its
# check only demands >= 1), so a fractional factor is written as computed and never rounded up: 384K is 1.5.
YARN_ORIG_CTX = 262144                                # what the model was trained for: plain RoPE up to here
CONTEXT_MAX = 1048576                                 # ... and the elastic YaRN reach at factor 4 (1M)
CONTEXT_STEP = 65536                                  # past 262144 a --context moves in these steps
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
# U6 (the live T4 configs' tuning the setup left out), classified per key in ./scratch/u6-keys.txt with the engine
# line each one is read at.  GENERIC: a feature that helps any GPU, NVIDIA and AMD alike - the conversation park
# without the zero fill (conversation_copy.hpp:153, on by default, live records it), the elastic batch slots the
# expert cache gives room to and takes back (generate.cpp:213-253) - with _GIVE the engine says "vision_lend=1"
# and lends CUDA0's VRAM to the picture encoder (generate.cpp:13121-13164): this is what pictures need; the slot
# checkpoint grid and the message-boundary checkpoint (generate.cpp:298, 20516).  Nothing of it does harm without
# --batch slots (the slot keys just do nothing).
GENERIC_ENV = {"SAGUARO_PARK_FAST": "1", "SAGUARO_ELASTIC_SLOTS": "1", "SAGUARO_ELASTIC_SLOTS_GIVE": "1",
               "SAGUARO_ELASTIC_IDLE": "1", "SAGUARO_ELASTIC_IDLE_S": "300", "SAGUARO_SLOT_GRID": "6",
               "SAGUARO_CACHE_MESSAGE_BOUNDARY": "1"}
GENERIC_ARGS = ["--conversation-cache-mib", "auto", "--conversation-cache-slots", "auto",
                "--conversation-cache-min-tokens", "1024"]      # (parked conversations: one RAM pool, 64 of them,
                                                                #  only from 1024 tokens - generate.cpp:983-1012)
# T4-ONLY: sm_75 tuning, only where every card is compute capability 7.5 (the T4_ENV gate): the one-block argmax
# rows the engine takes on sm_7x only with this opt-in (verify_kernels.cu:990), and the GDN prompt recurrence per
# key head, sm_80+ by default and sm_75 with =1 (prefill/kernels.cu:849).
T4_ONLY_ENV = {"SAGUARO_ARGMAX_ROWS_SM70": "1", "SAGUARO_GDN_KEYHEAD": "1"}
# PASCAL: every card compute capability 6.x (Tesla P40 / P100, GTX 10): the sm_6x Q8 rows for the model and the
# MTP drafter (proven on a P40: IQ3_S + MTP ~20 tok/s); DENSE_F32 only on IQ3_S, where it was measured exact there.
PASCAL_ENV = {"SAGUARO_Q8_SM60": "1", "SAGUARO_MTP_Q8_SM60": "1"}
PASCAL_ENV_QUANT = {"gsq-iq3_s": {"SAGUARO_DENSE_F32": "1"}}
# MULTI-GPU: the layer split's pipeline (stage pools, NUMA loads, P2P boundaries, reads beside the batch slots and
# their shared chunks, pipelined drafts, direct slot copies, the split's CPU read share) - what the live split runs
# and none of it fires without stages (batch_read.hpp:97: SLOT_ADMIT "not on the pipelined slot groups ... on a
# layer split"; piped_mtp needs !stages.empty(), generate.cpp:5977; SHORT_READS only where multi_gpu,
# generate.cpp:9148).  Values are the live configs' own; BATCH_READS_OVERTAKE and PARK_STAGING_MIB are read
# nowhere and stay out.
SPLIT_ENV = {"SAGUARO_STAGE_SERVE": "1", "SAGUARO_PARALLEL_LOAD": "1", "SAGUARO_P2P_HANDOFF": "1",
             "SAGUARO_SLOT_ADMIT": "1", "SAGUARO_SLOT_ADMIT_CHUNK": "4096", "SAGUARO_SLOT_ADMIT_MULTI": "1",
             "SAGUARO_READ_YIELD": "1", "SAGUARO_READ_YIELD_CHUNK": "4096", "SAGUARO_BATCH_READS": "1",
             "SAGUARO_BATCH_READS_MAX": "8", "SAGUARO_BATCH_READS_FAST": "512",
             "SAGUARO_BATCH_READS_FAST_SHRINK": "512", "SAGUARO_BATCH_READS_DECODE_WINDOWS": "0",
             "SAGUARO_BATCH_READS_DECODE_SHARE": "0.3", "SAGUARO_FAST_LANE": "1", "SAGUARO_FAST_LANE_MAX": "96",
             "SAGUARO_PIPED_MTP": "1", "SAGUARO_PIPED_MTP_DRAFTS": "2", "SAGUARO_PIPED_MTP_MAX": "4",
             "SAGUARO_SLOT_COPY_DIRECT": "1", "SAGUARO_SHORT_READS": "1"}
# the per-quant differences between the two live refs, kept per quant: the Q8_0 box never ran
# SAGUARO_PARALLEL_LOAD_REGISTER and its server (not the engine: serve/qfn_engine.py:93) collects the wrap spill
# the UD-Q4 box does not; live-ud-q4 caches prompts at 64 checkpoints, live-q8 runs the default.
SPLIT_ENV_QUANT = {"ud-q4_k_xl": {"SAGUARO_PARALLEL_LOAD_REGISTER": "overlap"},
                   "q8_0": {"SAGUARO_BATCH_WRAP_SPILL": "1"}}
QUANT_ARGS = {"ud-q4_k_xl": ["--prompt-cache", "64"]}
# TOPOLOGY, computed and only on a multi-socket CPU: the pool's workers default to the host socket's cores
# (pool.cpp:143,181), so a split spanning sockets names them across both - every physical core bar the one the
# host spins on (2 x 10-core E5-2660 v2 -> 19, ./ref/live-ud-q4.json) - and the NUMA keys ride with it
# (numa.cpp:100, pool.cpp:762; one node: nothing to place and no pool to split).
TOPOLOGY_ENV = {"SAGUARO_NUMA_PLACE": "1", "SAGUARO_NUMA_POOL": "0"}
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


def manifest_quants() -> list:
    """Every quant the manifest names (the CLI lays out more than the two measured splits)."""
    try:
        with open(MANIFEST, encoding="utf-8") as fh:
            return sorted((json.load(fh).get("quants") or {})) or sorted(MEASURED_SPLITS)
    except (OSError, ValueError):
        return sorted(MEASURED_SPLITS)


def resident_budget_gib(ram_gb: float, quant: str | None = None) -> int:
    """--resident-budget-gib for a PC with `ram_gb` GiB of RAM: all but RAM_SPARE_GB (measured on 91.7 GiB: 71
    answered 32% faster than 64 with 10.8 GiB still free; the engine takes no more than the experts), and never
    more than the model's own expert bytes (its files, from the manifest)."""
    tier = max(8, int(ram_gb - RAM_SPARE_GB))
    experts = quant_gib(quant) if quant else 0.0
    return min(tier, int(experts)) if experts > 0 else tier


def page_cache_tier(ram_gb: float, quant: str) -> bool:
    """True when the OS page cache can hold every expert with RAM_SPARE_GB still untouched - then no pinned tier is
    written and the file cache IS the tier, which is what the live 252 GB box runs (./ref/live-*.json carry no
    --resident-budget-gib); the engine reads the experts through the cache and needs no budget to say so."""
    experts = quant_gib(quant)
    return bool(experts) and bool(ram_gb) and experts + RAM_SPARE_GB <= ram_gb


def kv_vram_gib(kv_resident: int = KV_RESIDENT) -> float:
    """GiB the resident K/V window of the whole model takes: --kv-resident cells x QSA layers x 1,056 B."""
    return kv_resident * KV_BYTES / GIB


def dense_gib(quant: str) -> float:
    """GiB of the model's dense weights at `quant`: the measured 3.9 GiB at UD-Q4_K_XL, scaled by the files' size
    (Q8_0's are ~1.7x).  3.9 for a quant or a manifest that does not read."""
    base, size = quant_gib("ud-q4_k_xl"), quant_gib(quant)
    return DENSE_GIB * size / base if base and size else DENSE_GIB


def kv_resident_gib(cells: int) -> float:
    """GiB of VRAM --kv-resident N costs: the same resident K/V window kv_vram_gib lays out - one constant
    serves both (v1.2.1-B review finding 2: the engine's own INFO kv_bytes_token= line, not a separate
    measured figure - qsa_kv_bytes_per_cell x the 12 QSA layers, 12,672 B a cell at --kv int8)."""
    return kv_vram_gib(cells)


def yarn_scale(n: int) -> float | None:
    """The YaRN --rope-scale a context `n` past the trained YARN_ORIG_CTX needs: n / 262144 (384K: 1.5,
    512K: 2, 1M: 4 - the 64K steps fall in quarters, and the engine's parser takes a double, so nothing is
    rounded). None at or below the trained context: the config keeps its factor, the elastic engine never
    reads the YaRN table inside 262144."""
    return n / YARN_ORIG_CTX if n > YARN_ORIG_CTX else None


def yarn_scale_text(n: int) -> str:
    """The --rope-scale value written for a context of `n`: past 262144 the computed factor as it is
    ("2", "1.5", "4"); at or below it the 4 every config written so far carries (with SAGUARO_YARN_ELASTIC
    it never fires below the trained context - the 256K configs stay byte-identical)."""
    scale = yarn_scale(n)
    return "4" if scale is None else "%g" % scale


def kv_pinned_gib(n: int, par: int = 1) -> float:
    """GiB of pinned RAM the K/V of `par` conversations of `n` tokens each costs at --kv int8: the engine's
    own KV_BYTES (~12.7 KB a token, measured) times the context times the answers at once - ~13 GB of it a
    conversation at 1M. The newest --kv-resident cells of each conversation sit on the cards too, but the
    whole cache is pinned RAM either way."""
    return n * KV_BYTES * max(1, int(par)) / GIB


def stage_bounds(cards: list, quant: str = "ud-q4_k_xl") -> list:
    """The layer split's stage bounds as layer numbers: [0, the first layer of each later card, ..., 48];
    one card: [0, 48]."""
    split = layer_split(cards, quant)
    return [0] + ([int(b) for b in split.split(",")] if split else []) + [LAYERS]


def card_min_gib(cards: list, quant: str = "ud-q4_k_xl") -> list:
    """The TRUE minimum VRAM of every card of this layout (GiB): its share of the dense weights and of the
    minimum resident K/V - by the layer split it would serve - plus the activations and the reserve.  Below it
    the engine cannot load the card at all; above it, everything not needed for these is expert cache."""
    b, dense, kv = stage_bounds(cards, quant), dense_gib(quant), kv_resident_gib(KV_RESIDENT_MIN)
    return [(b[i + 1] - b[i]) / LAYERS * (dense + kv) + ACTIVATIONS_GIB + RESERVE_GIB for i in range(len(cards))]


def ram_min_gib(quant: str = "ud-q4_k_xl") -> float:
    """The TRUE minimum RAM (GiB): the dense weights, the engine's buffers (a 32K context's pinned K/V among
    them) and the system.  The experts stream - VRAM, the RAM tier, the SSD - so RAM below the comfortable
    MIN_RAM_GB costs speed, not the run."""
    return dense_gib(quant) + RAM_BUFFER_GIB + RAM_SYSTEM_GIB


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


def pascal_only(cards: list) -> bool:
    """True where every card is compute capability 6.x: the gate for PASCAL_ENV."""
    return bool(cards) and all(60 <= _cc(g) < 70 for g in cards)


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


def unsupported(cards: list, ram_gb: float, quant: str | None = None) -> str | None:
    """Why these cards ({"name", "vram_gb"}) and this RAM cannot run Qwen3.8-Flash-Next on saguaro-qfn AT ALL,
    in one plain line carrying the numbers, or None (v1.2.1-B: refuse only what cannot run).  A card whose VRAM
    is not known (0) is not a refusal (the engine measures at its start); ram_gb 0: not read yet.  The refusal
    is the true minimum (card_min_gib / ram_min_gib): between it and the comfortable layout (every card
    MIN_VRAM_GB and MIN_RAM_GB of RAM - the layout as measured) the model RUNS and streams most of its experts
    from the RAM tier and the SSD, the same for a quant whose experts are more than these cards' VRAM and this
    PC's RAM together hold - streaming experts from RAM and the SSD is the engine's point (mortar runs Q8_0 on
    16 GB + 96 GB at 14-16 tok/s, SSD-bound; QFN ran on a Tesla P40 with 32 GB of RAM)."""
    q = quant or "ud-q4_k_xl"
    mins, b = card_min_gib(cards, q), stage_bounds(cards, q)
    for i, (g, m) in enumerate(zip(cards, mins)):
        v = float(g.get("vram_gb") or 0)
        if 0 < v < m:
            share = (b[i + 1] - b[i]) / LAYERS
            return (f"Qwen3.8-Flash-Next on saguaro-qfn cannot run on this PC's {g.get('name') or 'GPU'}: its "
                    f"{v:.0f} GB cannot hold the {m:.1f} GB this card of the layout must ({share * 100:.0f}% of "
                    f"the layers): its share of the {dense_gib(q):.1f} GiB of dense weights and of the "
                    f"{kv_resident_gib(KV_RESIDENT_MIN):.2f} GiB minimum resident K/V, {ACTIVATIONS_GIB} GB of "
                    "activations and " + f"{RESERVE_GIB:.1f} GB kept free beside the expert cache")
    if ram_gb and ram_gb < ram_min_gib(q):
        return (f"Qwen3.8-Flash-Next on saguaro-qfn cannot run with {ram_gb:.0f} GB of RAM: it needs at least "
                f"about {ram_min_gib(q):.0f} GB - the dense weights ({dense_gib(q):.1f} GiB), the engine's "
                f"buffers ({RAM_BUFFER_GIB:.0f} GB) and the system ({RAM_SYSTEM_GIB:.0f} GB); the experts stream "
                "from the cards' VRAM, the RAM tier and the SSD")
    return None


def comfortable(cards: list, ram_gb: float, quant: str | None = None) -> bool:
    """The layout as measured (the rule that refused before v1.2.1-B, now the COMFORTABLE level): every card of
    MIN_VRAM_GB and MIN_RAM_GB of RAM or more (a fact of 0: not read yet, no complaint).  Between this and the
    true minimum the plan is a slow one: the setup scales it down, warns loudly and confirms - it does not
    refuse.  v1.2.1-D: the RAM side is quant-aware - the scale-down (drafting off, a 32K context, --kv-resident
    4096) was tuned on the quants that stream most of their experts, those with MORE experts than the measured
    PC's whole RAM (UD-Q4_K_XL's 111 GiB, Q8_0's 175, CYBER-FROST's 105); a smaller quant was measured usable
    on a quarter of that RAM: a 24 GB Tesla P40 with 31 GB of RAM answered GSQ IQ3_S (77.9 GiB of experts) at
    ~15-16 tok/s over a 262K context WITH drafting (ref/REPORT-p40-batch.md item 3) - the pace of the measured
    Q8_0 box.  So on a PC below MIN_RAM_GB only a quant whose experts are more than MIN_RAM_GB is slow for the
    RAM; a smaller one gets the layout as laid out (the card side is unchanged).  No quant, or a manifest that
    does not read: no waiver, the rule as before."""
    return (all(not float(g.get("vram_gb") or 0) or float(g["vram_gb"]) >= MIN_VRAM_GB for g in cards)
            and (not ram_gb or ram_gb >= MIN_RAM_GB
                 or bool(quant) and 0.0 < quant_gib(quant) < MIN_RAM_GB))


def slow_why(cards: list, ram_gb: float, quant: str) -> str:
    """Why a layout between the minimum and the comfortable one decodes slowly, in one plain line: the expected
    cause the setup warns with ("most experts read from RAM / the SSD").  A quant whose experts are more than
    these cards' VRAM and this PC's RAM together hold is named with its numbers here - warn, not refuse."""
    parts = []
    small = [float(g["vram_gb"]) for g in cards if 0 < float(g.get("vram_gb") or 0) < MIN_VRAM_GB]
    if small:
        parts.append(f"its {min(small):.0f} GB card{'s hold' if len(small) > 1 else ' holds'} few of the "
                     "model's layers' experts in VRAM (the layout was measured on 16 GB ones)")
    if ram_gb and ram_gb < MIN_RAM_GB:
        parts.append(f"{ram_gb:.0f} GB of RAM pins only a thin expert tier (the layout was measured on about 96)")
    experts, room = quant_gib(quant), ram_gb + sum(float(g.get("vram_gb") or 0) for g in cards)
    if experts and room and experts > room:
        parts.append(f"{quant}: its {experts:.0f} GiB of experts are {experts - room:.0f} GiB more than these "
                     f"cards' VRAM and this PC's RAM together hold (~{room:.0f} GiB)")
    return ", ".join(parts)


def layer_split(cards: list, quant: str = "ud-q4_k_xl") -> str | None:
    """The config's "layer_split" for these cards (the FIRST LAYER of each later card; None for one card): the live
    4x Tesla T4 configs' own values on four "16 GB" cards of compute capability 7.5, else the 48 layers shared out
    in proportion to each card's VRAM (proportional_split)."""
    n = len(cards)
    if n <= 1:
        return None
    if n == 4 and quant in MEASURED_SPLITS and all(t4_sized(g) for g in cards) and sm75_only(cards):
        return MEASURED_SPLITS[quant]
    return proportional_split(cards)


def proportional_split(cards: list) -> str:
    """The 48 layers across these cards in proportion to their VRAM: a stage caches its own layers' experts in the
    card's VRAM, so a card with less VRAM takes fewer layers - by count alone the smaller of two unequal cards
    would get the same layers as the bigger one and more weight than it can hold.  Equal cards get the even split
    ("24" for two, "16,32" for three, "12,24,36" for four); a card whose VRAM reads 0 counts like the others
    (nothing to weigh).  Every card gets at least one layer and the bounds strictly rise."""
    n = len(cards)
    vram = [float(g.get("vram_gb") or 0) for g in cards]
    if any(v <= 0 for v in vram):
        vram = [1.0] * n
    total = sum(vram) or float(n)
    out, cum = [], 0.0
    for i, v in enumerate(vram[:-1]):
        cum += v
        b = round(LAYERS * cum / total)
        b = min(max(b, (out[-1] if out else 0) + 1), LAYERS - (n - 1 - i))
        out.append(b)
    return ",".join(str(b) for b in out)


def _layout_runs(cards: list) -> bool:
    """True when every card of this proposed layout clears its own minimum (a VRAM of 0: not read, no
    complaint) - what the equal-group fallback of best_gpus weighs group against single card."""
    return all(not float(g.get("vram_gb") or 0) or float(g["vram_gb"]) >= m
               for g, m in zip(cards, card_min_gib(cards)))


def best_gpus(cards: list) -> list:
    """The cards the setup takes without being told: one big card alone (it caches its experts without a split
    over the bus), else up to MAX_GROUP equal cards for a split - more cards, more experts in VRAM - else simply
    the largest one (one 16 GB card runs it).  [] for no cards.  v1.2.1-B: no size filter - a card below the
    comfortable 16 GB is the layout's "slow", not a refusal (unsupported() refuses only under the true
    minimum).  v1.2.1-B review finding 1: equal_group ranks by count, not size, so beside one comfortable card
    the "equal group" can be two tiny cards that cannot hold their own share of the split - when the group
    stands below its own card_min_gib and the largest card alone stands above its single-card minimum, the
    single card is taken; a pair of tiny cards with nothing better keeps the split, which is what gets refused."""
    have = sorted(cards, key=lambda g: (-round(float(g.get("vram_gb") or 0)), g.get("index", 0)))
    if not have:
        return []
    if float(have[0].get("vram_gb") or 0) >= 20.0 or len(have) == 1:
        return have[:1]
    group = equal_group(have)
    if group and not _layout_runs(group) and _layout_runs(have[:1]):
        return have[:1]
    return group or have[:1]


def layout_env(split: bool, t4: bool, quant: str, cpu: dict | None, pascal: bool = False) -> dict:
    """The live configs' tuning env for this layout, by class (./scratch/u6-keys.txt): GENERIC on every card;
    T4-ONLY only where every card is sm_75; MULTI-GPU only on a layer split (its per-quant additions among them);
    TOPOLOGY only on a multi-socket CPU."""
    env = dict(GENERIC_ENV)
    if split:
        env.update(SPLIT_ENV)
        env.update(SPLIT_ENV_QUANT.get(quant, {}))
    if t4:
        env.update(t4_env(quant))
        env.update(T4_ONLY_ENV)
    if pascal:
        env.update(PASCAL_ENV)
        env.update(PASCAL_ENV_QUANT.get(quant, {}))
    if multi_socket(cpu):
        env.update(TOPOLOGY_ENV)
    return env


def multi_socket(cpu: dict | None) -> bool:
    """True when the CPU's facts (qfn_fit's caller reads them: {"sockets", "cores"}) say more than one package:
    the NUMA keys and the across-sockets pool workers exist only then; None / unread: no."""
    return bool(cpu) and int(cpu.get("sockets") or 0) >= 2


def pool_workers(cpu: dict | None) -> int | None:
    """--pool-workers for this CPU: every physical core across the sockets minus the one the host loop spins on
    (the live box: 2 x 10 cores -> 19); None where the CPU is one socket or was not read."""
    if not multi_socket(cpu):
        return None
    return max(2, int(cpu.get("cores") or 0) - 1)


def plan(cards: list, ram_gb: float, quant: str = "ud-q4_k_xl", cpu: dict | None = None) -> dict:
    """The layout for these cards: {"fits", "reason", "gpus", "n_gpus", "layer_split" (None for one card),
    "context" (to recommend), "kv_resident", "resident_budget_gib", "args" (the engine's flags beside the generic
    ones: the conversation cache every PC gets, the split's --trim-stage-weights and --prefill-pipe, the quant's
    --prompt-cache, --pool-workers on a multi-socket CPU), "env" (the live configs' tuning by class - see
    layout_env), "t4", "measured", "said" (one plain line) and v1.2.1-B's slow-zone keys: "slow" (fits, but
    between the true minimum and the comfortable layout - the context and the K/V window come scaled down),
    "slow_why" (the expected cause in one line: most experts read from RAM / the SSD) and "turned_off" (what the
    setup scales down: drafting, --kv-resident, the context - and says so).  The cards it lists are the ones
    handed in -
    best_gpus() picks those, plan() lays them out; nothing is read from the hardware (the CPU's socket/core counts
    come in as `cpu` - saguaro.py's qfn_cpu_info reads them from /sys)."""
    why = ("Qwen3.8-Flash-Next on saguaro-qfn runs on one GPU, or up to " + str(MAX_GROUP) +
           " equal ones splitting the layers: no cards were named") if not cards else unsupported(cards, ram_gb, quant)
    # v1.2.1-B: between the minimum (unsupported) and the comfortable layout (comfortable) the setup scales the
    # layout down - drafting off, a smaller --kv-resident, a context that fits - says what it turned off and asks
    # for a loud confirmation; it does not refuse what can run.  v1.2.1-D: comfortable() now weighs in the quant,
    # so the same thin-RAM PC reads "slow" for a big streaming quant and "fits as laid out" for a small one (the
    # P40 box's IQ3_S measured ~15-16 tok/s at 262K with drafting - it is not the scale-down's slow case).
    slow = why is None and not comfortable(cards, ram_gb, quant)
    slow_ = slow_why(cards, ram_gb, quant) if slow else None
    n = len(cards)
    split = layer_split(cards, quant) if why is None else None
    full_ctx = DEFAULT_CONTEXT if n <= 1 else SPLIT_CONTEXT
    ctx = SCALED_CONTEXT if slow else full_ctx
    t4 = sm75_only(cards)
    # --resident-budget-gib: a pinned tier of RAM - 20 (never above the model's expert bytes) only while the page
    # cache cannot hold the experts; when it can, no tier: the file cache is it, as the live 252 GB box runs it.
    budget = resident_budget_gib(ram_gb, quant) if ram_gb and not page_cache_tier(ram_gb, quant) else None
    if slow and budget is not None:                # (v1.2.1-B review finding 3: the tier's 8 GiB floor sits beside
        room = int(ram_gb - ram_min_gib(quant))    #  the minimum's own budget on a thin-RAM PC - a 16 GB one clears
        budget = min(budget, room) or None         #  the 15.9 GB minimum but could not pin 8 GiB and run too.  On a
                                                   #  slow layout the tier shrinks to what the RAM minimum leaves,
                                                   #  and where it leaves nothing there is no tier at all: the
                                                   #  experts stream through the page cache there)
    # the live 4x Tesla T4 configs' own splits: four cards, all sm_75, the quant one of them ran - the pipe rate
    # belongs to THAT layout (a computed split that happened to hit the same bounds on other cards gets "1")
    live = bool(split) and n == 4 and t4 and split in set(MEASURED_SPLITS.values())
    args = ([] if n <= 1 else
            ["--trim-stage-weights", "--prefill-pipe", MEASURED_PREFILL_PIPE.get(quant, "1") if live else "1"])
    if why is None:
        args += GENERIC_ARGS
        if t4:                                         # (the per-quant args come from the live T4 boxes too)
            args += QUANT_ARGS.get(quant, [])
        workers = pool_workers(cpu)
        if workers:
            args += ["--pool-workers", str(workers)]
    amd = str((cards[0] if cards else {}).get("arch") or "").startswith("gfx")
    measured = why is None and (live or (n == 1 and amd and t4_sized(cards[0]) and ram_gb >= 90.0))
    tier_said = ("the next ones read through the OS file cache, the experts all fitting in RAM"
                 if page_cache_tier(ram_gb, quant) else
                 f"the next ones in a {budget} GiB RAM tier, the rest from the SSD" if budget is not None else
                 "the next ones read through the OS file cache - this PC's RAM leaves no room for a pinned tier"
                 if ram_gb else "the next ones in a RAM - 20 GiB RAM tier, the rest from the SSD")
    said = ("no cards named" if not cards else
            f"one card: the experts it holds in VRAM, {tier_said}, a {ctx // 1024}K context" if n <= 1 else
            f"{n} cards: the {LAYERS} layers split across them ({split}), each caching its own layers' experts, "
            f"a {ctx // 1024}K context")
    said += " - below the measured layout: expect slow decode" if slow else ""
    turned = (["drafting off",                                # (v1.2.1-B review finding 5: no (--mtp-max-t 1)
               f"--kv-resident {min(ctx, KV_RESIDENT_MIN)} instead of {min(full_ctx, KV_RESIDENT)}",   #  here - the
               f"a {ctx // 1024}K context instead of {full_ctx // 1024}K"] if slow else [])            #  flag is
                                                          #  only written where a drafter exists, and only
                                                          #  write_config_qfn knows that; a layout with no
                                                          #  drafter drafts off either way)
    return {"fits": why is None, "reason": why, "gpus": list(cards), "n_gpus": n, "layer_split": split,
            "context": ctx, "kv_resident": min(ctx, KV_RESIDENT_MIN if slow else KV_RESIDENT),
            "resident_budget_gib": budget,
            "args": args, "env": layout_env(bool(split), t4 and why is None, quant, cpu, pascal_only(cards)) if why is None else {},
            "t4": t4, "measured": measured, "said": said,
            "slow": slow, "slow_why": slow_, "turned_off": turned}


def main(argv=None) -> int:
    """The same from a shell, for checking a layout without a PC: cards as "VRAM[:CC]" GiB figures.
    tools/qfn_fit.py --ram 220 --quant ud-q4_k_xl 16 16:75 16 16"""
    import argparse
    ap = argparse.ArgumentParser(description="Qwen3.8-Flash-Next on saguaro-qfn: the fit and layout for these cards")
    ap.add_argument("cards", nargs="*", help="one card per GPU: VRAM GiB, optionally :compute-capability (16:75)")
    ap.add_argument("--ram", type=float, default=0.0, help="the PC's RAM in GiB (0: not known)")
    ap.add_argument("--quant", default="ud-q4_k_xl", choices=manifest_quants(), help="the quant to lay out")
    ap.add_argument("--sockets", type=int, default=0, help="the CPU's sockets (0: not known)")
    ap.add_argument("--cores", type=int, default=0, help="the CPU's physical cores across its sockets (0: not known)")
    a = ap.parse_args(argv)
    cards = []
    for i, spec in enumerate(a.cards):
        v, _, cc = spec.partition(":")
        cards.append({"index": i, "name": f"GPU {i}", "vram_gb": float(v), "arch": cc or 0})
    p = plan(cards, a.ram, a.quant, {"sockets": a.sockets, "cores": a.cores})
    print(f"{a.quant}: {quant_gib(a.quant):.0f} GiB of files")
    print(("fits: " + p["said"]) if p["fits"] else "does not fit: " + p["reason"])
    if p["fits"] and p["slow"]:
        print(f"slow: {p['slow_why']}")
        print("the setup scales down: " + ", ".join(p["turned_off"]))
    if p["fits"]:
        b = p["resident_budget_gib"]
        print(f"context {p['context']}, --kv-resident {p['kv_resident']} ({kv_vram_gib(p['kv_resident']):.1f} GiB "
              f"of K/V in VRAM), resident budget {b if b is not None else '-'} GiB")
        print(f"layer_split: {p['layer_split'] if p['layer_split'] else '-'}   args: {' '.join(p['args']) or '-'}")
        print(f"T4 env: {'yes' if p['t4'] else 'no'}" + ("   (measured on this layout)" if p["measured"] else ""))
    return 0 if p["fits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
