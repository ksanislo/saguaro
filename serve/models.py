"""serve/models.py - the web app's Models page: the run configs the setup wrote (saguaro-<model>.json beside
saguaro.py; the older maya-<model>.json name is still found - tools/config_names.py), what
each one runs with, the engine settings the page changes in them, and a model deleted (delete_plan, delete).

A config's settings, as the page shows them (None: the engine's own default, or "auto"):
  both:      context, images, gpus, reserve_mib, env
  GLM:       parallel, draft ("mtp" | "dflash2"), expert_cache, prefill, kv_streaming, kv_resident
  DeepSeek:  parallel, draft ("dspark" | "none"), tp, mmap ("auto" | "on" | "off"), prefill_chunk, cpu_threads
  MiniMax:   parallel, draft ("dspark" | "none"), tp (0: the layers split, no groups), kv_cache, kv_streaming,
             kv_resident, cpu_threads (maya-mm3 stops at a flag it does not know: only its own are written)
  Hy4-Flash (saguaro-hy4 stops at a flag it does not know - only its own are written): draft ("auto" | "mtp" |
             "dfly2" | "none": what its pack has), tp ("auto" | 0: the layers split | the group's size), reserve_gb,
             prefill_chunk (--chunk), cpu_threads, kv ("auto" | "q8" | "f16"); no parallel
  Inkling 2: prefill_chunk (--chunk), cpu_threads (--threads), kv ("f16" | "fp8": SAGUARO_IK2_KV), draft ("auto" |
             "dspark" | "none": --drafting), parallel (--parallel: one GPU holding every expert); one or two GPUs
             (--gpus kept with them); no reserve (saguaro-ik2 stops at a flag it does not know)
   QFN (saguaro-qfn, its own engine and family - CYBER-FROST's quant too; its server always loads the MTP drafter,
              and SAGUARO_EXPERT_RAM_GB does nothing here): draft ("mtp" | "none":
              off is --mtp-max-t 1 with --mtp kept), expert_cache, prefill, kv ("fp16" | "int8" | "k8v4" | "q4_0"),
              kv_resident (--kv-resident: the K/V positions kept in VRAM), ram_gib (the experts' RAM tier,
              --resident-budget-gib), parallel (the config's own "parallel" key, not an engine flag), images
              (the engine's --vision with the setup's vision entry)
apply() writes them back into the config's "args", "env", "gpu", "vision" and "parallel" - its other keys as they
are; keys the family does not have are refused ("not a setting of this model")."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import config_names  # noqa: E402  (tools/) the config name family: saguaro-<model>.json, the old maya- one found too
import maya_quants  # noqa: E402  (tools/) the quants' names users see, and each config's family

# the server's notes: SWITCH_FILE says saguaro.py which config to start next (a switch), DELETE_FILE says the next
# server to delete a config (it was the one running).  Written under these 1.2.2 names since C122; a note left by
# a server of before the rename (.maya-switch.json / .maya-delete.json) is still read (config_names.note_paths).
SWITCH_FILE = ".saguaro-switch.json"
DELETE_FILE = ".saguaro-delete.json"
LAST_GOOD = ".saguaro-last-good.json"       # what loaded successfully last: a failed load's page offers to start it
CONTEXTS = [8192, 32768, 65536, 131072]       # the setup's choices (a config's own size is offered too)
DS4_CONTEXTS = [32768, 65536, 131072, 262144, 524288, 1048576]   # ... for DeepSeek (saguaro.py's DS4_CONTEXTS)
MM3_CONTEXTS = DS4_CONTEXTS                                       # ... and MiniMax (saguaro.py's MM3_CONTEXTS)
QFN_CONTEXTS = [32768, 65536, 131072, 262144, 524288, 1048576]   # ... and Qwen3.8-Flash-Next on saguaro-qfn
                                                                  #  (saguaro.py's QFN_CONTEXTS: past 256K the
                                                                  #  elastic YaRN's 512K and 1M)
CONTEXT_MIN = 4096          # the dashboard's Context size: the smallest it offers ...
CONTEXT_FALLBACK_MAX = 262144   # ... and the largest when the model file does not say what it was trained for
# the largest context the dashboard offers without an "untested" warning, by engine: maya-ds4 its model's own 1M; the
# GLM engine's 256K, and maya-mm3's 256K (MiniMax-M3.1 is offered up to its 1M; past 256K it has not been run end to
# end yet)
HY4_CONTEXTS = [8192, 32768, 65536, 131072]                       # ... Hy4-Flash (tools/hy4_fit.py's CONTEXTS)
IK2_CONTEXTS = [32768, 65536, 131072, 262144]                     # ... and Inkling 2 Mini (tools/ik2_fit.py's)
# (hy4, ik2: the largest context their measured Tesla T4 layouts ran at)
CONTEXT_TESTED = {"ds4": 1048576, "mm3": 262144, "hy4": 131072, "ik2": 131072,
                  "qfn": 1048576}      # (QFN's 1M runs live on a 4 x T4 box: plain RoPE to 256K, YaRN past it)
CONTEXT_TESTED_DEFAULT = 262144
QFN_YARN_ORIG = 262144        # QFN's trained reach: plain RoPE up to here ...
QFN_YARN_MAX = 1048576        # ... and the elastic YaRN's top: past QFN_YARN_ORIG the slider says "YaRN, extended"
QFN_KV = ("fp16", "int8", "k8v4", "q4_0")                         # saguaro-qfn's --kv
MM3_KV = ("bf16", "int8", "fp8", "fp4")                           # maya-mm3 --kv-cache (one format for K and V)
# files the setup's quant list does not name, by their config's installer key - since the model-names review's
# nit 3 the same map maya_quants.quant_name reads, so the page's model_quant and a row's own quant name one map
OTHER_QUANTS = maya_quants.OTHER_QUANTS
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# variables that change which code the engine's process loads or runs (a library preloaded, the program search path):
# never set or changed from the page - with remote access on, the page is a network client; a config's own are kept
SECRET = re.compile(r"key|token|secret|password", re.I)   # (the server's report: the same names left out)
MASK = "***"
ENV_LOADER = re.compile(r"^(?:LD_\w*|DYLD_\w*|PYTHON\w*|PATH|PATHEXT|BASH_ENV|ENV|IFS|COMSPEC|SYSTEMROOT)$", re.I)
GLM_KEYS = ("context", "images", "gpus", "reserve_mib", "env", "parallel", "draft", "expert_cache", "prefill",
            "kv_streaming", "kv_resident")
DS4_KEYS = ("context", "images", "gpus", "reserve_mib", "env", "parallel", "draft", "tp", "mmap", "prefill_chunk",
            "cpu_threads", "prompt_lookup", "dspark_conf")
MM3_KEYS = ("context", "images", "gpus", "env", "parallel", "draft", "tp", "kv_cache", "kv_streaming", "kv_resident",
            "cpu_threads", "reserve_gb", "prefill_chunk", "cache_gb", "dspark_conf")
# QFN's own field set (saguaro-qfn): its RAM tier is the engine argument --resident-budget-gib (the old Qwen path's
# SAGUARO_EXPERT_RAM_GB does nothing here), several answers at once are the config's own "parallel" key
QFN_KEYS = ("context", "images", "gpus", "reserve_mib", "env", "parallel", "draft", "expert_cache", "prefill",
            "kv", "kv_resident", "ram_gib")
IK2_KEYS = ("context", "images", "gpus", "env", "prefill_chunk", "cpu_threads", "kv", "draft", "parallel")
HY4_KEYS = ("context", "images", "gpus", "env", "draft", "tp", "reserve_gb", "prefill_chunk", "cpu_threads", "kv")
HY4_KV = ("auto", "q8", "f16")                                    # saguaro-hy4 --kv (the latent cache's rows)
# what the other engines have and Hy4's does not, said in plain words when a client asks for it
ONE_AT_A_TIME = {"hy4": "Answers at once: Hy4-Flash answers one request at a time for now"}
IK2_KV = ("f16", "fp8")                                           # saguaro-ik2's SAGUARO_IK2_KV (f16: its own)
IK2_OTHER_ENV = re.compile(r"^SAGUARO_(?:GLM|MM3|DS4)_")         # another engine's settings: not added to its configs
import ik2_fit as IK2F  # noqa: E402  (tools/) where drafting runs: the setup's own rules


def ik2_draft_of(args: list) -> str:
    """Inkling 2's drafting as the page shows it: auto (--drafting auto: measured once on this PC), dspark (--drafting
    on: the drafter and the prompt lookup, not measured), none."""
    how = flag(args, "--drafting") or ("on" if flag(args, "--draft") else "off")
    return {"auto": "auto", "on": "dspark"}.get(how, "none")


def ik2_card(cfg: dict) -> dict:
    """The card an Inkling 2 config runs on first ({"arch", "vram_gb"} as the setup reads them; {} when not known)."""
    ids = gpus_of(cfg) or [0]
    return next((g for g in machine_gpus("hip" if cfg.get("backend") == "hip" else "cuda") if g.get("index") == ids[0]),
                None) or {}


_ik2_ex: dict = {}


def ik2_experts_gib(cfg: dict) -> float:
    """GiB of the experts of an Inkling 2 config's GGUF: a file of the user's own (installer.ik2_file, e.g. our one-T4
    quant) read from its header once, else UD-Q4_K_XL's - as the setup's fit counts them."""
    path = flag(cfg.get("args") or [], "--model")
    if not path or not (cfg.get("installer") or {}).get("ik2_file"):
        return IK2F.EXPERTS_GIB
    if path not in _ik2_ex:
        _ik2_ex[path] = IK2F.experts_gib_of(path) or IK2F.EXPERTS_GIB
    return _ik2_ex[path]


def ik2_chunk_max(cfg: dict, args: list, kv: str) -> int | None:
    """The largest Prompt chunk an Inkling 2 config's card runs at its context (ik2_fit.chunk_max), or None."""
    card = ik2_card(cfg)
    return IK2F.chunk_max(float(card.get("vram_gb") or 0), _int(flag(args, "--max-context"), IK2F.CONTEXTS[0]), kv,
                          cfg.get("backend") == "hip", ik2_experts_gib(cfg))


# ------------------------------------------------------------------ the engine's arguments
def flag(args: list, name: str, default=None):
    """The value after `name` in the arguments, else `default`."""
    args = [str(a) for a in args]
    return args[args.index(name) + 1] if name in args[:-1] else default


def set_flag(args: list, name: str, value) -> list:
    """The arguments with `name value` (replaced where it was, else added at the end)."""
    out = [str(a) for a in args]
    if name in out[:-1]:
        out[out.index(name) + 1] = str(value)
    else:
        out += [name, str(value)]
    return out


def drop_flag(args: list, name: str) -> list:
    out, i, args = [], 0, [str(a) for a in args]
    while i < len(args):
        if args[i] == name and i + 1 < len(args):
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def _int(v, default=None):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _float(v, default=None):
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return default
    return f if f == f else default                    # (not NaN)


def family(cfg: dict) -> str:
    """"hy4", "ik2", "minimax", "deepseek", "qfn" or "glm" (maya_quants.family_of, as saguaro.py and the server
    read it: a config of the old Qwen path reads "qfn" too)."""
    return maya_quants.family_of(cfg)


def gpus_of(cfg: dict) -> list:
    g = cfg.get("gpu")
    if isinstance(g, list):
        return [int(x) for x in g if str(x).strip().isdigit()]
    if isinstance(g, int):
        return [g]
    if isinstance(g, str):
        return [int(x) for x in g.split(",") if x.strip().isdigit()]
    return []


def pack_of(cfg: dict) -> Path | None:
    args = cfg.get("args") or []
    p = flag(args, "--glm-pack") or flag(args, "--pack")
    return Path(p) if p else None


def model_file_of(cfg: dict) -> Path | None:
    """Inkling 2's GGUF (saguaro-ik2 reads it in place: --model, no pack)."""
    p = flag(cfg.get("args") or [], "--model") if family(cfg) == "ik2" else None
    return Path(p) if p else None


def home_of(cfg: dict) -> Path | None:
    """The folder Saguaro built beside a config's model files: its pack - or, for Inkling 2, the setup's saguaro-ik2
    folder (its tokenizer and the record of the GGUF it was made from)."""
    pack = pack_of(cfg)
    if pack is not None or family(cfg) != "ik2" or not cfg.get("tokenizer"):
        return pack
    tok = Path(str(cfg["tokenizer"]))
    return tok.parent if tok.name == "tokenizer" and (tok.parent / "sources.json").is_file() else None


def hy4_txt(cfg: dict) -> list:
    """A Hy4-Flash pack's hy4.txt as [key, value ...] lines ([] when it cannot be read)."""
    pack = pack_of(cfg)
    try:
        return [ln.split() for ln in (pack / "hy4.txt").read_text(encoding="utf-8", errors="replace").splitlines()]
    except (OSError, TypeError):
        return []


def hy4_has(cfg: dict, what: str) -> bool:
    """Whether a Hy4-Flash pack has its MTP block ("mtp": n_mtp 1), a DFly2 drafter ("dfly2": dfly_layers) or the
    vision tower ("vision": vision.dim) - what tools/hy4/pack_gguf.py found beside the model."""
    lines = hy4_txt(cfg)
    if what == "mtp":
        return any(ln[:2] == ["n_mtp", "1"] for ln in lines)
    return any(ln[:1] == [{"dfly2": "dfly_layers", "vision": "vision.dim"}[what]] for ln in lines)


def hy4_draft_of(args: list) -> str:
    """A Hy4-Flash config's drafting as the page shows it: auto (--drafting auto / on: the engine measures once which
    decodes fastest), mtp, dfly2 or none."""
    args = [str(a) for a in args]
    how = flag(args, "--drafting")
    if how in ("auto", "on"):
        return "auto"
    if "--mtp" in args or how == "mtp":
        return "mtp"
    return "dfly2" if "--dfly2" in args or how == "dfly2" else "none"


# ------------------------------------------------------------------ the context a model was trained for
def trained_context(model_dir) -> int | None:
    """The context the model was trained for (`<arch>.context_length` in its first GGUF's header; GLM-5.3-Flash:
    1,048,576): the top of the dashboard's Context size.  None when there is no readable GGUF."""
    try:
        from gguf_reader import GGUFFile                # (tools/, on the path above)
        shards = sorted(Path(model_dir).glob("*-00001-of-*.gguf")) or sorted(Path(model_dir).glob("*.gguf"))
        if not shards:
            return None
        md = GGUFFile(shards[0]).metadata
        n = md.get(f"{md.get('general.architecture', '')}.context_length")
        return int(n) if isinstance(n, int) and n > 0 else None
    except Exception:  # noqa: BLE001 - a header it can't read: the fallback range
        return None


def trained_context_ds4(pack) -> int | None:
    """DeepSeek's and MiniMax's: the context its pack says the model was trained for (ds4.txt's / mm3.txt's
    max_position_embeddings, from the model's config.json or a GGUF's context_length; V4.1-Flash and M3.1-Flash:
    1,048,576, their native reach), the top of the dashboard's Context size.  None when the pack does not say."""
    for name, key in (("ds4.txt", "max_position_embeddings"), ("mm3.txt", "max_position_embeddings"),
                      ("hy4.txt", "context_length")):
        try:
            with open(Path(pack) / name, encoding="utf-8", errors="replace") as f:
                for line in f:
                    k, _, v = line.strip().partition(" ")
                    if k == key:
                        n = int(v.split()[0])
                        return n if n > 0 else None
        except (OSError, ValueError, IndexError):
            continue
    return None


_trained: dict = {}


def trained_context_of(cfg: dict) -> int | None:
    """A config's model's trained context, as the server reads it when it runs that config (GLM: its GGUF header
    beside --glm-pack; DeepSeek and MiniMax: --pack's own file); kept once found (a header read is not free)."""
    args = [str(x) for x in cfg.get("args") or []]
    if "--glm-pack" in args[:-1]:
        key, read_it = ("gguf", str(Path(args[args.index("--glm-pack") + 1]).resolve().parent)), trained_context
    elif "--pack" in args[:-1]:
        key, read_it = ("pack", args[args.index("--pack") + 1]), trained_context_ds4
    else:
        return None
    if key not in _trained:
        n = read_it(key[1])
        if n is None:
            return None                                 # (not kept: the files may still be on their way)
        _trained[key] = n
    return _trained[key]


def context_range(trained: int | None, current: int, engine_kind: str = "") -> dict:
    """The dashboard's Context size slider for one model (Settings > Model for the running one, the Models page's
    Engine settings for each): CONTEXT_MIN up to what the model was trained for (else CONTEXT_FALLBACK_MAX), never
    below the size it has now; `tested`: past it the page says the size is untested on this engine.  QFN's engine
    runs elastic YaRN: its top is 1M whatever the files say (a live 4 x T4 box runs it), and the page gets a
    `yarn_above` mark: sizes past it are the extension, not the trained reach (the slider says "YaRN, extended").
    The other families are unchanged."""
    top = max(trained or CONTEXT_FALLBACK_MAX, int(current or 0))
    if engine_kind == "qfn":
        top = max(top, QFN_YARN_MAX)
    out = {"min": CONTEXT_MIN, "max": top, "trained": trained,
           "tested": CONTEXT_TESTED.get(engine_kind, CONTEXT_TESTED_DEFAULT)}
    if engine_kind == "qfn":
        out["yarn_above"] = QFN_YARN_ORIG
    return out


# ------------------------------------------------------------------ what a config can be switched to
def dflash_file(cfg: dict) -> Path | None:
    """GLM's DFlash2 drafter: the one the config drafts with, else one a setup downloaded into its models folder."""
    p = flag(cfg.get("args") or [], "--dflash")
    if p:
        return Path(p)
    models = (cfg.get("installer") or {}).get("models_dir")
    d = Path(models) / "GLM-5.3-Flash-DFlash2" if models else None
    try:
        found = sorted(d.glob("*.gguf")) if d is not None and d.is_dir() else []
    except OSError:
        found = []
    return found[0] if found else None


def ds4_has_drafter(cfg: dict) -> bool:
    """DeepSeek's pack holds the DSpark drafter (tools/calibrate_ds4.py's has_drafter)."""
    pack = pack_of(cfg)
    t = pack / "tensors.txt" if pack else None
    try:
        with t.open(encoding="utf-8", errors="replace") as f:
            return any(line.startswith("t mtp.") for line in f)
    except (OSError, AttributeError):
        return False


def ds4_has_vision(cfg: dict) -> bool:
    pack = pack_of(cfg)
    try:
        return any(line.startswith("vision.num_hidden_layers ")
                   for line in (pack / "ds4.txt").read_text(encoding="utf-8", errors="replace").splitlines())
    except (OSError, TypeError):
        return False


def qfn_mtp(cfg: dict) -> Path | None:
    """QFN's Lamina draft layer: the folder its config drafts with (--mtp, the setup's download beside the model);
    None when the config names none or that folder is not on this PC."""
    p = flag(cfg.get("args") or [], "--mtp")
    d = Path(p) if p else None
    return d if d is not None and d.is_dir() else None


def mm3_pack_has(cfg: dict, key: str) -> bool:
    """MiniMax's pack indexes `key` (mm3.txt: "dspark.num_hidden_layers" its drafter, "vision.hidden_size" its vision
    tower - saguaro.py's checks)."""
    pack = pack_of(cfg)
    try:
        return any(line.startswith(key + " ")
                   for line in (pack / "mm3.txt").read_text(encoding="utf-8", errors="replace").splitlines())
    except (OSError, TypeError):
        return False


def choices(cfg: dict) -> dict:
    """What this config can be set to beyond what it is: a drafter and an image encoder that exist on this PC."""
    if family(cfg) == "ik2":                           # (images: its "vision" entry, on or kept off)
        # (drafting: none where it does not run - Pascal, a streamed cache, no room, AMD's offload path; else auto,
        #  the drafter by hand where the setup downloaded it, none)
        args, card = cfg.get("args") or [], ik2_card(cfg)
        hip = cfg.get("backend") == "hip"
        off = IK2F.drafting_off_reason(None if hip else card.get("arch"), float(card.get("vram_gb") or 0),
                                       _int(flag(args, "--max-context"), IK2F.CONTEXTS[0]),
                                       (cfg.get("env") or {}).get("SAGUARO_IK2_KV") or IK2F.DEFAULT_KV, hip,
                                       ik2_experts_gib(cfg), len(set(gpus_of(cfg))) > 1) \
            or (_int(flag(args, "--tp"), 1) > 1 and "tensor parallel")   # (two GPUs, --tp: no drafting yet)
        return {"draft": ["none"] if off else ["auto"] + (["dspark"] if flag(args, "--draft") else []) + ["none"],
                "images": bool(cfg.get("vision") or cfg.get("vision_off"))}
    if family(cfg) == "hy4":                           # (the drafters its pack has; images: the pack's mmproj file)
        drafters = (["mtp"] if hy4_has(cfg, "mtp") else []) + (["dfly2"] if hy4_has(cfg, "dfly2") else [])
        return {"draft": (["auto"] if drafters else []) + drafters + ["none"], "images": hy4_has(cfg, "vision")}
    if family(cfg) == "minimax":
        return {"draft": ["dspark", "none"] if mm3_pack_has(cfg, "dspark.num_hidden_layers") else ["none"],
                "images": mm3_pack_has(cfg, "vision.hidden_size")}
    if family(cfg) == "deepseek":
        return {"draft": ["dspark", "none"] if ds4_has_drafter(cfg) else ["none"],
                "images": ds4_has_vision(cfg) or "--vision" not in (cfg.get("args") or [])}
    if family(cfg) == "qfn":                           # (images: the engine's --vision, the setup's vision entry)
        return {"draft": ["mtp", "none"] if qfn_mtp(cfg) is not None else ["none"],
                "images": bool(cfg.get("vision") or cfg.get("vision_off")
                               or "--vision" in [str(a) for a in (cfg.get("args") or [])])}
    f = dflash_file(cfg)
    return {"draft": ["mtp", "dflash2"] if f is not None else ["mtp"],
            "images": bool(cfg.get("vision") or cfg.get("vision_off"))}


# ------------------------------------------------------------------ settings <-> the config
def settings_of(cfg: dict) -> dict:
    args = cfg.get("args") or []
    env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    out = {"context": _int(flag(args, "--max-context"), 32768), "gpus": gpus_of(cfg), "env": env}
    if family(cfg) == "ik2":                           # (saguaro-ik2: --ctx is its other name for --max-context)
        out.update(context=_int(flag(args, "--max-context") or flag(args, "--ctx"), 32768),
                   images=bool(cfg.get("vision")), prefill_chunk=_int(flag(args, "--chunk")),
                   cpu_threads=_int(flag(args, "--threads")), draft=ik2_draft_of(args),
                   parallel=_int(flag(args, "--parallel"), 1),
                   kv="fp8" if env.get("SAGUARO_IK2_KV") in ("fp8", "e4m3") else "f16")
        return out
    if family(cfg) == "hy4":                           # (saguaro-hy4's own defaults where the config names none)
        tp = flag(args, "--tp", "auto")
        out.update(images=flag(args, "--vision") != "off" and hy4_has(cfg, "vision"), draft=hy4_draft_of(args),
                   tp="auto" if tp == "auto" else _int(tp, "auto"), reserve_gb=_float(flag(args, "--reserve-gb")),
                   prefill_chunk=_int(flag(args, "--chunk")), cpu_threads=_int(flag(args, "--cpu-threads")),
                   kv=flag(args, "--kv", "auto"))
        return out
    if family(cfg) == "minimax":                       # (maya-mm3's own defaults where the config names none)
        vision, kvs = flag(args, "--vision"), flag(args, "--kv-streaming")
        out.update(images=vision == "on" or (vision is None and mm3_pack_has(cfg, "vision.hidden_size")),
                   parallel=_int(flag(args, "--parallel"), 1),
                   draft="dspark" if flag(args, "--dspark", "off") not in ("off", "none") else "none",
                   tp=_int(flag(args, "--tp"), 0), kv_cache=flag(args, "--kv-cache", "int8"),
                   kv_streaming=kvs == "on" or (kvs is None and _int(flag(args, "--kv-resident"), 0) > 0),
                   kv_resident=_int(flag(args, "--kv-resident")), cpu_threads=_int(flag(args, "--cpu-threads")),
                   reserve_gb=_float(flag(args, "--reserve-gb")), prefill_chunk=_int(flag(args, "--chunk")),
                   cache_gb=_float(flag(args, "--cache-gb")), dspark_conf=_float(flag(args, "--dspark-conf")))
        return out
    if family(cfg) == "qfn":                           # (saguaro-qfn's own fields; see QFN_KEYS)
        ec, pf = flag(args, "--expert-cache", "auto"), flag(args, "--prefill", "auto")
        out.update(images=bool(cfg.get("vision")) or "--vision" in [str(a) for a in args],
                   reserve_mib=_int(flag(args, "--vram-reserve-mib")),
                   parallel=_int(cfg.get("parallel"), 1),
                   draft="mtp" if flag(args, "--mtp") and flag(args, "--mtp-max-t") != "1" else "none",
                   expert_cache=None if ec == "auto" else _int(ec), prefill=None if pf == "auto" else _int(pf),
                   kv=flag(args, "--kv", "fp16"), kv_resident=_int(flag(args, "--kv-resident")),
                   ram_gib=_int(flag(args, "--resident-budget-gib")))
        return out
    if family(cfg) == "deepseek":
        dspark = flag(args, "--dspark")
        out.update(images=flag(args, "--vision") != "off",
                   reserve_mib=_int(flag(args, "--reserve-mib")), parallel=_int(flag(args, "--parallel"), 1),
                   draft="none" if dspark == "none" or (dspark is None and not ds4_has_drafter(cfg)) else "dspark",
                   tp=_int(flag(args, "--tp"), 2), mmap=flag(args, "--mmap-experts", "auto"),   # (2: the engine's own)
                   prefill_chunk=_int(flag(args, "--prefill-chunk")), cpu_threads=_int(flag(args, "--cpu-threads")),
                   # (--suffix-draft: the prompt lookup's shortest match, 3 by the engine; 0 off)
                   prompt_lookup=_int(flag(args, "--suffix-draft"), 3) != 0,
                   dspark_conf=_float(flag(args, "--dspark-conf")))
        if out["mmap"] not in ("auto", "on", "off"):
            out["mmap"] = "auto"
        return out
    ec, pf = flag(args, "--expert-cache", "auto"), flag(args, "--prefill", "auto")
    out.update(images=bool(cfg.get("vision")), reserve_mib=_int(flag(args, "--vram-reserve-mib")),
               parallel=_int(flag(args, "--parallel"), 1), draft="dflash2" if flag(args, "--dflash") else "mtp",
               expert_cache=None if ec == "auto" else _int(ec), prefill=None if pf == "auto" else _int(pf),
               kv_streaming=glm_streams(cfg), kv_resident=_int(flag(args, "--kv-resident")))
    return out


def glm_streams(cfg: dict) -> bool:
    """GLM's KV streaming as the engine runs it: --kv-streaming on/off, else only a window given (--kv-resident or the
    environment's GLM_KV_RESIDENT) streams - a config from before the flag keeps every cell in VRAM."""
    args, env = cfg.get("args") or [], cfg.get("env") or {}
    kvs = flag(args, "--kv-streaming")
    if kvs is not None:
        return kvs != "off"
    return (_int(flag(args, "--kv-resident"), 0) or 0) > 0 or \
        (_int(env.get("SAGUARO_GLM_KV_RESIDENT"), 0) or 0) > 0


def _num(s: dict, key: str, lo: int, hi: int, what: str, none_ok: bool = True):
    v = s.get(key)
    if v is None or v == "":
        if none_ok:
            return None
        raise ValueError(f"{what}: a number is needed")
    if isinstance(v, bool) or _int(v) is None:
        raise ValueError(f"{what}: a whole number, please")
    v = _int(v)
    if not lo <= v <= hi:
        raise ValueError(f"{what}: from {lo} to {hi}")
    return v


def _dec(s: dict, key: str, lo: float, hi: float, what: str):
    """A decimal setting: None where empty (the engine's own), else a number from lo to hi."""
    v = s.get(key)
    if v is None or v == "":
        return None
    if isinstance(v, bool) or _float(v) is None:
        raise ValueError(f"{what}: a number, please")
    v = _float(v)
    if not lo <= v <= hi:
        raise ValueError(f"{what}: from {lo:g} to {hi:g}")
    return v


def qfn_yarn_follow(args: list) -> list:
    """The QFN args with --rope-scale kept in step with --max-context: past the config's own --yarn-orig-ctx
    the elastic YaRN extension reaches max / orig, so a context of 1M needs --rope-scale 4 (512K: 2,
    384K: 1.5; the engine parses the factor as a double, so it is written as computed, never rounded).  At or
    below it, and on a config without the YaRN pair, the args come back as they are (inside the trained
    context the elastic engine never reads the YaRN table, so the written factor stays).  One rule for every
    writer of a QFN context: the Models page (apply), the running server saving a live slider move
    (server._save_context - a 1M move that saved only --max-context left the next start spawning a context its
    factor cannot reach, under-scaled and silent), and the driver rewriting its spawn command
    (engines/qfn/serve/server.py's set_context, whose copy of this rule it matches - gate on the config's own
    orig and compute with the same number, review finding 3: the constant gate could hand a hand-edited
    config a factor under 1, which the engine refuses)."""
    yctx, yorig = _int(flag(args, "--max-context"), 0), _int(flag(args, "--yarn-orig-ctx"), 0)
    if yorig > 0 and yctx > yorig and flag(args, "--rope-scaling") == "yarn":
        return set_flag(args, "--rope-scale", "%g" % (yctx / yorig))
    return args


def apply(cfg: dict, settings: dict) -> dict:
    """The config with `settings` written into it (a ValueError says what is wrong, nothing changed).  Keys left out
    keep the config's value."""
    if not isinstance(settings, dict):
        raise ValueError("the settings must be a JSON object")
    fam = family(cfg)
    ds4 = fam == "deepseek"
    known = {"minimax": MM3_KEYS, "deepseek": DS4_KEYS, "qfn": QFN_KEYS, "ik2": IK2_KEYS,
             "hy4": HY4_KEYS}.get(fam, GLM_KEYS)   # (QFN has its own field set: QFN_KEYS)
    if fam in ONE_AT_A_TIME and settings.get("parallel") not in (None, "", 1):
        raise ValueError(ONE_AT_A_TIME[fam])
    if fam in ONE_AT_A_TIME:
        known += ("parallel",)                          # (1 is what they do: taken, nothing written)
    unknown = sorted(set(settings) - set(known))
    if unknown:
        raise ValueError(f"not a setting of this model: {', '.join(unknown)}")
    s = {**settings_of(cfg), **settings}
    can = choices(cfg)
    out = copy.deepcopy(cfg)
    args = [str(a) for a in out.get("args") or []]
    # (saguaro-ik2's --ctx: a config that names its context so keeps the one flag)
    ctx_flag = "--ctx" if fam == "ik2" and "--ctx" in args[:-1] and "--max-context" not in args[:-1] else "--max-context"
    args = set_flag(args, ctx_flag, _num(s, "context", 1024, 1048576, "Context size", none_ok=False))

    gpus = s.get("gpus")
    if not isinstance(gpus, list) or not gpus or not all(isinstance(g, int) and not isinstance(g, bool) and 0 <= g < 64
                                                        for g in gpus):
        raise ValueError("GPUs: pick at least one")
    if fam == "ik2" and len(set(gpus)) > 2:
        raise ValueError("GPUs: Inkling 2 runs on one or two")
    if gpus != gpus_of(cfg):
        out["gpu"] = sorted(set(gpus))

    env = s.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("Environment: name = value pairs")
    clean = {}
    for k, v in env.items():
        k = str(k).strip()
        if not k:
            continue
        if not ENV_NAME.match(k):
            raise ValueError(f"Environment: {k!r} is not a variable name (letters, digits and _)")
        if fam == "ik2" and IK2_OTHER_ENV.match(k) and k not in (cfg.get("env") or {}):
            raise ValueError(f"Environment: {k} is another engine's setting - Inkling 2's are SAGUARO_IK2_*")
        if str(v) == MASK and SECRET.search(k) and k in (cfg.get("env") or {}):
            v = (cfg.get("env") or {})[k]                # (the page's mask of a secret: its value as it was)
        if ENV_LOADER.match(k) and str((cfg.get("env") or {}).get(k)) != str(v):
            raise ValueError(f"Environment: {k} changes which programs or libraries the engine loads - not from the "
                             "page (the setup's --env sets it)")
        clean[k] = str(v)
    if clean:
        out["env"] = clean
    else:
        out.pop("env", None)

    reserve = _num(s, "reserve_mib", 0, 65536, "VRAM kept free") if fam not in ("minimax", "ik2", "hy4") else None
    if fam == "hy4":
        # (saguaro-hy4 stops at a flag it does not know: drafting, --tp, --reserve-gb, --chunk, --cpu-threads, --kv and
        #  --vision are all it gets - empty: the engine's own.  A config that names --gpus keeps it in step)
        n = len(set(gpus))
        if gpus != gpus_of(cfg) and "--gpus" in args[:-1]:
            args = set_flag(args, "--gpus", ",".join(str(i) for i in range(n))) if n > 1 else drop_flag(args, "--gpus")
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: this pack has no MTP block" if s.get("draft") == "mtp" else
                             "Drafting: this pack has no DFly2 drafter" if s.get("draft") == "dfly2" else
                             "Drafting: this pack has no drafter (the setup packs none)" if s.get("draft") == "auto"
                             else "Drafting: auto, MTP, DFly2 or none")
        args = [x for x in drop_flag(args, "--drafting") if x not in ("--mtp", "--dfly2")] + \
            (["--drafting", "auto"] if s["draft"] == "auto" else ["--mtp"] if s["draft"] == "mtp" else
             ["--dfly2"] if s["draft"] == "dfly2" else [])
        tp = s.get("tp")
        if tp in (None, "", "auto"):
            args = drop_flag(args, "--tp")
        elif isinstance(tp, int) and not isinstance(tp, bool) and tp > max(2, n):
            raise ValueError(f"Tensor parallel: a group of {tp} needs {tp} GPUs picked (Auto and Off fit any)")
        else:
            args = set_flag(args, "--tp", _num(s, "tp", 0, max(2, n), "Tensor parallel", none_ok=False))
        v = _dec(s, "reserve_gb", 0, 64, "VRAM kept free")
        args = set_flag(args, "--reserve-gb", f"{v:g}") if v is not None else drop_flag(args, "--reserve-gb")
        for key, name, lo, hi, what in (("prefill_chunk", "--chunk", 64, 65536, "Prompt chunk"),
                                        ("cpu_threads", "--cpu-threads", 1, 512, "CPU threads")):
            v = _num(s, key, lo, hi, what)
            args = set_flag(args, name, v) if v is not None else drop_flag(args, name)
        kv = str(s.get("kv") or "auto")
        if kv not in HY4_KV:
            raise ValueError("KV cache: auto, 8-bit or F16")
        args = drop_flag(args, "--kv") if kv == "auto" else set_flag(args, "--kv", kv)
        if s.get("images"):                             # (on is the engine's own when the pack has the tower: no flag)
            if not can["images"]:
                raise ValueError("Images: this pack has no vision tower (the setup packs the model with its mmproj "
                                 "file)")
            args = drop_flag(args, "--vision")
        elif can["images"] or "--vision" in args:
            args = set_flag(args, "--vision", "off")
    elif fam == "ik2":
        # (saguaro-ik2 stops at a flag it does not know: --gpus, --chunk, --threads and SAGUARO_IK2_KV are all it gets)
        n = len(set(gpus))
        args = set_flag(args, "--gpus", n) if n > 1 or "--gpus" in args[:-1] else args
        if s.get("images") and not out.get("vision"):
            if not out.get("vision_off"):
                raise ValueError("Images: this model was set up without its image encoder")
            out["vision"] = out.pop("vision_off")
        elif not s.get("images") and out.get("vision"):
            out["vision_off"] = out.pop("vision")       # (kept, so Images can be switched on again)
        for key, name, lo, hi, what in (("prefill_chunk", "--chunk", 16, 65536, "Prompt chunk"),
                                        ("cpu_threads", "--threads", 1, 512, "CPU threads")):
            v = _num(s, key, lo, hi, what)
            args = set_flag(args, name, v) if v is not None else drop_flag(args, name)
        kv = str(s.get("kv") or "f16")
        cap = ik2_chunk_max(cfg, args, kv)             # (the setup's own fit: tools/ik2_fit.py)
        if cap is not None and _int(flag(args, "--chunk"), 0) > cap:
            raise ValueError(f"Prompt chunk: at most {cap} here - every expert stays on this card with little room "
                             "left, and a larger chunk runs it out of GPU memory on a long prompt")
        if "kv" not in settings and "env" in settings:  # (only the env row sent: its value is the choice)
            kv = "fp8" if clean.get("SAGUARO_IK2_KV") in ("fp8", "e4m3") else "f16"
        if kv not in IK2_KV:
            raise ValueError("KV cache: " + ", ".join(f.upper() for f in IK2_KV))
        env2 = dict(out.get("env") or {})
        if kv != "f16" or "SAGUARO_IK2_KV" in env2:     # (written where it changes; the page's choice over the env row)
            env2["SAGUARO_IK2_KV"] = kv
        if env2:
            out["env"] = env2
        else:
            out.pop("env", None)
        # (drafting: auto, the drafter by hand (dspark: --drafting on) or none; written where it changes)
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: not on this GPU or layout (Pascal, two GPUs, an AMD card's offload path, a cache partly in RAM, tensor parallel)"
                             if can["draft"] == ["none"] else
                             "Drafting: this setup has no drafter (the setup downloads it)" if s.get("draft") == "dspark"
                             else "Drafting: auto, the drafter (DSpark) or none")
        if s["draft"] != ik2_draft_of(args):
            args = set_flag(args, "--drafting", {"auto": "auto", "dspark": "on"}.get(s["draft"], "off"))
        # (--parallel: n conversations at once where one GPU holds every expert - elsewhere the engine says so and
        #  answers one at a time; drafting is off while it is above 1)
        par = _num(s, "parallel", 1, 8, "Answers at once", none_ok=False)
        args = set_flag(args, "--parallel", par) if par > 1 or "--parallel" in args else args
    elif fam == "qfn":
        # (saguaro-qfn's own: the RAM tier is --resident-budget-gib, several answers at once are the config's
        #  "parallel" key (the driver turns it into --batch), and drafting off is --mtp-max-t 1 with --mtp kept)
        # C18: the Context size and the elastic YaRN factor stay in step - qfn_yarn_follow above, the same rule
        # the running server saves a live slider move with (server._save_context) and the driver's set_context
        # applies to its spawn command.
        args = qfn_yarn_follow(args)
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: this setup did not download the MTP draft layer (run the setup again)"
                             if s.get("draft") == "mtp" else "Drafting: MTP or none")
        if qfn_mtp(cfg) is not None:   # (loaded either way - the drafts are what "none" switches off)
            args = set_flag(args, "--mtp", qfn_mtp(cfg))
        if flag(args, "--mtp") is not None:
            args = drop_flag(args, "--mtp-max-t") if s["draft"] == "mtp" else set_flag(args, "--mtp-max-t", 1)
        par = _num(s, "parallel", 1, 8, "Answers at once", none_ok=False)
        if par > 1:
            out["parallel"] = par
        else:
            out.pop("parallel", None)          # (1 or absent: the engine answers one at a time either way)
        # Images: the engine's bare --vision flag with the config's vision entry (vision-qfn with the model's
        # mmproj); off keeps the entry as "vision_off", so Images can be switched on again
        if s.get("images") and not out.get("vision"):
            if not out.get("vision_off"):
                raise ValueError("Images: this model runs text only: its image encoder (vision-qfn with the "
                                 "model's mmproj file) is not set up here")
            out["vision"] = out.pop("vision_off")
        elif not s.get("images") and out.get("vision"):
            out["vision_off"] = out.pop("vision")
        args = [x for x in args if x != "--vision"] + (["--vision"] if out.get("vision") else [])
        args = set_flag(args, "--vram-reserve-mib", reserve) if reserve is not None else \
            drop_flag(args, "--vram-reserve-mib")
        ec = _num(s, "expert_cache", 1, 1000000, "Expert cache")
        args = set_flag(args, "--expert-cache", "auto" if ec is None else ec)
        pf = _num(s, "prefill", 64, 65536, "Prompt chunk")
        args = set_flag(args, "--prefill", "auto" if pf is None else pf)
        kv = str(s.get("kv") or "fp16")
        if kv not in QFN_KV:
            raise ValueError("KV cache: " + ", ".join(f.upper() for f in QFN_KV))
        args = set_flag(args, "--kv", kv) if kv != "fp16" or "--kv" in args else args
        kr = _num(s, "kv_resident", 1024, 1048576, "K/V kept in VRAM")
        args = set_flag(args, "--kv-resident", kr) if kr is not None else drop_flag(args, "--kv-resident")
        rg = _num(s, "ram_gib", 1, 4096, "RAM for the experts")
        args = set_flag(args, "--resident-budget-gib", rg) if rg is not None else \
            drop_flag(args, "--resident-budget-gib")
    elif fam == "minimax":
        # (maya-mm3 --parallel: n conversations at once - each its own KV cache in VRAM, so no KV streaming and no DSpark
        #  while it is above 1; those two settings stay as chosen for the day it is 1 again; written where it changes)
        par = _num(s, "parallel", 1, 8, "Answers at once", none_ok=False)
        args = set_flag(args, "--parallel", par) if par > 1 or "--parallel" in args else args
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: this pack has no DSpark drafter" if s.get("draft") == "dspark"
                             else "Drafting: DSpark or none")
        if s["draft"] == "dspark" or "--dspark" in args:
            args = set_flag(args, "--dspark", "on" if s["draft"] == "dspark" else "off")
        if s.get("images") and not can["images"]:
            raise ValueError("Images: this pack has no vision tower")
        args = set_flag(args, "--vision", "on" if s.get("images") else "off") \
            if can["images"] or "--vision" in args else args
        # (0: the layers split across the GPUs; 1, 2 or 4: the tensor-parallel path in groups of that many)
        tp = _num(s, "tp", 0, 4, "Tensor parallel", none_ok=False)
        if tp == 3:
            raise ValueError("Tensor parallel: 1, 2 or 4 GPUs a group, or 0 to split the layers")
        args = set_flag(args, "--tp", tp) if tp else drop_flag(args, "--tp")
        kv = str(s.get("kv_cache") or "int8")
        if kv not in MM3_KV and kv != flag(cfg.get("args") or [], "--kv-cache"):   # (a K,V mix the config has: kept)
            raise ValueError("KV cache: " + ", ".join(f.upper() for f in MM3_KV))
        args = set_flag(args, "--kv-cache", kv) if kv != "int8" or "--kv-cache" in args else args
        args = set_flag(args, "--kv-streaming", "on" if s.get("kv_streaming") else "off")
        kr = _num(s, "kv_resident", 1024, 1048576, "Positions kept in VRAM")
        args = set_flag(args, "--kv-resident", kr) if kr is not None and s.get("kv_streaming") else \
            drop_flag(args, "--kv-resident")
        ct = _num(s, "cpu_threads", 1, 512, "CPU threads")
        args = set_flag(args, "--cpu-threads", ct) if ct is not None else drop_flag(args, "--cpu-threads")
        # (empty: the engine's own - --reserve-gb 1.0, 0.3 with --tp; --chunk the prefill calibration's; --cache-gb 1.5,
        #  read with --tp only; --dspark-conf 0.7)
        for key, name, lo, hi, what in (("reserve_gb", "--reserve-gb", 0, 64, "VRAM kept free"),
                                        ("cache_gb", "--cache-gb", 0, 256, "Expert cache"),
                                        ("dspark_conf", "--dspark-conf", 0, 1, "Draft confidence")):
            v = _dec(s, key, lo, hi, what)
            args = set_flag(args, name, f"{v:g}") if v is not None else drop_flag(args, name)
        ch = _num(s, "prefill_chunk", 16, 65536, "Prompt chunk")
        args = set_flag(args, "--chunk", ch) if ch is not None else drop_flag(args, "--chunk")
    elif ds4:
        # (maya-ds4 --parallel: n conversations at once - each its own context's caches in VRAM; written where it changes)
        par = _num(s, "parallel", 1, 8, "Answers at once", none_ok=False)
        args = set_flag(args, "--parallel", par) if par > 1 or "--parallel" in args else args
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: this pack has no DSpark drafter" if s.get("draft") == "dspark"
                             else "Drafting: DSpark or none")
        args = set_flag(args, "--dspark", "on" if s["draft"] == "dspark" else "none") \
            if ds4_has_drafter(cfg) or "--dspark" in args else args
        if s.get("images"):
            if not can["images"]:
                raise ValueError("Images: this pack has no vision tower")
            args = drop_flag(args, "--vision")
        else:
            args = set_flag(args, "--vision", "off")
        # (a group of up to every GPU picked - the calibration's 2, 4, 8, 16 or all of them; 2 on one GPU runs its
        #  layers alone, as before)
        tp = _num(s, "tp", 1, max(2, len(set(gpus))), "Tensor parallel", none_ok=False)
        args = set_flag(args, "--tp", tp)
        args = set_flag(args, "--reserve-mib", reserve) if reserve is not None else drop_flag(args, "--reserve-mib")
        if s.get("mmap") not in ("auto", "on", "off"):
            raise ValueError("Experts VRAM does not hold: auto, on or off")
        args = drop_flag(args, "--mmap-experts") if s["mmap"] == "auto" else set_flag(args, "--mmap-experts", s["mmap"])
        for key, name, lo, hi, what in (("prefill_chunk", "--prefill-chunk", 16, 65536, "Prompt chunk"),
                                        ("cpu_threads", "--cpu-threads", 1, 512, "CPU threads")):
            v = _num(s, key, lo, hi, what)
            args = set_flag(args, name, v) if v is not None else drop_flag(args, name)
        # the prompt lookup: on keeps a config's own shortest match (else the engine's 3), off is --suffix-draft 0
        if not isinstance(s.get("prompt_lookup"), bool):
            raise ValueError("Prompt lookup: on or off")
        if not s["prompt_lookup"]:
            args = set_flag(args, "--suffix-draft", 0)
        elif not _int(flag(args, "--suffix-draft"), 0):
            args = drop_flag(args, "--suffix-draft")
        conf = _dec(s, "dspark_conf", 0, 1, "Draft confidence")
        args = set_flag(args, "--dspark-conf", f"{conf:g}") if conf is not None else drop_flag(args, "--dspark-conf")
    else:
        par = _num(s, "parallel", 1, 8, "Answers at once", none_ok=False)
        args = set_flag(args, "--parallel", par) if par > 1 or "--parallel" in args else args
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: the DFlash2 drafter is not on this PC (the setup's --draft dflash2 downloads "
                             "it)" if s.get("draft") == "dflash2" else "Drafting: MTP or DFlash2")
        if s["draft"] == "dflash2":
            args = set_flag(args, "--dflash", dflash_file(cfg))
        else:
            args = drop_flag(args, "--dflash")
        if s.get("images") and not out.get("vision"):
            if not out.get("vision_off"):
                raise ValueError("Images: this model was set up without its image encoder (the setup adds it)")
            out["vision"] = out.pop("vision_off")
        elif not s.get("images") and out.get("vision"):
            out["vision_off"] = out.pop("vision")       # (kept, so Images can be switched on again)
        args = set_flag(args, "--vram-reserve-mib", reserve) if reserve is not None else \
            drop_flag(args, "--vram-reserve-mib")
        # (a flag the config does not have is written only when its value changes: a config from an older setup, run
        #  by the engine it was built with, keeps its command - that engine may not know the newer flags)
        ec = _num(s, "expert_cache", 1, 1000000, "Expert cache")
        if ec is not None or "--expert-cache" in args:
            args = set_flag(args, "--expert-cache", "auto" if ec is None else ec)
        pf = _num(s, "prefill", 64, 65536, "Prompt chunk")
        if pf is not None or "--prefill" in args:
            args = set_flag(args, "--prefill", "auto" if pf is None else pf)
        if "--kv-streaming" in args or bool(s.get("kv_streaming")) != glm_streams(cfg):
            args = set_flag(args, "--kv-streaming", "on" if s.get("kv_streaming") else "off")
        kr = _num(s, "kv_resident", 1024, 1048576, "Positions kept in VRAM")
        args = set_flag(args, "--kv-resident", kr) if kr is not None and s.get("kv_streaming") else \
            drop_flag(args, "--kv-resident")
    out["args"] = args
    return out


def shown_env(env: dict | None) -> dict:
    """A config's environment as the page gets it: a secret's value (a name with key, token, secret or password in it,
    as the server's report leaves them out) masked - a save that sends the mask back keeps the config's value."""
    return {str(k): (MASK if SECRET.search(str(k)) and str(v) else str(v)) for k, v in (env or {}).items()}


def shown_settings(cfg: dict) -> dict:
    return {**settings_of(cfg), "env": shown_env(cfg.get("env"))}


def command(cfg: dict) -> dict:
    """What the config runs: the engine, its arguments and its environment (the page's preview, secrets masked)."""
    return {"exe": Path(str(cfg.get("exe") or "maya")).name, "args": [str(a) for a in cfg.get("args") or []],
            "env": shown_env(cfg.get("env"))}


# ------------------------------------------------------------------ the list
def config_paths(root: Path) -> list:
    """The installed models' configs, the most recently used first (saguaro.py's configs()).  Both name families
    (tools/config_names.py: saguaro-<model>.json, the older maya-<model>.json still found, a model under both names
    is the new one); beside them the page keeps its own files - <config>.shared-settings.json and a model's live
    Turbo sidecar <config>.turbo.json - and neither is a model."""
    try:
        return sorted(config_names.list_configs(root), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []


def read(path: Path) -> dict:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


def write(path: Path, cfg: dict):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def folder_size(pack: Path | None) -> int | None:
    """The model's files: its folder's (the GGUF shards beside the pack) and the pack's, one level each."""
    if pack is None:
        return None
    total = 0
    for d in (pack.parent, pack):
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total or None


def files_size(gguf: Path | None) -> int | None:
    """Inkling 2's files: its GGUF's folder, one level (the GGUF and the mmproj beside it)."""
    return _files_in(gguf.parent) or None if gguf is not None else None


def _files_in(d: Path) -> int:
    """The bytes of a folder's own files (one level, links not followed)."""
    try:
        with os.scandir(d) as it:
            return sum(e.stat(follow_symlinks=False).st_size for e in it if e.is_file(follow_symlinks=False))
    except OSError:
        return 0


def hy4_size(cfg: dict) -> int | None:
    """Hy4-Flash's files: the ones its pack indexes, wherever they sit in the model's folder, and the pack."""
    pack = pack_of(cfg)
    if pack is None:
        return None
    total = sum(_size(p) for p in pack_shards(pack)) + _files_in(pack)
    return total or folder_size(pack)


def quant_label(cfg: dict, path: Path) -> str:
    """The files a config runs, as shown after the model's name ("Maya-XS", "UD-Q8_K_XL", "EXL3 3.5 bpw" - the setup's
    names, maya_quants); else its installer key, or the config's own name for one the setup did not write."""
    fam, key = family(cfg), maya_quants.config_key(cfg)
    what = maya_quants.tier(fam, key) or maya_quants.OTHERS.get((fam, key)) or OTHER_QUANTS.get((fam, str(key).lower()))
    return str(what or key or config_names.bare(path.stem).removesuffix("-hip"))


def missing(cfg: dict) -> str | None:
    """What keeps a config from starting (saguaro.py stops at it - a switch to it would leave no model running): None when
    its engine and pack are named and the pack is on this PC."""
    if family(cfg) == "ik2":                           # (its GGUF and the tokenizer folder beside it)
        gguf, tok = model_file_of(cfg), cfg.get("tokenizer")
        if not cfg.get("exe") or gguf is None:
            return "its engine or model file is not set"
        if not gguf.is_file():
            return f"its model file is not on this PC ({gguf})"
        if not tok or not (Path(tok) / "vocab.json").is_file():
            return f"its tokenizer folder is not on this PC ({tok})"
        return None
    pack = pack_of(cfg)
    if not cfg.get("exe") or pack is None:
        return "its engine or pack is not set"
    if not pack.is_dir():
        return f"its model files are not on this PC ({pack.parent})"
    return None


def needs(cfg: dict) -> dict | None:
    """What the page says a Hy4-Flash or Inkling 2 config runs on and what it does not do yet: {"cards": how many it
    is set up for, "measured": its layout is one measured on Tesla T4s (the setup's record; None for a config the
    setup did not write: nothing claimed), "limits": plain lines}."""
    fam = family(cfg)
    if fam not in ("hy4", "ik2"):
        return None
    inst = cfg.get("installer") or {}
    if fam == "hy4":
        limits = ["one conversation at a time", "no saved conversations yet", "pictures not measured on Tesla T4s yet"]
    else:
        limits = ["several conversations at once only on one GPU that holds every expert"]
    measured = inst.get("measured")
    return {"cards": len(set(gpus_of(cfg))) or None, "measured": measured if isinstance(measured, bool) else None,
            "limits": limits}


def describe(path: Path, running: str | None) -> dict:
    cfg = read(path)
    pack = pack_of(cfg)
    gguf = model_file_of(cfg)
    why = missing(cfg)
    inst = cfg.get("installer") or {}
    script = next((s.name for s in (path.with_name(f"run-{path.stem}.sh"), path.with_name(f"run-{path.stem}.bat"))
                   if s.is_file()), None)
    try:
        used = path.stat().st_mtime
    except OSError:
        used = None
    cur = shown_settings(cfg)
    setup = dict(inst.get("as_set_up") or cur)
    if family(cfg) == "qfn":                           # (an as_set_up from before QFN's own field set: its CYBER-
        setup = {k: v for k, v in setup.items() if k in QFN_KEYS}   # FROST keys (ram_gb!) would be refused on Reset)
    setup["env"] = shown_env(setup.get("env"))
    name_pretty, name_quant = maya_quants.display_names(cfg)   # (the two parts a page names with, as /health says them:
                                                               #  the row says the model's name as the header does)
    return {"file": path.name, "path": str(path), "model": cfg.get("model_name") or "", "family": family(cfg),
            "quant": quant_label(cfg, path), "model_name_pretty": name_pretty, "model_quant": name_quant,
            "backend": cfg.get("backend") or "cuda", "written": inst.get("written"),
            "used": used, "running": running is not None and str(path.resolve()) == str(Path(running).resolve()),
            "port": cfg.get("port"), "script": script,
            "folder": str(pack.parent) if pack else str(gguf.parent) if gguf else None,
            "size": hy4_size(cfg) if family(cfg) == "hy4" else folder_size(pack) if pack else files_size(gguf),
            "ok": why is None, "missing": why,
            **({"needs": needs(cfg)} if needs(cfg) else {}),
            "settings": cur, "setup": setup, "choices": choices(cfg), "command": command(cfg),
            "contexts": {"deepseek": DS4_CONTEXTS, "minimax": MM3_CONTEXTS,
                         "qfn": QFN_CONTEXTS, "hy4": HY4_CONTEXTS, "ik2": IK2_CONTEXTS}.get(family(cfg), CONTEXTS),
            # (the Engine settings' Context size slider: the same range as Settings > Model's for this model)
            "context_range": context_range(trained_context_of(cfg), cur.get("context"),
                                           {"deepseek": "ds4", "minimax": "mm3", "hy4": "hy4",
                                            "ik2": "ik2", "qfn": "qfn"}.get(family(cfg), ""))}


_gpu_cache: dict = {}


def machine_gpus(backend: str = "cuda") -> list:
    """This PC's GPUs as the setup numbers them ([{index, name, vram_gb}]), asked once a minute."""
    now = time.time()
    hit = _gpu_cache.get(backend)
    if hit and now - hit[0] < 60:
        return hit[1]
    found = []
    try:
        import setup as SU                              # (the project's setup.py: nvidia-smi / the KFD topology)
        raw = SU.amd_gpus() if backend == "hip" else SU.gpus()
        found = [{"index": int(g["index"]), "name": g.get("name"), "vram_gb": g.get("vram_gb"),
                  **({"arch": g["arch"]} if g.get("arch") else {})} for g in raw or []]
    except (ImportError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        found = []
    _gpu_cache[backend] = (now, found)
    return found


def listing(root: Path, running: str | None) -> dict:
    models = [describe(p, running) for p in config_paths(root)]
    backend = next((m["backend"] for m in models if m["running"]), models[0]["backend"] if models else "cuda")
    gpus = machine_gpus(backend)
    seen = {g["index"] for g in gpus}
    for m in models:                                   # (a GPU a config names that the PC does not list now)
        for i in m["settings"]["gpus"]:
            if i not in seen:
                seen.add(i)
                gpus.append({"index": i, "name": None, "vram_gb": None})
    return {"models": models, "gpus": sorted(gpus, key=lambda g: g["index"]), "contexts": CONTEXTS}


def own_defaults(path: Path) -> bool:
    """A switch to the config at `path` (the Models page): its model's own sampling and thinking level
    (maya_quants.MODEL_DEFAULTS, as the setup writes them) written into it where they differ - a config from an older
    setup, or with another model's values - the one before kept beside it (<config>.prev).  True when it changed."""
    cfg = read(path)
    if not cfg:
        return False
    sampling, effort = maya_quants.model_defaults(cfg)
    budget = maya_quants.thinking_budget_for(cfg)      # (GLM's 8192 where none was set, or still the old 32768)
    if cfg.get("sampling") == sampling and cfg.get("reasoning_effort") == effort and budget is None:
        return False
    p = Path(path)
    p.with_name(p.name + ".prev").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    st = p.stat()
    write(p, {**cfg, "sampling": sampling, "reasoning_effort": effort,
              **({} if budget is None else {"thinking_budget": budget})})
    os.utime(p, (st.st_atime, st.st_mtime))
    return True


def _commit(path: Path, cfg: dict, new: dict) -> dict:
    """`new` written over the config at `path` (`cfg` as it was): its settings before kept beside it (<config>.prev), and
    the first time the page changes it, what the setup wrote (installer.as_set_up).  "changed": the engine's command or
    environment is not what it was - a restart is due (a Turbo marker alone is not one)."""
    engine = ("args", "env", "gpu", "vision", "vision_off", "parallel")
    changed = any(new.get(k) != cfg.get(k) for k in engine)
    if new == cfg:
        return {"changed": False}
    inst = new.setdefault("installer", {})
    if "as_set_up" not in inst:
        inst["as_set_up"] = settings_of(cfg)
    prev = Path(path).with_name(Path(path).name + ".prev")
    prev.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    st = Path(path).stat()
    write(path, new)
    os.utime(path, (st.st_atime, st.st_mtime))         # (its time says when it last ran: a start picks the latest)
    return {"changed": changed, "prev": prev.name}


def save(path: Path, settings: dict) -> dict:
    """The settings written into the config at `path` (its settings before kept beside it, <config>.prev, and the
    first time the page changes it, what the setup wrote: installer.as_set_up, the page's Reset)."""
    cfg = read(path)
    if not cfg:
        raise ValueError(f"{Path(path).name}: not a config Saguaro can read")
    return _commit(path, cfg, apply(cfg, settings))


# ------------------------------------------------------------------ Turbo (the page's Turbo button)
# The fastest settings the engine has, for people who play with speed and run benchmarks.  What the engine needs a start
# for is written into the config (the same "settings saved, Saguaro starts again" flow); the config keeps a "turbo" note:
# what it was before, so that switching Turbo off puts every one of these back.  The request-level half (greedy
# decoding, the fastest thinking level, coding prompts) lives in the page (serve/web/turbo-core.js).
TURBO_CONTEXT = 32768                               # GLM and MiniMax: a smaller context leaves VRAM to the experts ...
TURBO_CONTEXT_FAMILIES = ("glm", "minimax")         # ... DeepSeek's context is cheap, Qwen keeps its own
# Turbo trades quality for speed on purpose (benchmarks, playing with the engine): it goes past the safeguards.
# Expert skipping, where the engine has it (docs/SETTINGS.md): GLM's MISS_SKIP leaves out the light experts VRAM does not
# hold on layers 23 and up, at the highest value the docs measured (0.16: 30.9 vs 27.4 tok/s on Maya-S24, KL 0.075;
# _FROM stays 23 - the docs warn the early layers notice it far more); DeepSeek's RAM_SKIP the same rule for every
# expert VRAM does not hold, at the highest value documented (Maya-XS, DSpark: 27.8 -> 39.7 tok/s at 0.14).  MiniMax and
# Qwen have no skipping.  DeepSeek's DSPARK_CUM is lowered with the draft floor below (longer verified windows).
# Not used: SAGUARO_GLM_TRUNK_KEEP(_MAP) (its layer map is per model, a uniform value hurts the early layers),
# SAGUARO_*_DISK_SKIP_* (only for experts that fit neither VRAM nor RAM), SAGUARO_GLM_DFLASH_MIN/_CUM (meant for per-request
# tuning, no measured fast value).  The loop guard (SAGUARO_LOOP_GUARD) is deliberately NOT touched: it costs no speed and
# stops greedy runaway loops, which would ruin a benchmark.
# Where the whole model is in VRAM those skips have nothing to leave out, so Turbo also caps the routed experts a token
# computes, wherever they live (SAGUARO_*_RESIDENT_TOPK, docs/SETTINGS.md; the weight rule SAGUARO_*_RESIDENT_SKIP_W is not
# used: on the AMD Strix Halo test PC 0.20 already gave garbage, the rank cap degrades more gently).  The values are the
# fastest whose output was still sound there (a three.js page that parses with `node --check`, short factual answers):
# DeepSeek IQ1_M keeps 3 of its 6 experts a token (40.8 -> 49.5 tok/s with Turbo's other settings), GLM Maya-M-Derisked 4
# of its 8 (33.2 -> 38.8; 3 gave a page with a duplicate import and slower drafts).
TURBO_ENV = {"glm": {"SAGUARO_GLM_MISS_SKIP_W": "0.16", "SAGUARO_GLM_MISS_SKIP_FROM": "23", "SAGUARO_GLM_RESIDENT_TOPK": "4"},
             "deepseek": {"SAGUARO_DS4_RAM_SKIP_W": "0.14", "SAGUARO_DS4_DSPARK_CUM": "0.3", "SAGUARO_DS4_RESIDENT_TOPK": "3"}}
# the drafter that goes fastest where the pack has it: DSpark (DeepSeek, MiniMax), MTP (QFN).  GLM keeps the drafter the
# setup measured for this PC (MTP or DFlash2: which is faster depends on the GPUs).
TURBO_DRAFT = {"deepseek": "dspark", "minimax": "dspark", "qfn": "mtp"}
# Turbo's live half for engines whose engine-side levers are per-request rather than in the config: switching Turbo on
# (serve/server.py's set_turbo) puts the model's live Turbo (serve/turbo_live.py, the sidecar beside the config) on
# with these options and restores what it was when Turbo goes off.  QFN has no expert-skipping environment knobs:
# its lossy speed is turbo_drop / turbo_skip on every request the engine answers without its own Turbo.
TURBO_LIVE_DEFAULT = {}                             # (none: QFN's drop gains little and costs quality - it stays the user's own choice)
# DSpark's draft confidence floor (default 0.7), lowered so that longer windows are verified
TURBO_DSPARK_CONF = 0.3
TURBO_DSPARK_FAMILIES = ("deepseek", "minimax")
# The smallest quants (GLM and DeepSeek Maya-XS, experts at 1-2 bits) keep every expert a token routes to: on 3x V100 the
# cap gained at most 2-6% there, and GLM Maya-XS with 4 of 8 experts invented three.js classes and rambled.
TURBO_NO_TOPK = ("maya-xs",)
TURBO_NONE = ("ik2", "hy4")                         # no Turbo: neither engine skips experts; no drafter set up


def turbo_smallest_quant(cfg: dict) -> bool:
    """Whether cfg runs one of TURBO_NO_TOPK's quants (the setup's record of the quant, else the pack's folder)."""
    inst = cfg.get("installer") or {}
    names = [str(inst.get(k) or "") for k in ("ds4_quant", "quant", "model_dir")]
    args = cfg.get("args") or []
    names += [str(args[i + 1]) for i, a in enumerate(args[:-1]) if a in ("--pack", "--glm-pack")]
    return any(t in n.lower().replace("_", "-") for n in names for t in TURBO_NO_TOPK)


def turbo_patch(cfg: dict) -> dict:
    """The engine settings Turbo changes in `cfg` (only those that differ from what it has now), for apply()."""
    fam, cur, can = family(cfg), settings_of(cfg), choices(cfg)
    patch = {}
    if fam in TURBO_NONE:
        return patch
    if fam in TURBO_CONTEXT_FAMILIES and (cur.get("context") or 0) > TURBO_CONTEXT:
        patch["context"] = TURBO_CONTEXT
    if (cur.get("parallel") or 1) > 1:                 # (one answer at a time is the fastest one)
        patch["parallel"] = 1
    want = TURBO_DRAFT.get(fam)
    if want and want in can["draft"] and cur.get("draft") != want:
        patch["draft"] = want
    if fam in TURBO_DSPARK_FAMILIES and (want in can["draft"] or cur.get("draft") == "dspark")             and cur.get("dspark_conf") != TURBO_DSPARK_CONF:
        patch["dspark_conf"] = TURBO_DSPARK_CONF
    if fam == "deepseek" and cur.get("prompt_lookup") is False:
        patch["prompt_lookup"] = True
    env = TURBO_ENV.get(fam)
    if env and turbo_smallest_quant(cfg):              # (no expert cap on the 1-2 bit quants: see TURBO_NO_TOPK)
        env = {k: v for k, v in env.items() if not k.endswith("_RESIDENT_TOPK")}
    if env and any(cur["env"].get(k) != v for k, v in env.items()):
        patch["env"] = {**cur["env"], **env}
    return patch


def turbo_held(patch: dict) -> bool:
    """Whether `patch` (turbo_patch's) is only "one answer at a time": the server holds that itself, no restart."""
    return set(patch) == {"parallel"}


def turbo_one_at_a_time(cfg: dict) -> bool:
    """Whether Turbo is on in `cfg` with one answer at a time held by the server (turbo_held)."""
    return turbo_on(cfg) and bool(cfg["turbo"].get("one_at_a_time"))


def turbo_on(cfg: dict) -> bool:
    return isinstance(cfg.get("turbo"), dict)


def turbo_info(cfg: dict) -> dict:
    """What the page needs: whether Turbo is on, and what switching it on would change (before -> after)."""
    cur = settings_of(cfg)
    if family(cfg) in TURBO_NONE and not turbo_on(cfg):   # ("supported": the page shows no Turbo button)
        return {"on": False, "family": family(cfg), "plan": {}, "env": {}, "supported": False}
    if turbo_on(cfg):
        n = cfg["turbo"]                                # ("off_restarts": switching off changes the engine's start)
        back = any((n.get(k) or None) != (cfg.get(k) or None) for k in ("args", "env", "parallel"))
        return {"on": True, "family": family(cfg), "since": n.get("since"), "plan": {}, "off_restarts": back}
    patch = turbo_patch(cfg)
    if turbo_held(patch):
        return {"on": False, "family": family(cfg), "plan": {}, "env": {}, "one_at_a_time": True}
    return {"on": False, "family": family(cfg),
            "plan": {k: {"from": cur.get(k), "to": v} for k, v in patch.items() if k != "env"},
            "env": {k: v for k, v in (patch.get("env") or {}).items() if cur["env"].get(k) != v}}


def turbo_enable(path: Path, live_before: dict | None = None) -> dict:
    """Turbo on: its settings written into the config at `path`, and the note that undoes them.  "changed": the engine
    must start again."""
    cfg = read(path)
    if not cfg:
        raise ValueError(f"{Path(path).name}: not a config Saguaro can read")
    if turbo_on(cfg):
        raise ValueError("Turbo is already on")
    if family(cfg) in TURBO_NONE:
        raise ValueError(f"Turbo: {'Hy4-Flash' if family(cfg) == 'hy4' else 'Inkling 2'}'s engine has none (no expert "
                         "skipping, one answer at a time)")
    cur, patch = settings_of(cfg), turbo_patch(cfg)
    held = turbo_held(patch)                           # (the server admits one answer at a time: nothing to start again)
    if held:
        patch = {}
    new = apply(cfg, patch) if patch else copy.deepcopy(cfg)
    new["turbo"] = {"since": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "before": {k: cur.get(k) for k in patch},            # (the settings Turbo changed, as they were)
                    "args": list(cfg.get("args") or []), "env": cfg.get("env"),   # (and the command and environment whole)
                    "parallel": cfg.get("parallel"),   # (... with QFN's Answers-at-once key, a config key like these)
                    "applied": {"args": list(new.get("args") or []), "env": new.get("env")}}
    if held:
        new["turbo"]["one_at_a_time"] = True
    if live_before:
        new["turbo"]["live_before"] = dict(live_before)   # (the live Turbo as it was: kept across the restart)
    return _commit(path, cfg, new)


def turbo_disable(path: Path) -> dict:
    """Turbo off: the config as it was before.  The command and environment come back whole when nothing else changed
    them since; if they were changed meanwhile (the context raised, a setting edited), only the settings Turbo changed go
    back to what they were."""
    cfg = read(path)
    if not cfg:
        raise ValueError(f"{Path(path).name}: not a config Saguaro can read")
    note = cfg.get("turbo")
    if not isinstance(note, dict):
        raise ValueError("Turbo is not on")
    applied = note.get("applied") or {}
    if list(cfg.get("args") or []) == applied.get("args") and cfg.get("env") == applied.get("env"):
        new = copy.deepcopy(cfg)
        new["args"] = list(note.get("args") or [])
        if note.get("env"):
            new["env"] = note["env"]
        else:
            new.pop("env", None)
        if "parallel" in note:                           # (QFN's Answers at once: a config key Turbo took out)
            if note["parallel"] is None:
                new.pop("parallel", None)
            else:
                new["parallel"] = note["parallel"]
    else:
        new = apply(cfg, dict(note.get("before") or {}))
    new.pop("turbo", None)
    return _commit(path, cfg, new)


# ------------------------------------------------------------------ deleting a model (the page's Delete)
# Off the list: its config, its start scripts and what Saguaro keeps beside it.  With its files: what the setup downloaded
# for it and the pack Saguaro built - never a file another model here runs from, nor one outside the models folder the
# setup downloaded into (a GGUF of your own stays; its pack goes).  Its DFlash2 drafter goes too: another model drafting
# with the same file drafts with MTP from then on (its --dflash taken out).
GGUF_PART = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)


def _res(p) -> Path | None:
    try:
        return Path(p).expanduser().resolve() if p else None
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def _within(p: Path, d: Path) -> bool:
    """`p` is `d` or in it."""
    return p == d or d in p.parents


def drafter_of(cfg: dict) -> Path | None:
    """The DFlash2 drafter a config drafts with, or the one its setup downloaded for it (installer.draft)."""
    p = flag(cfg.get("args") or [], "--dflash")
    if p:
        return Path(p)
    return dflash_file(cfg) if (cfg.get("installer") or {}).get("draft") == "dflash2" else None


def vision_files(cfg: dict) -> list:
    """GLM's image encoder: its mmproj and vocab files (on or switched off)."""
    out = []
    for key in ("vision", "vision_off"):
        v = cfg.get(key)
        if isinstance(v, dict):
            out += [Path(v[k]) for k in ("mmproj", "model") if isinstance(v.get(k), str) and v[k]]
    return out


def pack_shards(pack: Path) -> list:
    """The model files a pack indexes, in its folder: its sources.json's list, else the GGUF its native_experts.txt
    names (every part of a split one)."""
    names = []
    try:
        rec = json.loads((pack / "sources.json").read_text(encoding="utf-8")).get("files") or []
        names = [r["name"] for r in rec if isinstance(r, dict) and isinstance(r.get("name"), str)]
    except (OSError, ValueError, AttributeError):
        names = []
    if not names:
        try:
            with open(pack / "native_experts.txt", encoding="utf-8") as f:
                m = re.search(r"absolute offsets in ([^,)]+)", f.readline())
        except OSError:
            m = None
        first = Path(m.group(1).strip()).name if m else None
        part = GGUF_PART.match(first) if first else None
        names = ([f"{part.group(1)}-{i:05d}-of-{part.group(3)}.gguf" for i in range(1, int(part.group(3)) + 1)]
                 if part else [first] if first else [])
    inside = hy4_pack_files(pack)
    if inside:
        return inside
    return [pack.parent / Path(n).name for n in names]   # (by name: nothing outside the pack's folder)


def hy4_pack_files(pack: Path) -> list:
    """The model files a Hy4-Flash pack's tensors.txt reads (its `file N <path>` lines, relative to the model's
    folder): only those inside that folder."""
    if not (pack / "hy4.txt").is_file():
        return []
    out = []
    try:
        with open(pack / "tensors.txt", encoding="utf-8", errors="replace") as f:
            for ln in f:
                if not ln.startswith("file "):
                    if ln.startswith("t "):
                        break
                    continue
                rel = Path(ln.split(None, 2)[2].strip())
                if not rel.is_absolute() and ".." not in rel.parts:
                    out.append(pack.parent / rel)
    except (OSError, IndexError):
        return []
    return out


def runs_from(cfg: dict) -> list:
    """Every path a config runs from: its pack and the model files in it, its drafter, image encoder and tokenizer, and
    the folder or GGUF its setup named."""
    pack = pack_of(cfg)
    found = ([pack] + pack_shards(pack)) if pack else []
    inst = cfg.get("installer") or {}
    found += [Path(inst[k]) for k in ("model_dir", "gguf", "gguf_dir") if isinstance(inst.get(k), str) and inst[k]]
    found += [Path(v) for v in (flag(cfg.get("args") or [], "--dflash"), flag(cfg.get("args") or [], "--mtp"),
                                cfg.get("tokenizer")) if isinstance(v, str) and v]
    if family(cfg) == "ik2":                           # (saguaro-ik2 runs from its GGUF itself: no pack)
        found += [Path(v) for v in (flag(cfg.get("args") or [], "--model"),) if v]
    found += vision_files(cfg)
    return [r for r in map(_res, found) if r is not None]


def _size(p: Path) -> int:
    try:
        st = os.lstat(p)
    except OSError:
        return 0
    if not stat.S_ISDIR(st.st_mode):
        return st.st_size if stat.S_ISREG(st.st_mode) else 0
    total = 0
    for dp, _, files in os.walk(p):                     # (symlinked folders: not followed)
        for n in files:
            try:
                st = os.lstat(os.path.join(dp, n))
            except OSError:
                continue
            # (a hard link another model's folder also holds - the Engram tables the setup links between folders - frees
            # nothing when this one goes: the data stays with the last link, so it is not counted)
            total += st.st_size if stat.S_ISREG(st.st_mode) and st.st_nlink <= 1 else 0
    return total


def _links_into(roots: list, p: Path) -> bool:
    """Whether a symbolic link under one of `roots` (another model's folders) points into `p`: deleting `p` would leave
    it dangling (the setup links the Engram tables between model folders when it cannot hard link them)."""
    seen = 0
    for r in roots:
        if not r.is_dir():
            continue
        for dp, dirs, files in os.walk(r):
            for n in dirs + files:
                seen += 1
                f = os.path.join(dp, n)
                if os.path.islink(f) and _within(_res(f) or Path(f), p):
                    return True
            if seen > 20000:
                return False
    return False


def delete_plan(root: Path, path: Path) -> dict:
    """What deleting the config at `path` with its files would delete: {"ok", "why" (when not ok), "items": [{"path",
    "what" (folder, pack, gguf, vision, drafter), "bytes"}], "bytes", "folder", "gguf" (model files), "pack", "vision",
    "drafter" (its bytes, or None), "drafter_users" (the other configs drafting with it), "kept" (what stays, said)}."""
    path = Path(path)
    cfg = read(path)
    me = _res(path)
    others = [(p, read(p)) for p in config_paths(root) if _res(p) != me]
    refs = [r for _, c in others for r in runs_from(c)]
    homes = [h.parent for h in (_res(home_of(c)) for _, c in others) if h is not None]   # (their folders: links in them)
    out = {"ok": False, "why": None, "items": [], "bytes": 0, "folder": None, "gguf": 0, "pack": False,
           "vision": False, "drafter": None, "drafter_users": [], "kept": []}
    models = _res((cfg.get("installer") or {}).get("models_dir"))
    pack = _res(home_of(cfg))
    if models is None or pack is None:
        out["why"] = "Saguaro's setup did not download this model's files, so it does not know which ones are its own"
        return out
    folder = pack.parent
    out["folder"] = str(folder)
    project = _res(Path(root))
    items: list = []

    def shared(p: Path) -> bool:                        # (another model runs from it, or from something in it, or links to it)
        return any(_within(r, p) for r in refs) or _links_into(homes + refs, p)

    def covered(p: Path) -> bool:
        return any(_within(p, it) for it, _ in items)

    def add(p: Path, what: str):
        if not covered(p) and (p.exists() or p.is_symlink()):
            items.append((p, what))

    downloaded = _within(folder, models)
    if (downloaded and folder != models and not shared(folder) and not _within(project, folder)
            and not _within(Path.home().resolve(), folder)):
        add(folder, "folder")                           # (its own folder: the files, the pack, the image encoder)
    else:                                               # (a folder other models' files are in too, or your own)
        if not shared(pack):
            add(pack, "pack")
        if downloaded:
            for f in pack_shards(pack):
                if not shared(f):
                    add(f, "gguf")
        else:
            out["kept"].append(f"Your own model files in {folder} stay; only the pack Saguaro built from them goes.")
    for v in map(_res, vision_files(cfg)):
        if v is not None and _within(v, models) and not shared(v):
            add(v, "vision")
    d = _res(drafter_of(cfg))
    if d is not None and d.is_file() and _within(d, models):
        out["drafter_users"] = [p.name for p, c in others if _res(flag(c.get("args") or [], "--dflash")) == d]
        add(d, "drafter")
    for p, what in items:
        n = _size(p)
        out["items"].append({"path": str(p), "what": what, "bytes": n})
        out["bytes"] += n
        if what == "drafter":
            out["drafter"] = n
        elif what == "vision":
            out["vision"] = True
        elif what == "pack":
            out["pack"] = True
        elif what == "gguf":
            out["gguf"] += 1
        else:                                           # (its folder: the GGUF files in it, the pack, the encoder)
            try:
                out["gguf"] += sum(1 for e in p.iterdir() if e.suffix.lower() == ".gguf" and e.is_file())
                out["pack"] = out["pack"] or pack.is_dir()
                out["vision"] = out["vision"] or any(v is not None and _within(v, p) and v.exists()
                                                     for v in map(_res, vision_files(cfg)))
            except OSError:
                pass
    out["ok"] = True
    return out


def delete(root: Path, path: Path, files: bool = False) -> dict:
    """The config at `path` off the list: it, its start scripts, the settings kept beside it (.prev,
    .shared-settings.json), and its old-name shadow if one sits there (a hand-made pair: left, the next start's
    rename would move it back and resurrect the deleted model); with `files`, delete_plan's files first (a file
    that cannot be deleted keeps the config, and says which).  {"deleted", "freed" (bytes), "drafter_users"
    (now drafting with MTP)}."""
    path = Path(path)
    freed, users, errors = 0, [], []
    if files:
        plan = delete_plan(root, path)
        if not plan["ok"]:
            raise ValueError(plan["why"])
        for it in plan["items"]:
            p = Path(it["path"])
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                freed += it["bytes"]
            except FileNotFoundError:
                pass
            except OSError as e:
                errors.append(f"{p} ({e.strerror or e})")
                continue
            if it["what"] == "drafter":
                try:
                    p.parent.rmdir()                    # (its folder, when nothing else is in it)
                except OSError:
                    pass
                for name in plan["drafter_users"]:      # (another model drafting with it: with MTP from now on)
                    other = Path(root) / name
                    cfg = read(other)
                    if not cfg:
                        continue
                    other.with_name(other.name + ".prev").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
                    st = other.stat()
                    write(other, {**cfg, "args": drop_flag(cfg.get("args") or [], "--dflash")})
                    os.utime(other, (st.st_atime, st.st_mtime))
                    users.append(name)
        if errors:
            raise OSError("could not delete " + "; ".join(errors) + " - the model stays on the list")
    stale = [path, path.with_name(f"run-{path.stem}.sh"), path.with_name(f"run-{path.stem}.bat"),
             path.with_name(path.name + ".prev"), path.with_name(path.stem + ".shared-settings.json")]
    if path.name.startswith(config_names.NEW):         # (its maya- shadow, the other half of a hand-made pair)
        shadow = path.with_name(config_names.OLD + path.stem[len(config_names.NEW):] + ".json")
        stale += [shadow, shadow.with_name(f"run-{shadow.stem}.sh"), shadow.with_name(f"run-{shadow.stem}.bat"),
                  shadow.with_name(shadow.name + ".prev"), shadow.with_name(shadow.stem + ".shared-settings.json")]
    for p in stale:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    return {"deleted": path.name, "freed": freed, "files": bool(files), "drafter_users": users}


# ------------------------------------------------------------------ files no model here runs from
# What the models folders hold that no config here names: a download whose config was deleted by hand or by an older
# version, a quant set up again into another folder.  Only what is in a folder the setup downloaded into: at its top a
# folder or a file of 100 MB or more (a download's log is not listed), and one level down in a folder another model
# runs from only a folder with a pack of its own (GLM's quants side by side in GLM-5.3-Flash/: DeepSeek's dspark/ and
# vision/ beside its GGUF folder are read through the pack, so a folder without one is never taken for unused).  Never
# a hidden one, the project or the home folder.  Saguaro may not have downloaded it: the page says what is in it, and
# deletes one at a time.
LEFTOVER_MIN = 10 ** 8
PACK_MARKS = ("sources.json", "tensors.txt", "mm3.txt", "ds4.txt", "native_experts.txt", "index.txt")


def models_dirs(root: Path) -> list:
    """The folders the setup downloaded into: each config's, and the setup's own (Saguaro-data/models beside the project,
    or an earlier setup's Maya-data/models)
    - not the home folder, nor one the project is in."""
    project, home = _res(root), Path.home().resolve()
    found = [_res((read(p).get("installer") or {}).get("models_dir")) for p in config_paths(root)]
    found += [_res(Path(root).parent / d / "models") for d in ("Saguaro-data", "Maya-data")]
    out = []
    for d in found:
        if d is None or d in out or not d.is_dir() or _within(home, d) or (project and _within(project, d)):
            continue
        out.append(d)
    return out


def _has_pack(p: Path) -> bool:
    """A pack Saguaro built is in `p` (or `p` is one), two folders deep at most."""
    try:
        return any(next(p.glob(pat + m), None) is not None for pat in ("", "*/", "*/*/") for m in PACK_MARKS)
    except OSError:
        return False


def leftovers(root: Path) -> dict:
    """{"folders": the models folders, "items": [{"path", "name", "bytes", "folder" (a folder, else a file), "pack" (a
    pack Saguaro built is in it), "hub" (downloaded from the Hugging Face Hub), "modified"}], "bytes"}: what no model here
    runs from, largest first."""
    project, home = _res(root), Path.home().resolve()
    refs = [r for p in config_paths(root) for r in runs_from(read(p))]
    dirs, items = models_dirs(root), []

    def entries(d: Path) -> list:
        try:
            return [e for e in d.iterdir() if not e.name.startswith(".") and not e.is_symlink()]
        except OSError:
            return []

    def add(e: Path):
        n = _size(e)
        if n >= LEFTOVER_MIN:
            items.append({"path": str(e), "name": e.name, "bytes": n, "folder": e.is_dir(),
                          "pack": e.is_dir() and _has_pack(e), "hub": (e / ".cache" / "huggingface").is_dir(),
                          "modified": e.stat().st_mtime})

    for d in dirs:
        for e in entries(d):
            try:
                if not (e.is_dir() or e.is_file()) or (project and _within(project, e)) or _within(home, e):
                    continue
                if not any(_within(r, e) for r in refs):
                    add(e)
                elif e.is_dir() and not any(_within(e, r) for r in refs):   # (another model runs from a part of it)
                    for c in entries(e):
                        if c.is_dir() and not any(_within(r, c) for r in refs) and _has_pack(c):
                            add(c)
            except OSError:
                continue
    items.sort(key=lambda i: -i["bytes"])
    return {"folders": [str(d) for d in dirs], "items": items, "bytes": sum(i["bytes"] for i in items)}


def delete_leftover(root: Path, path) -> dict:
    """One of leftovers()'s items deleted, checked again first (a model set up since may run from it now).
    {"deleted", "freed" (bytes)}."""
    item = next((i for i in leftovers(root)["items"] if i["path"] == str(path or "")), None)
    if item is None:
        raise ValueError(f"{path} is not a file no model uses here (gone, or a model runs from it now)")
    p = Path(item["path"])
    if item["folder"]:
        shutil.rmtree(p)
    else:
        p.unlink()
    return {"deleted": str(p), "freed": item["bytes"]}
