"""QFN's drafter check: does the Lamina MTP layer make THIS PC faster?  (saguaro.py's --calibrate and
--speed-calibration on a Qwen3.8-Flash-Next config.)

The server always loads the Lamina draft layer - the folder the config's --mtp names (tools/qfn/download.json's
"mtp"); its drafts are turned off with --mtp-max-t 1 (windows of one token, prompt-lookup drafts only).  Whether
the drafts pay off depends on the PC: how much of the model the GPUs' memory holds, and on the CPU, the RAM and
the SSD.  So it is measured here, like CYBER-FROST's MTP check and DeepSeek's DSpark check (calibrate_ds4.py), on
the same harness (calibrate_draft.py: the same warm-up and questions, the sustained speed):

  - the config twice: drafting ON (its own --mtp args as they stand) and OFF (--mtp-max-t 1) - one engine start
    each through QFN's batch driver (serve/qfn_engine.py, the way serve/server.py's start_qfn starts it);
  - first a warm-up that is not counted, then fresh questions about a document, each answered at length, until
    each side has written MIN_TOKENS tokens: its speed is all those tokens over all the time spent writing them;
  - the drafts are kept only when they win by more than MIN_GAIN - a tie keeps decoding without them;
  - the engine reads a copy of this PC's expert profile and writes back to the copy only: the calibration's
    answers do not lead the next start's placement.

    python tools/qfn_calibrate.py saguaro-qwen3.8-flash-next-ud-q4_k_xl.json    # measure and print; ./saguaro.sh
                                                                             # --speed-calibration applies it
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_draft as CD  # noqa: E402  (requests, measure: the same warm-up and questions as the others')

MIN_GAIN = 0.03            # the drafts must beat no-drafting by this much (CYBER-FROST's MTP check uses 3% too)
MIN_TOKENS = 1024          # tokens each side writes after its warm-up (the DSpark check's number)
DOC_TOKENS = 512           # the document's length (each question reads it again: kept short)


def option(args: list, name: str) -> str | None:
    return args[args.index(name) + 1] if name in args[:-1] else None


def without(args: list, name: str) -> list:
    out = list(args)
    while name in out[:-1]:
        i = out.index(name)
        del out[i:i + 2]
    return out


def mtp_dir(cfg: dict) -> Path | None:
    """The Lamina draft layer this config drafts with: its own --mtp folder (the setup downloads it into its
    models folder; the live configs name it by hand - ./ref/*.json).  None: no --mtp in its args at all."""
    args = [str(x) for x in (cfg.get("args") or [])]
    d = option(args, "--mtp")
    return Path(d) if d else None


def has_drafter(cfg: dict) -> bool:
    """The config's --mtp folder on this PC with its files (dense.txt is the layer's index - every Lamina
    download of it carries it, tools/qfn/download.json)."""
    d = mtp_dir(cfg)
    return d is not None and (d / "dense.txt").is_file()


def with_drafting(cfg: dict, on: bool) -> dict:
    """The config drafting with the Lamina layer (on: its --mtp args as they stand) or without its drafts (off:
    --mtp-max-t 1 - the drafter stays loaded, its windows hold one token, only prompt-lookup drafts run)."""
    out = copy.deepcopy(cfg)
    args = [str(x) for x in (out.get("args") or [])]
    args = without(args, "--mtp-max-t")
    if not on:
        args += ["--mtp-max-t", "1"]
    out["args"] = args
    return out


def profile_copy(cfg: dict, tmp: Path) -> dict:
    """The config reading (and saving back) a copy of this PC's expert profile in `tmp`: the calibration's
    answers neither lead the next start's placement nor overwrite what the runs learned."""
    out = copy.deepcopy(cfg)
    args = [str(x) for x in (out.get("args") or [])]
    if not any(f in args[:-1] for f in ("--expert-profile", "--expert-profile-save")):
        return out
    dst = tmp / "expert-profile.bin"
    src = option(args, "--expert-profile")
    if src and src != "none" and Path(src).is_file() and not dst.exists():
        shutil.copyfile(src, dst)
    for flag in ("--expert-profile", "--expert-profile-save"):
        if flag in args[:-1]:
            args = without(args, flag) + [flag, str(dst)]
    out["args"] = args
    return out


def decide(off: float, on: float, min_gain: float = MIN_GAIN) -> str:
    """Which side the measurements keep: "on" when the drafts beat no-drafting by more than `min_gain`, else
    "off" - a tie keeps decoding without drafts, as the other engines' speed calibrations do."""
    return "on" if off > 0 and on > off * (1.0 + min_gain) else "off"


def start_qfn_engine(cfg: dict):
    """`start_engine` for both calibration harnesses: calibrate_draft's hands it the config of the side it measures,
    calibrate.py's the args list; either way it starts QFN the way its server does (serve/server.py's start_qfn):
    the batch driver serve/qfn_engine.py, its engine_args (the config's, the layer split, --batch from "parallel",
    --pool-cores, the draft settings) and its child_env."""

    def start(side):
        c = dict(side) if isinstance(side, dict) else {**cfg, "args": list(side)}
        c = {**c, "args": [str(x) for x in c.get("args") or []]}
        from serve import qfn_engine as QE
        return QE.StrataEngine(c["exe"], QE.engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=QE.child_env(c))

    return start


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Both sides measured, without the drafts first: {"off": {...}, "on": {...}, "keep": "on" or "off", "gain":
    the drafts' speed over no-drafting's - 1}.  `start_engine(cfg)` returns a started engine (a stand-in in
    tests).  Raises when a side could not be measured."""
    if start_engine is None:
        start_engine = start_qfn_engine(cfg)
    saved = CD.MIN_TOKENS, CD.DOC_TOKENS
    CD.MIN_TOKENS, CD.DOC_TOKENS = MIN_TOKENS, DOC_TOKENS
    res: dict = {}
    try:
        warm, questions, n_doc = CD.requests(cfg)
        for i, (key, on, label) in enumerate((("off", False, "without drafting"), ("on", True, "the Lamina layer"))):
            with tempfile.TemporaryDirectory(prefix="qfn-speed-") as tmp:   # (each side from the PC's own profile)
                c = profile_copy(with_drafting(cfg, on), Path(tmp))
                res[key] = CD.measure(c, label, warm, questions, n_doc, start_engine, say, f"[{i + 1}/2] ")
    finally:
        CD.MIN_TOKENS, CD.DOC_TOKENS = saved
    for key, label in (("off", "without drafting"), ("on", "the Lamina layer")):
        if res[key]["tokens"] < MIN_TOKENS // 2:
            raise RuntimeError(f"{label} wrote only {res[key]['tokens']} tokens: too few for a sustained speed")
    o, d = res["off"]["tok_s"], res["on"]["tok_s"]
    res["keep"] = decide(o, d)
    res["gain"] = round(d / o - 1.0, 3) if o > 0 else None
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: qfn_calibrate.py <saguaro-qwen3.8-flash-next-*.json> (older maya- names work too)")
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    if mtp_dir(cfg) is None:
        sys.exit("this config has no --mtp folder: it starts without the Lamina drafter")
    print(json.dumps(run(cfg), indent=1))
