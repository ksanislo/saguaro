"""serve/server.py - plan v0.3 P8: OpenAI and Anthropic endpoints over any engine that maps token ids to tokens.

    python -m serve.server --engine mock --port 8095            (a scripted engine, for clients and tests)
    python -m serve.server --config saguaro.json --port 8080    (the real engine, resident: the config's exe picks the
                                                                 driver - engine_kind: the batch driver for a QFN
                                                                 config, the line driver for every other engine)

Endpoints: POST /v1/chat/completions and /v1/responses (OpenAI, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), POST /tokenize and /detokenize, GET /v1/models, GET /models, GET /props, GET /slots, GET /health, GET /mcp. One sequence at a time behind a FIFO (plan: one resident sequence).
Tools from MCP servers (serve/mcp.py, `"mcp_servers"` in the config or --mcp-config) are offered only to requests that
ask for them with `"strata_mcp": true` - the web app does; other clients see exactly the API they always saw.
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths) go through `vision-glm` (the model's mmproj file) and reach the engine as
embeddings (`GENI`).  JPEG/PNG/BMP/GIF go straight in; WebP, TIFF, AVIF, ... (agents like omp send WebP) are
converted to PNG first with Pillow.  DeepSeek-V4.1 (maya-ds4, `INFO vision=ds4`) runs its vision tower in the engine:
the server only resizes the pictures (serve/ds4_vision.py) and sends their pixels with `GENI`; so does
MiniMax-M3.1-Flash (maya-mm3 with --vision on, `INFO vision=mm3`, serve/mm3_vision.py).
Requests whose prompt plus max tokens exceed the engine's context are REJECTED with 400, never truncated.
An unset (or 0, or -1) max tokens means "unlimited": whatever the prompt leaves of the context.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `saguaro-glm --serve` process resident (weights, expert arena and VRAM tier load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.
"""
from __future__ import annotations

def engine_kind(cfg: dict) -> str | None:
    """The driver a config needs: "batch" for a Qwen3.8-Flash-Next config - its engine is the QFN batch binary
    (engines/qfn: saguaro-qfn, driven by serve/qfn_engine.py's batch protocol) - and "line" for every
    other engine config (today's maya engine, the one-token-at-a-time line protocol); None with no config.  The
    signal is the engine binary: only the batch driver's name ends in qfn; the configs' exe is what runs the model."""
    exe = str(cfg.get("exe") or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    if exe.endswith(".exe"):
        exe = exe[:-4]
    return "batch" if exe.endswith("qfn") else ("line" if exe else None)


# ---- a Qwen3.8-Flash-Next config (engine_kind: batch) runs its driver in THIS server (serve/qfn_engine.py: QFN's own
# batch-protocol driver, extracted from engines/qfn by tools/saguaro/extract_qfn_engine.py).  (The old SAGUARO_QFN_SHIM
# path, an os.execv into engines/qfn/serve/server.py, is gone: that server no longer imports here, and every start of
# a batch config has run the extracted driver since it was written.)
import argparse
import atexit
import collections
import contextlib
import dataclasses
import base64
import hashlib
import hmac
import codecs
import itertools
import json
import math
import os
import queue
import socket
import re
import shutil
import subprocess
import struct
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
import socketserver
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))   # run as a script (run-<model>.bat) as well as a module
from serve import ds4_vision, gpu_map, hy4_vision, logctx, loop_guard, mm3_vision, model_api, model_pull, models, savings, turbo_live  # noqa: E402

ENGINE_VISION = (ds4_vision.Vision, mm3_vision.Vision)   # the engines that run their vision tower themselves
#                                                          (hy4_vision.Vision is an mm3_vision.Vision)
HY4_TEXT_ONLY = "Hy4-Flash with its pictures off"         # (no vision tower loaded: the errors' name for it)
# the engines without the live per-request Turbo (saguaro-ik2 stops at its keys, saguaro-hy4 ignores them): exact
NO_LIVE_TURBO = {"ik2": "Inkling 2", "hy4": "Hy4-Flash"}
# saguaro-hy4's per-PC files where the config's env names none (child_env): the CPU lane's 4-bit copies' verdict,
# the prompt route's rates, --drafting auto's choice
HY4_LANE_FILE = ROOT / "data" / "hy4-cpu-lane.txt"
HY4_ROUTE_FILE = ROOT / "data" / "hy4-prompt-route.txt"
HY4_DRAFT_FILE = ROOT / "data" / "hy4-drafting.txt"
HY4_MEASURE_DOCS = ("docs/SETTINGS.md", "README.md")      # (the text its first start measures them on)
HY4_MEASURE_CHARS = 200_000
# saguaro-ik2's --drafting auto: this PC's choice (SAGUARO_IK2_DRAFT_FILE) and the requests it is timed on
IK2_DRAFT_FILE = ROOT / "data" / "ik2-drafting.txt"
# what the engines write while they run (not in git): saguaro-ik2's drafting requests, its AMD GEMM picks
VAR_DIR = ROOT / "var"
IK2_DRAFT_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "The current weather in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city's name."}},
                   "required": ["city"]}}}
IK2_DRAFT_QS = ("Explain in a few paragraphs how a refrigerator moves heat out of its inside.",
                "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
                "What is the weather in Paris and in Rome right now? Use the tool for each city.")


def hy4_hot_file(cfg: dict) -> Path:
    """saguaro-hy4's HY4_HOT scores for this config's model (each layer's experts a start places in VRAM): a file a
    model in the data folder, named by the model's folder."""
    args = cfg.get("args") or []
    pack = Path(args[args.index("--pack") + 1]) if "--pack" in args[:-1] else None
    name = "" if pack is None else pack.parent.name if pack.name in ("saguaro-hy4", "maya-hy4", "pack") else pack.name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "model"
    return ROOT / "data" / f"hy4-expert-hot-{name}.bin"


HY4_HOT_SEEDS = ROOT / "data" / "hy4-hot-seed"            # (the scores shipped, a file a size: hy4_hot_seed)
#                                                          (Maya-S and -L: Maya-XS's copied, until learned on their own packs)


def hy4_hot_seed(cfg: dict) -> Path | None:
    """The HY4_HOT scores shipped for this config's model - data/hy4-hot-seed/hy4-expert-hot-<folder>.bin, the folder's
    name as hy4_hot_file has it (a layout's folder, Hy4-Flash-Maya-XS-8GB, takes its size's) - for a PC that has no
    file of its own yet: its first start then places the experts as every start after it.  None: no such set."""
    name = hy4_hot_file(cfg).name
    stem = name[len("hy4-expert-hot-"):-len(".bin")]
    size = re.match(r"(Hy4-Flash-Maya-[A-Za-z0-9]+)-", stem)
    for s in dict.fromkeys([stem] + ([size.group(1)] if size else [])):
        p = HY4_HOT_SEEDS / f"hy4-expert-hot-{s}.bin"
        if p.is_file():
            return p
    return None


def hy4_measure_ids(cfg: dict) -> Path | None:
    """The text saguaro-hy4 measures this PC's prompt route on at its first start here (HY4_MEASURE_IDS), before it
    says READY: this install's docs in the model's tokens, written once into the data folder.  None: no tokenizer,
    or the file cannot be written - the engine then measures while it answers."""
    tdir = Path(cfg.get("tokenizer") or "")
    vocab = tdir / "vocab.json"
    if not cfg.get("tokenizer") or not vocab.is_file():
        return None
    try:
        tag = hashlib.md5(vocab.read_bytes()).hexdigest()[:10]
    except OSError:
        return None
    path = ROOT / "data" / f"hy4-measure-ids-{tag}.txt"
    if path.is_file():
        return path
    try:
        import strata_tokenizer as ST
        text = ""
        for name in HY4_MEASURE_DOCS:
            p = ROOT / name
            if p.is_file() and len(text) < HY4_MEASURE_CHARS:
                text += p.read_text(encoding="utf-8", errors="replace")[:HY4_MEASURE_CHARS - len(text)] + "\n\n"
        ids = ST.Tokenizer.from_dir(tdir).encode(text)
        if len(ids) < 8192:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")   # (per process: two servers never share the tmp path)
        tmp.write_text(f"# saguaro: {', '.join(HY4_MEASURE_DOCS)} in the model's tokens ({len(ids)}) - the text "
                       "saguaro-hy4 measures a PC's prompt route on at its first start there\n" +
                       ",".join(map(str, ids)) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path
    except Exception:  # noqa: BLE001 - no text: the engine measures while it answers
        return None


HY4_PICTURE_NOTE = "[a picture was here; pictures are off on this server, so it was not read]"


def hy4_tool_pictures_noted(messages: list) -> list:
    """Hy4-Flash with no vision tower loaded: the pictures (and videos) in tool results as a short text note, so the
    agent's turn goes on instead of failing at every retry.  A user's picture is left: the 400 that says why."""
    out = []
    for m in messages:
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, list) and \
                any(isinstance(i, dict) and i.get("type") in ("image", "video") for i in c):
            m = {**m, "content": [{"type": "text", "text": HY4_PICTURE_NOTE}
                                  if isinstance(i, dict) and i.get("type") in ("image", "video") else i for i in c]}
        out.append(m)
    return out
from serve.frontend import (CALL_END, CALL_START, MINIMAX_NS, ChatTemplate, Event, OutputParser,  # noqa: E402
                            EFFORT_WORDS, QWEN_EFFORT, anthropic_to_messages, check_messages,
                            deepseek_effort_kwargs, deepseek_images,
                            effort_kwargs,
                            effort_level, effort_name, literal_tags, mark_literals, media_of, openai_to_messages,
                            unmark_literals, INK_CALL, INK_END, INK_EOS, INK_MODEL, INK_OTHER_TEXT, INK_TEXT)
from serve.mcp import McpCancelled, hub_from_config  # noqa: E402
from serve.responses import (ResponseStore, input_items, input_page, normalize_request,  # noqa: E402
                             new_response, response_collect, response_events)
from serve.response_config import ENV_KEYS as RESPONSE_ENV_KEYS, limits_from_env  # noqa: E402
from serve.response_jobs import BackgroundJobs, JobError  # noqa: E402
from serve.response_store import StoreBusy  # noqa: E402
from serve.update import RESTART_EXIT, Updater  # noqa: E402
import config_names  # noqa: E402  (tools/) the config name family: saguaro-<model>.json, the old maya- one found too
import maya_quants  # noqa: E402  (tools/) the quants' names users see (the model first: "GLM-5.3 · Maya-S")

try:                        # Saguaro's release (the dashboard's About; the engine's INFO carries none)
    SAGUARO_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip() or None
except OSError:
    SAGUARO_VERSION = None
MAC_FREE_SETTLE_BYTES = 1 << 30    # macOS: the old engine's memory is back once free memory rises by less than this ...
MAC_FREE_POLL_S = 0.5              # ... between two reads this far apart


def mac_free_bytes() -> int | None:
    """What macOS could give a process now (vm_stat: free, inactive, purgeable and speculative pages), as maya-metal's
    free_memory() counts it; None when vm_stat can't say."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    size = re.search(r"page size of (\d+) bytes", out)
    counts = [re.search(rf"^Pages {k}:\s+(\d+)", out, re.M) for k in ("free", "inactive", "purgeable", "speculative")]
    if not size or not all(counts):
        return None
    return sum(int(c.group(1)) for c in counts) * int(size.group(1))


def wait_gpu_release(pid: int, timeout: float = 60.0):
    """An engine that has ended still holds its GPU memory for a few seconds - the driver frees tens of GB of VRAM and
    its pinned RAM tier after the process is gone - and an engine started at once finds the cards full (the split
    search measures them before anything loads: the context reload's new engine got 0.09 GB and failed).  Waits until
    no GPU lists the process any more (NVML), or a few seconds where NVML can't say (AMD).  On a Mac no GPU lists it,
    but its unified memory comes back over seconds too, and maya-metal sizes its expert pool from what is free when it
    starts (a restart after a crash came back with a smaller pool, partly streaming from the SSD and MTP off): waits
    until free memory stops rising."""
    if sys.platform == "darwin":
        t0, prev = time.time(), mac_free_bytes()
        while prev is not None and time.time() - t0 < timeout:
            time.sleep(MAC_FREE_POLL_S)
            cur = mac_free_bytes()
            if cur is None or cur - prev < MAC_FREE_SETTLE_BYTES:
                break
            prev = cur
        return
    try:
        from serve.telemetry import gpu_pids
    except ImportError:
        gpu_pids = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        pids = gpu_pids() if gpu_pids else None
        if pids is None:
            time.sleep(max(0.0, 5.0 - (time.time() - t0)))
            return
        if pid not in pids:
            if time.time() - t0 > 1:
                print(f"[saguaro] the GPUs freed the engine's memory in {time.time() - t0:.0f} s", flush=True)
            return
        time.sleep(0.5)
    print(f"[saguaro] the GPUs still list the old engine after {timeout:.0f} s; starting anyway", flush=True)


def nvml_pids(pid: int | None, listed: set) -> set:
    """The pids NVML may list a process the server started under (the GPU map): its own - or, when NVML lists
    processes but not that one, also those of every process it started in turn: an engine run through a launcher (a
    script that does not exec it, `nsys launch`, a debugger) holds its GPU memory in a child of the process the server
    started.  psutil finds them (setup installs it); without it, the pid alone."""
    if pid is None:
        return set()
    if pid in listed or not listed:
        return {pid}
    try:
        import psutil
        return {pid} | {c.pid for c in psutil.Process(pid).children(recursive=True)}
    except Exception:  # noqa: BLE001 - no psutil, or the process has just ended: the pid alone
        return {pid}


# the dashboard's Context size range (serve/models.py: the Models page's Engine settings offer the same one)
CONTEXT_MIN, CONTEXT_FALLBACK_MAX = models.CONTEXT_MIN, models.CONTEXT_FALLBACK_MAX
CONTEXT_TESTED, CONTEXT_TESTED_DEFAULT = models.CONTEXT_TESTED, models.CONTEXT_TESTED_DEFAULT
# the engine log's lines a report carries (saguaro.py's --report reads the same ones)
REPORT_LINES = re.compile(r"glm fast:|glm prefill: (?:CUDA|HIP)|glm split|glm stat|glm slots|glm vram|ds4 serve: prompt read|glm prefill: \d|ERR|error|"
                          r"failed|out of memory|WARNING", re.I)
IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
CTX_SLACK = 8               # `saguaro-glm --serve` rejects prompt + max_new + 8 > context: keep the same margin here
TEXT_FILE_MIN, TEXT_FILE_MAX = 512 * 1024, 16 * 1024 * 1024


def text_file_limit(context: int) -> int:
    """The largest text file the chat page attaches (#635): the context holds at most this many bytes of text (about 4
    bytes a token), at least the 512 KB the page always took and at most 16 MB.  The page sends the file as text, so
    the real limit is the model's context; the page says so when a file is over it."""
    return max(TEXT_FILE_MIN, min(TEXT_FILE_MAX, int(context or 0) * 4))
# The live tok/s is a rate over a window, not a mean since the first token: a mean reads ~1/elapsed at the first
# token (the Monitor showed five-digit numbers) and then undershoots for the first second of every answer.  5 s: drafted
# tokens come in bursts of 1-4 a step, and over 2 s the rate swung +-25% around the engine's steady one
RATE_WINDOW_S = 5.0
PEAK_MIN_SPAN_S = 2.0     # a peak only from an answer or read this old: younger windows read bursts as rates
RATE_MIN_SPAN_S = 0.25      # younger than this there is no rate yet: the mean so far, with the span floored here
# A request that waited this long in the FIFO gets a line in the server window when its turn comes: with nothing else
# logged, a queue that had stopped moving (and requests piling up behind it) was indistinguishable from a busy server.
QUEUE_LOG_S = 5.0
# A queued request's keep-alive while it waits for its first turn (#483): often enough for any stream-idle timeout,
# seldom enough to cost nothing (queue_ping_seconds: SAGUARO_QUEUE_PING_S sets another).
QUEUE_PING_S = 2.0
# The queue watchdog (Service.start_queue_watch): how often it looks, the age at which the first waiter means the
# queue has stopped moving, and how often it says so again while it stays that way.
QUEUE_TICK_S = 15.0
QUEUE_WEDGED_S = 120.0
QUEUE_WARN_S = 60.0
# The Monitor's Conversations rows (Service.convs): a request continues a conversation only when its prompt starts
# with exactly that conversation's LAST PROMPT (minus the open reply's generation tail) - the part a later turn
# re-sends byte-for-byte, just what the engine's own prefix reuse keys on (a finished assistant turn re-renders with
# small differences, so it is not part of the key).  An opening, or a first message, is no identity: the longest
# last-prompt a request starts with wins, none -> a new row.  The table keeps the recent CONV_MAX with an LRU.
CONV_MAX = 64
CONV_IDLE_S = 2 * 3600.0


# ------------------------------------------------------------------------------------------------ engines
class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event) -> Iterator[int]: ...


PEAK_MIN_TOKENS = 32   # the Monitor's peak decode speed: answers this long or longer (a few tokens' speed says little)
PEAK_MIN_READ_TOKENS = 512   # the Monitor's peak prefill speed: reads of at least this many (a small read's rate says little)


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token.  Given a list of
    scripts, each request gets the next one and the last one repeats (a tool call, then the answer after it)."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0,
                 tok_s: float | None = None):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        # tok_s: it writes at about that speed and reports its clock as the engine's DONE does (the Monitor's speeds;
        # `--engine mock` takes SAGUARO_MOCK_TOK_S)
        self.tok_s = tok_s
        end = tokenizer.encode(IM_END, parse_special=True)
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in ([script] if isinstance(script, str)
                                                                                 else script)]
        self.script, self.turns, self._runs = self.scripts[0], 0, 0
        self.last_prompt: list[int] = []
        # it samples nothing, so every request's bias is as good as applied; and a scripted token is certain: logprob 0,
        # the only token its row has (logprobs)
        self.info = {"logit_bias": 1, "logprobs": TOP_LOGPROBS_MAX}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        n_top = logprobs_asked(sampling)
        t0, steps = time.time(), self.script[:max_new]
        self._runs += 1
        for i, t in enumerate(steps):
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            if self.tok_s:
                time.sleep((0.8 + 0.4 * ((t * 7919 + self._runs * 104729) % 101) / 100) / self.tok_s)   # (uneven)
                if i == len(steps) - 1:                 # (its DONE, before the last token: the server stops there)
                    self.last = {"generated": i + 1, "decode_ms": (time.time() - t0) * 1000,
                                 "prompt_ms": len(ids) * 0.5, "reused": 0}
            if n_top is not None:
                t = LpToken(t)
                t.lp = (0.0, [(int(t), 0.0)] if n_top > 0 else [])
            yield t


# R122: the queue priorities (a request's "priority" field, its X-Saguaro-Priority header, the config's
# "priority"): a person's question steps an agent's long answer aside at once and runs first.  Everything
# without one is normal, so an old client's order is exactly the plain FIFO it always was.
PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW = 2, 1, 0
PRIORITY_NAMES = {PRIORITY_HIGH: "high", PRIORITY_NORMAL: "normal", PRIORITY_LOW: "low"}


def priority_level(value, field="priority") -> int:
    """One of "high" / "normal" / "low" as its level, or a ValueError naming the field (a 400).  Exactly the
    three (Ken's ONE name): "background" and "idle" are refused like any other wrong value."""
    if isinstance(value, str) and value.strip().lower() in PRIORITY_NAMES.values():
        return {"high": PRIORITY_HIGH, "normal": PRIORITY_NORMAL, "low": PRIORITY_LOW}[
            value.strip().lower()]
    raise ValueError(f'{field}: "high", "normal" or "low"')


_ENV_S: dict = {}      # name -> (the raw env string, what it parsed to): see _env_seconds


def _env_seconds(name: str, default: float) -> float:
    """A seconds setting from the environment, parsed only when its raw string changed - the queue checks read
    them under the FIFO's lock, once per waiter per grant check, and re-parsing there is pure noise. Unset,
    empty or unparseable: the default; nothing under these goes below 0."""
    raw = os.environ.get(name)
    hit = _ENV_S.get(name)
    if hit is not None and hit[0] == raw:
        return hit[1]
    try:
        value = max(0.0, float(raw))
    except (TypeError, ValueError):
        value = float(default)
    _ENV_S[name] = (raw, value)
    return value


def priority_age_seconds() -> float:
    """SAGUARO_PRIORITY_AGE_S (0 = off): a low waiter that has waited this many seconds counts as normal.
    Off by default (Ken's rule): a low request waits as long as a higher-priority one uses or waits for the
    model, however long - it is never dropped or errored for waiting, only by its client, a stop or a shutdown."""
    return _env_seconds("SAGUARO_PRIORITY_AGE_S", 0.0)


def priority_idle_seconds() -> float:
    """SAGUARO_PRIORITY_IDLE_S (30): the idle rule (R122) - a low request starts, or resumes after a pause, only
    when nothing normal or high runs or waits AND none finished in the last this many seconds (0 = no grace):
    a person's follow-up a few seconds after an answer finds the model where they left it."""
    return _env_seconds("SAGUARO_PRIORITY_IDLE_S", 30.0)


class FairLock:
    """The service's FIFO: waiters get the engine in the order they asked (threading.Lock promises no order), and the
    holder can see that someone is waiting (`waiting()`, the fair slices).  acquire/release/with as threading.Lock.

    With an engine that runs several conversations at once (READY ... multi=<n>) it has n places: a request takes
    one (`units=1`), and what needs the engine to itself - an image encode, a context reload, an update - takes them
    all (the default, and `with`).  Places go in arrival order: a waiter for all of them holds back the ones behind
    it, so it is not starved.

    Entries carry a priority (`enqueue(units, priority)`, `acquire(priority=...)`, high 2 / normal 1 / low 0,
    normal the default): the highest priority is granted first, FIFO inside one priority, and an all-places waiter
    holds back the ones behind it of its own or a lower priority - with equal priorities the order is the plain
    FIFO's, unchanged.  A low waiter older than SAGUARO_PRIORITY_AGE_S counts as normal (aging), and a low one
    starts or resumes only while the model is idle (SAGUARO_PRIORITY_IDLE_S, `_idle_wait`); `waiting(above=p)`
    counts the waiters that would step ahead of a runner at level p, and `step_aside(p)` is the runner's own
tick decision: a low one gives its place up whenever a normal or high one runs OR waits - even when a place
     is free - a normal one only for a waiting high one that does not fit (claim_yield's counting, and of several
     running normals it is the one that started last), a high never."""

    def __init__(self, capacity: int = 1):
        self._cv = threading.Condition(threading.Lock())
        self._queue: collections.deque = collections.deque()   # [token, units, asked at, queued again, priority]
        self.capacity = max(1, int(capacity))
        self._used = 0
        self._yields = 0                                       # runners that decided to give their place up (claim_yield)
        self._one = False                                      # one holder at a time, whatever the capacity (Turbo)
        self._held: list = []                                  # one entry per grant: (the priority it was taken at,
        #  the thread that took it) - the idle rule's "nothing normal or high runs", and the order the running
        #  requests started their turns in: of several running normals it is the one that started LAST that steps
        #  aside for a waiting high one (Ken's "then the normal one that started last", `_last_normal`), and the
        #  ident is how a runner names itself about that without a handle plumbed through _turn and _sliced - a
        #  request's whole run, FIFO turn included, lives on one thread
        self._last_finish = 0.0                                # when the last normal or high request ended (release
        #  with its priority): the idle rule's grace counts from here; 0: since before the process started

    def set_one(self, on: bool):
        """One holder at a time while on (places and their sizes stay the capacity's, so a flip mid-run is safe)."""
        with self._cv:
            self._one = bool(on)
            self._cv.notify_all()

    def _fits(self, k: int) -> bool:
        return self._used + k <= self.capacity and not (self._one and self._used > 0)

    def set_capacity(self, n: int):
        with self._cv:
            n = max(1, int(n))
            if n != self.capacity:
                print(f"[saguaro] the engine's places: {self.capacity} -> {n} ({self._used} in use)", flush=True)
            self.capacity = n
            # A place held for "all of them" was sized by the old capacity: a smaller engine cannot hold more than it
            # has, or the FIFO would stay jammed (waiting for a place that is never free) - the releases that follow
            # give their places back and are clamped in release().
            self._used = min(self._used, self.capacity)
            self._cv.notify_all()

    def _units(self, units):
        return self.capacity if units is None else max(1, min(int(units), self.capacity))

    def _eff(self, e) -> int:
        """What priority a waiter counts at now: with SAGUARO_PRIORITY_AGE_S > 0 a low one that has waited
        that long has aged into normal; 0 (the default) = no promotion by waiting (an entry from before the
        priorities is normal)."""
        p = e[4] if len(e) > 4 else PRIORITY_NORMAL
        age = priority_age_seconds()
        if p == PRIORITY_LOW and 0 < age <= time.monotonic() - e[2]:
            return PRIORITY_NORMAL
        return p

    def _first(self, me) -> bool:
        """Whether `me` is the head of the grant order (the highest priority first, FIFO inside one priority):
        nothing that counts waits ahead of it at its own level or above, and nothing above it waits behind it.
        Entries in the deque are in arrival order, so a plain FIFO (all one priority) means exactly what the
        head of the deque meant before."""
        p = self._eff(me)
        ahead = True
        for e in self._queue:
            if e is me:
                ahead = False
                continue
            q = self._eff(e)
            if q > p or (q == p and ahead):
                return False
        return True

    def acquire(self, blocking: bool = True, timeout: float = -1, units: int | None = None,
                priority: int | None = None) -> bool:
        k = self._units(units)
        p = PRIORITY_NORMAL if priority is None else priority
        with self._cv:
            # the fast path only when nobody of an equal or higher priority waits: a new normal request may step
            # over a queued low one, but never over its own kind or above a high one; and a low one takes it only
            # while the model is idle (the idle rule, `_idle_wait`)
            if self._fits(k) and not any(self._eff(e) >= p for e in self._queue) and self._idle_wait(p) == 0.0:
                self._used += k
                self._held.append((p, threading.get_ident()))    # (running at p: the idle rule counts it, and
                return True                                      #  the start order names who steps aside first)
            if not blocking:
                return False
        me = self.enqueue(units, p)
        if self.wait_turn(me, timeout):
            return True
        self.leave(me)
        return False

    def _idle_wait(self, p: int) -> float:
        """Seconds a request at level p still waits under the idle rule (R122) - 0: it may start or resume now.
        A LOW one runs only while the model is idle: nothing normal or high holds a place, and none of them
        finished in the last SAGUARO_PRIORITY_IDLE_S seconds (0 = no grace), so a person's follow-up a few
        seconds after an answer finds the model where they left it.  -1: a normal or high request has the
        engine - its release wakes the line.  Waiters are not looked at: the grant order already keeps a low
        one behind every normal or high one in line.  Called under the lock; `self._held` carries one (level,
        thread) entry per live grant, and a release naming no level is a context manager or a maintenance path -
        it holds every place, so nothing else can have a grant to lose."""
        if p > PRIORITY_LOW:
            return 0.0
        if any(lvl > PRIORITY_LOW for lvl, _ in self._held):
            return -1.0
        return max(0.0, priority_idle_seconds() - (time.monotonic() - self._last_finish))

    def enqueue(self, units: int | None = None, priority: int | None = None) -> object:
        """A place in line (wait_turn waits for it, leave gives it up): a waiter that wakes up now and then (to send
        a heartbeat) keeps its place."""
        me = [object(), self._units(units), time.monotonic(), False,     # (token, units, asked at: the queue watchdog's age, queued again by a fair slice)
              PRIORITY_NORMAL if priority is None else priority]         # (R122: its place in the grant order)
        with self._cv:
            self._queue.append(me)
        return me

    def claim_yield(self, slice_s: float, above: int | None = None) -> bool:
        """The fair slices' decision, made one runner at a time: True when this runner should give its place up now,
        and then counted until it does (yield_place) or says it did not (unclaim_yield).  Without the count, every
        runner past its slice saw the same waiter and gave up its place for it: with 4 conversations running, one
        short request cut all 4, each to read all it had written again.  The waiters that count are the ones that
        asked for a place (a request that has not run yet), and those a slice put back in line once they have waited
        a slice themselves (long answers still take turns); what they need, less the places free and the runners
        already on their way out, is how many more runners give up a place.

        `above` (R122): a priority preemption counts only the waiters above that level - a runner is never cut for
        its own or a lower priority; with it left out (the fair slices) every waiter counts, as before."""
        now = time.monotonic()
        with self._cv:
            need = sum(min(e[1], self.capacity) for e in self._queue
                       if (above is None or self._eff(e) > above)
                       and (not (len(e) > 3 and e[3]) or now - e[2] >= slice_s))
            free = 0 if self._one and self._used > 0 else self.capacity - self._used
            if need - free - self._yields > 0:
                self._yields += 1
                return True
            return False

    def unclaim_yield(self):
        with self._cv:
            self._yields = max(0, self._yields - 1)

    def _unhold(self, p):
        """Giving one `_held` entry back (called under the lock): THIS thread's grant first - the (level, thread)
        pairs make the giver's own findable - at level p, or at any level when p is None; failing that any entry
        at p (a place taken on one thread and given back on another: the old behaviour the odd maintenance path
        needs); and with no level named and nothing of this thread's held, the newest, which is what a release
        without a level always popped."""
        tid = threading.get_ident()
        for i in range(len(self._held) - 1, -1, -1):
            lvl, th = self._held[i]
            if th == tid and (p is None or lvl == p):
                return self._held.pop(i)
        if p is not None:
            for i in range(len(self._held) - 1, -1, -1):
                if self._held[i][0] == p:
                    return self._held.pop(i)
            return None
        return self._held.pop() if self._held else None

    def _last_normal(self):
        """The thread that took its turn last among the running NORMAL grants - `_held` is appended in grant
        order, so the newest NORMAL entry names the one that started last (Ken's "then the normal one that
        started last" steps aside for a waiting high one); None with no normal running. Called under the lock."""
        for lvl, _tid in reversed(self._held):
            if lvl == PRIORITY_NORMAL:
                return _tid
        return None

    def step_aside(self, p: int, slice_s: float = 0.0) -> bool:
        """The runner's preemption decision on its token tick (R122, Ken's "who steps aside, exactly"): True when
        a runner at level p gives its place up now.  A LOW one does so whenever ANY normal or high request runs
        or waits - even when a place is free: low is "only while idle", and with parallel > 1 it would otherwise
        keep the card's speed from the people using it (every low runner steps aside on its own, all of them,
        while a person is around).  A NORMAL one only for a waiting HIGH one that does
        fit - `claim_yield`'s counting cuts just as many runners as that waiter needs, and a free place is its
        place, not a reason to cut anybody; when more than one normal runs, the one that STEPS ASIDE is the one
        that started its turn LAST (Ken's choice of victim, `_last_normal`) - an earlier one keeps its place and
        an earlier runner's tick simply does not claim.  A HIGH one never.  A True claims the decision like
        `claim_yield` does: the runner then either gives its place up (`yield_place`) or says it did not
        (`unclaim_yield`); the low one's claim needs no need/free arithmetic - it gives its place up beside a
        running person, where nobody is waiting at all."""
        if p >= PRIORITY_HIGH:
            return False
        if p == PRIORITY_LOW:
            with self._cv:
                if not (self._idle_wait(PRIORITY_LOW) != 0.0        # a normal or high holds a place, or the grace
                        or any(self._eff(e) > PRIORITY_LOW for e in self._queue)):   # (aged ones count)
                    return False
                self._yields += 1                               # (counted, as a fair slice's claim is: the place
                return True                                     #  is on its way out - nobody else must give one)
        with self._cv:
            if self._last_normal() != threading.get_ident():
                return False                        # (a normal that started earlier: not this one's place to give)
        return self.waiting(above=p) > 0 and self.claim_yield(slice_s, above=p)

    def highest_active(self):
        """The highest priority running or waiting (waiters aged), None with nothing at all: whose name a
        stepping-aside line logs - a low answer cut while a normal one runs has no WAITER to name."""
        with self._cv:
            levels = [lvl for lvl, _ in self._held] + [self._eff(e) for e in self._queue]
        return max(levels, default=None)

    def yield_place(self, units: int | None = 1, priority: int | None = None) -> object:
        """A runner's place given up, and a place in line behind whoever waits now, in one step (the claim ends with it:
        the free place is counted from here).  It keeps its priority (R122): the resumed request waits with the same
        claim it had before it gave way, so a high-priority answer cut for a person's question does not land behind
        the agents' low queue."""
        k = self._units(units)
        p = PRIORITY_NORMAL if priority is None else priority
        me = [object(), k, time.monotonic(), True, p]
        with self._cv:
            self._queue.append(me)
            self._used -= min(k, self._used)
            self._unhold(p)                               # (waiting in line again, not running: the idle rule -
            self._yields = max(0, self._yields - 1)       #  and its OWN grant entry goes: the pairs are the key)
            self._cv.notify_all()
        return me

    def wait_turn(self, me, timeout: float = -1) -> bool:
        """Wait until `me` is first in line and its places are free, then hold them (True); False after `timeout`,
        still in line - and False at once when `me` was given up (leave) or is no longer in the line: a wedged FIFO
        (waiting for a place that never comes free) is what a queue that stops moving was.  A low entry (R122)
        also waits out the idle rule (`_idle_wait`): the sleep is capped by what is left of its grace, since
        only time clears that part."""
        with self._cv:
            end = None if timeout is None or timeout < 0 else time.monotonic() + timeout
            while True:
                if me not in self._queue:
                    return False
                p = me[4] if len(me) > 4 else PRIORITY_NORMAL
                #  (its OWN level, not `_eff`: with SAGUARO_PRIORITY_AGE_S > 0 a low waiter that has aged into
                #  normal grants in normal order and cuts runners like a normal one (_eff is what _first and
                #  waiting count with), but it still waits out the idle grace - the grace is the model's rest
                #  after a person, not a queue position - and the grant is recorded at the raw level, which is
                #  what the request's release(priority=) hands back to _held)
                grace = self._idle_wait(p)
                if grace == 0.0 and self._first(me) and self._fits(min(me[1], self.capacity)):
                    self._queue.remove(me)
                    me[1] = min(me[1], self.capacity)
                    self._used += me[1]
                    self._held.append((p, threading.get_ident()))   # (running at p: the idle rule counts it, and
                    self._cv.notify_all()                   #  the start order names who steps aside first)
                    return True
                left = None if end is None else end - time.monotonic()
                if left is not None and left <= 0:
                    return False
                if grace > 0:                               # grace left: sleep no longer than that, then look again
                    left = grace if left is None else min(left, grace)
                self._cv.wait(left)

    def leave(self, me):
        with self._cv:
            if me in self._queue:
                self._queue.remove(me)
                self._cv.notify_all()                   # the next one may now be first in line

    def release(self, units: int | None = None, priority: int | None = None):
        """Giving a place back.  `priority` (R122): the level the place was taken at - the request path passes it,
        and a normal or high request that ends here starts the idle rule's grace (`_idle_wait`); a release
        without it (an encode, a reload, the context manager) is no request that finished and starts none."""
        k = self._units(units)
        with self._cv:
            if self._used < k:
                # A capacity that shrank under a "all places" holder (the engine restarted with fewer conversations)
                # gives back what it has: raising here would leave the place held for ever - and the next release
                # would too, with every request behind it waiting for a place that never comes.
                print(f"[saguaro] a place was given back that is not held ({k} of {self._used} in use, capacity "
                      f"{self.capacity}): the engine's conversations changed while it ran", flush=True)
                k = self._used
            self._used -= k
            if priority is None:
                if self._held:                            # (no level given: this thread's own newest grant - and
                    self._unhold(None)                    #  the newest at all when it holds none of them here,
            else:                                         #  the release-without-a-level of before)
                self._unhold(priority)                    # (this thread's own at that level first: the pairs make
                #  the giver's entry findable; a level with no entry at all leaves _held alone, as before)
            if priority is not None and priority > PRIORITY_LOW:
                self._last_finish = time.monotonic()      # (a person's answer finished: the low grace starts)
            self._cv.notify_all()

    def locked(self) -> bool:
        return self._used > 0

    def waiting(self, above: int | None = None) -> int:
        """The waiters in line; with `above` (R122), only those of a HIGHER priority than that level - a runner asks
        this before it even looks at giving its place up.  Under the lock like every other view of the line (the
        values are advisory either way: `claim_yield` re-decides under the lock; this just reads the deque as the
        thread that appends or removes an entry left it)."""
        with self._cv:
            if above is None:
                return len(self._queue)
            return sum(1 for e in self._queue if self._eff(e) > above)

    def highest_waiting(self):
        """The highest priority waiting (aged), None with an empty line: whose name a preemption logs."""
        with self._cv:
            return max((self._eff(e) for e in self._queue), default=None)

    def all_low_waiting(self) -> bool:
        """True when the line holds waiters and every one of them is a low request (R122, the queue watchdog):
        a low request waiting a long time behind the people is Ken's rule (no promotion by waiting), not a
        wedged FIFO - the WARNING would fire on ordinary busy evenings.  With an age setting on an aged low
        waiter counts as normal, and the line is worth printing again."""
        with self._cv:
            return bool(self._queue) and all(self._eff(e) == PRIORITY_LOW for e in self._queue)

    def oldest_wait(self) -> float:
        """Seconds the request first in line has been waiting (0 when nobody waits): 0 while it is being served."""
        with self._cv:
            if not self._queue:
                return 0.0
            asked = self._queue[0][2] if len(self._queue[0]) > 2 else None
            return time.monotonic() - asked if isinstance(asked, float) else 0.0

    def held_by(self) -> str:
        """What has the engine (for a line when the queue stops moving): the places in use of its capacity."""
        with self._cv:
            return f"{self._used} of {self.capacity} place(s) in use"

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def queue_ping_seconds() -> float:
    """SAGUARO_QUEUE_PING_S: how often a request waiting for its first turn sends a keep-alive (QUEUE_PING_S)."""
    try:
        return max(0.01, float(os.environ.get("SAGUARO_QUEUE_PING_S", "") or QUEUE_PING_S))
    except ValueError:
        return QUEUE_PING_S


def wait_event_seconds() -> float:
    """SAGUARO_WAIT_EVENT_S (15): R122 - a stream whose request was queued at least once, or paused for a higher
    priority (Ken's trigger: wait_events' `waited` gate), keeps its ": keep-alive" comments, and once this long goes
    by without a real event one empty event goes out too (an empty chat.completion.chunk, an Anthropic ping, a
    response.in_progress): clients that count only events stay. Never text in the answer; a request that never
    waited sends nothing extra, however long it takes."""
    try:
        return max(0.01, float(os.environ.get("SAGUARO_WAIT_EVENT_S", "") or 15))
    except ValueError:
        return 15.0


def wait_space_seconds() -> float:
    """SAGUARO_WAIT_SPACE_S (20): R122 - a non-streaming request that has waited this long (or was paused once by a
    higher priority) gets its 200 and headers early, and a space every SAGUARO_WAIT_EVENT_S until the JSON: spaces
    before a JSON body are valid, and an error after the headers cannot change the status any more (the log says so)."""
    try:
        return max(0.01, float(os.environ.get("SAGUARO_WAIT_SPACE_S", "") or 20))
    except ValueError:
        return 20.0


def wait_events(items, empty, waited=None):
    """R122: a stream's keep-alive comments (the None items) stay as they are; once SAGUARO_WAIT_EVENT_S goes by
    without a real event, one empty event follows them - `empty()` builds it, and the answer text is never touched.
    `waited` (Ken's decision on the trigger): a callable saying whether THIS request is one that waits - queued at
    least once, or paused; while it says no, the None items stay plain comments and the stream is framed exactly as
    before R122, a long prompt read (whose PP ticks are None items too) included. `waited=None`: every gap counts,
    as the queued case does anyway. Closing this closes what it wraps."""
    last, every = time.monotonic(), wait_event_seconds()
    try:
        for c in items:
            if c is None:
                yield None
                if (waited is None or waited()) and time.monotonic() - last >= every:
                    yield empty()
                    last = time.monotonic()
            else:
                last = time.monotonic()
                yield c
    finally:
        close = getattr(items, "close", None)
        if close is not None:
            close()


def fair_slice_seconds() -> float:
    """SAGUARO_FAIR_SLICE_S: a request that has been answering this long while others wait gives the engine to the next
    one and queues again behind them (0, the default: it keeps the engine to the end, as before)."""
    try:
        return max(0.0, float(os.environ.get("SAGUARO_FAIR_SLICE_S", "0") or 0))
    except ValueError:
        return 0.0


def fair_read_seconds() -> float:
    """SAGUARO_FAIR_READ_S: a prompt read this long while another request waits behind it (sent to the engine, nothing
    heard from it yet) pauses at the engine's next chunk - the engine keeps what it read - and goes on with the rest
    once that request has started (0, the default: a prompt is read to the end).  The design is upstream PR #144's."""
    try:
        return max(0.0, float(os.environ.get("SAGUARO_FAIR_READ_S", "0") or 0))
    except ValueError:
        return 0.0


def fair_read_yield_seconds() -> float:
    """SAGUARO_FAIR_READ_YIELD_S: how long a paused prompt holds back at most for the requests it paused for (30)."""
    try:
        return max(0.0, float(os.environ.get("SAGUARO_FAIR_READ_YIELD_S", "30") or 0))
    except ValueError:
        return 30.0


class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (issue #27: on Linux, the out-of-memory killer)."""


class LendTimeout(ValueError):
    """A VLEND / VRECLAIM got no answer within its wait (PLAN-vision-fast-start review 4): the engine was busy reading
    its prompt and did not reach a settle.  Distinct from an ERR refusal (the engine alive and saying "no room"): on a
    timeout _vision_up waits once more, and a lending engine is not started without room.  (A refusal stays a ValueError.)"""


class ContextOverflow(ValueError):
    """A request too long for the context (rejected, never truncated): a 400 with OpenAI's error code and wording -
    "maximum context length is N tokens" - which OpenAI-compatible clients (opencode, pi-ai, OpenClaw) recognise as a
    context overflow and answer by compacting the conversation (#432)."""
    code = "context_length_exceeded"


def error_body(e: Exception, kind: str = "invalid_request_error") -> dict:
    """An error response's body: OpenAI's shape, with the exception's `code` when it has one (ContextOverflow)."""
    err = {"type": kind, "message": str(e)}
    if getattr(e, "code", None):
        err["code"] = e.code
    return {"error": err}


# saguaro-ik2's ERR after which it ends (a GPU failure in a request: cu_fail), so the server starts a fresh one
ENGINE_ENDS_AFTER_ERR = "the engine starts again for the next request"
# the engine's ERR when a route met a NaN or an infinity (Glm5Model::route_ok): it has dropped that conversation's state
NONFINITE = "non-finite"

# what Service._sliced yields when the loop guard ends an answer (a loop that came back): run() ends it, "stop"
LOOP_STOP = object()
SEGMENT = object()          # separates one engine call's tokens from the next's in segments() (run's drain)
# ... and the answer it gets when that happens before any answer text (never a silent end after the thinking)
LOOP_STOP_NOTE = "(The answer was stopped: the model kept repeating itself.)"
# a request's max_tokens lowers the thinking budget only at or below this (Service.run)
THINK_TINY_LIMIT = 1024
# A reasoning block that reaches the thinking budget is closed at the end of its sentence or line, not mid-sentence
# (#613: an agent's turn closed in the middle of a formula, and the answer after it was garbled): Service._sliced
# closes it at the first boundary past the budget, and the engine's own close (its think_budget) only comes this many
# tokens later, for reasoning with no boundary at all - at most half the room the request's max_tokens leaves after
# the budget, so the answer keeps the rest.
THINK_GRACE = 512
# A model that writes the server's own markers inside its thinking (it quotes them while it works on them) ends the
# engine there: the engine stops at its end-of-sequence token, the server at its stop markers - and the agent dies
# mid-thinking.  Inside the thinking such a marker is the model's text, not the turn's end: the server skips it, and
# at the engine's own EOS token it goes on from the quote with this note as the server's (Service.run), at most
# THINK_RESTARTS times.  The answer's own markers still end the turn, in the answer.
THINK_RESTARTS = 2
THINK_NOTE = ("[The server's note, not the user: your thinking was cut off there - an end-of-turn marker written "
              "inside the thinking is a quote, not the turn's end. Do not stop: go on with the thinking, right where "
              "it stopped.]")
# A model whose thinking ends the turn even after the restarts above leaves nothing for the answer; the turn goes out
# with a reasoning block and no content at all, which an agent client retries as an empty answer.  As the loop guard
# does (LOOP_STOP_NOTE), the turn says so in one line instead of ending empty.
THINK_STOP_NOTE = "(The answer was stopped: the model's thinking ended the turn without an answer.)"
_THINK_BOUNDARY = re.compile(r"(?:\n|(?<![\d.])[.!?][\"')\]]*|[。！？])[ \t]*$")


_WEAK_BOUNDARY = re.compile(r"[.!?][\"')\]]*[ \t]*")


def think_stop_tables(tok) -> str:
    """The engine's soft close (#746): the ids that end a sentence or line, as GEN keys - `think_stop` (the token's own
    text ends one), `think_weak` (a bare . ! ? : one only when the token before does not end in a digit or a dot) and
    `think_dig` (the tokens that end in a digit or a dot).  "" for a tokenizer with no token list."""
    n = len(getattr(tok, "tokens", None) or ())
    if not n:
        return ""
    stop, weak, dig = [], [], []
    for t in range(n):
        try:
            text = tok.decode([t])
        except Exception:
            continue
        if not text:
            continue
        if text[-1].isdigit() or text[-1] == ".":
            dig.append(t)
        if _WEAK_BOUNDARY.fullmatch(text):
            weak.append(t)
        elif think_boundary(text):
            stop.append(t)
    j = lambda v: ",".join(map(str, v))
    return f" think_stop={j(stop)} think_weak={j(weak)} think_dig={j(dig)}" if stop or weak else ""


def think_boundary(text: str) -> bool:
    """The reasoning so far (its last characters will do) ends a line or a sentence: a newline, or . ! ? after
    something other than a digit or another dot (not "3." of "3.0", not "..."), closing quotes or brackets allowed
    after it."""
    return bool(_THINK_BOUNDARY.search(text))

# ---- log-probabilities (OpenAI's `logprobs` / `top_logprobs`; the DeepSeek and GLM engines' GEN logprobs=<n> and LP lines).
# The engine reports, for each token it writes, the token's logprob and its row's n most likely, from the model's
# softmax after the request's penalties, at temperature 1 and before top_k / top_p / min_p (OpenAI's convention: the
# same numbers whatever the request's temperature).  Only a request that asks gets them; an engine that does not say
# INFO logprobs=<n> is refused with a 400.
TOP_LOGPROBS_MAX = 20
LOGPROB_FLOOR = -9999.0           # OpenAI's logprob for a token the model gives no chance (-inf is no JSON number)


def logprobs_of(req: dict | None) -> int | None:
    """The engine's logprobs=<n> for a request (n alternatives a token, 0..20), None when it does not ask.
    OpenAI's chat completions: "logprobs": true asks, "top_logprobs" (0..20, 0 when absent) says how many alternatives;
    top_logprobs without logprobs true, or a value of the wrong type, is a 400 (ValueError) naming the field.  (The
    Responses API's include / top_logprobs arrive here in this form: serve/responses.py's normalize_request.)"""
    lp, top = (req or {}).get("logprobs"), (req or {}).get("top_logprobs")
    if lp is not None and not isinstance(lp, bool):
        raise ValueError("logprobs: true or false")
    if isinstance(top, str) and re.fullmatch(r"\s*\d+\s*", top):
        top = int(top)
    if top is not None:
        if isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= TOP_LOGPROBS_MAX:
            raise ValueError(f"top_logprobs: an integer from 0 to {TOP_LOGPROBS_MAX}")
        if lp is not True:
            raise ValueError("top_logprobs: it needs \"logprobs\": true")
    return (top or 0) if lp is True else None


def logprobs_asked(req: dict | None) -> int | None:
    """logprobs_of() for a request already checked (Service.check_request): None for one it would refuse."""
    try:
        return logprobs_of(req)
    except ValueError:
        return None


class LpToken(int):
    """A token id from an engine asked for logprobs: the id itself, as every token goes through the server (the stop
    ids, the loop guard, fair slices), with its LP line's figures in `lp` - (its logprob, [(id, logprob), ...] its
    row's most likely, most likely first) - or None when the engine wrote none for it."""
    lp = None


def parse_lp(fields: list[str]):
    """An LP line's fields after its tag (`LP [<rid>]`): `<id> <logprob> <id>:<logprob> ...` -> (id, logprob,
    [(id, logprob), ...]); None for a line that does not read (cut short)."""
    try:
        top = []
        for f in fields[2:]:
            i, _, v = f.partition(":")
            top.append((int(i), float(v)))
        return int(fields[0]), float(fields[1]), top
    except (IndexError, ValueError):
        return None


def lp_attach(tok: LpToken, fields: list[str]) -> LpToken:
    """The token's LP line (its fields after the tag) into tok.lp, when the line is that token's."""
    got = parse_lp(fields)
    if got is not None and got[0] == int(tok):
        tok.lp = (got[1], got[2])
    return tok


def ik2_decode_stats(line: str) -> dict | None:
    """saguaro-ik2's "decode stats: N tokens, H expert hits, M misses (..)" (SAGUARO_IK2_STATS=1, before its DONE) as
    DONE's hit fields: the experts the answer's tokens found in VRAM, of those they asked for."""
    m = re.match(r"decode stats: \d+ tokens, (\d+) expert hits, (\d+) misses", line)
    return {"hits": int(m.group(1)), "lookups": int(m.group(1)) + int(m.group(2))} if m else None


def ik2_log_facts(text: str, ctx: int) -> dict:
    """What saguaro-ik2's start lines (its log) say of where the model went, under the names the other engines' INFO
    lines give them (the Monitor's facts): the routed experts, the VRAM cache's slots over its GPUs and its size, the
    experts kept in pinned host memory, the K / V cache's size at this context."""
    out: dict = {}
    m = re.search(r"saguaro ik2: (\d+) layers.*?, (\d+) experts \(", text)
    experts = int(m.group(1)) * int(m.group(2)) if m else 0
    pools = {int(g): (int(n), float(gib)) for g, n, gib in
             re.findall(r"expert cache, GPU (\d+): (\d+) slots across \d+ pools? \(([\d.]+) GiB\)", text)}
    if experts:
        out["experts"] = experts
        if pools:
            out["expert_slots"] = min(experts, sum(n for n, _ in pools.values()))
            out["cache_gb"] = round(sum(g for _, g in pools.values()), 2)
    m = re.search(r"experts in pinned .*?memory: ([\d.]+) GiB", text)
    if m and experts:
        out["ram_slots"], out["ram_gb"] = experts, float(m.group(1))
    m = re.search(r"kv cache: \S+ \(([\d.]+) GiB", text)
    if m and ctx > 0:
        out["kv_gb"], out["kv_ctx"] = float(m.group(1)), int(ctx)
    return out


def ik2_vision_gpu(cards: list, need: int, free_of) -> int | bool:
    """Where Inkling 2's on-demand image encoder starts: the config's GPU with the most free memory when that holds
    what the encoder measured (its number, as nvidia-smi numbers them), else False - the CPU."""
    free, gpu = max(((free_of(g) or 0), -i, g) for i, g in enumerate(cards))[::2]
    return int(gpu) if free >= need else False


def ik2_stat(info: dict, last: dict) -> dict | None:
    """An answer's figures as the STAT reading the Monitor's "Where the experts live" draws (saguaro-ik2 prints none
    while it writes: this is the finished answer's, from its DONE and "decode stats:" lines)."""
    n, ms, look = int(last.get("generated") or 0), float(last.get("decode_ms") or 0), int(last.get("lookups") or 0)
    experts, slots = int(info.get("experts") or 0), int(info.get("expert_slots") or 0)
    whole = experts > 0 and slots >= experts            # (every expert in VRAM: its tokens ask the cache for none)
    if n <= 0 or ms <= 0 or not (look or whole):
        return None
    hits = int(last.get("hits") or 0)
    st = {"tok_s": round(1000.0 * n / ms, 2), "ms_tok": round(ms / n, 2),
          "vram_hit": round(hits / look, 4) if look else 1.0, "ram_fetch": round((look - hits) / n, 3), "disk": 0.0}
    if experts and slots:
        st.update(vram_used=slots, ram_used=experts - slots, disk_gb=0.0)
        if info.get("ram_gb") and info.get("cache_gb"):
            st.update(vram_gb=min(float(info["cache_gb"]), float(info["ram_gb"])),
                      ram_gb=round(max(0.0, float(info["ram_gb"]) - float(info["cache_gb"])), 2))
    return st


def engine_token(field: str) -> int:
    """A T line's id.  Below 0 it is no token: an engine whose sample failed (maya-ds4 before #619, or one an
    update could not rebuild) wrote -1 as one - an error for the request, never an id for the tokenizer and the
    loop guard (#805: an IndexError there ended a 351K-token prompt's request with a traceback)."""
    t = int(field)
    if t < 0:
        raise ValueError(f"the engine could not pick a token (it wrote the id {t}: its sampling failed - the engine's "
                         "log says why)")
    return t


def token_piece(tok, t: int) -> bytes:
    """One token's bytes (a logprobs entry's `bytes`): the tokenizer's own, else its decode; b"" for an id it lacks."""
    try:
        if hasattr(tok, "token_bytes"):
            return tok.token_bytes(int(t))
        return tok.decode([int(t)]).encode("utf-8")
    except (IndexError, KeyError, ValueError):
        return b""


def logprob_entry(tok, t, n_top: int) -> dict | None:
    """OpenAI's logprobs entry for token t - {token, logprob, bytes, top_logprobs: [{token, logprob, bytes}, ...]} -
    from its LP line (LpToken.lp; at most n_top alternatives); None for a token without one (the loop guard's own
    end of thinking).  A token split inside a UTF-8 character has the replacement character in `token` and its own
    bytes in `bytes`, as OpenAI's do."""
    lp = getattr(t, "lp", None)
    if lp is None:
        return None

    def piece(i, v):
        b = token_piece(tok, i)
        return {"token": b.decode("utf-8", "replace"), "logprob": v if math.isfinite(v) else LOGPROB_FLOOR,
                "bytes": list(b)}
    e = piece(int(t), lp[0])
    e["top_logprobs"] = [piece(i, v) for i, v in lp[1][:n_top]]
    return e


def model_management_setting(cfg: dict | None, env=None) -> bool:
    """llama.cpp's model management (/models/load, /models/unload, /models/sse: load the model, free its GPU memory,
    watch its state from an API client or a command): on unless the run config has "model_management": false or
    SAGUARO_MODEL_MANAGEMENT says 0 / off / false / no; the environment, when set, wins over the config."""
    on = ("1", "on", "true", "yes")
    env = os.environ if env is None else env
    v = env.get("SAGUARO_MODEL_MANAGEMENT")
    if v is not None and str(v).strip():
        return str(v).strip().lower() in on
    v = (cfg or {}).get("model_management")
    return bool(v) if v is not None else True


def load_by_name_setting(cfg: dict | None, env=None) -> bool:
    """Load by name: a chat request whose "model" names another installed config switches to it first, the way the
    Models page's Switch and llama.cpp's /models/load do, so a client that sends a model id (opencode, Hermes) gets
    that model, like llama-swap.  OFF unless the run config has "load_by_name": true or SAGUARO_LOAD_BY_NAME says
    1 / on / true / yes (the environment, when set, wins over the config), and it needs model management on
    (`model_management_setting`): the switch is model management's own."""
    on = ("1", "on", "true", "yes")
    env = os.environ if env is None else env
    v = env.get("SAGUARO_LOAD_BY_NAME")
    if v is not None and str(v).strip():
        want = str(v).strip().lower() in on
    else:
        want = bool((cfg or {}).get("load_by_name"))
    return want and model_management_setting(cfg, env)


def loop_guard_setting(cfg: dict | None, env=None) -> bool:
    """The loop guard (serve/loop_guard.py), on unless SAGUARO_LOOP_GUARD says 0 / off / false / no
    or the run config has "loop_guard": false; the environment, when set, wins over the config."""
    off = ("0", "off", "false", "no", "none")
    env = os.environ if env is None else env
    v = env.get("SAGUARO_LOOP_GUARD")
    if v is not None and str(v).strip():
        return str(v).strip().lower() not in off
    v = (cfg or {}).get("loop_guard")
    return str(v).strip().lower() not in off if isinstance(v, str) else v is not False and v != 0


def priority_setting(cfg: dict | None):
    """The run config's "priority" (R122): the queue level a request without its own runs at - "low" makes
    every request on this server step aside for other clients' normal ones (a llama-swap alias of the same model
    sets it in its params).  None: normal, as ever; a bad value refuses to start the server."""
    v = (cfg or {}).get("priority")
    if v is None:
        return None
    try:
        return priority_level(v)
    except ValueError as e:
        raise SystemExit(f"[saguaro] config priority: {e}")


def request_priority(sampling) -> int:
    """The level this request runs at: what Service.check_request put in "_priority" (R122); a caller that never
    went through it (an older integration, a test) is normal."""
    p = (sampling or {}).get("_priority")
    return p if p in (PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW) else PRIORITY_NORMAL


def client_stop_strings(req: dict | None) -> list[str]:
    """A request's own stop strings: OpenAI's "stop" (a string or a list) or Anthropic's "stop_sequences" (from
    Strata #454 - they were ignored, so a client that relied on them got text past its marker)."""
    s = (req or {}).get("stop")
    if s is None:
        s = (req or {}).get("stop_sequences")
    if isinstance(s, str):
        s = [s]
    return [x for x in s if isinstance(x, str) and x][:16] if isinstance(s, list) else []


# the penalties' window at most: what the engines count (GLM's and DeepSeek's samplers keep the last 4096 tokens) -
# a request's penalty_last_n / repeat_last_n of -1 (llama.cpp's "the whole context") asks for this
PENALTY_WINDOW_MAX = 4096

# a request's sampling fields and their JSON type (True: an integer); llama.cpp's names for two of them
SAMPLING_TYPES = (("temperature", False), ("top_p", False), ("min_p", False), ("presence_penalty", False),
                  ("frequency_penalty", False), ("repetition_penalty", False), ("top_k", True),
                  ("penalty_last_n", True), ("seed", True))


def check_request_sampling(req: dict) -> None:
    """A request's sampling fields, read the way the run config's are, or a 400 (ValueError) that names the field.
    Some clients send numbers as strings ("temperature": "0.7"): read and used.  A value that cannot be read as its
    type ("temperature": "warm", "top_k": 40.5, "seed": true) is refused - before, it replaced the configured default
    and the engine line left it out, so the answer was silently greedy.  Ranges behave as before (a temperature <= 0 is
    greedy, top_k 0 or above 64 the widest list, a seed <= 0 none) and an absent or null field is not checked.
    llama.cpp's repeat_penalty / repeat_last_n are taken as repetition_penalty / penalty_last_n (those win); the window
    means what it means there (penalty_keys): 0 = no penalties, -1 = the whole context, below -1 refused."""
    if not isinstance(req, dict):
        return
    if req.get("repeat_penalty") is not None and req.get("repetition_penalty") is None:
        req["repetition_penalty"] = req["repeat_penalty"]
    window_key = "penalty_last_n"
    if req.get("penalty_last_n") is None and req.get("repeat_last_n") is not None:
        req["penalty_last_n"] = req["repeat_last_n"]
        window_key = "repeat_last_n"
    for key, integer in SAMPLING_TYPES:
        value = req.get(key)
        if value is None:
            continue
        what = f"{window_key if key == 'penalty_last_n' else key}: {'an integer' if integer else 'a number'}"
        if isinstance(value, str) and value.strip():
            try:
                num = float(value)
            except ValueError:
                raise ValueError(what) from None
            if num != num or num in (float("inf"), float("-inf")):
                raise ValueError(what)
            if integer:
                if num != int(num):
                    raise ValueError(what)
                num = int(num)
            elif re.fullmatch(r"\s*[+-]?\d+\s*", value):
                num = int(value)
            req[key] = value = num
        if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
            raise ValueError(what)
    if isinstance(req.get("penalty_last_n"), int) and req["penalty_last_n"] < -1:
        raise ValueError(f"{window_key}: -1 (the whole context), 0 (no penalties) or a positive number of tokens")
    guard = req.get("loop_guard")                   # the loop guard for this request only (serve/loop_guard.py)
    if guard is not None and not isinstance(guard, bool):
        if isinstance(guard, str) and guard.strip().lower() in ("true", "false", "on", "off", "1", "0"):
            req["loop_guard"] = guard.strip().lower() in ("true", "on", "1")
        elif isinstance(guard, int) and guard in (0, 1):
            req["loop_guard"] = bool(guard)
        else:
            raise ValueError("loop_guard: true or false")
    # PLAN-turbo-live: this request's lossy options (serve/turbo_live.py): {"drop": N, "skip": "off"|"ssd"|"ram"}
    # (OpenAI's extra-body form: a top-level field), checked and kept in its canonical form; absent = the model's
    # live Turbo setting (Service.check_request), off = today's exact request
    if "turbo" in req:
        req["turbo"] = turbo_live.normalize_turbo(req.get("turbo"))


def logit_bias_of(value, vocab_size: int | None = None) -> dict[int, float]:
    """OpenAI's logit_bias - {"<token id>": bias, ...}, bias -100..100 - as {id: bias}; {} for none.  A bias of -100
    bans the token.  Anything else is a 400 (ValueError) naming the field, before anything is sent."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('logit_bias: an object of {"<token id>": bias}')
    out = {}
    for token, bias in value.items():
        if isinstance(token, bool) or not re.fullmatch(r"[0-9]+", str(token)):
            raise ValueError("logit_bias: token ids are non-negative integers")
        tid = int(token)
        if tid > 2 ** 31 - 1 or (vocab_size and tid >= vocab_size):
            raise ValueError(f"logit_bias: token {tid} is outside the model's vocabulary")
        if isinstance(bias, str):
            try:
                bias = float(bias)
            except ValueError:
                raise ValueError("logit_bias: biases are numbers from -100 to 100") from None
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not -100 <= bias <= 100:
            raise ValueError("logit_bias: biases are numbers from -100 to 100")
        out[tid] = float(bias)
    if vocab_size and sum(b <= -100 for b in out.values()) >= vocab_size:
        raise ValueError("logit_bias: it cannot ban the whole vocabulary")
    return out


# A FROZEN engine (Strata #1317).  Silence alone proves nothing: on a slow PC the engine reads a long prompt chunk for
# minutes without a line.  But one that prints nothing for ENGINE_STALL_S and in that time uses no CPU, reads or writes
# no disk and leaves its GPUs idle is not slow, it is stuck (a deadlock, a driver stall), and waiting cannot help: it is
# ended, the request gets an error, and the next request starts it again.  One that is silent but working is never
# ended.  SAGUARO_ENGINE_STALL_S sets it (0 = off); it needs psutil (setup installs it), without it nothing is ended.
def _stall_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("SAGUARO_ENGINE_STALL_S", "90")))
    except ValueError:
        return 90.0


ENGINE_STALL_S = _stall_seconds()
# A request whose OWN lines stop (T / PP / DONE / ERR for its rid) is not slow, it is lost: the engine is working on
# other requests, so the engine-wide stall check above never fires, and this one waited for a line that no longer came
# while it held its place in the FIFO - four of those under --parallel and every request behind them waited for ever
# (nothing was logged, nothing expired).  After REQUEST_QUIET_S with no line of its own the request ends with an error
# (its place is given back) and the timeout is logged.  A prompt read reports PP lines as it goes, so quiet means lost,
# not busy; the waiting requests an engine admits later print "waits for a context" ... and 900 s is far past any read.
def _quiet_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("SAGUARO_REQUEST_QUIET_S", "900")))
    except ValueError:
        return 900.0


REQUEST_QUIET_S = _quiet_seconds()


# The longest a stopped request's own lines (T / PP / DONE) are waited for when it is drained after a cancel with
# --parallel: a STOP an engine does not honour (or a rid it has already let go) would otherwise hold the FIFO place for
# ever.  With one conversation the drain waits for the DONE (its lines share one queue with the next request's) and
# this is only when it says it is still waiting; a stuck engine there is ended by the stall watchdog (ENGINE_STALL_S).
def _drain_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("SAGUARO_DRAIN_S", "60")))
    except ValueError:
        return 60.0


DRAIN_S = _drain_seconds()
STALL_CPU_SHARE = 0.02      # the engine may use 2% of one core over the window and still count as stuck: its idle
                            # threads' polls (the GLM service threads sleep 2 ms at a time when quiet); work is cores
STALL_IO_EPS_B = 1 << 20    # bytes it may read or write in that window
STALL_GPU_BUSY = 5.0        # a GPU busier than this (%, NVML) is work


def engine_frozen(base: tuple[float, int], now: tuple[float, int], window_s: float) -> bool:
    """True when two (CPU seconds, disk bytes) samples of the engine process, window_s apart, show no work between."""
    return (now[0] - base[0]) <= max(0.5, STALL_CPU_SHARE * window_s) and (now[1] - base[1]) <= STALL_IO_EPS_B


# SAGUARO_DS4_RESTORE_POINTS (4): the DeepSeek engine keeps restore points where a later prompt is likely to leave this
# one - its GEN line carries marks=<pos>,..., tail=<pos> and shared=<pos> - besides the end of the last prompt, the
# position before it and the anchor it keeps anyway.  N (1-16) turns them on, and is also the engine's cap a
# conversation (it reads the same setting); 0: no keys, every GEN line as before.  SAGUARO_DS4_RESTORE_TAIL (512): how far
# before the prompt's end the tail point sits (0: none); it goes as tail=, not in marks=, because the engine leaves it
# out where its read stop costs more than reading those tokens again (most experts outside VRAM, #668).
RESTORE_LEAST = 1024        # no point before this many tokens: such a start is read again in about the time a point costs


def restore_point_settings(env=None) -> tuple[bool, int]:
    """(on, tail) from SAGUARO_DS4_RESTORE_POINTS and SAGUARO_DS4_RESTORE_TAIL."""
    env = os.environ if env is None else env

    def whole(name, default):
        try:
            return max(0, int(str(env.get(name, default)).strip()))
        except ValueError:
            return default
    return whole("SAGUARO_DS4_RESTORE_POINTS", 4) > 0, whole("SAGUARO_DS4_RESTORE_TAIL", 512)


def restore_turns(env=None) -> int:
    """SAGUARO_DS4_RESTORE_TURNS (6; 0: only the last user turn, as before)."""
    env = os.environ if env is None else env
    try:
        return max(0, int(str(env.get("SAGUARO_DS4_RESTORE_TURNS", 6)).strip()))
    except ValueError:
        return 6


RESTORE_RECENT = 100000   # the agent context engines' fresh tail (hermes-lcm: 32 messages, at most 100K tokens)


def deepseek_restore_points(ids, user_id, system_id=None, tail=512, least=RESTORE_LEAST, assistant_id=None,
                            recent=RESTORE_RECENT, spacing=65536, turns=None) -> dict:
    """Where the DeepSeek engine keeps restore points for this prompt -> {"marks": [pos, ...], "tail": pos, "shared":
    pos} (any may be missing; {}: none).  From the prompt's ids as the template wrote them: a turn opens with its role's special
    token (<｜User｜>, a tool result's turn too; <｜System｜> for a system message after the first user turn), and text
    inside a message never encodes to one (render_ids), so the positions are the template's own.

    - the start of the last user turn: its last message edited, or sent again after another answer, goes back there;
    - "tail": `tail` tokens before the end, when the last turn is longer than that: a new question after the same long
      document in the same message (the engine keeps it where its read stop costs less than reading those tokens);
    - "shared": the end of the system prompt and its tools (the first user turn): another conversation behind the same
      start goes back there, and the engine gives that point up last.
    Positions under `least` are left out: a start that short is read again in about the time a point costs."""
    if user_id is None or len(ids) < 2:
        return {}
    first = last = None
    starts = []   # every message start after the first user turn (an agent's context engine cuts at any of them: a
    for i, t in enumerate(ids):   # tool loop is one user turn of many assistant / tool messages)
        if t == assistant_id and first is not None:
            starts.append(i)
        if t == user_id:
            if first is not None:
                starts.append(i)
            first = i if first is None else first
            last = i
        elif t == system_id and first is not None:
            starts.append(i)
            last = i
    marks, out = [], {}
    if first is not None and first >= least:
        out["shared"] = first
    if last is not None and last >= least and last != first:
        marks.append(last)
    # SAGUARO_DS4_RESTORE_TURNS (6): up to that many earlier turn starts besides the last, spread over the conversation
    # (a compaction or chain summary that keeps the history up to one of them reads only what follows it)
    k = restore_turns() if turns is None else turns
    early = [p for p in starts if p >= least and last is not None and p < last]   # (not the reply's own opener)
    if k > 0 and early:
        # the start nearest `recent` tokens before the end first (where a context engine keeping its fresh tail cuts),
        # the rest spread over the conversation
        # the rest on a fixed grid (the last message start at or before each multiple of `spacing`), newest first: the
        # same positions turn after turn, so an incremental turn finds them held and asks for no new read stops
        near = min(early, key=lambda p: abs((len(ids) - recent) - p))
        grid = []
        for g in range(spacing, len(ids), spacing):
            q = [p for p in early if p <= g]
            if q and q[-1] not in grid:   # (the near point may be one too: the window does not shift)
                grid.append(q[-1])
        pick = [near] + grid[::-1][:k - 1]
        marks.extend(p for p in pick if p not in marks)
        marks.sort()
    t = len(ids) - tail
    if tail > 0 and t >= least and (last is None or t > last):
        out["tail"] = t
    if marks:
        out["marks"] = sorted(marks)
    return out


def narrate_start(log_path: str, offset: int, args: list, done: threading.Event, heartbeat=20.0) -> None:
    """While the engine starts, say in the server window what it is doing, from its log: the start reads tens of GB
    into RAM and locks part of it for the GPU, and on many PCs everything is slow or frozen for a minute or more -
    people closed the window thinking it had hung.  The warning comes at that step, not after it."""
    gb = 0.0
    if "--native" in args:                              # about the size of the experts it will read
        try:
            gb = os.path.getsize(args[args.index("--native") + 1]) / 1e9
        except (OSError, IndexError):
            pass
    size = f"about {gb:.0f} GB" if gb >= 1 else "tens of GB"
    t0 = last = time.time()
    said = set()

    def say(key, text):
        nonlocal last
        if key not in said:
            said.add(key)
            last = time.time()
            print(text, flush=True)

    say("weights", "[saguaro] starting the engine: reading the model's weights ...")
    pos = offset
    while not done.wait(0.5):
        try:
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk.count(b"\n"):
            cut = chunk.rfind(b"\n") + 1
            pos += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if "PLE on" in line or "expert arena:" in line:
                    say("arena", f"[saguaro] loading the experts into RAM ({size}) and locking part of them for the GPU.\n"
                                 "         YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.\n"
                                 "         Please wait and don't close this window; the browser opens when it is ready.")
                elif " loaded " in line and "GiB at" in line:
                    say("loaded", "[saguaro] experts loaded: " + line.split(" loaded ", 1)[1].strip() +
                        f" ({time.time() - t0:.0f} s so far)")
                elif "expert cache " in line and " slots, " in line and "auto" not in line:
                    n = line.split("expert cache ", 1)[1].split(";")[0].replace(" slots,", " experts,").strip()
                    say("cache", f"[saguaro] filling the GPU's expert cache ({n}) ...")
                elif "session is up" in line:
                    say("up", "[saguaro] almost ready ...")
                elif "WARNING - " in line:   # the engine's own (e.g. Windows' commit limit capping the RAM tier)
                    print("[saguaro] WARNING: " + line.split("WARNING - ", 1)[1].strip(), flush=True)
        if time.time() - last > heartbeat:
            last = time.time()
            print(f"[saguaro] still starting ({time.time() - t0:.0f} s) - please wait ...", flush=True)


class StrataEngine:
    """The resident engine: `saguaro-glm --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then
    `DONE ...`.  Requests are serialized by the service's FIFO, so one pipe is enough.

    Per-request sampling rides the same line as engine-side keys between max_new and the ids
    (`temperature=F top_p=F top_k=N seed=N`, the engine's own spelling).  An absent temperature keeps the
    engine's default, which is greedy; `temperature=0` means the same thing, so it is not forwarded.
    """

    gpu_busy = None   # () -> bool: the engine's GPUs are working (the service's telemetry); kept across restarts
    multi = 0         # READY ... multi=<n> (set per start; the class default for engines built without __init__)

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None):
        self.spawn = (exe, list(args), cwd, log, env)   # to start it again after it died (issue #27)
        paths = {k: v for k, v in zip(args, args[1:]) if k in ("--native", "--pack")}
        self.model_path = paths.get("--native") or paths.get("--pack", "pack/full")
        if Path(exe).name.startswith(("saguaro-ik2", "maya-ik2")) and "--model" in args[:-1]:   # (its GGUF: /props)
            self.model_path = args[args.index("--model") + 1]
        self.log_path = log
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        loading = threading.Event()                     # set once READY: the narrator below stops
        log_start = os.path.getsize(log) if log else 0
        if log:
            threading.Thread(target=narrate_start, args=(log, log_start, args, loading),
                             daemon=True).start()
        self.proc = subprocess.Popen([exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env)
        track_spawned(self.proc)
        self.max_context = 0
        self.can_stop = False            # the engine honours a STOP line mid-request (READY <ctx> stop)
        self.can_pause = False           # ... and PAUSE: a STOP that keeps the state with what was written (fair slices)
        self.multi = 0                   # READY ... multi=<n>: n conversations at once (requests tagged rid=<id>)
        self._routes: dict[int, queue.Queue] = {}       # rid -> its lines (multi)
        self._routes_lock = threading.Lock()
        self._rids = itertools.count(1)
        self._tls = threading.local()                   # the request this thread is generating (pause())
        self._sent: dict[int, list] = {}                # rid -> [sent at, heard anything] (fair reads: who waits)
        self.last = {}
        self.info = {}                   # INFO key=value facts (engine 0.1.8+): kv, expert slots, ... (Monitor tab)
        self.vram = []                   # VRAM lines: what each part holds on its GPU (the Monitor's GPU map)
        self.prefill_tok_s_mean = None
        self.progress = None             # (read, total, reuse, ms) prompt progress while a prompt is read, from PP lines
        try:                             # a ready-made engine's BUILD.json says its version
            self.info["version"] = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        except (OSError, ValueError):
            self.info["version"] = None
        head = []                        # its other lines before READY (saguaro-ik2's start lines)
        for line in self.proc.stdout:
            if line.startswith("NOTE "):
                print("[saguaro] " + line[5:].strip(), flush=True)
            elif not line.startswith(("INFO ", "VRAM ", "READY")) and len(head) < 400:
                head.append(line)
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            if line.startswith("VRAM "):
                part = gpu_map.parse(line)
                if part:
                    part["at"] = time.monotonic()   # (the Monitor says how fresh each card's split is)
                    self.vram.append(part)
            if line.startswith("READY"):
                f = line.split()
                self.max_context = int(f[1])
                self.can_stop = "stop" in f[2:]
                self.can_pause = "pause" in f[2:]
                self.multi = next((int(x[6:]) for x in f[2:] if x.startswith("multi=") and x[6:].isdigit()), 0)
                break
        loading.set()
        if self.max_context <= 0:
            # its output closed: it is gone, even before the OS says so (as _pump) - alive() is False at once, not
            # only after the process is reaped (a busy CI runner saw it alive right after a failed restart)
            self.ended = True
            raise RuntimeError("the engine exited before it was ready" + (f" (see {log})" if log else ""))
        # (#1026) a start that came out short of its VRAM reserve (INFO reserve_short=<MiB>) - the engine measured
        # what it allocates after its experts' plan and keeps that free from its next start on: started once more
        # now, so the first start after an update or a new setting never serves with the reserve used up
        if self.info.get("reserve_short") and not (env or os.environ).get("SAGUARO_RESERVE_RESTARTED"):
            print(f"[saguaro] this start came out short of the VRAM reserve: the engine starts once more and keeps "
                  f"{self.info['reserve_short']} MiB more free beside it (measured on this PC)", flush=True)
            self.close()
            if self.log is not subprocess.DEVNULL:
                self.log.close()
            self.__init__(exe, args, cwd, log, dict(env if env is not None else os.environ, SAGUARO_RESERVE_RESTARTED="1"))
            return
        # (from PR #41, midhatn) a locally built engine can sit next to another release's BUILD.json: engines that
        # report their own version (INFO engine=, 0.1.8+) win, the manifest stays the fallback for older ones
        if self.info.get("engine"):
            self.info["version"] = str(self.info["engine"])
        if log:
            self._context_cost(log_start)
        if self.info.get("engine_kind") == "ik2":
            # saguaro-ik2 says where the model went on its stdout, before INFO: the Monitor's facts, and the log's
            for k, v in ik2_log_facts("".join(head), self.max_context).items():
                self.info.setdefault(k, v)                # (an engine that says it itself, INFO: its own figure)
            self._ik2_log(head)
        # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True).start()

    def _context_cost(self, offset: int):
        """What this start's context costs in VRAM, from the GLM engine's own start lines ("CUDA0 VRAM before the
        expert pool: ... state/KV (32768 ctx) 0.66"), summed over its GPUs: the dashboard scales it to the context
        sizes it offers (measured here, so it fits this model, this split and these cards)."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                text = f.read(8 << 20).decode("utf-8", "replace")
        except OSError:
            return
        per = {}
        for m in re.finditer(r"\b((?:CUDA|HIP)\d+) VRAM before the expert pool:.*?state/KV \((\d+) ctx\) ([\d.]+)", text):
            per[m.group(1)] = (int(m.group(2)), float(m.group(3)))
        if per:
            self.info["kv_ctx"] = max(c for c, _ in per.values())
            self.info["kv_gb"] = round(sum(g for _, g in per.values()), 3)

    def _ik2_eof(self):
        """After QUIT, saguaro-ik2 only: its stdin closed.  Its exit waits for the thread that reads stdin (exit()
        flushes every stream, and that thread holds stdin's lock inside its read), so it ends at the end of input."""
        if (getattr(self, "info", None) or {}).get("engine_kind") == "ik2":
            try:
                self.proc.stdin.close()
            except (OSError, ValueError):
                pass

    def _ik2_done(self, stats):
        """saguaro-ik2's "decode stats:" with the DONE after it: the answer's hit fields, and the Monitor's reading.  An
        engine that says its own path (INFO path=) prints STAT and DONE's hit fields itself: nothing is made up here."""
        if stats and not self.info.get("path"):
            if stats["lookups"]:
                self.last.update(stats)
            st = ik2_stat(self.info, self.last)
            if st:
                self._parse_stat("STAT " + " ".join(f"{k}={v}" for k, v in st.items()))

    def _ik2_log(self, lines):
        """saguaro-ik2's plain stdout lines (its start, a request's stats) into the engine log, where the other
        engines' are."""
        log = getattr(self, "log", None)
        if hasattr(log, "write"):
            try:
                log.write("".join(lines))
                log.flush()
            except (OSError, ValueError):
                pass

    def set_context(self, n: int):
        """The next start's context: --max-context in the command (restart() runs it)."""
        exe, args, cwd, log, env = self.spawn
        args = list(args)
        if "--max-context" in args[:-1]:
            args[args.index("--max-context") + 1] = str(int(n))
        else:
            args += ["--max-context", str(int(n))]
        self.spawn = (exe, args, cwd, log, env)

    def stop(self, timeout: float = 60.0):
        """Ask the engine to end (QUIT: it frees its GPU memory itself), and end it if it does not in time - then wait
        until the GPUs have that memory back."""
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=timeout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            try:
                self.proc.kill()
                self.proc.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.ended = True
        wait_gpu_release(self.proc.pid)

    def _pump(self, proc=None, lines=None):
        # This start's process and line queue: a later start (restart) has its own.  An engine's output can close
        # well after a new one started - GLM's link mode: the AMD GPU's helper program shares the lead's output and
        # ends a little after it - and that late end must not mark the NEW engine ended (every request then started
        # the engine again, about a minute each)
        proc = proc if proc is not None else self.proc
        lines = lines if lines is not None else self.lines
        for line in proc.stdout:
            if proc is not self.proc:                   # replaced: what is left of its output is nobody's
                continue
            if line.startswith("VRAM "):
                self._vram_line(line)
                continue
            # END (maya-ds4, ahead of a request's DONE): a GPU's state is unknown and the engine ends once that DONE
            # is out - taken as ended now, so the next request, even one already waiting, starts a fresh engine
            # instead of writing to this one while it exits
            if line.startswith("END "):
                print("[saguaro] " + line[4:].strip(), flush=True)
                self.ended = True
                continue
            if self.multi and self._route(line):
                continue
            lines.put(line)
        if proc is not self.proc:                       # an engine already replaced: the running one is not ended
            return
        self.ended = True                               # its output closed: it is gone, even before the OS says so
        lines.put(None)
        with self._routes_lock:
            for q in self._routes.values():
                q.put(None)

    def _vram_line(self, line: str) -> None:
        """A VRAM line after the start (the engine's periodic re-print of a card whose split moved - elastic
        contexts, VLEND, a picture's room; PLAN-vram-live): the Monitor's map of that GPU follows it."""
        part = gpu_map.parse(line)
        if part:
            part["at"] = time.monotonic()
            hit = [v for v in self.vram if v.get("dev") == part["dev"]]
            if hit:
                self.vram = [part if v.get("dev") == part["dev"] else v for v in self.vram]
            else:
                self.vram = self.vram + [part]      # (a card that only appears at runtime: the map gains it)

    _TAGGED = ("T ", "LP ", "PP ", "DONE ", "ERR ")

    def _route(self, line: str) -> bool:
        """Several conversations at once: a request's own lines (T / LP / PP / RESUME / DONE / ERR <rid> ...) go to its
        queue, the engine's (STAT, NOTE) are handled here; True when the line was taken."""
        if line.startswith(self._TAGGED):
            f = line.split(None, 2)
            if len(f) >= 2 and f[1].isdigit():
                with self._routes_lock:
                    q = self._routes.get(int(f[1]))
                if q is not None:
                    q.put(line)
                    return True
                return line.startswith(("T ", "LP ", "PP ", "DONE "))   # a request already gone: dropped
            return False
        if line.startswith("STAT "):
            self._parse_stat(line)
            return True
        if line.startswith("NOTE "):
            print("[saguaro] " + line[5:].strip(), flush=True)   # (glm tags its loop guard's NOTE with r<rid>: the
            return True                                          #  printed message says which answer it closed)
        if line.startswith("RESUME "):
            # (glm, before a restored read: this many of the prompt's tokens are not read again. The reuse base rides
            # the PP lines for these engines ("no RESUME base needed", the solo reader), so the line is only taken -
            # routed to its own request when one is live: it says that read began - and never left for the shared
            # queue, which in multi mode nothing reads)
            f = line.split()
            if len(f) >= 3 and f[1].isdigit():
                with self._routes_lock:
                    q = self._routes.get(int(f[1]))
                if q is not None:
                    q.put(line)
            return True
        return False

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), a GPU's fault it
        caught after a request (exit 3), a GPU out of memory in a request
        (saguaro-ik2: exit 4), Saguaro's stall watchdog, else RAM."""
        tail = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace")
        except (OSError, TypeError):
            pass
        for line in reversed(tail.splitlines()):
            if "issue #29" in line:
                return ("The engine stopped itself because it had stopped making progress - a hang it caught. Its log "
                        "line: " + line.strip() + " - please report it at github.com/ksanislo/saguaro/issues.")
            if "ran out of memory during this request" in line:
                return ("The engine ran out of GPU memory during the request and ended (its GPU state is not known "
                        "after that); a fresh engine starts with the next request. A smaller --chunk, context or prompt "
                        "needs less memory. Its log line: " + line.strip())
            if "a GPU's state is unknown" in line:
                return ("The engine stopped itself because a GPU took a fault inside a kernel (an illegal address or "
                        "the like), after which that GPU's state is lost; a fresh engine starts with the next request. "
                        "Its log line: " + line.strip() + " - the lines above it name the GPU and the error; please "
                        "report it at github.com/ksanislo/saguaro/issues.")
        if getattr(self, "stalled", False):         # (Saguaro's own stall watchdog ended it: _end)
            return ("Saguaro ended it because it had stopped making progress (no output and no CPU, disk or GPU work); "
                    "the lines at the end of its log say where it was - please report it at "
                    "github.com/ksanislo/saguaro/issues.")
        if getattr(getattr(self, "proc", None), "returncode", None) in (-2, -15):   # (#539: read as a lack of RAM)
            return ("A signal from outside ended it (SIGINT or SIGTERM: Ctrl+C, kill, a service manager), not a lack "
                    "of RAM.")
        return ("The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | "
                "grep -i -E 'killed process|out of memory'); Windows slows down instead. Close other programs or use a "
                "smaller model (Q2_0 / IQ2_XS).")

    def alive(self) -> bool:
        return not getattr(self, "ended", False) and self.proc.poll() is None

    def _activity(self):
        """(CPU seconds, disk bytes read + written) the engine process has used so far; None without psutil."""
        try:
            import psutil
            p = psutil.Process(self.proc.pid)
            t, io = p.cpu_times(), p.io_counters()
            return t.user + t.system, io.read_bytes + io.write_bytes
        except Exception:  # noqa: BLE001 - no psutil or no access: nothing to decide on
            return None

    def _stall_check(self, heard: float, state: dict) -> str | None:
        """Called while the engine is silent (since `heard`, monotonic): a baseline soon after its last line, and at
        ENGINE_STALL_S of silence the comparison - no CPU, disk or GPU work in all that time: why it is ended."""
        t = time.monotonic()
        age = t - heard
        if "base" not in state:
            state["base"], state["t"], state["next"] = self._activity(), t, ENGINE_STALL_S
            return None
        if state["base"] is None or age < state["next"]:
            return None
        now = self._activity()
        try:
            gpu = bool(self.gpu_busy()) if self.gpu_busy is not None else False
        except Exception:  # noqa: BLE001 - a telemetry hiccup must not decide anything
            gpu = False
        if now is not None and not gpu and engine_frozen(state["base"], now, t - state["t"]):
            return (f"the engine said nothing for {age:.0f} s and used no CPU, disk or GPU in that time, so it was stuck "
                    "and has been ended (SAGUARO_ENGINE_STALL_S sets this, 0 = off)")
        state["base"], state["t"], state["next"] = now, t, age + ENGINE_STALL_S   # working: look again a window later
        return None

    def _end(self, why: str):
        """End a stuck engine now; the next request starts it again (Service.run)."""
        print(f"[saguaro] {why}", flush=True)
        self.ended = self.stalled = True
        try:
            self.proc.kill()
            self.proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def exit_code(self):
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def restart(self):
        """Start the engine again (the same command) after it died; the new process has its own line queue."""
        try:
            self.proc.kill()
            self.proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            pass
        wait_gpu_release(self.proc.pid)
        with self._routes_lock:                         # requests still waiting on the dead one: it is gone
            for q in self._routes.values():
                q.put(None)
        info = dict(self.info)
        # The rid counter and the routes go on across the start: __init__ makes a new counter (from 1) and a new
        # table, and a request that comes in while the old ones are still unwinding would then be given the number of
        # one of them - its lines went to THAT request's queue, the request whose number it took popped its entry, and
        # it waited for its own lines for ever, holding its place in the FIFO (four of those under --parallel and no
        # request starts any more, the queue filling with requests nothing ever takes: the engine restart was the
        # trigger, the silent wait the bug).
        rids, routes, routes_lock, sent = self._rids, self._routes, self._routes_lock, self._sent
        ctx = self.max_context
        self.ended = self.stalled = False
        try:
            self.__init__(*self.spawn)
        except Exception:
            # (#351: a start that fails - a GPU that fell off the bus - left max_context 0, so every later request
            #  failed its size check with "exceeds the context (0)" before ensure_engine could try again; the context
            #  stays what it was, the next request starts it again or says why it cannot)
            self.max_context = ctx
            self.info = {**info, **self.info}
            self._rids, self._routes, self._routes_lock, self._sent = rids, routes, routes_lock, sent
            raise
        self._rids, self._routes, self._routes_lock, self._sent = rids, routes, routes_lock, sent
        self.info = {**info, **self.info}

    @property
    def last(self) -> dict:
        """The last DONE's figures.  With several conversations at once (multi), the figures of the request this thread
        runs (#310): requests decoded together finish within a step of each other, and one shared field let a request
        report another's timings and cached tokens.  Each request thread has its own (an empty one until its DONE)."""
        if self.multi:
            t = self._tls
            if not hasattr(t, "last"):
                t.last = {}
            return t.last
        return self._last

    @last.setter
    def last(self, v: dict):
        if self.multi:
            self._tls.last = v
        else:
            self._last = v

    def _parse_done(self, line):
        f = line.split()
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}
        if len(f) >= 9:                                   # the conversation cache's fields (engine 0.1.3+)
            self.last.update(drafts_accepted=int(f[6]), drafts_offered=int(f[7]), reused=int(f[8]))
        if len(f) >= 11:                                  # decode hit rate fields
            self.last.update(hits=int(f[9]), lookups=int(f[10]))

    def _parse_stat(self, line):
        """STAT key=value ... (the GLM engine, every few tokens): live expert-tier counters for the Monitor."""
        st = {}
        for kv in line.split()[1:]:
            k, _, v = kv.partition("=")
            try:
                st[k] = float(v)
            except ValueError:
                st[k] = v
        st["time"] = time.time()
        self.stat = st
        hist = getattr(self, "stat_history", None)
        if hist is None:
            hist = self.stat_history = collections.deque(maxlen=120)
        hist.append(st)

    IK2_GEN_KEYS = ("temperature", "top_p", "top_k", "seed", "spec")   # all saguaro-ik2 reads on a GEN line

    def gen_keys(self, sampling: dict) -> str:
        """sampling_keys for this engine: saguaro-ik2 gets only the keys it reads (no budget, penalties, Turbo, gt=)."""
        keys = self.sampling_keys(sampling)
        if (getattr(self, "info", None) or {}).get("engine_kind") != "ik2":
            return keys
        out = [w for w in keys.split() if w.partition("=")[0] in self.IK2_GEN_KEYS]
        # its own top_k: up to its 1000 candidates, and 0 is "off" (the other engines' cap of 64 is not its)
        tk = sampling.get("top_k")
        out = [f"top_k={min(tk, 1000)}" if w.startswith("top_k=") else w for w in out
               if not (w.startswith("top_k=") and tk < 1)]
        return "".join(" " + w for w in out)

    @staticmethod
    def sampling_keys(sampling: dict) -> str:
        keys = ""
        t = sampling.get("temperature")
        tb, te = sampling.get("_think_budget"), sampling.get("_think_end")   # the run config's (StrataService.run)
        think = f" think_budget={tb} think_end={te}" if isinstance(tb, int) and isinstance(te, int) and tb > 0 and \
            te >= 0 else ""
        soft, stops = sampling.get("_think_soft"), sampling.get("_think_stops")
        if think and isinstance(soft, int) and soft > 0 and isinstance(stops, str) and stops:
            think += f" think_soft={soft}{stops}"   # (#746: the engine closes at the sentence's end, a prefix of the state)
        # setup's calibration (tools/calibrate.py): engine settings for this request only, measured without a restart
        tune, tune_keys = sampling.get("strata_tune"), ""
        if isinstance(tune, dict):
            for k in ("pcie_frac", "spec_min_p", "spec_cum"):
                v = tune.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
                    tune_keys += f" {k}={float(v)!r}"
            sd = tune.get("suffix_draft")   # the DeepSeek engine's prompt lookup beside DSpark: off (0) / on (1)
            if isinstance(sd, int) and not isinstance(sd, bool) and sd in (0, 1):
                tune_keys += f" suffix_draft={sd}"
            ct = tune.get("cpu_threads")   # the GLM engine's CPU lane threads (more: SAGUARO_GLM_CPU_LANE_MAX)
            if isinstance(ct, int) and not isinstance(ct, bool) and ct > 0:
                tune_keys += f" cpu_threads={ct}"
            rc = tune.get("read_chunks")   # ... and the pieces it reads an expert from the SSD in (x3)
            if isinstance(rc, int) and not isinstance(rc, bool) and 1 <= rc <= 16:
                tune_keys += f" read_chunks={rc}"
            mw = tune.get("miss_skip_w")   # the GLM engine's light-miss bar (SAGUARO_GLM_MISS_SKIP_W) for this request
            if isinstance(mw, (int, float)) and not isinstance(mw, bool) and 0.0 <= float(mw) <= 1.0:
                tune_keys += f" miss_skip_w={float(mw)!r}"
            mf = tune.get("miss_skip_from")   # ... and its first layer (SAGUARO_GLM_MISS_SKIP_FROM)
            if isinstance(mf, int) and not isinstance(mf, bool) and 0 <= mf <= 1000:
                tune_keys += f" miss_skip_from={mf}"
            sw = tune.get("spec")   # the GLM engine's DFlash decode: its verify window (the anchor and drafts), 2..16;
            if isinstance(sw, int) and not isinstance(sw, bool) and 1 <= sw <= 16:   # 1: no drafting (the token loop)
                tune_keys += f" spec={sw}"
            sn = tune.get("spec_n_min")   # ... drafts it keeps whatever their confidence
            if isinstance(sn, int) and not isinstance(sn, bool) and 0 <= sn <= 15:
                tune_keys += f" spec_n_min={sn}"
        bias = sampling.get("logit_bias")      # {id: bias}, checked by the handler (logit_bias_of)
        bias_key = (" logit_bias=" + ",".join(f"{int(i)}:{float(b):g}" for i, b in sorted(bias.items()))
                    if isinstance(bias, dict) and bias else "")
        penalty_keys = StrataEngine.penalty_keys(sampling)
        if isinstance(t, (int, float)) and not isinstance(t, bool) and float(t) <= 0.0:
            # temperature 0 is greedy: the engine reads temperature=0 as that whatever else the line says, and no other
            # sampler key goes (with only a config's top_p the engine took its sampled path: not deterministic) - but
            # the penalties do: they change the logits greedy picks from (#273: a greedy answer's loops)
            return (" temperature=0" + penalty_keys + think + tune_keys + turbo_live.turbo_keys(sampling) +
                    StrataEngine.projection_key(sampling) + bias_key + StrataEngine.restore_keys(sampling) +
                    StrataEngine.logprob_key(sampling) + StrataEngine.reply_key(sampling))
        if isinstance(t, (int, float)) and float(t) > 0.0:
            keys += f" temperature={float(t)!r}"
        tp = sampling.get("top_p")
        if isinstance(tp, (int, float)) and float(tp) < 1.0:
            keys += f" top_p={float(tp)!r}"
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            # the engine's sampled path keeps at most 64 candidates: 0 ("off") and wider lists get all 64
            keys += f" top_k={tk if 1 <= tk <= 64 else 64}"
        mp = sampling.get("min_p")
        if isinstance(mp, (int, float)) and 0.0 < float(mp) <= 1.0:
            keys += f" min_p={float(mp)!r}"
        keys += penalty_keys
        seed = sampling.get("seed")
        if isinstance(seed, int) and seed > 0:
            keys += f" seed={seed}"
        keys += think + tune_keys + turbo_live.turbo_keys(sampling)
        return (keys + StrataEngine.projection_key(sampling) + bias_key + StrataEngine.restore_keys(sampling) +
                StrataEngine.logprob_key(sampling) +
                StrataEngine.reply_key(sampling))

    @staticmethod
    def penalty_keys(sampling: dict) -> str:
        """The repetition / frequency / presence penalties (llama.cpp's repeat_penalty and repeat_last_n are read as
        repetition_penalty and penalty_last_n): only the ones that change something, and their window with them.  ""
        when none does - the GEN line is then the one it was without them.  The window as llama.cpp's: none given = the
        last 64 tokens, 0 = no penalties, -1 = the whole context - the engines count at most PENALTY_WINDOW_MAX."""
        keys = ""
        pln = sampling.get("penalty_last_n")
        if isinstance(pln, int) and not isinstance(pln, bool) and pln == 0:
            return keys
        rp = sampling.get("repetition_penalty")
        rp_on = isinstance(rp, (int, float)) and float(rp) != 1.0
        pf = sampling.get("frequency_penalty")
        pf_on = isinstance(pf, (int, float)) and float(pf) != 0.0
        pp = sampling.get("presence_penalty")
        pp_on = isinstance(pp, (int, float)) and float(pp) != 0.0
        if rp_on:
            keys += f" penalty_repeat={float(rp)!r}"
        if pf_on:
            keys += f" penalty_freq={float(pf)!r}"
        if pp_on:
            keys += f" penalty_present={float(pp)!r}"
        if rp_on or pf_on or pp_on:
            # a penalty without a window counts over nothing: the engine's default is the last 64 tokens
            if isinstance(pln, int) and not isinstance(pln, bool) and pln > 0:
                keys += f" penalty_last_n={pln}"
            elif pln == -1 and not isinstance(pln, bool):
                keys += f" penalty_last_n={PENALTY_WINDOW_MAX}"
            else:
                keys += " penalty_last_n=64"
        return keys

    @staticmethod
    def restore_keys(sampling: dict) -> str:
        """`marks=<pos>,...`, `tail=<pos>` and `shared=<pos>`: the DeepSeek engine's restore points for this prompt
        (Service.run puts them in `_restore` with SAGUARO_DS4_RESTORE_POINTS; no other engine gets them).  "" without: the
        line as before."""
        pts = sampling.get("_restore")
        if not isinstance(pts, dict):
            return ""

        def pos(v):
            return isinstance(v, int) and not isinstance(v, bool) and v > 0
        marks = [p for p in pts.get("marks") or () if pos(p)]
        tail, shared = pts.get("tail"), pts.get("shared")
        return ((" marks=" + ",".join(str(p) for p in marks) if marks else "") +
                (f" tail={tail}" if pos(tail) else "") + (f" shared={shared}" if pos(shared) else ""))

    @staticmethod
    def logprob_key(sampling: dict) -> str:
        """`logprobs=<n>`: an LP line after each of the request's T lines, n alternatives in each (OpenAI's logprobs /
        top_logprobs, checked by the handler: logprobs_of).  "" when the request does not ask: the line as before."""
        n = logprobs_asked(sampling)
        return f" logprobs={n}" if n is not None else ""

    @staticmethod
    def reply_key(sampling: dict) -> str:
        """`gt=<n>`: the prompt's last n tokens open the reply (prepare) - the engine keeps the conversation's state
        before them, so the next request continues it even when this answer is cancelled or dropped from the history."""
        v = (sampling or {}).get("_gen_tail")
        return f" gt={v}" if isinstance(v, int) and not isinstance(v, bool) and v > 0 else ""

    @staticmethod
    def projection_key(sampling: dict) -> str:
        """`cvec=0|1`: the experimental-speed-projection control vector for this request, when the engine was
        started with one (--control-vector-scaled; an engine without one ignores the key).  Absent = on."""
        on = sampling.get("experimental_speed_projection")
        return f" cvec={int(on)}" if isinstance(on, bool) else ""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        """Yields token ids, and None as a heartbeat every 10 s while the engine is quiet (reading a long prompt):
        the HTTP layer turns it into an SSE comment, which keeps clients' watchdogs calm and notices a client that
        has gone.  A consumer that stops early (or `cancel`) makes the engine STOP, so it does not run to max_new."""
        if self.multi:
            yield from self._generate_multi(ids, max_new, sampling, cancel, embeddings)
            return
        self.progress = None
        self.prefill_tok_s_mean = None
        if getattr(self, "_tls", None) is not None:   # (this request's own prompt progress: Service._progress)
            self._tls.own_pp = None
        # an image request takes the same sampling keys as text (#75: it used to decode greedily whatever was asked)
        head = f"GENI {int(max_new)}{self.gen_keys(sampling or {})} {embeddings}" if embeddings else \
            f"GEN {int(max_new)}{self.gen_keys(sampling or {})}"
        try:
            self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
            self.proc.stdin.flush()
        except OSError:                                  # the pipe is gone: the engine died (not the client)
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        done = False
        heard, stall = time.monotonic(), {}               # the last line's time; the stall watchdog's samples
        # logprobs: a token is held until its LP line, which the engine writes right after its T line (an engine that
        # writes none for it: the token goes on at the next other line, without them)
        want_lp, lp_wait = logprobs_asked(sampling) is not None, None
        ik2, ik2_stats = (getattr(self, "info", None) or {}).get("engine_kind") == "ik2", None
        try:
            while True:
                try:
                    line = self.lines.get(timeout=10)
                except queue.Empty:
                    if lp_wait is not None:
                        t, lp_wait = lp_wait, None
                        yield t
                    if cancel.is_set():
                        return
                    if ENGINE_STALL_S > 0:
                        why = self._stall_check(heard, stall)
                        if why:
                            done = True                   # no STOP and no drain: nothing is listening
                            self._end(why)
                            raise EngineDied(why)
                    yield None
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                heard = time.monotonic()
                stall.clear()
                if lp_wait is not None and not line.startswith("LP "):
                    t, lp_wait = lp_wait, None
                    yield t
                if line.startswith("T "):
                    if cancel.is_set():
                        return
                    if want_lp:
                        lp_wait = LpToken(engine_token(line[2:]))
                    else:
                        yield engine_token(line[2:])
                elif line.startswith("LP "):
                    if lp_wait is not None:
                        t, lp_wait = lp_attach(lp_wait, line.split()[1:]), None
                        yield t
                elif line.startswith("PP "):
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        # (read, total, reuse, ms): prompt progress, one per chunk: also a heartbeat (the
                        # lines reset the 10 s wait, so without this a long prompt would send no keep-alives at all).
                        # `reuse` (engines that print the field): what the conversation cache already held,
                        # `ms` the prompt read's own clock - both go to the stream's prompt_progress
                        self.progress = (int(f[1]), int(f[2]),
                                         int(f[5]) if len(f) >= 6 and f[5].isdigit() else 0,
                                         float(f[3]) if len(f) >= 4 else 0.0)
                        self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                        if getattr(self, "_tls", None) is not None:
                            self._tls.own_pp = (self.progress, self.prefill_tok_s_mean)
                            cb = getattr(self._tls, "on_read", None)   # (engine-total read rate: Service._note_read)
                            if cb:
                                cb(self.progress[0] - self.progress[2], self.prefill_tok_s_mean)
                    if cancel.is_set():                   # lines reset the 10 s wait, so without this a long prompt
                        return                            # would send no keep-alives at all)
                    yield None
                elif line.startswith("STAT "):
                    self._parse_stat(line)
                elif line.startswith("NOTE "):                # what happens to the kept conversations
                    print("[saguaro] " + line[5:].strip(), flush=True)
                elif line.startswith("DONE"):
                    self._parse_done(line)
                    self._ik2_done(ik2_stats)
                    done = True
                    return
                elif line.startswith("ERR"):
                    done = True
                    self._done_after_err()        # (maya-ds4 / maya-mm3 end the request with a DONE after it)
                    if ENGINE_ENDS_AFTER_ERR in line:
                        self._wait_end()
                    raise ValueError(line[4:].strip())
                elif ik2 and line.startswith(("decode stats:", "prefill stats:", "prefill sources:")):
                    ik2_stats = ik2_decode_stats(line) or ik2_stats
                    self._ik2_log([line])
        finally:
            if not done:                                  # the consumer stopped early: stop the engine, drain to DONE
                if self.can_stop:
                    try:
                        self.proc.stdin.write("STOP\n")
                        self.proc.stdin.flush()
                    except OSError:
                        pass
                # Not bounded like the multi one: one conversation shares one line queue, so going on before this
                # request's DONE handed its lines (T ... DONE) to the next request, and every request after it read
                # the one before's answer (a request cancelled 70 s into a long prompt: the engine looks
                # at STOP between the prompt's chunks, and a 32K chunk deep in a long context takes over a minute).
                # The engine is busy with it either way, so the drain waits; only a stuck engine (the stall watchdog,
                # SAGUARO_ENGINE_STALL_S) ends it.  DRAIN_S is when it says so.
                t0 = heard = time.monotonic()
                said, stall = False, {}
                while True:
                    try:
                        line, quiet = self.lines.get(timeout=5), False
                    except queue.Empty:
                        line, quiet = None, True
                    # (looked at on every line too: a prompt's progress lines go on while its chunk is read)
                    if not said and time.monotonic() - t0 >= DRAIN_S:
                        said = True
                        print(f"[saguaro] stopped, and the engine has not said the request ended after {DRAIN_S:.0f} s "
                              "(SAGUARO_DRAIN_S) - it is still working on it (a long prompt is stopped between its "
                              "chunks); waiting for it", flush=True)
                    if quiet:
                        why = self._stall_check(heard, stall) if ENGINE_STALL_S > 0 else None
                        if why:
                            self._end(why)
                            break
                        continue
                    heard = time.monotonic()
                    stall.clear()
                    if line is None:
                        self.lines.put(None)                # (the engine is gone: the next reader hears it too)
                        break
                    if line.startswith("STAT "):
                        self._parse_stat(line)   # the final snapshot can follow EOS or the consumer's last token
                        continue
                    if line.startswith("ERR"):
                        self._done_after_err()
                        break
                    if ik2 and line.startswith(("decode stats:", "prefill stats:", "prefill sources:")):
                        ik2_stats = ik2_decode_stats(line) or ik2_stats   # (an answer that ended at its stop token:
                        self._ik2_log([line])                             #  its last lines are read here)
                        continue
                    if line.startswith("DONE"):
                        self._parse_done(line)
                        self._ik2_done(ik2_stats)
                        if said:
                            print(f"[saguaro] the stopped request ended after {heard - t0:.0f} s", flush=True)
                        break

    def _wait_end(self):
        """An engine that ends after its ERR (saguaro-ik2's GPU failure in a request): its exit awaited here, still
        holding the request's place, so the next request finds it ended and starts a fresh one (ensure_engine) instead
        of writing to the one on its way out."""
        try:
            self.proc.wait(timeout=15)
        except (subprocess.TimeoutExpired, OSError, AttributeError):
            pass
        self.ended = True

    # How long, after an ERR, the reader waits for the DONE that may follow it: maya-ds4 and maya-mm3 print one right
    # after (in the same flush), the GLM engine none for most errors - so a short wait, not DRAIN_S.
    ERR_DONE_S = 0.5

    def _done_after_err(self):
        """After an ERR (single stream): take the DONE the engine may print after it, so the next request does not read
        that stale DONE as its own end (it returned an empty answer).  STAT / NOTE lines meanwhile are handled; the
        wait ends at another line, at the engine's end or after ERR_DONE_S."""
        end = time.monotonic() + self.ERR_DONE_S
        while True:
            try:
                line = self.lines.get(timeout=max(0.01, end - time.monotonic()))
            except queue.Empty:
                return
            if line is None:
                self.lines.put(None)
                return
            if line.startswith("DONE"):
                return
            if line.startswith("STAT "):
                self._parse_stat(line)
            elif line.startswith("NOTE "):
                print("[saguaro] " + line[5:].strip(), flush=True)
            else:
                return                                  # (not this request's end: dropped, as before)

    def command(self, line: str, expect: str, timeout: float = 60.0) -> str:
        """One control line between requests (the GLM engine's VLEND / VRECLAIM) -> its answer line; ValueError on
        the engine's ERR or no answer.  The caller holds the service's FIFO, so no request is reading the lines."""
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except OSError:
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        end = time.time() + timeout
        while True:
            try:
                got = self.lines.get(timeout=max(0.1, end - time.time()))
            except queue.Empty:
                raise ValueError(f"the engine did not answer {line.split()[0]}") from None
            if got is None:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
            if got.startswith(expect):
                return got.strip()
            if got.startswith("ERR"):
                raise ValueError(got[4:].strip())

    def wrap(self):
        """The server is stopping: a request still thinking closes its reasoning at the next token and answers."""
        if (getattr(self, "info", None) or {}).get("engine_kind") == "ik2":   # (saguaro-ik2 has no WRAP: not sent)
            return
        try:
            self.proc.stdin.write("WRAP\n")
            self.proc.stdin.flush()
        except OSError:
            pass

    def _generate_multi(self, ids, max_new, sampling, cancel, embeddings=None):
        """generate() on an engine that runs several conversations: this request's GEN carries rid=<id>, its lines come
        back through its own queue (_route), and STOP <id> ends it alone."""
        rid = next(self._rids)
        q: queue.Queue = queue.Queue()
        with self._routes_lock:
            self._routes[rid] = q
            self._sent[rid] = [time.monotonic(), False]
        self._tls.rid = rid
        self._tls.pp = None                 # this request's last prompt-progress report (fair reads)
        self.progress = None
        self.prefill_tok_s_mean = None
        self._tls.own_pp = None
        keys = self.gen_keys(sampling or {}) + f" rid={rid}"
        head = f"GENI {int(max_new)}{keys} {embeddings}" if embeddings else f"GEN {int(max_new)}{keys}"
        done = False
        try:
            try:
                self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
                self.proc.stdin.flush()
            except OSError:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
            heard, stall = time.monotonic(), {}
            # how long a wait before the checks (the engine-wide stall watchdog, this request's own quiet): 10 s, or
            # less when SAGUARO_REQUEST_QUIET_S is set low (so a test - or a tuned server - sees it in its own time)
            poll_s = 10.0 if REQUEST_QUIET_S <= 0 else min(10.0, max(0.2, REQUEST_QUIET_S / 2))
            want_lp, lp_wait = logprobs_asked(sampling) is not None, None   # (as generate(): a token waits for its LP)
            while True:
                try:
                    line = q.get(timeout=poll_s)
                except queue.Empty:
                    if lp_wait is not None:
                        t, lp_wait = lp_wait, None
                        yield t
                    if cancel.is_set():
                        return
                    if ENGINE_STALL_S > 0:
                        why = self._stall_check(heard, stall)
                        if why:
                            done = True
                            self._end(why)
                            raise EngineDied(why)
                    quiet = time.monotonic() - heard
                    if REQUEST_QUIET_S > 0 and quiet >= REQUEST_QUIET_S:
                        # The engine is alive (the other requests' lines reach them), so silence for this rid is this
                        # request's loss: end it here rather than hold the FIFO place for a line that is not coming.
                        print(f"[saguaro] request {rid}: the engine has said nothing of it for {quiet:.0f} s - stopping it "
                              f"and giving its place in the queue back (SAGUARO_REQUEST_QUIET_S={REQUEST_QUIET_S:.0f})",
                              flush=True)
                        try:
                            self.proc.stdin.write(f"STOP {rid}\n")
                            self.proc.stdin.flush()
                        except OSError:
                            pass
                        done = True                        # nothing of its own to drain: later lines are dropped
                        raise ValueError(f"the engine never answered this request ({quiet:.0f} s without a line of "
                                         "its own); it was stopped")
                    yield None
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                heard = time.monotonic()
                stall.clear()
                with self._routes_lock:
                    if rid in self._sent:
                        self._sent[rid][1] = True
                f = line.split()
                if lp_wait is not None and not line.startswith("LP "):
                    t, lp_wait = lp_wait, None
                    yield t
                if line.startswith("T "):
                    if cancel.is_set():
                        return
                    if want_lp:
                        lp_wait = LpToken(engine_token(f[2]))
                    else:
                        yield engine_token(f[2])
                elif line.startswith("LP "):
                    if lp_wait is not None:
                        t, lp_wait = lp_attach(lp_wait, f[2:]), None
                        yield t
                elif line.startswith("PP "):
                    if len(f) >= 4 and f[2].isdigit() and f[3].isdigit():
                        self.progress = (int(f[2]), int(f[3]),
                                         int(f[6]) if len(f) >= 7 and f[6].isdigit() else 0,
                                         float(f[4]) if len(f) >= 5 else 0.0)
                        self._tls.pp = self.progress
                        self.prefill_tok_s_mean = float(f[5]) if len(f) >= 6 else None
                        if getattr(self, "_tls", None) is not None:
                            self._tls.own_pp = (self.progress, self.prefill_tok_s_mean)
                            cb = getattr(self._tls, "on_read", None)   # (engine-total read rate: Service._note_read)
                            if cb:
                                cb(self.progress[0] - self.progress[2], self.prefill_tok_s_mean)
                    if cancel.is_set():
                        return
                    yield None
                elif line.startswith("DONE "):
                    self._parse_done("DONE " + line.split(None, 2)[2])
                    done = True
                    return
                elif line.startswith("ERR "):
                    done = True
                    raise ValueError(line.split(None, 2)[2].strip() if len(f) > 2 else "engine error")
        finally:
            if not done:                                  # the consumer stopped early: stop this request, drain it
                try:
                    self.proc.stdin.write(f"STOP {rid}\n")
                    self.proc.stdin.flush()
                except OSError:
                    pass
                # The drain is bounded: an engine that does not honour the STOP (or whose DONE for this rid was
                # already dropped) left this loop waiting for a line that never came, holding the request's place in
                # the FIFO for ever.  Past DRAIN_S: say so, drop the route, and let the next request run.
                end = time.monotonic() + DRAIN_S
                while True:
                    try:
                        line = q.get(timeout=max(0.05, end - time.monotonic()))
                    except queue.Empty:
                        print(f"[saguaro] request {rid}: stopped, and the engine did not say it had ended within "
                              f"{DRAIN_S:.0f} s (SAGUARO_DRAIN_S) - going on; the engine may still be writing it",
                              flush=True)
                        break
                    if line is None or line.startswith("ERR "):
                        break
                    if line.startswith("DONE "):
                        self._parse_done("DONE " + line.split(None, 2)[2])
                        break
            with self._routes_lock:
                if self._routes.get(rid) is q:                # (a restarted engine's table may hold another one)
                    del self._routes[rid]
                self._sent.pop(rid, None)
            self._tls.rid = None

    def waiting_behind(self, seconds: float) -> set:
        """Several conversations: the other requests sent this long ago that have heard nothing yet (no token, no read
        progress) - they wait behind this thread's prompt read."""
        me, now = getattr(self._tls, "rid", None), time.monotonic()
        with self._routes_lock:
            return {r for r, (t, h) in self._sent.items() if r != me and not h and now - t >= seconds}

    def started(self, rids) -> bool:
        """Those requests have heard from the engine (their read began) or are gone."""
        with self._routes_lock:
            return all(r not in self._sent or self._sent[r][1] for r in rids)

    def pause(self, keep: bool = False):
        """End the request in flight at its next step and keep the conversation's state with what it wrote (the
        fair slices: prompt + written tokens then continues with one token to read).  The tokens it still writes
        before its DONE come as usual.  keep (a request's first slice): the engine first sets its prompt's state
        aside for the chat's next turn, whose history does not repeat the answer as written (no reasoning)."""
        rid = getattr(getattr(self, "_tls", None), "rid", None) if self.multi else None
        try:
            if rid is not None:
                self.proc.stdin.write(f"PAUSE {rid}{' keep' if keep else ''}\n")
            else:
                self.proc.stdin.write("PAUSE keep\n" if keep else "PAUSE\n")
            self.proc.stdin.flush()
        except OSError:
            pass

    def stop(self):
        """End the request in flight at its next step (its answer is cut there)."""
        if self.can_stop:
            try:
                self.proc.stdin.write("STOP\n")
                self.proc.stdin.flush()
            except OSError:
                pass

    def close(self):
        """QUIT, then wait for the engine to end: with kept conversations (SAGUARO_GLM_SLOT_KEEP) it first writes the
        one it holds, so it gets SAGUARO_ENGINE_QUIT_S seconds (50; llama-swap's unloadTimeout should exceed it)."""
        wait = float(os.environ.get("SAGUARO_ENGINE_QUIT_S", "50") or 50)
        try:
            if self.can_stop:                           # a request in flight ends at its next step first: QUIT
                self.proc.stdin.write("STOP\n")         # alone lets it run to max_tokens before the engine ends
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self._ik2_eof()
            end = time.monotonic() + wait
            src = getattr(self, "lines", None)          # after start-up the pump thread owns stdout: read its queue
            while self.proc.poll() is None and time.monotonic() < end:
                if src is not None:
                    try:
                        line = src.get(timeout=max(0.1, min(1.0, end - time.monotonic())))
                    except queue.Empty:
                        continue
                else:
                    line = self.proc.stdout.readline()
                if not line:
                    break
                if line.startswith("NOTE "):
                    print("[saguaro] " + line[5:].strip(), flush=True)
            self.proc.wait(timeout=max(0.1, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(f"[saguaro] the engine did not end within {wait:.0f} s: stopped", flush=True)
            self.proc.kill()
        except Exception as e:  # noqa: BLE001 - its stdin gone (it had ended already: Ctrl+C reaches it too) or the like
            if self.proc.poll() is None:
                print(f"[saguaro] the engine could not be told to stop ({type(e).__name__}): stopped", flush=True)
                self.proc.kill()


PICTURE_DIR_PREFIXES = ("saguaro-vision-", "saguaro-ds4-pictures-", "saguaro-hy4-pictures-", "saguaro-mm3-pictures-")
PICTURE_DIR_PID = ".saguaro-pid"       # the server that owns the folder (a second server running at once keeps its own)


def picture_dir(prefix: str) -> Path:
    """A folder in the temp directory for the pictures a server reads (cached by hash while it runs).  It goes when the
    server ends (any way out of main(); #1180) - on tmpfs these folders were RAM that was never given back."""
    d = Path(tempfile.mkdtemp(prefix=prefix))
    with contextlib.suppress(OSError):
        (d / PICTURE_DIR_PID).write_text(str(os.getpid()), encoding="ascii")
    atexit.register(shutil.rmtree, str(d), True)
    return d


def clean_stale_picture_dirs(older_than: float = 86400.0, temp: str | None = None, now: float | None = None) -> int:
    """At start: the picture folders of earlier runs that ended without removing theirs (kill -9, a power cut): only
    ours (the prefixes above), only in the temp directory, only when older than a day and their server is gone.  The
    number removed."""
    root = Path(temp or tempfile.gettempdir())
    now = time.time() if now is None else now
    n = 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    for d in entries:
        try:
            if not d.name.startswith(PICTURE_DIR_PREFIXES) or d.is_symlink() or not d.is_dir():
                continue
            if now - d.stat().st_mtime < older_than:
                continue
            try:
                pid = int((d / PICTURE_DIR_PID).read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                pid = 0
            if pid and pid != os.getpid():
                try:
                    import psutil
                    if psutil.pid_exists(pid):
                        continue                         # (its server still runs)
                except ImportError:
                    pass
            shutil.rmtree(d, ignore_errors=True)
            n += not d.exists()
        except OSError:
            continue
    return n


class Vision:
    """The resident image encoder: `vision-glm` (llama.cpp mtmd + the mmproj file) reads `ENC <image> <out>`
    and `VID <video> <out>` lines and writes each picture's embeddings (a video: one record per frame); results
    are cached by the source's hash, so a conversation that sends the same picture again (every turn, with most
    clients) encodes it once.

    On demand (start=False, `lend`/`reclaim` set - the GLM engine's VLEND / VRECLAIM): no process until a picture
    needs encoding; the engine first empties the tail of its expert cache on the first GPU and frees it, the encoder
    starts in that memory, and once the request's pictures are encoded (`active()`) it ends and the engine takes the
    memory back.  The decode keeps the whole cache between pictures (~18% faster on Mercury than with the encoder
    resident), for a second or two of start-up per request with new pictures."""

    def __init__(self, cfg: dict, log=None, env: dict | None = None, start: bool = True):
        # (no "model": no text model - Inkling 2's encoder reads its pictures from the mmproj alone)
        args = [cfg["exe"], "--mmproj", cfg["mmproj"]] + (["--model", cfg["model"]] if cfg.get("model") else [])
        if cfg.get("gpu"):
            args.append("--gpu")
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        if cfg.get("max_tokens"):
            args += ["--max-tokens", str(cfg["max_tokens"])]
        if cfg.get("no_flash_attn"):   # GLM-5.3's encoder on a GPU: full-precision attention (cosine 0.9997 vs 0.999)
            args.append("--no-flash-attn")
        self.args, self.log, self.env = args, log, env
        self.dir = picture_dir("saguaro-vision-")
        self.proc = None
        self.lend = self.reclaim = None   # on demand: callables (the engine's VLEND / VRECLAIM), wired by main()
        # on demand without lending (Inkling 2): a callable, True when the GPU's free memory holds the encoder now -
        # it then starts there, else on the CPU (no GPU visible, `cpu_threads`); it ends after the request's pictures
        self.room = None
        self.cpu_threads = 0
        self.on_gpu = None                # (where the last on-demand start ran: True GPU, False CPU)
        self.owed = False                 # the engine's memory is not back yet (a reclaim that kept failing)
        self.lock = threading.RLock()
        self.cache: dict[str, tuple[Path, int]] = {}
        if start:
            self._start()

    def _start(self):
        if self.lend is not None:
            self.settle()
            self.lend()
        args, env = self.args, self.env
        if self.room is not None:
            where = self.room()                           # (True / False, or the GPU's number: Inkling 2 on two cards)
            self.on_gpu = where is not False and where is not None
            if self.on_gpu and where is not True:
                env = dict(env if env is not None else os.environ, CUDA_VISIBLE_DEVICES=str(int(where)))
            print("[saguaro] images: reading the pictures on " +
                  (("the GPU" if where is True else f"GPU {int(where)}") if self.on_gpu else "the CPU"), flush=True)
            if not self.on_gpu:
                args = [x for x in args if x != "--gpu"] + \
                    (["--threads", str(self.cpu_threads)] if self.cpu_threads and "--threads" not in args else [])
                env = dict(env if env is not None else os.environ, CUDA_VISIBLE_DEVICES="")
        try:
            self.proc = subprocess.Popen(args + (["--no-warmup"] if self.lend is not None or self.room is not None
                                                 else []),
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.log or subprocess.DEVNULL, text=True, encoding="utf-8",
                                         bufsize=1, env=env)
            track_spawned(self.proc)
            line = self.proc.stdout.readline()
            if not line.startswith("READY"):
                raise RuntimeError("the vision encoder did not start: " + line.strip())
        except Exception:
            self._stop()
            raise

    def _stop(self):
        if self.proc is not None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self.reclaim is not None:
            self.owed = True
            for _ in range(40):          # the driver frees an ended process's memory within moments
                if self.settle():
                    return
                time.sleep(0.25)
            print("[saguaro] the vision encoder's GPU memory is not back yet: prompts are read token by token until "
                  "it is", flush=True)

    def settle(self) -> bool:
        """The engine takes back the memory an earlier encode borrowed (true when nothing is owed)."""
        if not self.owed:
            return True
        try:
            self.reclaim()
            self.owed = False
        except ValueError:
            pass
        return not self.owed

    @contextlib.contextmanager
    def active(self):
        """The request's pictures are encoded inside: an on-demand encoder ends at the exit and the engine has its
        memory back before the request runs."""
        with self.lock:
            try:
                yield self
            finally:
                if (self.lend is not None or self.room is not None) and self.proc is not None:
                    self._stop()

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            return base64.b64decode(source.split(",", 1)[1])
        if source.startswith(("http://", "https://")):
            req = urllib.request.Request(source, headers={"User-Agent": "maya"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        path = source[7:] if source.startswith("file://") else source
        if path and os.path.isfile(path):
            return Path(path).read_bytes()
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    @staticmethod
    def normalize(data: bytes) -> bytes:
        """The formats vision-glm's decoder (stb_image) reads pass through; anything else is converted to PNG."""
        if data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
                data[:6] in (b"GIF87a", b"GIF89a"):
            return data
        try:
            import io
            from PIL import Image
        except ImportError:
            raise ValueError("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and "
                             "GIF work without it") from None
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception as e:
            raise ValueError(f"the image could not be read ({e})") from None
        if im.mode in ("RGBA", "LA", "P") and "transparency" in im.info or im.mode in ("RGBA", "LA"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    def encode(self, source: str) -> tuple[Path, int]:
        """-> (embeddings file, number of image tokens)."""
        data = self.normalize(self.load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            if key in self.cache:
                if self.cache[key][0].is_file():
                    return self.cache[key]
                del self.cache[key]
            if self.proc is None:
                self._start()
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            img.write_bytes(data)
            self.proc.stdin.write(f"ENC {img} {out}\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline().strip()
            img.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            self.cache[key] = (out, int(line.split()[1]))
            if len(self.cache) > 64:                                   # oldest first
                old = next(iter(self.cache))
                self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    VIDEO_FPS = 4.0                                        # (llama.cpp's mtmd default: the frames a second it reads)

    @staticmethod
    def video_container(data: bytes) -> str | None:
        """The container a video's first bytes name - MP4 / MOV, WebM / MKV or AVI - else None.  Only those reach
        ffmpeg: anything else (a playlist is text) could make it open other files or URLs while it reads."""
        if data[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):
            return "mp4"
        if data[:4] == b"\x1a\x45\xdf\xa3":
            return "webm"
        if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
            return "avi"
        return None

    @staticmethod
    def video_seconds(path: Path) -> float | None:
        """A video file's duration by ffprobe (files only - no other protocol), None when it does not say."""
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_entries",
                                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                               capture_output=True, text=True, timeout=30)
            return float(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else None
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return None

    def encode_video(self, source: str) -> tuple[Path, list[int]]:
        """-> (embeddings file, the frames' token counts), one SVE1 record per frame."""
        import shutil
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise ValueError("videos need ffmpeg and ffprobe on this PC (Linux: sudo apt install ffmpeg; Windows: winget "
                             "install Gyan.FFmpeg; macOS: brew install ffmpeg), then restart Saguaro")
        data = self.load(source)
        if self.video_container(data) is None:
            raise ValueError("this video's format is not one Saguaro reads: send MP4, MOV, WebM, MKV or AVI")
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            if key in self.cache:
                if self.cache[key][0].is_file():
                    return self.cache[key]
                del self.cache[key]
            vid, out = self.dir / f"{key}.vid", self.dir / f"{key}.sve"
            vid.write_bytes(data)
            # a long video is many frames (every one an image's tokens): refused before it is decoded
            limit = max(1, int(os.environ.get("SAGUARO_VIDEO_MAX_FRAMES", "240") or 240))
            seconds = self.video_seconds(vid)
            if seconds is not None and seconds * self.VIDEO_FPS > limit:
                vid.unlink(missing_ok=True)
                raise ValueError(f"the video is {seconds:.0f} s long: {seconds * self.VIDEO_FPS:.0f} frames at "
                                 f"{self.VIDEO_FPS:g} a second, more than the {limit} Saguaro reads (SAGUARO_VIDEO_MAX_FRAMES "
                                 f"sets it; every frame takes an image's share of the context)")
            if self.proc is None:
                self._start()
            self.proc.stdin.write(f"VID {vid} {out}\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline().strip()
            vid.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the video could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            # the frames' token counts come from the record headers: int32 {magic, n_tokens, nx, ny, n_embd} each
            counts = []
            with open(out, "rb") as f:
                while hdr := f.read(20):
                    magic, n, _, _, n_embd = struct.unpack("<5i", hdr)
                    if magic != 0x31455653 or n_embd <= 0:
                        raise ValueError("not an SVE1 embeddings file")
                    counts.append(n)
                    f.seek(n * n_embd * 4, 1)
            self.cache[key] = (out, counts)
            if len(self.cache) > 64:                                   # oldest first
                old = next(iter(self.cache))
                self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
        shutil.rmtree(self.dir, ignore_errors=True)    # (#1180: the pictures it kept go with the server)


def gpu_free_bytes(gpu: int) -> int | None:
    """The free memory of NVIDIA GPU `gpu` (nvidia-smi's numbering) now, or None when nvidia-smi cannot say."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0]) << 20 if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def amd_gpu_free_bytes(gpu: int, sysfs: str | None = None, meminfo: str = "/proc/meminfo") -> int | None:
    """The free memory of AMD GPU `gpu` (HIP's numbering, Linux) now: its VRAM's, as the amdgpu driver says
    (mem_info_vram_total / _used); on an APU whose memory is the RAM (a carve-out under 8 GiB beside a larger GTT)
    what MemAvailable leaves beyond the max(8 GiB, 10% of the RAM) the engines keep free.  None when it does not read."""
    from serve import telemetry   # (its KFD -> render node numbering, as the Monitor reads the cards)

    def num(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None
    dev = telemetry.amd_device_dir(gpu, sysfs)
    if not dev:
        return None
    total, used = num(os.path.join(dev, "mem_info_vram_total")), num(os.path.join(dev, "mem_info_vram_used"))
    gtt = num(os.path.join(dev, "mem_info_gtt_total"))
    if total is not None and total < (8 << 30) and gtt is not None and gtt > total:
        try:
            with open(meminfo, encoding="utf-8") as f:
                kb = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in f if ":" in line)
                      if k in ("MemTotal", "MemAvailable")}
            margin = max(8 << 30, kb["MemTotal"] * 1024 // 10)
            return max(0, kb["MemAvailable"] * 1024 - margin)
        except (OSError, ValueError, KeyError):
            return None
    return total - used if total is not None and used is not None else None


def vision_plan(cfg: dict, env: dict, cache_file: Path):
    """(vision config, the encoder's environment, on demand?, footprint, the ENGINE's environment) for a config with
    pictures.  The one place the engine's environment gets its picture-related keys (the GLM engine's lend size; the
    server and the bench both start the engine from it).  maya-ik2 lends no memory and plans none for pictures: its
    start - context, cache, drafter - is the same with pictures on or off; the encoder takes what is free when a
    picture arrives, else the CPU."""
    vcfg = cfg.get("vision") or None
    venv = dict(env)
    if vcfg and vcfg.get("gpu") and gpu_list(cfg) and cfg.get("backend") not in ("hip", "metal"):
        venv["CUDA_VISIBLE_DEVICES"] = str(gpu_list(cfg)[0])   # the encoder on the first GPU only
    # the GLM engine lends the encoder the tail of its expert cache while it encodes, sized by what the encoder
    # measured on this GPU ("resident": true keeps the encoder running beside the model, holding its VRAM)
    # (Metal: one memory shared with the engine, which answers ERR to VLEND / VRECLAIM: the encoder stays resident)
    on_demand = bool(vcfg) and vcfg.get("gpu") and not vcfg.get("resident") and cfg.get("backend") != "metal"
    engine_env, fp = env, None
    if on_demand:
        fp = vision_footprint(vcfg, venv, cache_file, (gpu_list(cfg) or [0])[0])
        if fp is None:
            vcfg, on_demand = dict(vcfg, gpu=False), False
        else:
            vcfg = dict(vcfg, max_tokens=fp[0])
            if models.family(cfg) != "ik2":
                engine_env = dict(env, SAGUARO_GLM_VISION_LEND_MB=str((fp[1] >> 20) + 1))
    return vcfg, venv, bool(on_demand), fp, engine_env


def engine_environment(cfg: dict, cache_file: Path) -> dict:
    """The environment the engine is started with for this config: child_env plus the picture keys."""
    return vision_plan(cfg, child_env(cfg), cache_file)[4]


def vision_footprint(vcfg: dict, env: dict, cache_file: Path, gpu: int) -> tuple[int, int] | None:
    """(image token cap, bytes) for the on-demand vision encoder on THIS machine's first GPU: measured once by
    `vision-glm --measure` (before the model loads, so the GPU is otherwise idle) and cached per encoder file, cap
    and GPU.  A cap the config does not set is the largest of 4096 / 2048 / 1024 image tokens whose footprint stays
    within 15% of the card (a 32 GB card keeps 4096, an 8-12 GB card gets 1024).  None: no GPU to measure on, or
    no measurement (the encoder failed or printed no well-formed MEM line)."""
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    if not isinstance(cache, dict):                     # a hand-edited or damaged file: measured again
        cache = {}
    st = Path(vcfg["mmproj"]).stat()
    caps = [int(vcfg["max_tokens"])] if vcfg.get("max_tokens") else [4096, 2048, 1024]
    for i, cap in enumerate(caps):
        key = f"{Path(vcfg['mmproj']).resolve()}|{st.st_size}|{int(st.st_mtime)}|gpu{gpu}|cap{cap}|" \
              f"fa{0 if vcfg.get('no_flash_attn') else 1}"
        m = cache.get(key)
        if not isinstance(m, dict) or not all(isinstance(m.get(k), int) and m[k] > 0 for k in ("bytes", "total")):
            m = None
        if m is None:
            args = [vcfg["exe"], "--mmproj", vcfg["mmproj"]] + \
                (["--model", vcfg["model"]] if vcfg.get("model") else []) + \
                ["--gpu", "--max-tokens", str(cap), "--measure"] + (["--no-flash-attn"] if vcfg.get("no_flash_attn") else [])
            print(f"[saguaro] images: measuring the vision encoder on this GPU ({cap} image tokens) ...", flush=True)
            try:
                r = subprocess.run(args, capture_output=True, text=True, timeout=600, env=env)
                # only a measurement that ran to its end (an encoder whose warm-up failed exits 1 without one)
                f = next((l.split() for l in r.stdout.splitlines() if l.startswith("MEM ")), None) \
                    if r.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                f = None
            if f is None or len(f) != 4 or not all(x.isdigit() for x in f[1:]) or int(f[3]) == 0:
                return None
            # its weights and work buffers, the context this process had (and whatever else used the card then -
            # the safe side), and 256 MB of slack
            m = cache[key] = {"bytes": int(f[1]) + int(f[2]) + (256 << 20), "total": int(f[3])}
            try:
                cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            except OSError:
                pass
        if len(caps) == 1 or m["bytes"] <= 0.15 * m["total"] or i == len(caps) - 1:
            return cap, int(m["bytes"])
    return None


trained_context = models.trained_context            # (the context a model was trained for: serve/models.py)
trained_context_ds4 = models.trained_context_ds4


def gpu_list(cfg: dict) -> list[int]:
    """The config's "gpu": one card (2), or several for a layer split ([0, 2] or "0,2"), numbered as nvidia-smi
    numbers them; [] when it names none."""
    g = cfg.get("gpu")
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    return [int(str(x).strip()) for x in items if str(x).strip() != ""]


def engine_args(cfg: dict) -> list[str]:
    """The engine's arguments: the config's, and with several GPUs the layer split across them ("layer_split" in the
    config: "auto" by default, or the first layer of each later GPU's share, e.g. "18" or "16,32")."""
    args = list(cfg["args"])
    # (saguaro-ik2 deals its layers itself, --gpus: it stops at a flag it does not know)
    if len(gpu_list(cfg)) > 1 and "--layer-split" not in args and maya_quants.family_of(cfg) != "ik2":
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    return args


def ik2_draft_ids(cfg: dict) -> Path | None:
    """The requests saguaro-ik2 times its --drafting auto on (SAGUARO_IK2_DRAFT_IDS): IK2_DRAFT_QS in the model's own
    chat template with thinking off (the last with IK2_DRAFT_TOOL offered), a line of token ids each, written once
    into the data folder.  None: not written - the engine then says it did not measure."""
    tdir = Path(cfg.get("tokenizer") or "")
    vocab, tpl = tdir / "vocab.json", tdir / "chat_template.jinja"
    if not cfg.get("tokenizer") or not vocab.is_file() or not tpl.is_file():
        return None
    try:
        tag = hashlib.md5(vocab.read_bytes() + tpl.read_bytes() + repr(IK2_DRAFT_QS).encode()).hexdigest()[:10]
    except OSError:
        return None
    path = VAR_DIR / f"ik2-draft-ids-{tag}.txt"
    if path.is_file():
        return path
    try:
        import strata_tokenizer as ST
        from serve.frontend import ChatTemplate
        tok, chat = ST.Tokenizer.from_dir(tdir), ChatTemplate(tpl)
        lines = []
        for i, q in enumerate(IK2_DRAFT_QS):
            tools = [IK2_DRAFT_TOOL] if i == len(IK2_DRAFT_QS) - 1 else None
            text = chat.render([{"role": "user", "content": q}], tools=tools, enable_thinking=False,
                               reasoning_effort="none")
            lines.append(",".join(map(str, tok.encode(text, parse_special=True))))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        tmp.write_text("# saguaro: requests in the model's chat template, a line each - the answers saguaro-ik2's "
                       "--drafting auto times at its first start on a PC\n" + "\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path
    except Exception:  # noqa: BLE001 - none: the engine drafts without measuring
        return None


def ik2_lt_store(cfg: dict) -> Path | None:
    """saguaro-ik2's measured hipBLASLt picks (AMD, SAGUARO_IK2_LT_STORE): var/<GGUF's name>-lt-picks.txt, so the
    model folder is not written at run time; an lt-picks.txt an earlier version left beside the GGUF is copied in
    first.  None: no var folder (the engine then keeps them beside the GGUF, as before)."""
    args = cfg.get("args") or []
    if "--model" not in args[:-1]:
        return None
    gguf = Path(args[args.index("--model") + 1])
    path = VAR_DIR / f"{gguf.stem}-lt-picks.txt"
    try:
        VAR_DIR.mkdir(parents=True, exist_ok=True)
        old = gguf.parent / "lt-picks.txt"
        if not path.exists() and old.is_file():
            shutil.copyfile(old, path)
    except OSError:
        return None
    return path


def child_env(cfg: dict) -> dict:
    """The engine's environment: the CUDA libraries setup installed (pip's nvidia packages, or the toolkit that
    compiled it) first on the library search path."""
    env = dict(os.environ)
    if gpu_list(cfg) and cfg.get("backend") == "hip":   # AMD: numbered as HIP numbers them (setup's KFD order)
        env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    elif gpu_list(cfg) and cfg.get("backend") != "metal":   # issue #51: the GPU(s) to run on, numbered as nvidia-smi does; CUDA's
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # own default order (fastest first) can number the cards otherwise
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    # engine settings the config carries (AMD: the GEMM tuning table): each setting has one name (issue #686)
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    # Hy4-flash: what its engine measures once on this PC (the CPU lane's 4-bit copies, the prompt route's rates,
    # --drafting auto's choice) and HY4_HOT's scores are kept in files - the config's env names them, else the data
    # folder's; the prompt route is measured before READY on this install's docs (HY4_MEASURE_IDS)
    if models.family(cfg) == "hy4":
        env.setdefault("HY4_Q4X_FILE", str(HY4_LANE_FILE))
        env.setdefault("HY4_ROUTE_FILE", str(HY4_ROUTE_FILE))
        env.setdefault("HY4_DRAFT_FILE", str(HY4_DRAFT_FILE))
        env.setdefault("HY4_HOT_FILE", str(hy4_hot_file(cfg)))
        # (... and, where this PC has no such file yet, the scores shipped for the model - a first start places the
        #  experts as every start after it; a file of its own wins)
        seed = hy4_hot_seed(cfg)
        if seed is not None:
            env.setdefault("HY4_HOT_SEED", str(seed))
        if "HY4_MEASURE_IDS" not in env:
            ids = hy4_measure_ids(cfg)
            if ids is not None:
                env["HY4_MEASURE_IDS"] = str(ids)
    # Inkling 2 (--drafting auto): which of no drafting, the prompt lookup and the drafter decodes fastest - measured
    # once on this PC at the first start, remembered in the data folder (off on Volta / Turing: the engine says so)
    if models.family(cfg) == "ik2" and "--drafting" in (cfg.get("args") or []):
        env.setdefault("SAGUARO_IK2_DRAFT_FILE", str(IK2_DRAFT_FILE))
        if "SAGUARO_IK2_DRAFT_IDS" not in env:
            ids = ik2_draft_ids(cfg)
            if ids is not None:
                env["SAGUARO_IK2_DRAFT_IDS"] = str(ids)
    if models.family(cfg) == "ik2" and cfg.get("backend") == "hip" and "SAGUARO_IK2_LT_STORE" not in env:
        lt = ik2_lt_store(cfg)
        if lt is not None:
            env["SAGUARO_IK2_LT_STORE"] = str(lt)
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}
    special_ids = {"eos": 256 + 1}         # (its <|im_end|>: the engine's own end of a turn, as the packs' eos is theirs)

    def encode(self, text, parse_special=False, plain=()):
        out, i = [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if parse_special and text.startswith(s, i) and not any(a <= i < b for a, b in plain):
                    out.append(256 + k)
                    i += len(s)
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            raw += self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


# ------------------------------------------------------------------------------------------------ core
class Detokenizer:
    """Incremental decode: each token's bytes go through an incremental UTF-8 decoder, which emits the complete
    characters and holds a multi-byte character split across tokens until it is complete (invalid bytes become
    U+FFFD, as a whole decode with errors="replace" makes them).  Constant time per token - the old re-decode of
    every generated id cost 2 ms per token after 8K tokens and 4 ms after 16K (perf-review F-1).  A tokenizer
    without `token_bytes` (the tests' byte tokenizer) keeps the re-decode."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if hasattr(tok, "token_bytes") else None

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


def speculation_modes(info: dict, dspark=None, lookup=None) -> list:
    """The drafters an engine runs with, for the Monitor's Speculation: DSpark (DeepSeek's server says, MiniMax's engine
    itself), DFlash2 or MTP (GLM's INFO dflash= / mtp=: its DFlash decode takes the request first, else MTP; QFN's
    mtp=) and the prompt lookup (QFN's INFO lookup=, DeepSeek's --suffix-draft).  Empty: drafting is off."""
    def on(v):
        try:
            return float(v) > 0
        except (TypeError, ValueError):
            return False
    modes = []
    ds = dspark if dspark is not None else info.get("dspark")
    if on(ds):
        modes.append("DSpark")
    if on(info.get("dflash")):
        modes.append("DFlash2")
    elif on(info.get("mtp")):
        modes.append("MTP")
    elif on(info.get("dfly2")):                        # (saguaro-hy4's DFly2 drafter: its INFO dfly2=)
        modes.append("DFly2")
    lk = lookup if lookup is not None else info.get("lookup")
    if on(lk):
        modes.append("Lookup")
    return modes


def request_spec_off(sampling) -> bool:
    """A request that switches drafting off for itself (the GLM engine's spec=1: the plain token loop)."""
    tune = (sampling or {}).get("strata_tune")
    return isinstance(tune, dict) and tune.get("spec") == 1 and not isinstance(tune.get("spec"), bool)


def conv_hist_hash(ids: list[int]) -> bytes:
    """A conversation's identity: a hash of the tokens a later turn must start with - its last prompt minus the open
    reply's generation tail (blake2b of the int32 bytes).  A request continues a conversation only when its prompt
    starts with exactly that - never with its opening alone."""
    return hashlib.blake2b(array("i", ids).tobytes(), digest_size=8).digest()


def conv_first_words(messages, n: int = 10) -> str:
    """The Monitor's conversation row's label: the first user message's first words ("" when there is none)."""
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
        w = " ".join(str(c or "").split()[:n])
        if w:
            return w[:96] + ("..." if len(w) > 96 else "")
    return ""


class Conversation:
    """One row of the Monitor's Conversations card: one conversation's turns - found by CONTINUATION, never by an
    opening: cont_len + cont_hash are its last turn's prompt minus the generation tail, and a request continues it
    only when its prompt starts with exactly that (the part a client re-sends byte-for-byte; the answer re-renders
    with small differences, so it is no key).  What its last turn did (speeds, tokens, context) and which turn the
    engine holds now (None: it waits idle)."""

    def __init__(self, cid: int, api, words: str):
        self.id, self.key, self.api, self.words = cid, str(cid), api, words
        self.cont_len, self.cont_hash = 0, None         # (0/None: no finished turn yet, nothing this could continue)
        self.turns, self.tokens_read, self.tokens_written = 0, 0, 0
        self.last_read_tok_s, self.last_decode_tok_s = None, None      # the last finished turn's rates
        self.prompt_last, self.out_last, self.ctx = 0, 0, 0            # its sizes: the idle row shows these
        self.stat = None                                                # the turn queued or with the engine now
        self.last_active = time.time()                                  # when one of its turns started or ended


class RequestStat:
    """One request's live figures: when its first answer token came (None while it reads the prompt) and its
    (time, generated) samples over RATE_WINDOW_S for the live decode rate."""

    def __init__(self):
        self.first_token = None
        self.spec_off = False                           # it asked for no drafting (spec=1)
        self.turbo = None                               # the request's lossy Turbo options (serve/turbo_live.py)
        self.base = None                                # (time, generated) at the window's start: the rate counts from it
        self.samples = collections.deque(maxlen=4096)   # the samples inside the window, oldest first
        # what the Monitor shows of this request (several answer at once: each its own, not the shared status's)
        self.phase, self.generated, self.prompt_tokens, self.max_tokens = "waiting", 0, 0, None
        # R122: its queue priority's name (the Monitor's small tag: high / low, normal shows nothing) and
        # whether it gave its place up for a request of a higher priority and waits for the engine again
        self.priority, self.paused_for_priority = None, False
        self.had_to_wait = False                          # (R122, Ken's trigger decision: this request waited in
        #  the FIFO at least once, or gave its place up - set where each happens below; the padding and the
        #  stream's empty events fire only once it is True, so a request that never waited is framed as in 1.2.1)
        self.started, self.progress, self.prefill_tok_s = None, None, None
        # the prompt read (the drivers' PP lines / the batch slots' read counters): tokens read so far - what the
        # conversation cache held stays out - the span the server saw them arrive over, and whether another chat
        # read beside it (then its own engine clock mixes the two and the Monitor labels the rate as shared)
        self.read_pos, self.read_t0, self.read_t1, self.read_shared = 0, None, None, False
        self.read_avg = None                            # the engine's mean over its read, from its last progress line
        self.read_last, self.read_prev_t, self.read_prev_n = None, None, 0   # its last observed read interval
        # the engine's OWN per-request read figures (PLAN-qfn-per-request-read-stats: PP/BRP new_read=/read_ms=,
        # RDONE / BRA at the read's end): counted tokens over the time its read actually ran - never a shared
        # reader's rate, never wall time since the engine took it
        self.read_eng_n, self.read_eng_ms = 0, None     # live pair (the last progress step's)
        self.read_engine = None                         # the finished read's exact (new tokens, read_ms)
        # U5 read heartbeat (PLAN-u5-read-heartbeat): the engine's last RHB report of the read in flight - stage
        # (staging the reused prefix from RAM, or reading new tokens), done/new (new tokens done of total),
        # reused, the rate over the last heartbeat interval and when it arrived (consumers ignore a stale one)
        self.read_hb = None
        self.read_cur = None        # its read interval in Service.reads while one runs (closed when the answer starts)
        self.at_engine = False                            # it holds an engine place now (a fair slice lets it go)
        self.engine_t0 = None                             # when the engine took it (the real read time starts here)
        # who this request is in the Monitor's "Right now": a stable id, the client it came from and when it queued
        self.id, self.api, self.queued_at = None, None, None
        self.conv = None                                # the Conversation its opening names (its row in the card)
        # shared with other answers (several at once): the most decoding beside it, and the engine's total rate
        # (summed over them) sampled while it decoded - "this chat 7 tok/s · engine 14 tok/s across 2 chats"
        self.shared_max, self.eng_sum, self.eng_n = 1, 0.0, 0

    def shared(self) -> dict | None:
        if self.shared_max < 2 or not self.eng_n:
            return None
        return {"chats_max": self.shared_max, "engine_tok_s": round(self.eng_sum / self.eng_n, 1)}

    def view(self, now) -> dict:
        """One row of the Monitor's Conversations: reading (with its own progress), writing or between its turns."""
        pr = self.progress
        if pr and self.first_token and pr[1]:            # (the engines' last PP is n-1: the last prompt token is read
            pr = (pr[1], pr[1])                          #  by the first answer step) - once answering, all n are read
        return {"state": "writing" if self.first_token else "reading" if self.started else "waiting",
                "phase": self.phase, "prompt_tokens": self.prompt_tokens, "generated": self.generated,
                "priority": self.priority, "paused_for_priority": self.paused_for_priority,   # (R122: the tag, and
                #  its words while it waits: "paused for a higher-priority request")
                "max_tokens": self.max_tokens, "prompt_read": pr[0] if pr else None, "prompt_total": pr[1] if pr else None,
                "prefill_tok_s": self.prefill_tok_s if not self.first_token else None,
                "prefill_shared": self.read_shared,     # (it read beside another chat: that rate is a shared clock's)
                "turbo": dict(self.turbo) if self.turbo else None,   # (the Monitor's Turbo badge: its options, none = exact)
                "tok_s": round(self.tok_s(now), 1) if self.first_token else None,
                "elapsed_s": round(now - self.started, 1) if self.started else None}

    def clear(self):
        self.base = None
        self.samples.clear()

    def _slide(self, now):
        while self.samples and self.samples[0][0] <= now - RATE_WINDOW_S:
            self.base = self.samples.popleft()

    def note(self, now, n):
        self.samples.append((now, n))
        self._slide(now)

    def tok_s(self, now) -> float:
        """From `base` (the last sample at or before now - RATE_WINDOW_S) to the newest.  Measuring from the first
        sample INSIDE the window read a burst after a long quiet stretch (tokens held up, then sent at once) as its size
        over a fraction of a second - a 300-token burst and a token 0.3 s later read ~1000 tok/s.  Younger than
        RATE_MIN_SPAN_S: the mean so far, span floored."""
        self._slide(now)
        start = self.base or (self.samples[0] if self.samples else None)
        if start is None:
            return 0.0
        t1, n1 = self.samples[-1] if self.samples else self.base
        if t1 - start[0] >= RATE_MIN_SPAN_S:
            return max(0.0, (n1 - start[1]) / (t1 - start[0]))
        return (n1 - start[1] + 1) / max(RATE_MIN_SPAN_S, now - start[0])


class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = False):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.fit_max_tokens = fit_max_tokens          # --fit-max-tokens: clamp the output cap instead of 400
        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block
        self.default_effort = None                    # the run config's `reasoning_effort`: a request without one
        self.priority_default = None                  # the run config's `priority` (R122): None = normal
        self.shared = {}                              # the web app's Chat settings for every client (POST /settings)
        self.shared_path = None                       # where they are kept between starts (next to the config)
        self.turbo_default = None                     # the model's live Turbo (PLAN-turbo-live): {"drop","skip"} or None
        self.turbo_path = None                        # its sidecar file (the engine config is never touched)
        self.turbo_live_before = None                 # Turbo on an engine whose half is live (QFN): the default before
        self.fifo = FairLock()
        self.stopping = False            # the server is ending: a request still queued is refused, not started
        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        self.response_limits = limits_from_env()
        self.responses = ResponseStore.from_limits(self.response_limits)
        self.response_jobs = BackgroundJobs(self.responses, self.response_limits)
        # pages on OTHER origins allowed to call /v1/* from a browser (CORS; the config's "cors_origins"): exact origins
        # like "http://127.0.0.1:8090", or "*" for every page.  Empty = no CORS headers at all.
        self.cors_origins: list[str] = []
        self.status = {"busy": False, "queued": 0}      # GET /status: what the model is doing right now
        self.active = 0                                 # requests with the engine now (several: READY multi=<n>)
        self.current_requests: set[RequestStat] = set()  # the requests decoding now: first token to the end of run()
        self.latest_request: RequestStat | None = None  # the request that last got the engine
        self.running: set[RequestStat] = set()          # every request with the engine now, reading or writing
        self.waiting: set[RequestStat] = set()          # requests in the FIFO for their first turn (the "Right now" rows)
        self._req_ids = itertools.count(1)              # stable per-request ids for the Monitor (live.chats)
        self.ended = collections.deque(maxlen=6)        # the last few that ended (kept for older readers of /metrics)
        # the Monitor's Conversations rows: recent conversations by their last turn (LRU, idle CONV_IDLE_S on gone;
        # the per-instance conv_max / conv_idle_s are what the tests turn down)
        self.convs: collections.OrderedDict = collections.OrderedDict()
        self._conv_ids = itertools.count(1)
        self.conv_max, self.conv_idle_s = CONV_MAX, CONV_IDLE_S
        self.history = collections.deque(maxlen=500)    # the last finished requests, newest last (GET /metrics)
        # since the server started (the Monitor's totals, issue #35)
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0,
                       "peak_decode": None,   # the fastest answer's decode: {tok_s, tokens, time} (PEAK_MIN_TOKENS+)
                       "peak_prefill": None}  # the fastest prompt read: {tok_s, tokens, time} (PEAK_MIN_READ_TOKENS+)
        # every prompt read as its own INTERVAL [stat, t_start, t_end, tokens, steps] (the drivers' read hooks fill
        # it): the window's count = in-progress reads' steps inside it + finished reads' tokens spread over their
        # overlap with it - a top-up lands ACROSS the read it belongs to, not in the window it happens to end in,
        # and the peaks come from this figure alone
        self.reads: collections.deque = collections.deque()   # the intervals, oldest first (_read_prune drops closed)
        self.read_wall_s, self.read_wall_since = 0.0, None   # wall seconds with >=1 prompt reading / >=1 answer writing
        self.decode_wall_s, self.decode_wall_since = 0.0, None
        # the loop guard (serve/loop_guard.py; SAGUARO_LOOP_GUARD=0 / "loop_guard": false: off) and what it did since
        # the server started: loops seen, and how each was broken (the Monitor's totals, /metrics, Prometheus)
        self.loop_guard = True
        self.loop_guard_counts = {"detected": 0, "penalised": 0, "thinking_closed": 0, "stopped": 0, "tool_repeats": 0,
                                  "tool_blocked": 0, "collapsed": 0, "kept_going": 0}
        self.last_timings = None                         # the last finished request's, llama.cpp's names (/v1/status)
        self.last_request_at = None                      # when a request last started or finished
        self.started_at = time.time()
        self.status_lock = threading.Lock()
        self.mcp = None                                  # serve/mcp.py's McpHub when MCP servers are configured
        self.config_path = None                          # the run config (a context change is saved into it)
        self.savings = None                              # serve/savings.py's Ledger: what the API would have charged
        self.model_dir = None                            # the model's folder (free disk on the dashboard)
        self.model_label = None                          # what runs, as users see it: "MiniMax M3.1 · Maya-XS"
        self.model_name_pretty = None                    # /health, /metrics: the model's display name on its own
        self.model_quant = None                          #   ("Qwen3.8 Flash Next") and its files' name as named
        #                                                  ("Q8_0"); None: this server does not say (the page's old text)
        self.supervisor = None                           # (llama-swap: it owns starts, stops, switches and configs)
        self.trained_context = None                      # what the model was trained for (its GGUF): the slider's top
        self.lookup = None                               # DeepSeek's prompt lookup (--suffix-draft above 0); the others' INFO says
        self.dspark = None                               # DeepSeek's: drafts with its DSpark blocks (its engine's INFO
        #                                                  does not say: the config's --dspark and the pack's drafter)
        self.reload = None                               # a context change: {state, from, to, started, error}
        # "idle_unload_s" / --idle-unload (as Strata's): after this long without a request the engine ends, giving back
        # its GPU memory and its pinned RAM tier, and the next request starts it again (0: never).  With
        # SAGUARO_GLM_SLOT_KEEP=1 the conversation in the model is kept on disk across it.
        self.idle_unload_s = 0.0
        self.keep_alive_s = None                          # a request's keep_alive (None: idle_unload_s; -1: never)
        self.puller = None                               # serve/model_pull.py: the one model download (the Models page, /api/pull)
        self.unloaded = False                            # the engine was ended by the idle unload (not by a crash)
        self._engine_start = threading.Lock()            # ensure_engine: one start, also with --parallel's requests
        self.vision_held = False                         # the elastic QFN encoder holds VRAM the expert cache gave it
        self.last_image_at = 0.0                         # when an image turn last ended (that encoder's idle stop)
        self._vision_enc = threading.Lock()              # the encode region of an elastic encoder that runs beside
        #                                                  the batch slots (VLEND): one encode, and the idle stop sees it
        self.updater = Updater(SAGUARO_VERSION)             # About > Updates: a newer release, and updating to it
        # The ids that end an answer server-side.  A string that is not a special token in this vocabulary
        # (GLM-5.3 has no <|im_end|>) encodes to its TEXT pieces, whose ids occur inside ordinary answers -
        # stopping on them would cut generation at any '<'.  Only single-token (real special) encodings count.
        def _special_ids(s: str) -> list[int]:
            ids = tokenizer.encode(s, parse_special=True)
            return ids if len(ids) == 1 else []
        # GLM ends an assistant turn with the next role's token (<|user|> / <|observation|>): the engine stops on it,
        # and it must not reach the answer's text either
        self.stop_ids = set(_special_ids(IM_END) + _special_ids("<|endoftext|>") + _special_ids("<|user|>") +
                            _special_ids("<|observation|>") +
                            # DeepSeek-V4.1 ends a turn with its EOS (and would start the next with <｜User｜>)
                            _special_ids("<｜end▁of▁sentence｜>") + _special_ids("<｜User｜>") +
                            # MiniMax-M3.1 with [e~[ (and would start the next turn with ]~b])
                            _special_ids("[e~[") + _special_ids("]~b]") +
                            # Hy4-flash with <｜hy_end:..｜> (and would start the next turn with <｜hy_start:..｜>)
                            (_special_ids("<｜hy_end%s｜>" % template.hy4_tag) +
                             _special_ids("<｜hy_start%s｜>" % template.hy4_tag)
                             if getattr(template, "hy4", False) else []) +
                            # Inkling 2 with <|content_model_end_sampling|>
                            _special_ids("<|content_model_end_sampling|>"))
        # DeepSeek's turn openers: where its last user turn starts and its system prompt ends (SAGUARO_DS4_RESTORE_POINTS)
        self.ds_user_id = (_special_ids("<｜User｜>") or [None])[0]
        self.ds_system_id = (_special_ids("<｜System｜>") or [None])[0]
        self.ds_assistant_id = (_special_ids("<｜Assistant｜>") or [None])[0]
        # ... and a model that spells a role marker out of ordinary pieces ('<|', 'user', '|>') has ended its turn just
        # the same: the text stops there.  Only markers that are special tokens in this vocabulary count.
        self.stop_texts = [s for s in ("<|user|>", "<|observation|>", "<|endoftext|>", "<｜end▁of▁sentence｜>", "<｜User｜>",
                                       "[e~[", "]~b]")
                           if _special_ids(s)]
        # the thinking budget (run config `thinking_budget`, tokens; 0 = none): a reasoning block that reaches it is
        # closed with the template's </think> and the answer follows - GLM at High can plan for 14k+ tokens
        self.think_budget = 0
        # (MiniMax-M3.1 closes its reasoning with </mm:think>; its vocabulary may hold a plain "</think>" too)
        # (Hy4-flash with its suffixed </think:..>, which the pack's tokenizer names </think> in what the server reads)
        end = _special_ids("</mm:think>") if getattr(template, "minimax", False) else \
            (_special_ids("</think%s>" % template.hy4_tag) or _special_ids("</think>")) \
            if getattr(template, "hy4", False) else _special_ids("</think>")
        self.think_end_id = end[0] if end else None
        # Inkling 2 leaves its reasoning at segment marks, not </think>: <|end_message|><|message_model|><|content_text|>
        # opens the answer (the close Service.run puts there for a tiny max_tokens); its other marks end it too
        ink = getattr(template, "inkling", False)
        close = [i for m in (INK_END, INK_MODEL, INK_TEXT) for i in _special_ids(m)] if ink else []
        self.ink_close = close if len(close) == 3 else None
        self.ink_marks = [i for m in (INK_TEXT, INK_CALL, INK_END, INK_EOS) + INK_OTHER_TEXT
                          for i in _special_ids(m)] if ink else []
        # the tokens the engine itself ends a request at, whatever the server does (see THINK_RESTARTS): a pack
        # tokenizer that names its "eos" (DeepSeek's, MiniMax's: their engines' --eos default) ends there only; the
        # GLM/Qwen engine ends at every turn-end token of its vocabulary (GLM: <|endoftext|>, <|user|>,
        # <|observation|>), and its tokenizer names them by their GGUF keys - all of stop_ids then
        sid = getattr(tokenizer, "special_ids", None) or {}
        self.eos_ids = {sid["eos"]} if isinstance(sid.get("eos"), int) else set(self.stop_ids)
        # the special literals that stay text inside a message (frontend.mark_literals): every one the vocabulary has
        self.literals = literal_tags(getattr(tokenizer, "special_tokens", None))

    def text_only(self) -> str | None:
        """Hy4-Flash while its engine has no vision tower loaded (--vision off, a pack without one, no room for a
        picture): its name for the errors' words; None for every other model."""
        if getattr(self, "vision", None) is None and getattr(getattr(self, "template", None), "hy4", False):
            return HY4_TEXT_ONLY
        return None

    def check_request(self, req: dict) -> None:
        """A request's sampling fields and logit_bias, checked (a ValueError is the 400) and kept in the form the engine
        line takes, before anything is sent.  A logit_bias needs an engine that applies it (INFO logit_bias=1): an
        older one would answer as if it had none."""
        check_request_sampling(req)
        # R122: the queue priority.  The body's "priority" ("high" | "normal" | "low", anything else a 400),
        # else the X-Saguaro-Priority header the handler passed in (the body wins), else the run config's
        # "priority", else normal.  OpenAI's "service_tier" is NOT this: it names a billing tier and is refused
        # where the dialect checks it (/v1/responses), it never steps another client's request aside.
        want, where = req.get("priority"), "priority"
        if want is None and req.get("_priority_header") is not None:
            want, where = req.get("_priority_header"), "X-Saguaro-Priority header"
        if want is not None:
            req["_priority"] = priority_level(want, where)
        else:
            req["_priority"] = self.priority_default if self.priority_default is not None else PRIORITY_NORMAL
        # PLAN-turbo-live: a request without its own Turbo takes the model's live one (the control panel's, no
        # config rewrite); one that says off keeps off - it overrides the global setting for itself
        if not isinstance(req.get("turbo"), dict):
            default = self.turbo_default   # (read once: a concurrent POST /api/turbo/options stores None between two reads and dict(None) is a 500)
            req["turbo"] = dict(default) if default else None
        if getattr(self, "engine_kind", None) in NO_LIVE_TURBO:   # (no Turbo in these engines: the request runs exact)
            req["turbo"] = None
        bias = logit_bias_of(req.get("logit_bias"), len(getattr(self.tok, "tokens", None) or ()) or None)
        if bias and (getattr(self.engine, "info", None) or {}).get("logit_bias") != 1:
            raise ValueError("logit_bias: this engine does not apply token biases - update Saguaro (./saguaro.sh --setup)")
        # ignore_eos (vLLM's and llama.cpp's): the tokens that end an answer are banned, so it runs to max_tokens - as
        # llama.cpp does it, through the logit bias, so the text past the model's natural end stays text it would write
        ignore = req.get("ignore_eos")
        if ignore is not None and not isinstance(ignore, bool):
            raise ValueError("ignore_eos: true or false")
        if ignore:
            try:
                limit = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)
            except (TypeError, ValueError):
                limit = 0
            if limit <= 0:
                raise ValueError("ignore_eos: needs max_tokens (the answer would run until the context is full)")
            if (getattr(self.engine, "info", None) or {}).get("logit_bias") != 1:
                raise ValueError("ignore_eos: this engine does not apply token biases - update Saguaro (./saguaro.sh --setup)")
            bias = {**bias, **{t: -100.0 for t in sorted(self.stop_ids | self.eos_ids)}}
        if "logit_bias" in req or ignore:
            req["logit_bias"] = bias
        # logprobs need an engine that writes them (INFO logprobs=<n>: the DeepSeek and GLM engines); another would answer
        # without them, which a client that asked could not tell from a bug
        n = logprobs_of(req)
        if isinstance(req.get("top_logprobs"), str):
            req["top_logprobs"] = n
        top = (getattr(self.engine, "info", None) or {}).get("logprobs")
        if n is not None and not (isinstance(top, int) and top >= n):
            # (an engine without them - QFN's: the answer comes without logprobs, as Strata's server did; /health's
            #  capabilities say so - a 400 would break clients that always ask)
            req.pop("logprobs", None)
            req.pop("top_logprobs", None)
            if not getattr(self, "_lp_said", False):
                self._lp_said = True
                print("[saguaro] logprobs asked, but this engine does not report them: answered without", flush=True)

    def capabilities(self) -> dict:
        """What this engine does, for the web app and clients (the same keys for every engine): logprobs (how many
        alternatives, 0: none), images, conversations at once, the engine's kind."""
        info = getattr(self.engine, "info", None) or {}
        top = info.get("logprobs")
        return {"engine": getattr(self, "engine_kind", None) or "maya",
                "logprobs": top if isinstance(top, int) else 0,
                "images": self.vision is not None,
                "parallel": max(1, int(getattr(self.engine, "multi", 0) or 1)),
                "max_context": getattr(self.engine, "max_context", None)}

    def render_ids(self, messages, tools, kwargs, suffix="") -> list[int]:
        """The template rendered for `messages` and `tools` (+ `suffix`, text the server appends to the prompt) and
        tokenized.  A special token's text inside a message or a tool is encoded as that text; only the literals the
        template (or the suffix) writes are special tokens."""
        marked, marked_tools, changed = mark_literals(messages, tools, self.literals)
        prompt = self.template.render(marked, tools=marked_tools, **kwargs)
        if kwargs.get("enable_thinking") is False and prompt.endswith("<think>"):
            # GLM-5.3's template opens the thinking block whatever it is told (no enable_thinking, and an effort it
            # does not know means 'max'): with thinking off the answer was written inside an open <think>.  Close
            # it empty, the form the same template writes for a past turn that did not think.
            prompt += "</think>"
        if not changed:
            return self.tok.encode(prompt + suffix, parse_special=True)
        prompt, plain = unmark_literals(prompt, self.literals)
        return self.tok.encode(prompt + suffix, parse_special=True, plain=plain)

    def set_shared(self, defaults) -> dict:
        """The Chat settings every client gets for what it leaves out; {} / None = clients use their own again."""
        self.shared = clean_shared_defaults(defaults)
        if self.shared_path:
            try:
                if self.shared:
                    Path(self.shared_path).write_text(json.dumps(self.shared, indent=1), encoding="utf-8")
                else:
                    Path(self.shared_path).unlink(missing_ok=True)
            except OSError as e:
                print(f"[saguaro] could not save the shared settings: {e}", flush=True)
        return self.shared

    def set_turbo_default(self, value) -> dict:
        """The model's live Turbo (PLAN-turbo-live, the control panel's per-model setting): lossy options every
        request that carries none of its own runs with (agents included).  It applies at once - no config rewrite,
        no reload - and is kept in a sidecar file for the next start; the engine's run config is not touched."""
        t = turbo_live.normalize_turbo(value)
        if getattr(self, "engine_kind", None) in NO_LIVE_TURBO and turbo_live.turbo_on(t):
            raise ValueError(f"Turbo: {NO_LIVE_TURBO[self.engine_kind]}'s engine has none - its requests run exact")
        self.turbo_default = turbo_live.turbo_on(t)     # (off, None: the model answers exact unless a request says otherwise)
        if self.turbo_path:
            try:
                if self.turbo_default:
                    Path(self.turbo_path).write_text(json.dumps(self.turbo_default, indent=1), encoding="utf-8")
                else:
                    Path(self.turbo_path).unlink(missing_ok=True)
            except OSError as e:
                print(f"[saguaro] could not save the live Turbo setting: {e}", flush=True)
        return dict(self.turbo_default) if self.turbo_default else None

    def turbo_skip_supported(self) -> bool:
        """PLAN-p5: whether this engine applies the skip half of the live Turbo.  QFN's applies drop only (its
        request line takes turbo_skip= and says the engine leaves it off), and an engine that gains it says so
        itself (INFO turbo_skip=1, kept in the driver's facts) - until then the page hides the Skip choice.
        Every other engine takes skip: the default is true."""
        if getattr(self, "engine_kind", None) != "qfn":
            return True
        return str((getattr(self.engine, "info", None) or {}).get("turbo_skip") or "") == "1"

    def with_shared(self, req: dict, api: str) -> dict:
        """The request with the shared thinking level and max tokens filled in where it has none of its own."""
        s = self.shared
        if not s:
            return req
        req = dict(req)
        if "max_tokens" in s and not req.get("max_tokens") and not req.get("max_completion_tokens"):
            req["max_tokens"] = s["max_tokens"]
        effort = s.get("reasoning_effort")
        if effort:
            if api == "openai":
                ctk = req.get("chat_template_kwargs") if isinstance(req.get("chat_template_kwargs"), dict) else {}
                if not req.get("reasoning_effort") and not req.get("reasoning") and \
                        "enable_thinking" not in ctk and "reasoning_effort" not in ctk:
                    req["reasoning_effort"] = effort
            elif not req.get("thinking") and not req.get("output_config"):
                if effort == "none":
                    req["thinking"] = {"type": "disabled"}
                else:
                    req["output_config"] = {"effort": effort}
        return req

    def close_for_restart(self):
        """Before the server ends for an update: the engine, the pictures encoder and the MCP servers stop first."""
        for part in (self.response_jobs, self.engine, self.vision, self.mcp):
            try:
                if part is not None and hasattr(part, "close"):
                    part.close()
            except Exception as e:  # noqa: BLE001 - ending anyway
                print(f"[saguaro] stopping {type(part).__name__}: {e}", flush=True)

    def ensure_engine(self):
        """The engine running (the caller holds the FIFO): started again after an idle unload, or after it died in an
        earlier request (issue #27) instead of failing every request.  With --parallel several requests hold a place
        each: the first one starts it, the others wait for that start."""
        if not hasattr(self.engine, "alive") or self.engine.alive():
            return
        with self._engine_start:
            if self.engine.alive():                      # (another request started it meanwhile)
                return
            if self.unloaded:
                print("[saguaro] loading the model again after an idle unload (a minute or so) ...", flush=True)
            else:
                code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
                print(f"[saguaro] the engine had stopped (exit code {code}); starting it again "
                      "(a minute or two) ...", flush=True)
            self.engine.restart()
            self.unloaded = False
            # the FIFO's places are the conversations the restarted engine reports: a start that came back with other
            # numbers (a different --parallel, no multi= at all) queued or over-admitted every request from here on
            places = max(1, int(getattr(self.engine, "multi", 0) or 1))
            if places != self.fifo.capacity:
                self.fifo.set_capacity(places)
            print("[saguaro] the engine is running again", flush=True)

    def _engine_ready(self) -> bool:
        """The engine runs, so its stdin takes a command (engines without an alive() are always ready here)."""
        return not hasattr(self.engine, "alive") or self.engine.alive()

    def _vision_lends(self) -> bool:
        """The QFN engine lends CUDA0's VRAM to the elastic encoder with VLEND / VRECLAIM (its INFO says so), so the
        encoder can start and stop while batch slots decode; without it the engine only takes VRAM between requests."""
        return (getattr(self.vision, "elastic", False) and hasattr(self.engine, "vision_lend")
                and str(getattr(self.engine, "info", {}).get("vision_lend") or "") == "1")

    def _vision_up(self):
        """The image turn's start for an elastic QFN encoder that is not running.  With a lending engine (VLEND) the
        caller holds its own FIFO place only: CUDA0's expert cache keeps elastic_mib free at its next quiet moment
        while the batch slots decode, then the encoder starts.  Without one the caller holds every FIFO place (no
        conversation decodes, the engine takes the VRAM command between requests).  If the cache cannot shrink (an
        engine that neither lends nor has --vram-elastic), the encoder starts anyway, as a resident one would."""
        v = self.vision
        if not getattr(v, "elastic", False) or v.alive():
            return
        t0, room = time.time(), ""
        if self._vision_lends() and self._engine_ready():
            # the engine may hold its raised floor after ANY VLEND (a refusal can leave the give half-done), so the
            # idle stop must send VRECLAIM even when the lend itself was refused
            self.vision_held = True
            # PLAN-vision-fast-start review 3: WHEN the command goes out and when its answer comes back - the line
            # pairs with the engine's "read by the stdin reader" line, so a wait between the two can be placed
            t_lend = time.monotonic()
            print(f"[saguaro] images: VLEND {v.elastic_mib} MiB sent to the engine (its VLENT line answers)",
                  flush=True)
            tries = 0
            while True:
                tries += 1
                try:
                    out = self.engine.vision_lend(v.elastic_mib)   # waits for the engine's VLENT line
                    print(f"[saguaro] images: VLENT after {(time.monotonic() - t_lend) * 1000:.0f} ms "
                          f"({out.get('cache_mib')} MiB mapped)", flush=True)
                    room = (f"the expert cache keeps {v.elastic_mib} MiB free for it "
                            f"({out.get('cache_mib')} MiB left mapped); ")
                    break
                except LendTimeout as e:
                    waited = (time.monotonic() - t_lend) * 1000
                    if tries == 1:
                        # review 4: keep the 45 s timeout, but on a timeout wait once more - the engine reads VLEND on
                        # its own thread now and answers at the next settle, so the first 45 s only covers a read that
                        # genuinely cannot pause yet; a second no-answer is the engine, not a queue (which is fixed)
                        print(f"[saguaro] images: no VLENT after {waited:.0f} ms ({e}); waiting for the engine once more",
                              flush=True)
                        t_lend = time.monotonic()
                        continue
                    # a lending engine that will not answer: the encoder is never started on a card with no room (the
                    # engine did not say vision_lend=0, so this is not the resident start-anyway path)
                    print(f"[saguaro] images: still no VLENT after another {waited:.0f} ms; the image encoder is not "
                          f"started without room", flush=True)
                    raise ValueError(f"the engine made no room for the image encoder ({e})") from None
                except ValueError as e:
                    # an ERR refusal (the engine is alive and says "no room"): start it anyway, as a resident one would
                    print(f"[saguaro] images: the lend was refused after {(time.monotonic() - t_lend) * 1000:.0f} ms "
                          f"({e})", flush=True)
                    room = f"the expert cache did not make room for it ({e}); "
                    break
        elif hasattr(self.engine, "vram_reserve") and self._engine_ready():
            try:
                out = self.engine.vram_reserve(v.elastic_mib)
                self.vision_held = True
                room = (f"the expert cache keeps {v.elastic_mib} MiB free for it "
                        f"({out.get('expert_cache_mib')} of {out.get('expert_cache_full_mib')} MiB left); ")
            except ValueError as e:
                room = f"the expert cache did not make room for it ({e}); "
            if self.vision_held:
                time.sleep(0.2)      # the free-VRAM counter shows the cache's release a moment later
        if not self.vision_held and hasattr(self.engine, "vision_room") and not self.engine.vision_room():
            # neither lend path is configured on this engine (no "INFO vision_lend=1" from the give path, no
            # --vram-elastic behind the VRAM command - its own start facts, serve/qfn_engine's vision_room): the
            # encoder would start onto a card with nothing to give it, which is how the reader got SIGKILLed
            # (U6's ./ref/vision-fail.log) - refuse the picture with a plain error instead
            raise ValueError("this engine lends the picture encoder nothing: no batch slots to give room from "
                             "and not started with --vram-elastic (its INFO says neither) - run the setup again "
                             "or add \"vram_elastic\": true")
        v.restart()
        print(f"[saguaro] images: {room}the image encoder started in {time.time() - t0:.1f} s", flush=True)

    @contextlib.contextmanager
    def _vision_active(self):
        """The encode region: the shared encoder's lend/reclaim bracket; QFN's elastic reader needs none (its up and
        down are this service's, around the encode)."""
        active = getattr(self.vision, "active", None)
        if active is None:
            yield self.vision
        else:
            with active():
                yield self.vision

    def _vision_keep_hot(self):
        """The cheap-restart state after an idle stop: the elastic reader's weight files held in the OS file cache
        for elastic_keep_s (the process itself cannot idle without GPU memory - see the Vision class).  It costs no
        VRAM, so it outlives the stop; the VRAM (VRECLAIM) already went back at elastic_idle_s."""
        keep_hot = getattr(self.vision, "keep_hot", None)
        if keep_hot is not None:
            try:
                keep_hot()
            except Exception as e:                    # a cold restart is the fallback, never a failed stop
                print(f"[saguaro] images: keeping the encoder's weights hot: {e}", flush=True)

    def _vision_idle_down(self, force: bool = False) -> str:
        """An elastic encoder idle for elastic_idle_s (or force) stops and the expert cache takes its VRAM back:
        "down", "busy" (an encode is running, or with a non-lending engine a request holds the engine, or a picture
        came more recently) or "nothing"."""
        v = self.vision
        if not getattr(v, "elastic", False) or not (self.vision_held or v.alive()):
            return "nothing"
        if not force and time.time() - self.last_image_at < v.elastic_idle_s:
            return "busy"
        if self._vision_lends():
            # VRECLAIM works while the slots decode: only an encode on the encoder itself may not run under it
            if not self._vision_enc.acquire(blocking=False):
                return "busy"
            try:
                v.unload()
                if self.vision_held and self._engine_ready():
                    out = self.engine.vision_reclaim()
                    print(f"[saguaro] images: the image encoder stopped after {v.elastic_idle_s:g} s without a "
                          f"picture; the expert cache took its VRAM back ({out.get('cache_mib')} MiB mapped)",
                          flush=True)
                self.vision_held = False
                self._vision_keep_hot()
                return "down"
            finally:
                self._vision_enc.release()
        if not self.fifo.acquire(blocking=False):     # every place, nobody even queued: the engine takes no answer now
            return "busy"
        try:
            v.unload()
            if self.vision_held and hasattr(self.engine, "vram_reserve") and self._engine_ready():
                out = self.engine.vram_reserve(0)
                print(f"[saguaro] images: the image encoder stopped after {v.elastic_idle_s:g} s without a picture; "
                      f"the expert cache took its VRAM back ({out.get('expert_cache_mib')} of "
                      f"{out.get('expert_cache_full_mib')} MiB)", flush=True)
            self.vision_held = False
            self._vision_keep_hot()
            return "down"
        finally:
            self.fifo.release()

    def start_vision_idle(self):
        """The idle-stop thread for an elastic encoder (does nothing for every other vision setup)."""
        v = self.vision
        if not getattr(v, "elastic", False):
            return
        print(f"[saguaro] images: the image encoder starts with the first picture and stops after {v.elastic_idle_s:g} s "
              f"without one (its weights stay hot for {getattr(v, 'elastic_keep_s', 0):g} s for a cheap restart); while "
              f"it runs the expert cache keeps {v.elastic_mib} MiB free for it", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, v.elastic_idle_s / 4)))
                try:
                    self._vision_idle_down()
                except Exception as e:                 # tried again at the next tick; the thread keeps running
                    print(f"[saguaro] images: stopping the idle image encoder: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def unload(self, idle_for: float | None = None) -> str:
        """End the engine between requests: "unloaded", "not loaded", "busy" (a request is running or waiting, a context
        reload is on, or - with idle_for - one ran more recently than that) or "unsupported"."""
        if not (hasattr(self.engine, "close") and hasattr(self.engine, "restart") and hasattr(self.engine, "alive")):
            return "unsupported"
        if not self.fifo.acquire(blocking=False):        # every place: nothing runs or waits for the engine
            return "busy"
        try:
            if not self.engine.alive():
                return "not loaded"
            with self.status_lock:
                if self.status.get("busy") or self.status.get("queued") or self.active:
                    return "busy"
                if self.reload and self.reload.get("state") in ("waiting", "running"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            # QUIT (close: with SAGUARO_GLM_SLOT_KEEP the engine first writes the conversation it holds), then the GPU has
            # its memory back.  (StrataEngine.stop is the request's STOP, not the engine's end.)
            self.engine.close()
            self.engine.ended = True
            pid = getattr(getattr(self.engine, "proc", None), "pid", None)
            if pid is not None:
                wait_gpu_release(pid)
            self.unloaded = True
            print(f"[saguaro] model unloaded{f' after {idle_for:.0f} s idle' if idle_for else ''} (GPU memory and the "
                  "RAM tier given back); the next request loads it again", flush=True)
            return "unloaded"
        finally:
            self.fifo.release()

    def idle_limit(self) -> float:
        """Seconds without a request before the model unloads: the last request's keep_alive, else the config's
        idle_unload_s; 0 = never."""
        ka = self.keep_alive_s
        if ka is None:
            return self.idle_unload_s
        return 0.0 if ka < 0 else ka

    def start_idle_unload(self):
        if not hasattr(self.engine, "close"):
            return
        if self.idle_unload_s:
            print(f"[saguaro] the model unloads after {self.idle_unload_s:g} s without requests", flush=True)

        def loop():
            while True:
                limit = self.idle_limit()
                time.sleep(max(1.0, min(30.0, (limit or 120.0) / 4)))
                limit = self.idle_limit()
                if not limit:
                    continue
                try:
                    self.unload(idle_for=limit)
                except Exception as e:  # noqa: BLE001 - tried again at the next turn; the thread keeps running
                    print(f"[saguaro] idle unload: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def reloading(self) -> bool:
        with self.status_lock:
            return bool(self.reload) and self.reload.get("state") in ("waiting", "running")

    def context_limits(self) -> dict:
        """The dashboard's Context size: the range it offers and the engine's measured cost of the current one."""
        info = dict(getattr(self.engine, "info", {}) or {})
        rng = models.context_range(self.trained_context, self.engine.max_context, str(info.get("engine_kind") or ""))
        with self.status_lock:
            reload = dict(self.reload) if self.reload else None
        # a streamed cache's RAM (KV streaming: maya-mm3's VRAM lines' kv_ram - every position's K / V in pinned RAM,
        # the engine's own count at this context) and the RAM free now: the dashboard scales the one against the other
        kv_ram = sum(int(p.get("kv_ram") or 0) for p in getattr(self.engine, "vram", None) or [])
        ram = {}
        if kv_ram:
            try:
                import psutil
                vm = psutil.virtual_memory()
                ram = {"ram_gb": round(vm.total / 2**30, 1), "ram_free_gb": round(vm.available / 2**30, 1)}
            except Exception:  # noqa: BLE001 - psutil is optional: no RAM figures, no RAM warning
                pass
        return {"context": self.engine.max_context, **rng,
                "kv_gb": info.get("kv_gb"), "kv_ctx": info.get("kv_ctx"),
                "kv_resident": info.get("kv_resident") or 0,
                # the engine's own K/V bytes of one position (every layer, the dtype it runs), and the pinned host
                # K/V it holds now (streaming engines; 0 when allocated lazily) - the dashboard scales these, never a constant
                "kv_bytes_token": info.get("kv_bytes_token"), "kv_ram_bytes": info.get("kv_ram_bytes"),
                "kv_ram_gb": round(kv_ram / 2**30, 3) if kv_ram else None, **ram,
                "vram_gb": info.get("vram_gb"), "vram_slots": info.get("vram_slots"), "reload": reload,
                "load_s": getattr(self.engine, "load_s", None)}   # (the engine's last load: what a reload takes here)

    def set_context(self, n: int) -> dict:
        """Reload the engine with an n-token context (the dashboard's Context size), on a thread: after the request in
        flight; new requests get a 503 meanwhile.  A start that fails (the context does not fit) puts the old one back;
        a good one is saved into the run config, so the next start keeps it."""
        lim = self.context_limits()
        if not lim["min"] <= n <= lim["max"]:
            raise ValueError(f"the context must be between {lim['min']} and {lim['max']} tokens")
        with self.status_lock:
            if self.reload and self.reload.get("state") in ("waiting", "running"):
                raise ValueError("the model is already reloading")
            self.reload = {"state": "waiting", "from": self.engine.max_context, "to": n, "started": time.time(),
                           "error": None}
            reload = dict(self.reload)
        threading.Thread(target=self._reload, args=(reload["from"], n), daemon=True).start()
        return reload

    def _restart_with(self, n: int):
        if hasattr(self.engine, "set_context"):
            self.engine.stop()
            self.engine.set_context(n)
            self.engine.restart()
        else:                                            # the mock engine: nothing to start
            time.sleep(float(os.environ.get("SAGUARO_MOCK_RELOAD_S", "2")))
            self.engine.max_context = n

    def _reload(self, old: int, n: int):
        with self.fifo:                                  # the request in flight finishes first
            with self.status_lock:
                self.reload["state"] = "running"
            print(f"[saguaro] reloading the model with a {n}-token context (was {old}) ...", flush=True)
            err = None
            t0 = time.time()
            try:
                self._restart_with(n)
                self.engine.load_s = round(time.time() - t0, 1)
            except Exception as e:  # noqa: BLE001 - any failed start: put the old context back
                err = f"the engine did not start with a {n}-token context ({e})"
                print(f"[saguaro] {err}; starting it again with {old} ...", flush=True)
                try:
                    self._restart_with(old)
                except Exception as e2:  # noqa: BLE001
                    err += f"; starting again with {old} failed too ({e2}) - the next request tries once more"
            places = max(1, int(getattr(self.engine, "multi", 0) or 1))   # the reloaded engine's conversations
            if places != self.fifo.capacity:
                self.fifo.set_capacity(places)
            if err is None:
                self._save_context(n)
                print(f"[saguaro] the model runs with a {n}-token context now", flush=True)
            with self.status_lock:
                self.reload.update(state="failed" if err else "done", error=err, ended=time.time(),
                                   context=self.engine.max_context)

    def _save_context(self, n: int):
        """--max-context in the run config (its other keys as they are), so the next start keeps the new size -
        and with it --rope-scale on a QFN config past its YaRN origin: the live move synced the driver's
        command but not this file, and the next start spawns from the file (models.qfn_yarn_follow)."""
        if not self.config_path:
            return
        try:
            p = Path(self.config_path)
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
            inst = cfg.setdefault("installer", {})       # (the first change from the page: what the setup wrote first,
            if "as_set_up" not in inst:                  #  the Models page's Reset - as models.save keeps it)
                inst["as_set_up"] = models.settings_of(cfg)
            args = list(cfg.get("args") or [])
            if "--max-context" in args[:-1]:
                args[args.index("--max-context") + 1] = str(int(n))
            else:
                args += ["--max-context", str(int(n))]
            args = models.qfn_yarn_follow(args)          # (a 1M slider move saves --rope-scale 4 with it: saved
                                                         #  args the saved factor can actually reach)
            cfg["args"] = args
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            os.replace(tmp, p)
        except (OSError, ValueError) as e:
            print(f"[saguaro] could not save the new context in {self.config_path}: {e}", flush=True)

    def think_budget_info(self) -> dict:
        """The dashboard's Thinking budget: tokens (0 = no limit), whether this model has a think block at all."""
        return {"budget": int(self.think_budget or 0),
                "supported": self.think_end_id is not None and getattr(self, "engine_kind", None) != "ik2",
                "saved": bool(self.config_path)}

    def set_think_budget(self, value) -> dict:
        """The dashboard's Thinking budget: 0 or 256..262144 tokens.  It applies from the next request (think_budget is
        read per request, no restart) and is saved into the run config - marked as the user's own, so a setup or a
        switch keeps it."""
        if self.think_end_id is None or getattr(self, "engine_kind", None) == "ik2":   # (saguaro-ik2 reads none)
            raise ValueError("this model has no thinking budget")
        n = maya_quants.check_thinking_budget(value)
        self.think_budget = n
        saved = False
        if self.config_path:
            p = Path(self.config_path)
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
            cfg["thinking_budget"] = n
            cfg.setdefault("installer", {})["thinking_budget_set"] = True
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            os.replace(tmp, p)
            saved = True
        print(f"[saguaro] thinking budget: {n or 'no limit'}" + (" (saved)" if saved else ""), flush=True)
        return {"budget": n, "supported": True, "saved": saved}

    def port_info(self) -> dict:
        """The dashboard's Port: where the server listens, and whether a new port can apply at once (saguaro.py started
        this server and starts it again on the saved port) or only from the next start."""
        host, port = getattr(self, "listen", None) or ("127.0.0.1", None)
        return {"port": port, "host": host, "restarts": bool(os.environ.get("SAGUARO_RESTART_ON_UPDATE")),
                "saved": bool(self.config_path), "restarting": getattr(self, "port_restart", None)}

    def set_port(self, n: int) -> dict:
        """The dashboard's Port: n saved into the run config (and Saguaro's start scripts for it), then - when saguaro.py
        started this server - the server ends with RESTART_EXIT after the request in flight, and saguaro.py starts it again
        on n.  The port must be free on this server's address now (a busy one: an error, nothing saved)."""
        host, port = getattr(self, "listen", None) or ("127.0.0.1", None)
        if not 1024 <= n <= 65535:
            raise ValueError("the port must be a number from 1024 to 65535")
        if n == port:
            raise ValueError(f"Saguaro listens on {n} already")
        if not self.config_path:
            raise ValueError("this server runs without a config, so a port cannot be saved: start it with --port")
        if getattr(self, "port_restart", None):
            raise ValueError("Saguaro is already moving to another port")
        import socket
        try:
            with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, n))
        except OSError as e:
            raise ValueError(f"port {n} is in use or not allowed here ({e.strerror or e})") from None
        self._save_port(n)
        restarts = bool(os.environ.get("SAGUARO_RESTART_ON_UPDATE"))
        if restarts:
            self.port_restart = n
            threading.Thread(target=self._restart_on_port, args=(n,), daemon=True).start()
        print(f"[saguaro] the port {n} is saved" + (": Saguaro starts again on it" if restarts else
              " - it applies from the next start"), flush=True)
        return {"port": n, "restarts": restarts}

    def _restart_on_port(self, n: int):
        time.sleep(0.5)                                  # (the answer to the page goes out first)
        with self.fifo:                                  # the request in flight finishes first
            print(f"[saguaro] starting again on port {n} ...", flush=True)
            self.close_for_restart()
            os._exit(RESTART_EXIT)

    def _save_port(self, n: int):
        """"port" in the run config, and --port in Saguaro's start scripts beside it that start this config (setup writes
        run-saguaro-<model>.sh / .bat with the port then chosen; a script would otherwise move it back)."""
        p = Path(self.config_path)
        cfg = json.loads(p.read_text(encoding="utf-8-sig"))
        cfg["port"] = int(n)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        os.replace(tmp, p)
        for script in list(p.parent.glob("run-*.bat")) + list(p.parent.glob("run-*.sh")):
            try:
                raw = script.read_bytes()                # (bytes: a .bat keeps its CR LF line ends)
                if p.name.encode() not in raw:
                    continue
                new = re.sub(rb"(--port\s+)\d+", lambda m: m.group(1) + str(int(n)).encode(), raw)
                if new != raw:
                    script.write_bytes(new)
            except OSError as e:
                print(f"[saguaro] could not change the port in {script.name}: {e}", flush=True)

    # -------------------------------------------------------------- the Models page (serve/models.py)
    def models_root(self) -> Path:
        """Where the configs are: beside this server's own (saguaro.py's folder)."""
        return Path(self.config_path).parent if self.config_path else ROOT

    def models_info(self) -> dict:
        """The configs the setup wrote, the running one marked, each with its engine settings; whether a switch or a
        change to the running one can start Saguaro again at once (saguaro.py started this server)."""
        out = models.listing(self.models_root(), self.config_path)
        out.update(restarts=bool(os.environ.get("SAGUARO_RESTART_ON_UPDATE")),
                   running=Path(self.config_path).name if self.config_path else None,
                   restarting=getattr(self, "model_restart", None),
                   deleted=getattr(self, "deleted", None),          # (the model running before, deleted at this start)
                   loaded=not hasattr(self.engine, "alive") or bool(self.engine.alive()),
                   can_unload=all(hasattr(self.engine, x) for x in ("close", "restart", "alive")),
                   pull=self.pull_slot().snapshot(),                # (the download running now, or the last one)
                   leftovers=models.leftovers(self.models_root()))
        return out

    def _model_path(self, name) -> Path:
        name = str(name or "")
        p = self.models_root() / name
        if (not config_names.is_config_name(name) or not p.is_file()):
            raise ValueError(f"no model config named {name!r} here")
        return p

    def _is_running(self, p: Path) -> bool:
        return bool(self.config_path) and p.resolve() == Path(self.config_path).resolve()

    def _restart_busy(self):
        if getattr(self, "model_restart", None) or getattr(self, "port_restart", None):
            raise ValueError("Saguaro is already starting again")
        if self.reloading() or self.updater.busy():
            raise ValueError("Saguaro is reloading or updating; try again when it is done")

    def model_settings(self, name, settings, dry_run: bool = False) -> dict:
        """The engine settings of the config `name` (the Models page): checked, and unless dry_run saved into it.  The
        running model's start Saguaro again with them (when saguaro.py started this server), after the request in flight;
        another model's apply when it is switched to.  dry_run: the command they make, nothing saved."""
        p = self._model_path(name)
        cfg = models.read(p)
        if not cfg:
            raise ValueError(f"{p.name}: not a config Saguaro can read")
        new = models.apply(cfg, settings)                 # (a ValueError: what is wrong, nothing saved)
        if dry_run:
            return {"command": models.command(new), "settings": models.shown_settings(new)}
        running = self._is_running(p)
        if running:
            self._restart_busy()
        out = models.save(p, settings)
        restarts = bool(running and out["changed"] and os.environ.get("SAGUARO_RESTART_ON_UPDATE"))
        if restarts:
            self._restart_into(p, "settings")
        print(f"[saguaro] {p.name}: engine settings saved" + (": Saguaro starts again with them" if restarts else
              " - they apply from its next start" if out["changed"] else " (nothing changed)"), flush=True)
        return {**out, "running": running, "restarts": restarts}

    def turbo_info(self) -> dict:
        """The Turbo button: whether it is on, what switching it on changes in the engine's settings, and whether this
        server can start the engine again by itself.  A server with no run config (--config) has only the request-level
        half: the page's greedy decoding, thinking and prompts."""
        restarts = bool(os.environ.get("SAGUARO_RESTART_ON_UPDATE"))
        if not self.config_path:
            return {"on": bool(getattr(self, "turbo_mem", False)), "engine": False, "plan": {}, "env": {},
                    "restarts": False, "family": None}
        cfg = models.read(Path(self.config_path))
        return {**models.turbo_info(cfg), "engine": True, "restarts": restarts,
                "restarting": getattr(self, "model_restart", None)}

    def set_turbo(self, on: bool) -> dict:
        """Turbo on or off.  The engine's part (models.turbo_enable / turbo_disable: the context, the expert skipping, the
        drafting, written into the run config) starts Saguaro again as a saved engine setting does, after the request in
        flight; nothing to restart where the engine already runs so (or without a run config).  QFN's lossy half is the
        live per-request Turbo, not the environment: switching on drafts and puts its live Turbo to its default,
        switching off restores what it was."""
        if not self.config_path:
            self.turbo_mem = bool(on)
            return self.turbo_info()
        p = Path(self.config_path)
        self._restart_busy()
        was = models.read(p) or {}
        out = models.turbo_enable(p, self.turbo_default) if on else models.turbo_disable(p)
        self.fifo.set_one(models.turbo_one_at_a_time(models.read(p) or {}))
        live = models.TURBO_LIVE_DEFAULT.get(models.family(models.read(p)))
        if live is not None:
            if on:
                self.turbo_live_before = dict(self.turbo_default) if self.turbo_default else None
                self.set_turbo_default(live)
            else:
                self.set_turbo_default(self.turbo_live_before or (was.get("turbo") or {}).get("live_before"))
                self.turbo_live_before = None
        restarts = bool(out["changed"] and os.environ.get("SAGUARO_RESTART_ON_UPDATE"))
        if restarts:
            self._restart_into(p, "settings")
        print(f"[saguaro] Turbo {'on' if on else 'off'}: {p.name}" + (": Saguaro starts again" if restarts else
              " - the engine settings apply from its next start" if out["changed"] else " (the engine needs no restart)"),
              flush=True)
        return {**self.turbo_info(), "restarting_now": restarts, "changed": out["changed"]}

    def switch_model(self, name) -> dict:
        """The Models page's Switch: the config `name` becomes the most recently used one (a start without --config
        picks it), and when saguaro.py started this server it ends with RESTART_EXIT after the request in flight, leaving
        saguaro.py a note (models.SWITCH_FILE) to start that config, on this port."""
        p = self._model_path(name)
        if self._is_running(p):
            raise ValueError(f"{p.name} is the model running now")
        cfg = models.read(p)
        why = models.missing(cfg)                        # (saguaro.py would stop at it: no model would run)
        if why:
            raise ValueError(f"{p.name} cannot start: {why} - run the setup again to repair it")
        self._restart_busy()
        if models.own_defaults(p):                       # (its model's own sampling and level, as the setup writes)
            print(f"[saguaro] {p.name}: its model's own sampling and thinking level written into it", flush=True)
        os.utime(p)
        restarts = bool(os.environ.get("SAGUARO_RESTART_ON_UPDATE"))
        if restarts:
            self._restart_into(p, "switch")
        print(f"[saguaro] {p.name} is the model to start" + (": Saguaro starts again with it" if restarts else
              " next (a start without --config picks it)"), flush=True)
        return {"to": p.name, "model": cfg.get("model_name") or "", "restarts": restarts}

    def delete_model(self, name, files: bool = False, dry_run: bool = False, then=None) -> dict:
        """The Models page's Delete: the config `name` off the list (models.delete), with `files` what its setup
        downloaded for it as well.  Not while Saguaro starts again.  The model running: Saguaro switches to `then` first, and
        the next server deletes it (models.DELETE_FILE) once the engine that ran from it is gone.  dry_run: what would
        go (models.delete_plan), nothing deleted."""
        p = self._model_path(name)
        running = self._is_running(p)
        if getattr(self, "model_restart", None):
            raise ValueError("Saguaro is starting again; try again when it is done")
        if dry_run:
            return {"file": p.name, "running": running, **models.delete_plan(self.models_root(), p)}
        if running:
            return self._delete_running(p, files, then)
        out = models.delete(self.models_root(), p, bool(files))
        print(f"[saguaro] {p.name} deleted" + (f", and its files ({out['freed'] / 2**30:.1f} GB)" if files else
              " (its files stay)") + "".join(f"; {n} drafts with MTP now (its DFlash2 drafter was deleted)"
                                             for n in out["drafter_users"]), flush=True)
        return out

    def _delete_running(self, p: Path, files: bool, then) -> dict:
        if not then:
            raise ValueError(f"{p.name} is the model running now: pick the model Saguaro switches to first")
        if not os.environ.get("SAGUARO_RESTART_ON_UPDATE"):
            raise ValueError("this server was not started by saguaro.sh or START-SAGUARO.bat, so it cannot switch to another "
                             f"model by itself: start another model, then delete {p.name}")
        if files:                                        # (checked now: the next server only reports)
            plan = models.delete_plan(self.models_root(), p)
            if not plan["ok"]:
                raise ValueError(plan["why"])
        cfg = models.read(p)                             # (its name, for the page once the config is gone)
        note = self.models_root() / models.DELETE_FILE
        note.write_text(json.dumps({"config": str(p.resolve()), "files": bool(files), "time": time.time(),
                                    "model": cfg.get("model_name") or "", "quant": models.quant_label(cfg, p)}),
                        encoding="utf-8")
        try:
            out = self.switch_model(then)                # (its checks: not this one, can start, not busy)
        except BaseException:
            note.unlink(missing_ok=True)
            raise
        print(f"[saguaro] {p.name} is deleted once {out['to']} runs" + (", with its files" if files else ""), flush=True)
        return {"deleting": p.name, "to": out["to"], "files": bool(files), "restarts": True}

    def finish_delete(self):
        """The model that ran before this start, deleted now that another one runs (the Models page deleted the
        running one: its note, models.DELETE_FILE, read once - a .maya-delete.json note of before the C122 rename
        is read too).  self.deleted says how it went, for the page."""
        notes = config_names.note_paths(self.models_root(), models.DELETE_FILE)
        try:
            note = json.loads(notes[0].read_text(encoding="utf-8"))
        except (IndexError, OSError, ValueError):
            return
        for f in notes:
            try:
                f.unlink()                               # (read once, under either name)
            except OSError:
                pass
        p = config_names.resolve(Path(str(note.get("config") or "")))
        if time.time() - float(note.get("time") or 0) > 900 or not p.is_file() or p.parent != self.models_root():
            return                                       # (stale, gone already, or not one of these configs)
        if self._is_running(p):                          # (the switch did not happen: it runs again)
            self.deleted = {"file": p.name, "error": "it is the model running now", "time": time.time()}
            print(f"[saguaro] {p.name} not deleted: it is the model running now", flush=True)
            return
        try:
            out = models.delete(self.models_root(), p, bool(note.get("files")))
        except (OSError, ValueError) as e:
            self.deleted = {"file": p.name, "error": str(e), "time": time.time()}
            print(f"[saguaro] {p.name} not deleted: {e}", flush=True)
            return
        self.deleted = {**out, "model": note.get("model") or "", "quant": note.get("quant") or "", "time": time.time()}
        print(f"[saguaro] {p.name} deleted" + (f", and its files ({out['freed'] / 2**30:.1f} GB)" if out["files"] else
              " (its files stay)"), flush=True)

    def pull_slot(self):
        """The model download slot (serve/model_pull.py); SAGUARO_PULL_COMMAND (a JSON list with {model}) replaces the
        setup's command - a test hook."""
        if self.puller is None:
            def command(ident, cfg):
                hook = os.environ.get("SAGUARO_PULL_COMMAND")
                if not hook:
                    return model_pull.setup_command(ident, cfg)
                return [a.replace("{model}", ident) for a in json.loads(hook)]
            self.puller = model_pull.Puller(self.models_root(), command)
        return self.puller

    def pull_model(self, ident) -> dict:
        cfg = models.read(Path(self.config_path)) if self.config_path else None
        return self.pull_slot().start(ident, cfg).snapshot()

    def delete_installed(self, name) -> dict:
        """DELETE /api/delete and the Models page: the model's config and its own files (models.delete_plan
        never lists a file another config runs from).  Not the running model, not during a download."""
        p = model_api.find_config(self.models_root(), name)
        if p is None:
            raise KeyError(name)
        if self._is_running(p):
            raise ValueError(f"{p.name} is the model running now: load another model first")
        if self.pull_slot().busy():
            raise ValueError("a model is downloading; delete when it is done")
        return self.delete_model(p.name, True)

    def model_listing(self) -> list:
        return models.listing(self.models_root(), self.config_path)["models"]

    def expires_at(self):
        limit = self.idle_limit()
        return (self.last_request_at or self.started_at) + limit if limit else None

    def delete_leftover(self, path) -> dict:
        """The Models page's files no model uses: one deleted (models.delete_leftover)."""
        out = models.delete_leftover(self.models_root(), path)
        print(f"[saguaro] {out['deleted']} deleted ({out['freed'] / 2**30:.1f} GB): no model here used it", flush=True)
        return out

    def _restart_into(self, p: Path, why: str):
        _, port = getattr(self, "listen", None) or ("127.0.0.1", None)
        note = {"config": str(p.resolve()), "port": port, "why": why, "time": time.time()}
        (self.models_root() / models.SWITCH_FILE).write_text(json.dumps(note), encoding="utf-8")
        self.model_restart = {"to": p.name, "why": why, "started": time.time()}
        threading.Thread(target=self._restart_for_models, args=(p.name, why), daemon=True).start()

    def _restart_for_models(self, name: str, why: str):
        time.sleep(0.5)                                  # (the answer to the page goes out first)
        with self.fifo:                                  # the request in flight finishes first
            print(f"[saguaro] starting again with {name} ..." if why == "switch" else
                  f"[saguaro] starting again with {name}'s new engine settings ...", flush=True)
            self.close_for_restart()
            os._exit(RESTART_EXIT)

    def storage(self) -> dict | None:
        """The model's folder and the free space on its drive (About)."""
        if not self.model_dir:
            return None
        try:
            import shutil
            u = shutil.disk_usage(self.model_dir)
            return {"path": str(self.model_dir), "free": u.free, "total": u.total}
        except OSError:
            return None

    def report(self) -> str:
        """A plain-text report for an issue (the dashboard's Copy report): versions, this PC, the engine's facts, the
        run config without its secrets, and the engine log's telling lines."""
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        hw, st = tel.get("now") or {}, tel.get("static") or {}
        gib = lambda b: f"{b / 2**30:.1f} GB" if isinstance(b, (int, float)) else "?"  # noqa: E731
        import platform
        out = [f"Saguaro {SAGUARO_VERSION or '?'} - report of {time.strftime('%Y-%m-%d %H:%M')}",
               f"System: {platform.platform()}, Python {platform.python_version()}",
               f"GPU: {st.get('gpu_name') or 'not readable'} ({gib(hw.get('gpu_mem_total'))} VRAM)",
               f"CPU: {st.get('cpu_name') or '?'}, {st.get('threads') or '?'} threads; RAM {gib(hw.get('ram_total'))}",
               f"Model: {self.model}" + (f" ({self.model_label})" if self.model_label else "") +
               f", context {self.engine.max_context}"]
        info = dict(getattr(self.engine, "info", {}) or {})
        if info:
            out.append("Engine: " + " ".join(f"{k}={v}" for k, v in sorted(info.items())))
        if self.config_path:
            try:
                cfg = json.loads(Path(self.config_path).read_text(encoding="utf-8-sig"))
                secret = re.compile(r"key|token|secret|password", re.I)
                env = {k: ("***" if secret.search(k) else v) for k, v in (cfg.get("env") or {}).items()}
                out.append(f"Config: {Path(self.config_path).name}: args {' '.join(map(str, cfg.get('args') or []))}; "
                           f"gpu {cfg.get('gpu')}; env {json.dumps(env)}")
            except (OSError, ValueError):
                pass
        log = getattr(self.engine, "log_path", None)
        if log:
            try:
                with open(log, "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - (2 << 20)))
                    lines = f.read().decode("utf-8", "replace").splitlines()
                keep = [ln for ln in lines if REPORT_LINES.search(ln)]
                out += ["", f"Engine log ({log}), the telling lines of the last 2 MB:"] + keep[-150:]
            except OSError:
                pass
        return "\n".join(out) + "\n"

    def start_telemetry(self):
        """The hardware sampler behind GET /metrics (serve/telemetry.py), recording this server's tok/s too."""
        if getattr(self, "telemetry", None) is None:
            from serve.telemetry import Telemetry
            self.telemetry = Telemetry(extra=lambda: {"tok_s": self._tok_s(), "tok_s_mean": self._tok_s_mean(),
                                                    "prefill_tok_s_mean": self._prefill_tok_s_mean(),
                                                    "prefill_tok_s": self._prefill_tok_s()},   # (the sparkline's read rate)
                                       gpu_index=int(getattr(self, "gpu_index", 0) or 0),
                                       gpu_indices=getattr(self, "gpu_indices", None),
                                       amd=getattr(self, "backend", None) == "hip",
                                       busy=lambda: bool(self.active or self.status.get("busy") or self.status.get("queued")))
            if isinstance(self.engine, StrataEngine):
                self.engine.gpu_busy = self._gpu_busy     # the stall watchdog's third sign of work
        self.start_queue_watch()

    def touch_telemetry(self) -> None:
        """A request starts: the hardware sampler returns to its once-a-second pace now (an idle server samples slower)."""
        tel = getattr(self, "telemetry", None)
        if tel is not None:
            tel.touch()

    def start_queue_watch(self):
        """A queue that has stopped moving says so, every QUEUE_WARN_S, with what holds the engine and what is waiting:
        a FIFO place that is never given back (a request waiting on an engine line that never came, a place leaked by
        an error) makes every later request wait for ever, and until this line appeared nothing said anything - the
        server looked idle, the requests piled up, and none of them expired."""
        if getattr(self, "_queue_watch", None) is not None:
            return

        def loop():
            warned = 0.0
            while True:
                time.sleep(QUEUE_TICK_S)
                try:
                    oldest, waiting = self.fifo.oldest_wait(), self.fifo.waiting()
                    low_only = self.fifo.all_low_waiting()   # (R122: a low request waiting long behind the
                    #  people is the no-aging rule, not a wedge: the WARNING is for the line a person waits in)
                except Exception:  # noqa: BLE001 - a diagnostic must never end the thread
                    continue
                if not waiting or oldest < QUEUE_WEDGED_S or low_only \
                        or time.monotonic() - warned < QUEUE_WARN_S:
                    continue
                warned = time.monotonic()
                with self.status_lock:
                    s = dict(self.status)
                    active = self.active
                doing = (f"a request is running ({s.get('phase') or 'working'}, {s.get('generated')} tokens so far)"
                         if s.get("busy") else "nothing is running")
                print(f"[saguaro] WARNING: {waiting} request(s) have waited for the engine for up to {oldest:.0f} s and "
                      f"{doing}; {self.fifo.held_by()}, {active} request(s) counted active. A request that holds the "
                      f"engine and waits on it for ever stops every request behind it - SAGUARO_REQUEST_QUIET_S / "
                      f"SAGUARO_DRAIN_S bound those waits, so this line is what a wedged FIFO would look like. "
                      f"Queue state: GET /status, /metrics (queued, active).", flush=True)

        self._queue_watch = threading.Thread(target=loop, daemon=True, name="strata-queue-watch")
        self._queue_watch.start()

    def _gpu_busy(self) -> bool:
        """Any of the engine's GPUs busier than STALL_GPU_BUSY in the telemetry's last sample (one a second)."""
        now = self.telemetry.now
        utils = [g.get("util") for g in now.get("gpus") or []] or [now.get("gpu_util")]
        return any(isinstance(u, (int, float)) and u > STALL_GPU_BUSY for u in utils)

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S seconds, summed over the requests decoding now.  0.0 while none is.
        Every reading is also the Monitor's live Decode figure, so one above the peak decode becomes the peak."""
        with self.status_lock:
            if not self.status.get("busy"):
                return 0.0
            now = time.time()
            rate = sum(r.tok_s(now) for r in self.current_requests)
            n = sum((r.samples[-1] if r.samples else r.base or (0, 0))[1] for r in self.current_requests)
            # a peak only from answers past their first PEAK_MIN_SPAN_S: a young window reads a burst as a rate
            if all(r.first_token is not None and now - r.first_token >= PEAK_MIN_SPAN_S for r in self.current_requests):
                self._note_peak(rate, n, now)
            return rate

    def _note_peak(self, tok_s, tokens, when):
        """The Monitor's peak decode (status_lock held): the highest decode rate it has shown - a running answer's live
        rate or a finished answer's - rounded as shown (0.1 tok/s), once the answers hold PEAK_MIN_TOKENS tokens."""
        tok_s = round(tok_s or 0.0, 1)
        peak = self.totals.get("peak_decode")
        if tok_s > 0 and tokens >= PEAK_MIN_TOKENS and (peak is None or tok_s > peak["tok_s"]):
            self.totals["peak_decode"] = {"tok_s": tok_s, "tokens": tokens, "time": when}

    def _tok_s_mean(self):
        """The whole-request mean since the first token (the old formula), kept so the two can be compared."""
        with self.status_lock:
            s = dict(self.status)
            r = self.latest_request if self.latest_request in self.current_requests else None
        if not s.get("busy") or r is None:
            return 0.0
        return s["generated"] / max(1e-6, time.time() - r.first_token)

    def _prefill_tok_s_mean(self):
        """Engine-reported mean over newly read tokens, excluding the cached prefix."""
        with self.status_lock:
            reading = self.status.get("busy") and self.latest_request not in self.current_requests
        return getattr(self.engine, "prefill_tok_s_mean", None) if reading else 0.0

    def _prefill_tok_s(self):
        """Engine-total prompt READ rate now, summed over every chat reading its prompt (the drivers' read hooks
        feed it, as the answer tokens feed _tok_s).  Every read is an interval: the window's count = the in-progress
        reads' steps counted inside it + each FINISHED read's tokens spread over its overlap with the window - the
        exact count over the time it really took, so three sparse-progress reads that end in the same second read as
        one engine at their true ~810 tok/s, not 1536 (live 11-20: a top-up landing at a read's end doubled the peak).
        0.0 while nothing reads a prompt; a reading above the peak becomes the peak, gated on the tokens counted
        inside RATE_WINDOW_S like decode gates on answer length."""
        with self.status_lock:
            if not self.status.get("busy"):
                return 0.0
            now = time.time()
            for st in self.running:                     # a read whose answer started is over: close its interval
                if st.first_token is not None:          # (from now on its tokens spread across the read itself)
                    self._read_close(st, st.first_token)
            readers = [r for r in self.running if r.at_engine and r.first_token is None]
            if not readers:
                return 0.0
            w0 = now - RATE_WINDOW_S
            n, t_lo = 0.0, None
            for r in self.reads:                        # finished reads: their tokens over their overlap with it
                t_start, t_end, tokens = r[1], r[2], r[3]
                if t_end is None or not tokens:
                    continue
                lo = max(t_start, w0)
                if t_end < lo:
                    continue
                n += tokens if t_end == t_start else tokens * (t_end - lo) / (t_end - t_start)
                t_lo = lo if t_lo is None else min(t_lo, lo)
            for st in readers:                          # reads still running: the steps counted inside the window
                e = st.read_cur
                if e is None or e[2] is not None or e[1] > now or not e[4] or e[4][-1][0] <= w0:
                    continue
                lo_n, lo_t, first_in = 0, e[1], None
                for te, ne in e[4]:
                    if te > w0:
                        first_in = (te, ne)
                        break
                    lo_n, lo_t = ne, te
                if first_in is None:
                    continue
                # the step that straddles the window's start counts only its share inside it (steps are spread
                # evenly over their own time, like a finished read over its span)
                te1, ne1 = first_in
                part = (ne1 - lo_n) * ((te1 - max(lo_t, w0)) / (te1 - lo_t)) if te1 > lo_t else (ne1 - lo_n)
                c = (e[4][-1][1] - ne1) + part
                if c <= 0:
                    continue
                n += c
                lo = max(lo_t, w0)
                t_lo = lo if t_lo is None else min(t_lo, lo)
            if not n or t_lo is None:                   # (a read paused past the window: it drained empty -
                return 0.0                              #  nothing read lately)
            span = now - t_lo
            total = n / span if span >= RATE_MIN_SPAN_S else 0.0
            if span >= PEAK_MIN_SPAN_S:
                self._note_peak_read(total, int(round(n)), now)
            return total

    def _read_open(self, stat, now):
        """One read of one request begins: its interval joins the list the window spreads reads over."""
        stat.read_cur = [stat, now, None, None, collections.deque(maxlen=4096), stat.read_pos]   # (stat, t_start,
        self.reads.append(stat.read_cur)                        #  t_end, tokens, steps (time, counted tokens), and
        self._read_prune(now)                                   #  the count the read started from (a leg's prefix)
        return stat.read_cur

    def _read_close(self, stat, at=None):
        """(status_lock held) the read is over - its answer's first token, a retry, or the request ending: the whole
        count stands with its span for the window to spread it over."""
        r = getattr(stat, "read_cur", None)
        if r is None or r[2] is not None:
            return
        r[3] = max(0, stat.read_pos - r[5])             # (what THIS read counted, not the absolute position: a
        r[2] = max(r[1], (time.time() if at is None else at) - 0.0)   #  slice's next leg starts from the held prefix)
        self._read_prune(time.time())

    def _read_exact(self, stat, tokens):
        """(status_lock held) the engine's exact new tokens for this request's read stand as its count, spread over
        the read's real span (the engine taking it -> its first token) - progress steps only ever previewed it."""
        if not tokens or stat.engine_t0 is None or stat.first_token is None or stat.first_token <= stat.engine_t0:
            return
        mine = [r for r in self.reads if r[0] is stat]
        r = mine[-1] if mine else None
        if r is None:
            return
        r[1], r[2], r[3] = stat.engine_t0, stat.first_token, tokens

    def _read_prune(self, now):
        """(status_lock held) closed reads no window can reach anymore leave the list; an open one always stays."""
        while self.reads and self.reads[0][2] is not None and self.reads[0][2] < now - 2 * RATE_WINDOW_S:
            self.reads.popleft()

    def _own_read_rate(self, st, now, n_own) -> float:
        """One chat's live read rate (PLAN-qfn-per-request-read-stats): once the engine's progress lines carry
        its own figures, its new_read over its own read_ms - the reader's wall time no longer mixes in what the
        chat spent queued or held at a piece boundary.  Without those fields (an older engine): its counted
        tokens over the time since the engine took it, as before."""
        if st.read_eng_ms and st.read_eng_ms > 0 and st.read_eng_n:
            return st.read_eng_n / (st.read_eng_ms / 1000.0)
        return n_own / (now - st.engine_t0) if n_own and st.engine_t0 and now > st.engine_t0 else 0.0

    def _read_counted(self, stat) -> int:
        """The prompt tokens this request's reads have counted (closed intervals plus an open one's steps) - what
        the Monitor's figures and the tests read back."""
        n = sum(r[3] or 0 for r in self.reads if r[0] is stat and r[2] is not None)
        r = getattr(stat, "read_cur", None)
        return n + (max(0, stat.read_pos - r[5]) if r is not None and r[2] is None else 0)

    def _read_start(self, stat):
        """(a driver's hook) the engine took this request - a QFN admission after its wait for a slot says so too:
        its read's interval starts HERE (a queue wait inside the turn is no read time), the previous leg closes."""
        now = time.time()
        with self.status_lock:
            self._read_close(stat, now)
            stat.engine_t0 = now
            self._read_open(stat, now)

    def _note_read_safe(self, *a):
        """_note_read for the drivers' read callbacks: a Monitor figure must never fail the request it watches."""
        try:
            self._note_read(*a)
        except Exception as e:                          # (logged once a kind, the answer goes on)
            if type(e).__name__ not in getattr(self, "_note_read_errs", set()):
                self._note_read_errs = getattr(self, "_note_read_errs", set()) | {type(e).__name__}
                print(f"[saguaro] read-rate bookkeeping failed ({type(e).__name__}: {e}); the request goes on", flush=True)

    def _note_hb_safe(self, *a):
        """_note_hb for the drivers' heartbeat hook: a Monitor figure must never fail the request it watches."""
        try:
            self._note_hb(*a)
        except Exception as e:                          # (logged once a kind, the answer goes on)
            if type(e).__name__ not in getattr(self, "_note_hb_errs", set()):
                self._note_hb_errs = getattr(self, "_note_hb_errs", set()) | {type(e).__name__}
                print(f"[saguaro] read-heartbeat bookkeeping failed ({type(e).__name__}: {e}); the request goes on",
                      flush=True)

    def _note_hb(self, stat, hb):
        """(a driver's hook, PLAN-u5-read-heartbeat) the engine's own about-once-a-second report of a read in
        flight: its stage (KV staging from RAM, or new tokens being read), done / new (new tokens done of the
        total), reused, and its rate over the last heartbeat interval.  It refreshes the chat's live figures
        every second - a one-chunk read, a long first chunk or a read mostly busy staging its K/V from RAM no
        longer shows nothing until it ends - and feeds the counted tokens in as a normal read step, so the
        engine-total window moves per second too.  No step of its own while it stages (done 0: nothing to count)."""
        stat.read_hb = dict(hb, at=time.time())
        n = hb.get("done") or 0
        if n > 0:
            self._note_read(stat, n, hb.get("rate"), False, False, hb.get("ms"))

    def _note_read(self, stat, n_read, tok_s=None, exact=False, prov=False, read_ms=None):
        """A read step of one request: the drivers' PP lines and the batch slots' read counters call this (through
        the hook run() installs on the engine's thread-local) with the prompt tokens IT has read now - the reused
        prefix left out.  It counts into THIS request's read interval (the engine's on_read_start, or its first
        step, opens one) - which is how the engine-total window spreads it - and the request's own read span.
        exact=True: the engine's own final figures for this read (its BRA line): they stand as THE count, topping the
        steps' suggestion up or correcting it, negative included - and they END the read: its interval closes here,
        so its tokens spread over the read itself instead of piling into the window it finished in.
        prov=True: a slot admission's BRP step: its progress may count from the prompt's start, so it only gives
        the live figure (the engine's own tok/s) and the read's wall span - it never counts tokens or a peak.
        read_ms (PLAN-qfn-per-request-read-stats): the engine's OWN milliseconds this request's read has run
        (PP/BRP read_ms=, RDONE / BRA ms=) beside its own counted tokens - when it comes, the request's rate IS
        new_read / read_ms: no shared reader's clock, no wall time since the engine took it."""
        now = time.time()
        if read_ms is not None and not prov:            # (a prov step is a position, not a counted read: no pair)
            if exact:
                prev = stat.read_engine
                # a leg of one request adds to it (BRA's read_ms is its own leg); a restarted read replaces it
                stat.read_engine = (n_read, prev[1] + read_ms) if prev and n_read >= prev[0] else (n_read, read_ms)
                stat.read_eng_n, stat.read_eng_ms = n_read, stat.read_engine[1]
            else:
                # the request's own pair: its tokens so far over the time its read has run - the finished legs'
                # clock included, the held time never (read_ms is only what ran)
                base_ms = stat.read_engine[1] if stat.read_engine and n_read >= stat.read_engine[0] else 0.0
                stat.read_eng_n, stat.read_eng_ms = n_read, base_ms + read_ms
        with self.status_lock:
            r = stat.read_cur
            if r is not None and r[2] is not None:      # (this read's interval closed with the answer: a new one)
                r = None
            if prov:
                if r is None:
                    r = self._read_open(stat, now)
                if stat.read_t0 is None:
                    stat.read_t0 = now
                stat.read_t1 = now
                if tok_s is not None:
                    stat.read_avg = tok_s
                    stat.read_prov = True
                return
            if exact:
                if r is None:
                    r = self._read_open(stat, now)
                stat.read_prov = False                  # (the engine's own count and clock from here)
                if sum(1 for x in self.running if x.at_engine and x.first_token is None) > 1:
                    stat.read_shared = True             # (two admissions read at once: label the request's own rate)
                if tok_s is not None:
                    stat.read_avg = tok_s
                    self._note_peak_read(tok_s, n_read, now)
                if stat.read_t0 is None:
                    stat.read_t0 = now
                stat.read_t1 = now
                d = n_read - stat.read_pos
                if d:
                    stat.read_pos = n_read
                    r[4].append((now, n_read))
                self._read_close(stat, now)             # (the engine's BRA ENDS this read: from here its tokens
                return                                  #  spread over the read itself, never at its end)
            if r is None:                               # the first step of a read the driver announced no start for
                r = self._read_open(stat, now)
            elif n_read < stat.read_pos:                # a new read of this request's prompt (a retry, a slice's leg)
                self._read_close(stat, now)
                stat.read_pos, stat.read_avg, stat.read_last, stat.read_prev_t = 0, None, None, None
                stat.read_eng_n, stat.read_eng_ms = 0, None   # (the engine's live pair restarts with the read)
                self._read_open(stat, now)              # (counts from nothing again; the engine's counter reset)
            d = n_read - stat.read_pos
            stat.read_pos = n_read
            if stat.read_t0 is None:
                stat.read_t0 = now
            stat.read_t1 = now
            if stat.read_prev_t is not None and now > stat.read_prev_t and stat.read_prev_n <= n_read:
                stat.read_last = (n_read - stat.read_prev_n) / (now - stat.read_prev_t)   # (its last interval's rate)
            stat.read_prev_t, stat.read_prev_n = now, n_read
            if tok_s is not None:                       # the engine's own mean over THIS request's read (its PP line)
                stat.read_avg = tok_s
                self._note_peak_read(tok_s, n_read, now)
            if d <= 0:
                return
            if sum(1 for x in self.running if x.at_engine and x.first_token is None) > 1:
                stat.read_shared = True                 # two chats read at once: neither's engine clock is just its own
            stat.read_cur[4].append((now, n_read))

    def _note_peak_read(self, tok_s, tokens, when):
        """The Monitor's peak Prefill (status_lock held): the highest read rate it has shown - a live window or a
        finished read's - rounded as shown, once a read holds PEAK_MIN_READ_TOKENS tokens (a small read's rate says
        little).  Kept like peak_decode."""
        tok_s = round(tok_s or 0.0, 1)
        peak = self.totals.get("peak_prefill")
        if tok_s > 0 and tokens >= PEAK_MIN_READ_TOKENS and (peak is None or tok_s > peak["tok_s"]):
            self.totals["peak_prefill"] = {"tok_s": tok_s, "tokens": tokens, "time": when}

    def _wall_tick(self):
        """The totals' wall clocks (status_lock held): the seconds during which at least one prompt was being read,
        and those during which at least one answer was being written.  The requests' summed prompt_ms / decode_ms add
        the same seconds once per chat; the totals' "N tokens at X tok/s" divides by these, so it stands under overlap.
        Called wherever a request takes or gives back an engine place, answers its first token, or ends."""
        now = time.time()
        reading = any(r.at_engine and r.first_token is None for r in self.running)
        writing = any(r.at_engine and r.first_token is not None for r in self.running)
        if reading and self.read_wall_since is None:
            self.read_wall_since = now
        elif not reading and self.read_wall_since is not None:
            self.read_wall_s += now - self.read_wall_since
            self.read_wall_since = None
        if writing and self.decode_wall_since is None:
            self.decode_wall_since = now
        elif not writing and self.decode_wall_since is not None:
            self.decode_wall_s += now - self.decode_wall_since
            self.decode_wall_since = None

    def _wall_seconds(self):
        """(status_lock held) (read_s, decode_s) to now - an open interval counts as it goes (nothing to close)."""
        now = time.time()
        read_s = self.read_wall_s + (now - self.read_wall_since if self.read_wall_since is not None else 0.0)
        decode_s = self.decode_wall_s + (now - self.decode_wall_since if self.decode_wall_since is not None else 0.0)
        return round(read_s, 3), round(decode_s, 3)

    def _chats_list(self, now):
        """The Monitor's "Right now": one row per request queued or with the engine, built from the per-request
        state (not the single shared status), both drivers - ids stable for the request's life.  Each row: its
        own read/write progress and rates, and - when the engine reports kv_bytes_token - the cache its context
        costs now, split VRAM / RAM where the cache streams (kv_resident positions in VRAM, the rest pinned RAM)."""
        info = getattr(self.engine, "info", None) or {}
        kbt = info.get("kv_bytes_token")
        kbt = float(kbt) if isinstance(kbt, (int, float)) and not isinstance(kbt, bool) and kbt > 0 else None
        win = info.get("kv_resident") if isinstance(info.get("kv_resident"), (int, float)) else 0
        stream = bool(kbt) and 0 < win < self.engine.max_context
        rows = []
        for st in sorted(self.waiting, key=lambda r: r.queued_at or 0.0):
            rows.append({"id": st.id, "cid": getattr(st, "cid", None), "label": st.api, "conv": st.conv.id if st.conv else None,
                         "state": "queued", "phase": None,
                         "priority": st.priority, "paused_for_priority": st.paused_for_priority,   # (R122)
                         "prompt_tokens": st.prompt_tokens or None, "prompt_read": None, "prompt_total": None,
                         "generated": 0, "max_tokens": st.max_tokens, "tok_s": None, "prefill_tok_s": None,
                         "turbo": dict(st.turbo) if st.turbo else None,
                         "prefill_shared": False, "context_used": st.prompt_tokens or 0,
                         "elapsed_s": round(now - st.queued_at, 1) if st.queued_at else None})
        for st in sorted(self.running, key=lambda r: r.started or 0.0):
            v = st.view(now)
            state, ph = v["state"], v["phase"] or ""
            if state == "writing":                # (the same phase detection the single live.phase uses, per request)
                state = "thinking" if ph == "thinking" else \
                    "tool" if ph.startswith("writing a tool call") or ph == "tool call complete" else "writing"
            # its own read: the engine's new_read / read_ms when its lines carry them (per-request read stats),
            # else the tokens counted for THIS chat over the time since the engine took it
            n_own = self._read_counted(st) if state == "reading" else 0
            rate = self._own_read_rate(st, now, n_own)
            entry = dict(v, id=st.id, label=st.api, conv=st.conv.id if st.conv else None, state=state,
                         prefill_tok_s=round(max(0.0, rate), 1) if state == "reading" and rate else None)
            hb = getattr(st, "read_hb", None)          # U5 read heartbeat: the engine's ~1-per-second report of
            if state == "reading" and hb and now - hb.get("at", 0.0) <= 3.0:   #  the read in flight (a stale one
                entry["read_stage"] = hb.get("stage")  #  is none: the engine stopped reporting, or went quiet)
                entry["read_done"], entry["read_new"] = hb.get("done"), hb.get("new")
                entry["read_reused"] = hb.get("reused")
            used = (st.prompt_tokens or 0) + (st.generated or 0)
            entry["context_used"] = used
            if kbt:
                entry["cache_vram_bytes"] = int(min(used, win) * kbt) if stream else int(used * kbt)
                entry["cache_ram_bytes"] = int(max(0, used - win) * kbt) if stream else 0
            rows.append(entry)
        return rows

    def _conv_prune(self, now):
        """(status_lock held) drop conversations idle past conv_idle_s, and the least recently used ones while the
        table is over conv_max - a turn in the engine always keeps its row."""
        for k in [k for k, c in self.convs.items() if c.stat is None and now - c.last_active > self.conv_idle_s]:
            del self.convs[k]
        while len(self.convs) > self.conv_max:
            gone = next((k for k, c in self.convs.items() if c.stat is None), None)
            if gone is None:
                break
            del self.convs[gone]

    def _conv_open(self, ids, api, words, stat, gen_tail=0) -> Conversation:
        """(status_lock held) the row this request continues: the LONGEST conversation whose last prompt - without
        the open reply's generation tail - this prompt starts with (the engine's prefix reuse works the same way:
        it never reaches past where two prompts diverge); none -> a row of its own (the table is an LRU, idle rows
        expire).  A fork: the branch that ran first advanced that row's last prompt, so the second no longer starts
        with it and takes its own row.  The turn count goes up the moment the request joins a row, so the row says
        "turn 14" while the fourteenth turn is still being written."""
        now = time.time()
        n = max(0, len(ids) - gen_tail)
        best = None
        for c in self.convs.values():                   # (the LRU holds <= CONV_MAX candidates; longest match wins)
            if 0 < c.cont_len <= n and c.cont_hash == conv_hist_hash(ids[:c.cont_len]):
                if best is None or c.cont_len > best.cont_len:
                    best = c
        if best is not None:
            self.convs.move_to_end(best.key)
            if words and not best.words:
                best.words = words
            best.api, best.stat, best.last_active = api or best.api, stat, now
            best.turns += 1
            self._conv_prune(now)
            return best
        c = Conversation(next(self._conv_ids), api, words or "")
        self.convs[c.key] = c
        c.stat, c.last_active, c.turns = stat, now, 1
        self._conv_prune(now)
        return c

    def _conv_idle(self, c: Conversation, stat=None, ids=None, gen_tail=0, prompt_n=0, out_n=0, reused=None,
                   read_tok_s=None, decode_tok_s=None):
        """(status_lock held) a conversation's turn ended: its row loses the request and waits idle with the turn's
        figures, its key - this turn's prompt minus the generation tail - recorded so the turn that resumes this
        conversation matches it.  A turn that forks from it (an edited retry: its prompt no longer starts with that
        one, and the branch that ran first advanced it) continues nothing and takes a row of its own.  When a later
        turn of the same conversation is already under way (the row is not this request's any more), only the
        counts are added."""
        now = time.time()
        mine = c.stat is stat
        if reused is not None:
            c.tokens_read += max(0, prompt_n - reused)
            c.prompt_last, c.out_last, c.ctx = prompt_n, out_n, prompt_n + out_n
        c.tokens_written += out_n
        if read_tok_s is not None:
            c.last_read_tok_s = read_tok_s
        if decode_tok_s is not None:
            c.last_decode_tok_s = decode_tok_s
        if mine and ids:
            n = max(0, len(ids) - gen_tail)
            c.cont_len, c.cont_hash = n, conv_hist_hash(list(ids[:n]))   # (the prompt the next turn must continue:
            self.convs.move_to_end(c.key)                                #  longer every turn of this conversation)
        if mine:
            c.stat, c.last_active = None, now
        self._conv_prune(now)

    def _convs_list(self, now) -> list:
        """(status_lock held) /metrics' live.conversations: one entry per conversation - the ones with a turn queued
        or in the engine first (their `turn` holds the per-request fields a row showed today), the idle ones after,
        most recently used first.  The cache split is the "Right now" rows' (kv_bytes_token, VRAM window)."""
        info = getattr(self.engine, "info", None) or {}
        kbt = info.get("kv_bytes_token")
        kbt = float(kbt) if isinstance(kbt, (int, float)) and not isinstance(kbt, bool) and kbt > 0 else None
        win = info.get("kv_resident") if isinstance(info.get("kv_resident"), (int, float)) else 0
        stream = bool(kbt) and 0 < win < self.engine.max_context
        self._conv_prune(now)
        live_rows, idle_rows = [], []
        for c in reversed(list(self.convs.values())):         # (most recently used first)
            e = {"id": c.id, "api": c.api, "words": c.words, "turns": c.turns,
                 "read_tok_s": c.last_read_tok_s, "decode_tok_s": c.last_decode_tok_s,
                 "tokens_read": c.tokens_read, "tokens_written": c.tokens_written}
            st = c.stat
            if st is not None:
                v = st.view(now)
                state, ph = v["state"], v["phase"] or ""
                if state == "writing":
                    state = "thinking" if ph == "thinking" else \
                        "tool" if ph.startswith("writing a tool call") or ph == "tool call complete" else "writing"
                # its own read: the engine's new_read / read_ms when its lines carry them (per-request read
                # stats), else its counted tokens over the time since the engine took it
                n_own = self._read_counted(st) if state == "reading" else 0
                rate = self._own_read_rate(st, now, n_own)
                e["state"] = state
                e["turn"] = dict(v, state=state, prefill_tok_s=(round(max(0.0, rate), 1)
                                     if state == "reading" and rate else None))
                used = (st.prompt_tokens or 0) + (st.generated or 0)
                live_rows.append((-(v["elapsed_s"] or 0), e, used))
            else:
                e["state"], e["idle_s"] = "idle", round(max(0.0, now - c.last_active), 1)
                e["prompt_last"], e["out_last"] = c.prompt_last, c.out_last
                idle_rows.append((e, c.ctx))
        live_rows.sort(key=lambda t: t[0])
        out = [(e, used) for _, e, used in live_rows] + idle_rows
        rows = []
        for e, used in out:
            e["context_used"] = used
            if kbt:
                e["cache_vram_bytes"] = int(min(used, win) * kbt) if stream else int(used * kbt)
                e["cache_ram_bytes"] = int(max(0, used - win) * kbt) if stream else 0
            rows.append(e)
        return rows

    def _engine_context(self, convs: list) -> dict | None:
        """The Monitor's Context panel as the ENGINE's context (PLAN-context-panel): every chat in a slot - active or
        idle-in-slot, the "Right now" and idle conversation rows - summed over the positions the engine can hold at
        once.  Used splits into VRAM (each chat's positions inside its KV window), RAM (its positions streamed to
        pinned RAM) and PARKED (conversations the engine parked in RAM outside its slots - its VRAM lines' `parked=`,
        their positions only as its own bytes per position says); a conversation saved on disk is in neither.
        Capacity is the BINDING limit: the engine's positions (max_context x its chat slots: QFN's --batch 8, MiniMax's
        --parallel 2, ...) or its live K/V budgets - (kv_room + kv_ram_room) / kv_bytes_token, its re-reported VRAM
        lines' fields; an engine without RAM streaming has no kv_ram_room and is VRAM only - whichever is smaller.
        None when the engine prints no VRAM lines: the page keeps its old per-request headline then."""
        eng = self.engine
        rows = list(getattr(eng, "vram", None) or [])
        if not rows:
            return None
        info = getattr(eng, "info", None) or {}
        kbt = info.get("kv_bytes_token")
        kbt = float(kbt) if isinstance(kbt, (int, float)) and not isinstance(kbt, bool) and kbt > 0 else None
        merged = gpu_map.merge(rows)                       # (two split parts on one card add up)
        kv_room = sum(int(r.get("kv_room") or 0) for r in merged)
        kv_ram_room = max((int(r.get("kv_ram_room") or 0) for r in rows), default=0)
        parked = max((int(r.get("parked") or 0) for r in rows), default=0)
        # PLAN-qfn-park-budget (QFN): how many conversations are parked and what they may hold now; absent elsewhere
        parked_n = max((int(r.get("parked_n") or 0) for r in rows), default=0) \
            if any("parked_n" in r for r in rows) else None
        park_budget = max((int(r.get("park_budget") or 0) for r in rows), default=0) \
            if any("park_budget" in r for r in rows) else None
        max_ctx = int(eng.max_context or 0)
        slots = max(int(getattr(eng, "multi", 0) or 0), int(getattr(eng, "batch", 0) or 0), 1)
        pos_cap = max_ctx * slots
        mem_cap = int((kv_room + kv_ram_room) / kbt) if kbt and (kv_room or kv_ram_room) else None
        if mem_cap is not None and mem_cap < pos_cap:      # what the engine's live budgets hold binds
            kind, cap = "memory", mem_cap
            which = "VRAM+RAM" if kv_room and kv_ram_room else ("RAM" if kv_ram_room else "VRAM")
        else:                                              # 8 x 1M positions bind (or no budget figures to compare)
            kind, which, cap = "positional", "", pos_cap
        win = info.get("kv_resident") if isinstance(info.get("kv_resident"), (int, float)) else 0
        stream = 0 < win < max_ctx                         # (the "Right now" rows' split, the same rule)
        used = vram_pos = ram_pos = 0
        for e in convs:
            n = int(e.get("context_used") or 0)
            used += n
            if stream:
                vram_pos += min(n, int(win))
                ram_pos += max(0, n - int(win))
            else:
                vram_pos += n
        parked_tok = int(parked / kbt) if kbt and parked else 0   # (its positions, as its own bytes say: the bar's PARKED)
        return {"capacity": {"tokens": cap, "kind": kind, "which": which, "slots": slots,
                             "max_context": max_ctx, "positional": pos_cap, "memory": mem_cap,
                             "kv_room": kv_room, "kv_ram_room": kv_ram_room},
                "used": {"tokens": used, "vram_tokens": vram_pos, "ram_tokens": ram_pos,
                         **({"vram_bytes": int(vram_pos * kbt), "ram_bytes": int(ram_pos * kbt)} if kbt else {})},
                "parked": {"bytes": parked, **({"tokens": parked_tok} if kbt and parked else {}),
                           **({"conversations": parked_n} if parked_n is not None else {}),
                           **({"budget_bytes": park_budget} if park_budget is not None else {})},
                "free": max(0, cap - used - parked_tok),
                # (QFN turn saves: each place's last saved position and time - main, slotN)
                **({"turn_saves": dict(getattr(eng, "turn_saves", None) or {})} if getattr(eng, "turn_saves", None) else {})}

    def metrics(self, all_requests=False) -> dict:
        """GET /metrics: what the Monitor tab shows - the engine's facts, what it is doing, the last requests, and
        the hardware (with a minute of history per series)."""
        rate = self._tok_s()                            # (first: a live rate above the peak is the peak in `totals`)
        pre_rate = self._prefill_tok_s()                # (the same for the read: 0.0 while nothing reads)
        with self.status_lock:
            s = dict(self.status)
            reading_n = sum(1 for r in self.running if r.at_engine and r.first_token is None)
            read_s, decode_s = self._wall_seconds()     # (the totals line divides by these, not by summed prompt_ms)
            hist = list(self.history)
            totals = dict(self.totals, read_s=read_s, decode_s=decode_s)
            guard = dict(self.loop_guard_counts, on=bool(self.loop_guard))   # (what the loop guard did since start)
            active = self.active
            writing = len(self.current_requests)        # (of the active ones, those past their prompt)
            spec_off = bool(self.current_requests) and all(r.spec_off for r in self.current_requests)
            prios = {r.priority for r in self.running}  # (R122: the live state says at which priority it runs,
            priority = prios.pop() if len(self.running) == 1 else None   #  and the Monitor tags high/low)
            paused = any(r.paused_for_priority for r in self.running)
            now = time.time()
            convs = self._convs_list(now)               # (one entry per conversation: the Monitor's Conversations)
            ended = list(self.ended)
            chats = self._chats_list(now)               # (every chat queued or with the engine, its own figures)
        progress = getattr(self.engine, "progress", None)
        if s.get("busy") and reading_n:
            state = "reading"                           # (aggregate for older code: any chat reading is reading)
        elif s.get("busy"):
            state = "generating"
        else:
            state = "unloaded" if self.unloaded else "idle"
        live = {"state": state, "queued": s.get("queued", 0), "active": active,   # (answering at once: --parallel)
                "writing": writing, "spec_off": spec_off,   # (the request(s) running asked for no drafting)
                "priority": priority, "paused_for_priority": paused,   # (R122: whose turn this is, and whether an
                #  answer is sitting out for one above it - the chats rows carry each request's own)
                "phase": s.get("phase") if s.get("busy") else None,
                "prompt_tokens": s.get("prompt_tokens") if s.get("busy") else None,
                "prompt_read": None, "prompt_total": None, "generated": s.get("generated") if s.get("busy") else None,
                "max_tokens": s.get("max_tokens") if s.get("busy") else None,
                "elapsed_s": round(now - s["started"], 1) if s.get("busy") and s.get("started") else None,
                "tok_s": round(rate, 1) if s.get("busy") and writing else None,
                "tok_s_mean": round(self._tok_s_mean(), 1) if state == "generating" else None,
                "prefill_tok_s_mean": getattr(self.engine, "prefill_tok_s_mean", None) if s.get("busy") else None,
                # the engine-total read rate (every chat reading, both drivers) and how many are: the Monitor's live
                # Prefill figure and its label - the mean above stays what one chat's engine reported (as before)
                "prefill_tok_s": round(pre_rate, 1) if s.get("busy") else None,
                "reading_n": reading_n,
                "chats": chats,                         # (every chat queued or with the engine: the "Right now" rows)
                "tok_s_window_s": RATE_WINDOW_S if s.get("busy") and writing else None}
        progress = getattr(self.engine, "progress", None)
        if state == "reading":
            rd = [c for c in chats if c["state"] == "reading" and c.get("prompt_total")]
            if rd:
                live["prompt_read"], live["prompt_total"] = rd[-1]["prompt_read"], rd[-1]["prompt_total"]
            elif reading_n == 1 and progress:           # (its PP line has not reached the view yet: the engine's own)
                live["prompt_read"], live["prompt_total"] = progress[0], progress[1]
        # U5 read heartbeat (PLAN-u5-read-heartbeat): what the reads in flight report about once a second -
        # stage (staging the reused prefix from RAM, or reading new tokens), done/new (new tokens done of the
        # total) and reused.  With most of a prompt reused the Monitor shows "restoring N cached · reading M
        # new" (and the stage) by these instead of a bare 0 rate while nothing has been read yet.
        live["read_states"] = [{"stage": c["read_stage"], "done": c.get("read_done"), "new": c.get("read_new"),
                                "reused": c.get("read_reused")}
                               for c in chats if c["state"] == "reading" and c.get("read_stage")]
        # several conversations at once: each running one's own figures (Conversations), and the counts summed over them
        live["conversations"], live["ended"] = convs, ended
        running_turns = [c["turn"] for c in convs if c.get("turn")]
        if len(running_turns) > 1:
            live["generated"] = sum(t["generated"] for t in running_turns)
        # the prompt the conversation cache already held (the Monitor's Context bar), from the engine's PP lines
        live["prompt_reused"] = progress[2] if s.get("busy") and progress and len(progress) > 2 else None
        engine = {"model": self.model, "model_label": self.model_label, "max_context": self.engine.max_context,
                  "model_name_pretty": self.model_name_pretty, "model_quant": self.model_quant,
                  "images": self.vision is not None,
                  **dict(getattr(self.engine, "info", {}) or {}), "saguaro_version": SAGUARO_VERSION,
                  "trained_context": self.trained_context,
                  **({"dspark": self.dspark} if self.dspark is not None else {})}
        # the drafters on (empty: drafting is off): the Monitor's Speculation says "off", or the draft acceptance
        engine["speculation"] = speculation_modes(engine, lookup=self.lookup)
        with self.status_lock:
            reload = dict(self.reload) if self.reload else None
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "history": {}, "static": {}}
        # the GLM engine's expert tiers (STAT lines): the latest reading and the recent series for the sparklines
        tiers = None
        stat = getattr(self.engine, "stat", None)
        if stat:
            sh = list(getattr(self.engine, "stat_history", []) or [])
            tiers = {"now": stat, "history": {k: [s.get(k) for s in sh] for k in
                                              ("tok_s", "ms_tok", "vram_hit", "ram_fetch", "disk", "promo",
                                               "cpu", "ram", "disk_mb", "engram_ms")}}   # (the last four: DeepSeek's)
        hw = dict(tel["now"])
        cards = hw.pop("vram", None)                      # (each card's processes: for the GPU map only)
        return {"engine": engine, "live": live, "tiers": tiers, "gpu_map": self._gpu_map(cards),
                "context": self._engine_context(convs),   # (the engine's Context panel: PLAN-context-panel)
                "requests": hist[::-1][:None if all_requests else 12],
                "requests_kept": len(hist), "totals": totals, "loop_guard": guard, "hardware": hw,
                "hardware_static":
                tel["static"], "history": tel["history"], "time": now, "reload": reload, "storage": self.storage(),
                "savings": self.savings.summary() if self.savings is not None else None}

    def _gpu_map(self, cards):
        """What fills each GPU (serve/gpu_map.py): the engine's VRAM lines and NVML's cards, with the engine's and the
        vision encoder's processes told apart from other programs; None for an engine without VRAM lines."""
        report = getattr(self.engine, "vram", None)
        if not report:
            return None
        proc = getattr(self.engine, "proc", None)
        v = self.vision
        vproc = getattr(v, "proc", None) if v is not None else None
        lent = v is not None and getattr(v, "lend", None) is not None and (vproc is not None or getattr(v, "owed", False))
        listed = {pid for c in cards or [] for pid in c.get("procs") or {}}
        return gpu_map.view(report, cards, nvml_pids(proc.pid if proc is not None else None, listed),
                            nvml_pids(vproc.pid if vproc is not None else None, listed), lent, now=time.monotonic())

    # the GLM engine's live STAT values (StrataEngine._parse_stat) exported as gauges: key -> help
    PROM_STAT = {"tok_s": "Decode rate over the running request, tokens/s",
                 "draft_rounds": "DSpark draft calls in the current decode interval (all requests with parallel serving)",
                 "draft_ms": "DSpark drafter measured host time in the current decode interval, ms",
                 "verify_ms": "DSpark verification forward measured host time in the current decode interval, ms",
                 "vram_hit": "Share of the routed experts found in VRAM (0..1)",
                 "ram_fetch": "Experts a token read from the RAM tier",
                 "disk": "Experts a token read from the SSD",
                 "promo": "Experts a token promoted into VRAM",
                 "cpu": "Experts a token computed on the CPU lane (DeepSeek)",
                 "disk_mb": "MB a token the storage read (DeepSeek, Linux)",
                 "engram_ms": "The n-gram memory's (Engram) read time a token, ms (DeepSeek)",
                 "eg_cache_rows": "Engram rows held in the host cache",
                 "eg_cache_cap": "Engram host cache capacity, rows",
                 "eg_cache_row_bytes": "Engram host cache payload bytes per row",
                 "eg_lookup_rows": "Consumer Engram row lookups since engine start",
                 "eg_lookup_hits": "Consumer Engram row lookups served without a file read since engine start",
                 "eg_lookup_reads": "Consumer Engram on-demand file reads since engine start (prefetch reads are separate)",
                 "slot_ram": "Parked DeepSeek conversations held in RAM",
                 "slot_disk": "Parked DeepSeek conversations held in the slot folder",
                 "slot_ram_bytes": "Parked DeepSeek conversation payload bytes in RAM",
                 "slot_ram_cap": "Parked DeepSeek conversation RAM budget, bytes; 0 means count-limited only",
                 "vram_used": "Expert slots in use in VRAM", "vram_slots": "Expert slots in VRAM",
                 "ram_used": "Expert slots in use in RAM", "ram_slots": "Expert slots in RAM"}
    PROM_STAT_COUNTERS = {"eg_lookup_rows", "eg_lookup_hits", "eg_lookup_reads"}

    def prometheus(self) -> str:
        """GET /metrics/prometheus: the totals and what is running in Prometheus' text format, under llama-server's
        metric names (llamacpp:*) so its dashboards work unchanged; the requests and the engine's expert tiers as
        maya:*."""
        with self.status_lock:
            s, t, active = dict(self.status), dict(self.totals), self.active
            guard = dict(self.loop_guard_counts)
        stat = dict(getattr(self.engine, "stat", None) or {})
        prompt_n = t["prompt_tokens"] - t["reused"]               # what was read, as llama.cpp counts it
        prompt_s, decode_s = t["prompt_ms"] / 1000, t["decode_ms"] / 1000
        rows = [
            ("llamacpp:prompt_tokens_total", "counter", "Prompt tokens read", prompt_n),
            ("llamacpp:prompt_tokens_cached_total", "counter", "Prompt tokens the kept conversation already held",
             t["reused"]),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt read time", prompt_s),
            ("llamacpp:tokens_predicted_total", "counter", "Tokens generated", t["output_tokens"]),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Generation time", decode_s),
            ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt read throughput in tokens/s",
             prompt_n / prompt_s if prompt_s else 0),
            ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s",
             t["output_tokens"] / decode_s if decode_s else 0),
            ("llamacpp:requests_processing", "gauge", "Requests running", max(active, int(bool(s.get("busy"))))),
            ("llamacpp:requests_deferred", "gauge", "Requests waiting their turn", s.get("queued") or 0),
            ("maya:requests_total", "counter", "Requests finished", t["requests"]),
            ("maya:decode_peak_tokens_seconds", "gauge",
             f"The fastest answer's generation throughput in tokens/s ({PEAK_MIN_TOKENS} tokens or more)",
             (t.get("peak_decode") or {}).get("tok_s") or 0),
            ("maya:prefill_peak_tokens_seconds", "gauge",
             f"The fastest prompt read throughput in tokens/s ({PEAK_MIN_READ_TOKENS} tokens read or more)",
             (t.get("peak_prefill") or {}).get("tok_s") or 0),
            ("maya:engine_alive", "gauge", "1 while the engine process runs",
             int(self.engine.alive()) if hasattr(self.engine, "alive") else 1),
            ("maya:loop_guard_detected_total", "counter", "Loops the loop guard saw in answers", guard["detected"]),
            ("maya:loop_guard_penalised_total", "counter",
             "Loops broken by dropping the repeats and a short repetition penalty", guard["penalised"]),
            ("maya:loop_guard_thinking_closed_total", "counter", "Loops in the thinking that closed it",
             guard["thinking_closed"]),
            ("maya:loop_guard_stopped_total", "counter", "Answers the loop guard ended (a loop that came back)",
             guard["stopped"]),
            ("maya:loop_guard_tool_repeats_total", "counter",
             "Tool calls made again and again with the same result that got a note after them", guard["tool_repeats"]),
            ("maya:loop_guard_tool_blocked_total", "counter",
             "Tool calls the loop guard did not let out (made again after its notes)", guard["tool_blocked"]),
            ("maya:loop_guard_kept_going_total", "counter",
             "Answers that would have ended the turn with nothing in a conversation that repeats itself, kept going",
             guard["kept_going"]),
            ("maya:loop_guard_collapsed_total", "counter",
             "Prompts the loop guard left the model's repeats out of (a conversation that repeats itself)",
             guard["collapsed"]),
        ]
        rows += [(f"maya:engine_{k}", "counter" if k in self.PROM_STAT_COUNTERS else "gauge", h, stat[k])
                 for k, h in self.PROM_STAT.items()
                 if isinstance(stat.get(k), (int, float)) and not isinstance(stat.get(k), bool)]
        for category, values in (("cache", self.responses.stats()), ("jobs", self.response_jobs.stats())):
            gauges = {"entries", "blocks", "bytes", "inflight_bytes", "peak_inflight_bytes", "max_entries",
                      "max_bytes", "ttl_s", "records", "running", "queued", "event_bytes", "readers"}
            rows += [(f"maya:responses_{category}_{key}", "gauge" if key in gauges else "counter",
                      f"Responses {category} {key}", value) for key, value in values.items()]
        return "".join(f"# HELP {name} {help_}\n# TYPE {name} {kind}\n{name} {value}\n"
                       for name, kind, help_, value in rows)

    def v1_status(self) -> dict:
        """GET /v1/status: what this server is and does, for a client that would rather ask than guess (a front-end
        that polls its OpenAI-compatible server's status, collabosm's for one): the model and its window, images,
        the APIs, what is running, and the last request's timings in llama.cpp's names.  /metrics has the rest."""
        with self.status_lock:
            s, totals = dict(self.status), dict(self.totals)
            last_t, last_at = (dict(self.last_timings) if self.last_timings else None), self.last_request_at
            active = self.active
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        hw, static = tel.get("now") or {}, tel.get("static") or {}
        places = max(1, int(getattr(self.engine, "multi", 0) or 0))   # requests the engine answers at once

        def scaled(v, unit, digits=0):
            return round(v / unit, digits) if isinstance(v, (int, float)) else None

        busy, ctx = bool(s.get("busy")), self.engine.max_context
        images = self.vision is not None
        return {
            "service": "maya", "model": self.model,
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            "concurrency": {"serving": places, "requested": places},   # more than that wait their turn
            "dialects": ["/v1/chat/completions", "/v1/responses", "/v1/messages"],
            "responses": {"cache": self.responses.stats(), "jobs": self.response_jobs.stats(),
                          "structured_outputs": {"mode": "validate-before-publish", "schema_draft": "2020-12",
                                                 "constrained_decoding": False}},
            "vision": {"enabled": images, "available": images,
                       "error": None if images or (getattr(self.engine, "info", {}) or {}).get("vision") != "off"
                       else "the model's vision tower did not load (the server's log says why)"},
            # (a Windows APU: Windows' commit - the page file - kept experts off the GPU; either engine's NOTE says so)
            "memory_note": (lambda i: None if not i.get("gpu_commit_capped_gb") else
                            f"{i['gpu_commit_capped_gb']} GB of experts stay off the GPU: Windows' page file is too small; "
                            f"a page file of {i.get('pagefile_hint_gb')} GB keeps them on the GPU")(
                                getattr(self.engine, "info", {}) or {}),
            "activity": {"requests": totals["requests"] + max(int(busy), active),
                         "in_flight": max(int(busy), active) + int(s.get("queued") or 0),
                         # the FIFO's state: a queue whose first waiter is old while nothing runs is stuck, not busy
                         "queue": {"waiting": int(s.get("queued") or 0),
                                   "first_waited_s": round(self.fifo.oldest_wait(), 1),
                                   "places": self.fifo.capacity},
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "machine": {
                "at": int(time.time()),
                "gpu": {"name": static.get("gpu_name"), "used_mib": scaled(hw.get("gpu_mem_used"), 2 ** 20),
                        "total_mib": scaled(hw.get("gpu_mem_total"), 2 ** 20), "util_pct": hw.get("gpu_util"),
                        "temp_c": hw.get("gpu_temp"), "power_w": hw.get("gpu_power")} if static.get("gpu_name") else None,
                "ram": {"used_gib": scaled(hw.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(hw.get("ram_total"), 2 ** 30, 1)} if hw.get("ram_total") else None}}

    def prepare(self, messages, tools, kwargs, max_new=None, cancel=None, guard=None):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context.  A tool_choice that requires a call ("_force_tool": "" any, else the tool's name)
        starts the answer with the call's opening, thinking off; that text goes back in kwargs["_prefill"] for run().
        With the loop guard (`guard`, else the server's) a tool call the model keeps making with the same result gets
        a note after that result (_tool_repeat); one it made again after TOOL_NUDGES such notes is blocked in this
        answer (kwargs["_blocked"] for run(), _block_suffixes); and in a conversation that repeats itself an answer
        may not end the turn short (kwargs["_blocked"]["_keep"], _keep_suffix)."""
        on = self.loop_guard if guard is None else guard
        blocked = loop_guard.tool_blocked(messages) if on else {}
        looping = bool(blocked) or bool(on and loop_guard.echoes(messages))
        asked = messages                            # (the request's own, before the prompt leaves anything out)
        if on:
            noted = self._tool_repeat(messages)
            if blocked and len(noted) == len(messages) and messages and messages[-1].get("role") == "user" \
                    and loop_guard.go_on(messages[-1]):     # "continue": which calls stay blocked, said at once
                noted = messages + [{"role": "user", "content": loop_guard.blocked_note(blocked)}]
            messages = self._collapse(messages) + noted[len(messages):]
        given, force = kwargs, kwargs.get("_force_tool")
        kwargs = {k: v for k, v in kwargs.items() if k != "_force_tool"}
        if force is not None and tools:
            kwargs["enable_thinking"] = False
        if self.default_effort and "reasoning_effort" not in kwargs and "enable_thinking" not in kwargs:
            kwargs = {**kwargs, **(deepseek_effort_kwargs(self.default_effort) if getattr(self.template, "deepseek", False)
                                  else effort_kwargs(self.default_effort))}   # the run config's thinking level
        prefill = ""
        if force is not None and tools:
            # GLM's form opens with the tool's name, Qwen's with <function=NAME>, DeepSeek's DSML with its calls block
            glm_calls = "<arg_key>" in self.template.source
            if getattr(self.template, "deepseek", False):
                prefill = "\n\n<｜DSML｜ calls>\n" + (f'<｜DSML｜ invoke name="{force}">\n' if force else "")
            elif getattr(self.template, "minimax", False):
                prefill = MINIMAX_NS + "<tool_call>\n" + (MINIMAX_NS + f'<invoke name="{force}">' if force else "")
            elif getattr(self.template, "inkling", False):
                # Inkling 2: a call is a model message of its own, <|message_model|>NAME<|content_invoke_tool_json|>{..}
                prefill = "<|message_model|>" + (force + "<|content_invoke_tool_json|>" if force else "")
            else:
                prefill = CALL_START + (force if glm_calls else "\n" + (f"<function={force}>\n" if force else ""))
        if not prefill and not tools and getattr(self.template, "inkling", False) and \
                (kwargs.get("enable_thinking") is False or kwargs.get("reasoning_effort") == "none"):
            # Inkling 2 has no switch for its thinking, only its effort, and at "none" it still thinks a little: with
            # thinking off the prompt opens the answer's text segment (with tools the first segment may be a call)
            kwargs = {**kwargs, "enable_thinking": False}
            prefill = "<|message_model|><|content_text|>"
        given["_prefill"] = prefill
        if prefill and getattr(self.template, "hy4", False):
            # Hy4-flash: GLM's form under its own suffixed tags, inside its <tool_calls:..> wrapper.  The prompt gets
            # the opening as its template writes it; the parser reads it in GLM's spelling, as it reads the answer
            given["_prefill"], prefill = CALL_START + (force or ""), self.template.hy4_call_open(force)
        given["_conv_label"] = conv_first_words(asked)   # (the Monitor's conversation rows: its first words)
        given["_blocked"] = self._block_suffixes(blocked, tools, kwargs) if blocked and force is None else None
        if looping and tools and force is None and not loop_guard.answers_user(asked):   # (an answer to the user ends)
            keep = self._keep_suffix(tools, kwargs)
            if keep is not None:                     # (+ what the model said before: stops_short's `seen`)
                given["_blocked"] = {**(given["_blocked"] or {}), "_keep": keep + (loop_guard.said_before(asked),)}
        if getattr(self.template, "hy4", False) and getattr(self, "vision", None) is None:
            messages = hy4_tool_pictures_noted(messages)   # (a tool result's picture: a text note, not a 400)
        ids = self.render_ids(messages, tools, kwargs, prefill)
        # the reply's opening: the tokens the generation prompt (and a thinking-off close or a forced call) adds
        # (not DeepSeek: maya-ds4 has no gt= key, so its prompts skip this second render)
        given["_gen_tail"] = 0
        try:
            if not getattr(self.template, "deepseek", False):
                bare = self.render_ids(messages, tools, {**kwargs, "add_generation_prompt": False}, "")
                if 0 < len(bare) < len(ids) and ids[:len(bare)] == bare:
                    given["_gen_tail"] = len(ids) - len(bare)
        except Exception:   # a template that cannot render without the generation prompt: the last token only
            pass
        self.embeddings.path = None
        media = media_of(messages)
        images = (deepseek_images(messages) if getattr(self.template, "deepseek", False)
                  else [src for kind, src in media if kind == "image"])
        if isinstance(self.vision, mm3_vision.Vision) and any(kind == "video" for kind, _ in media):
            raise ValueError(("Hy4-flash" if isinstance(self.vision, hy4_vision.Vision) else "MiniMax-M3.1")
                             + " reads pictures, not videos")
        if images and isinstance(self.vision, ENGINE_VISION):
            # DeepSeek: the engine encodes the pictures itself; each placeholder the template wrote (user text's are
            # plain text, see render_ids) becomes its picture's span, the resized pixels go in the GENI file
            pics = []
            for src in images:
                if cancel is not None and cancel.is_set():
                    raise ValueError("image preparation was cancelled")
                pics.append(self.vision.picture(src))
            if cancel is not None and cancel.is_set():
                raise ValueError("image preparation was cancelled")
            ids, starts = self.vision.expand(ids, pics)
            self.embeddings.path = self.vision.write(self.vision.dir / f"req-{uuid.uuid4().hex[:12]}.dsi", pics,
                                                     starts)
        elif media:
            if self.vision is None and getattr(self.template, "hy4", False):
                # (--vision off, a pack without the mmproj file, or no room for a picture: the engine's start says)
                raise ValueError(f"{HY4_TEXT_ONLY} reads text only, so it cannot read images or videos: send the "
                                 "request without them (describe the picture in text instead)")
            inkling = getattr(self.template, "inkling", False)
            if self.vision is None and inkling:
                # Inkling 2's GGUF has no vision tower: the image encoder (the config's "vision" entry) reads pictures
                raise ValueError(f"{self.model} on saguaro-ik2 reads text only without the image encoder (a \"vision\" "
                                 "entry in its config: the encoder and the model's mmproj file), so it cannot read "
                                 "images or videos: send the request without them (describe the picture in text "
                                 "instead)")
            if inkling and any(kind == "video" for kind, _ in media):
                raise ValueError(f"{self.model} reads pictures, not videos")
            if self.vision is None:
                raise ValueError("this server was started without the vision encoder (run setup again and choose "
                                 "'vision'), so it cannot read images or videos")
            if any(kind == "video" for kind, _ in media) and not hasattr(self.vision, "encode_video"):
                raise ValueError("this model's picture reader reads pictures, not videos")
            # the template's media markers: GLM-5.3 writes <|begin_of_image|><|image|><|end_of_image|> and
            # <|begin_of_video|><|video|><|end_of_video|>, Qwen <|vision_start|><|image_pad|><|vision_end|> and
            # <|vision_start|><|video_pad|><|vision_end|>
            glm = "<|begin_of_image|>" in self.template.source
            image_pad, image_start = ("<|image|>", "<|begin_of_image|>") if glm else (IMAGE_PAD, VISION_START)
            video_pad, video_start = ("<|video|>", "<|begin_of_video|>") if glm else ("<|video_pad|>", VISION_START)
            if inkling:
                # Inkling 2 writes <|content_image|><|unused_200054|><|end_message|>: the one placeholder becomes the
                # picture's image tokens (each takes one of the encoder's rows); pictures only
                image_pad, image_start = "<|unused_200054|>", "<|content_image|>"
            pad = self.tok.encode(image_pad, parse_special=True)[0]
            start = self.tok.encode(image_start, parse_special=True)[0]
            vp = vstart = iend = None
            if not inkling:
                vp = self.tok.encode(video_pad, parse_special=True)[0]
                vstart = self.tok.encode(video_start, parse_special=True)[0]
                # every frame is an independent image: the image's own begin / end markers around its pads
                iend = self.tok.encode("<|end_of_image|>" if glm else "<|vision_end|>", parse_special=True)[0]
            frame = (start, pad, iend)
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
            lend = self._vision_lends()
            # a lending engine (VLEND) gives the room while the batch slots decode, so the encode takes its own place
            # in the FIFO; the others need the engine to themselves (units=None: every place, as encoding always did)
            with self._turn([False, False], cancel, units=1 if lend else None):
                if cancel is not None and cancel.is_set():
                    raise ValueError("image preparation was cancelled")
                self.ensure_engine()                     # (the encoder borrows the engine's memory: it must run)
                # one encode on the encoder at a time (these requests now run beside each other), and the idle stop
                # waits for an encode through the very same lock
                enc_lock = self._vision_enc if lend else contextlib.nullcontext()
                with enc_lock:
                    self._vision_up()                    # (an elastic encoder: the cache frees room, then it starts)
                    with self._vision_active():
                        enc = []
                        combined = self.vision.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
                        try:
                            with open(combined, "wb") as target:
                                for kind, src in media:
                                    if cancel is not None and cancel.is_set():
                                        raise ValueError("image preparation was cancelled")
                                    item = self.vision.encode(src) if kind == "image" else self.vision.encode_video(src)
                                    enc.append((kind, item))
                                    # Own the bytes before the next encode can evict this cache entry. active()
                                    # holds the encoder lock, so another preparation cannot evict it mid-copy.
                                    with open(item[0], "rb") as source:
                                        shutil.copyfileobj(source, target)
                        except OSError as e:
                            combined.unlink(missing_ok=True)
                            raise BadBody(503, "vision embeddings could not be prepared. Retry the request.") from e
                        except BaseException:
                            combined.unlink(missing_ok=True)
                            raise
                        finally:
                            self.last_image_at = time.time()  # (a turn that failed still restarts the idle clock)
                if cancel is not None and cancel.is_set():
                    combined.unlink(missing_ok=True)
                    raise ValueError("image preparation was cancelled")
            literal = self.tok.encode(image_pad, parse_special=False)
            encoded = [e for kind, e in enc if kind == "image"]
            # one <|image_pad|> per image -> one per image token.  Only the markers the template writes for an image
            # (right after <|vision_start|>) are images: the same text inside a message (an agent reading these docs,
            # #150) is kept as plain text, or it took an image's place and the counts no longer matched.
            out, k = [], 0
            for j, t in enumerate(ids):
                if t == pad and j > 0 and ids[j - 1] == start and k < len(encoded):
                    out += [pad] * encoded[k][1]
                    k += 1
                elif t == pad:
                    out += literal
                else:
                    out.append(t)
            if k != len(encoded):
                combined.unlink(missing_ok=True)
                raise ValueError("the prompt and its images do not match")
            ids = out
            # each <|video|> marker -> its frames, every frame an independent image wrapped in the image's own
            # markers (GLM-5.3 reads a video as a row of pictures; no timestamp text between the frames)
            videos = [e for kind, e in enc if kind == "video"]
            if videos:
                vliteral = self.tok.encode(video_pad, parse_special=False)
                out, k = [], 0
                for j, t in enumerate(ids):
                    if t == vp and j > 0 and ids[j - 1] == vstart and k < len(videos):
                        for n in videos[k][1]:
                            out += [frame[0]] + [pad] * n + [frame[2]]
                        k += 1
                    elif t == vp:
                        out += vliteral
                    else:
                        out.append(t)
                if k != len(videos):
                    combined.unlink(missing_ok=True)
                    raise ValueError("the prompt and its videos do not match")
                ids = out
            self.embeddings.path = combined
        if self.engine.max_context <= 0:   # (a reload / /models/load: the start has not reported its context yet)
            raise BadBody(503, "the model is starting; try again in a moment")
        room = self.engine.max_context - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ContextOverflow(f"maximum context length is {self.engine.max_context} tokens; prompt "
                                      f"({len(ids)} tokens) leaves no room to answer in the context; requests are "
                                      "never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise ContextOverflow(f"maximum context length is {self.engine.max_context} tokens; prompt "
                                      f"({len(ids)} tokens) + max tokens ({max_new}) exceeds the context; requests "
                                      "are never truncated")
            max_new = max(1, room)          # --fit-max-tokens: a shorter completion beats a 400
        thinking = kwargs.get("enable_thinking", True) is not False
        if thinking and self.think_end_id is not None:
            # (the engine ends at its EOS token: a marker the model quotes inside its thinking would end the turn
            # there - run() goes on from the quote with THINK_NOTE, if the template opens the next thinking)
            resume = self._keep_suffix(tools, kwargs, THINK_NOTE)
            if resume is not None and resume[1]:
                given["_think_resume"] = resume
        return ids, thinking, max_new

    def chat_prompt(self, req: dict):
        """A chat request's template messages, tools and kwargs as prepare() gets them (the run config's thinking
        level included), for measuring its prompt without running it."""
        messages, tools, kw = openai_to_messages(req, deepseek=getattr(self.template, "deepseek", False))
        kw = {k: v for k, v in kw.items() if k != "_force_tool"}
        if req.get("strata_mcp") is True and self.mcp is not None:
            own = {t.get("name") for t in tools or []}
            tools = (tools or []) + self.mcp.template_tools(exclude=own) or None
        if self.default_effort and "reasoning_effort" not in kw and "enable_thinking" not in kw:
            kw = {**kw, **(deepseek_effort_kwargs(self.default_effort) if getattr(self.template, "deepseek", False)
                                  else effort_kwargs(self.default_effort))}
        return messages, tools, kw

    def tokenize(self, req: dict) -> dict:
        """POST /tokenize (llama.cpp's and vLLM's): the ids of `content` (llama.cpp's name) or `prompt` (vLLM's), or
        of a chat request's whole prompt (`messages`, with its `tools`: chat_prompt, as /api/tokens counts it - a
        picture as its marker).  A special token's text in the text is that token unless parse_special is false;
        add_special (or add_special_tokens) puts the tokenizer's BOS first."""
        if "messages" in req:
            messages, tools, kw = self.chat_prompt(req)
            ids = self.render_ids(messages, tools, kw)
        else:
            text = req.get("content", req.get("prompt"))
            if not isinstance(text, str):
                raise ValueError("tokenize needs content (or prompt) as a string, or messages")
            ids = self.tok.encode(text, parse_special=req.get("parse_special", True) is not False)
            bos = (getattr(self.tok, "special_ids", None) or {}).get("bos")
            if (req.get("add_special") or req.get("add_special_tokens")) and bos is not None:
                ids = [bos] + ids
        return {"tokens": ids, "count": len(ids), "max_model_len": self.engine.max_context}

    def detokenize(self, req: dict) -> dict:
        """POST /detokenize: the text of `tokens`, as `content` (llama.cpp's name) and `prompt` (vLLM's)."""
        ids = req.get("tokens")
        if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in ids):
            raise ValueError("detokenize needs tokens as a list of token ids")
        text = self.tok.decode(ids)
        return {"content": text, "prompt": text}

    def count_tokens(self, req: dict) -> dict:
        """POST /api/tokens: how many tokens this chat request's prompt takes - the same template, thinking level,
        tools and tokenizer as prepare(), with nothing run (no image encoder: a picture or a video counts as its
        marker, so the count is approximate then).  The web app's context meter."""
        messages, tools, kw = self.chat_prompt(req)
        if getattr(self.template, "deepseek", False) or isinstance(self.vision, mm3_vision.Vision):
            # (DeepSeek, MiniMax: the engine's own vision tower - a picture's span is its resize's)
            sources = (deepseek_images(messages) if getattr(self.template, "deepseek", False)
                       else [src for kind, src in media_of(messages) if kind == "image"])
            n, approximate = len(self.render_ids(messages, tools, kw)), bool(sources)
            if sources and isinstance(self.vision, ENGINE_VISION):
                try:
                    n += sum(self.vision.picture(src).tokens - 1 for src in sources)
                    approximate = False
                except (ValueError, OSError):
                    pass
            return {"tokens": n, "images": len(sources), "approximate": approximate,
                    "max_context": self.engine.max_context}
        media = media_of(messages)
        return {"tokens": len(self.render_ids(messages, tools, kw)), "images": len(media),
                "approximate": bool(media), "max_context": self.engine.max_context}

    @staticmethod
    def _say_malformed(parser):
        """A tool call the parser could not read (it went out as text, or as an incomplete call: OutputParser): the
        whole body in the server window (#613)."""
        while parser.malformed:
            body = parser.malformed.pop(0)
            print(f"[saguaro] a malformed tool call, sent on as written instead of failing the answer "
                  f"({len(body)} characters): {body!r}", flush=True)

    def _note(self, n, evs, stat):
        with self.status_lock:
            s = self.status
            s["generated"] = n
            now = time.time()
            if stat.first_token is None:
                stat.first_token = now
                self.current_requests.add(stat)
                self._wall_tick()                       # (its read became a write: the wall clocks follow)
            stat.note(now, n)                           # this request's window over the last RATE_WINDOW_S
            stat.generated = n
            k = len(self.current_requests)
            if k > 1:
                stat.shared_max = max(stat.shared_max, k)
                if now - getattr(stat, "eng_t", 0.0) >= 0.5:   # (the engine's total twice a second: cheap enough)
                    stat.eng_t = now
                    stat.eng_sum += sum(r.tok_s(now) for r in self.current_requests)
                    stat.eng_n += 1
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                else:
                    continue
                stat.phase = s["phase"]
            for ev in evs:
                s["tail"] = (s["tail"] + (ev.text or ""))[-600:]
            if stat.phase == "reading the prompt":
                stat.phase = "writing"

    def _progress(self, last_print, stat, every=1.0):
        """A progress line in the server window every `every` seconds while a request runs - its own figures (with
        several at once the shared status mixed one request's count with another's clock)."""
        now = time.time()
        pp = getattr(getattr(self.engine, "_tls", None), "own_pp", None)   # (this thread's prompt progress, its rate)
        with self.status_lock:
            if pp and stat.first_token is None:
                stat.progress, stat.prefill_tok_s = pp
            if now - last_print < every:
                return last_print
            r = stat.view(now)
        el = r["elapsed_s"] or 0
        if stat.first_token is None:
            pr = stat.progress or getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{r['prompt_tokens']:,}"   # prefix counts as read (#29)
            print(f"[saguaro] reading the prompt: {done} tokens, {el:.0f} s so far", flush=True)
        else:
            rate = r["generated"] / max(1e-6, now - stat.first_token)
            print(f"[saguaro] {r['phase']}: {r['generated']} of max {r['max_tokens']} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    @contextlib.contextmanager
    def _turn(self, hold, cancel=None, units=1, priority=None):
        """The engine's turn for one request: the FIFO, given back at the end unless a fair slice already did.  The
        place is taken (and given up) inside the try: an exception while waiting - a FIFO that throws instead of
        waiting - left it in the line for ever, and every request behind it waited for a turn nothing would take."""
        place = None
        try:
            if hold[0]:
                pass                                    # (the turn taken already: _queue_wait's, with keep-alives)
            elif cancel is None:
                self.fifo.acquire(units=units, priority=priority)
                hold[0] = True
            else:
                place = self.fifo.enqueue(units, priority)
                while not cancel.is_set():
                    if self.fifo.wait_turn(place, timeout=0.05):
                        hold[0] = True
                        break
            yield
        finally:
            if place is not None:
                self.fifo.leave(place)
            if hold[0]:
                hold[0] = False
                self.fifo.release(units=units, priority=priority)   # (R122: its level - a normal or high request
                #  that ends here is a finished answer: the idle rule's grace for the low queue starts with it)

    def _queue_wait(self, hold, cancel, units=1, priority=None, stat=None) -> Iterator[tuple[str, object]]:
        """A request's wait in the FIFO for its first turn, as ("ping", None) every QUEUE_PING_S - the streams write
        them as SSE comments (": keep-alive"), which every client's parser ignores (#483: a request queued behind a
        long answer sent nothing at all, and a client with a stream-idle timeout gave up on a turn that was only
        waiting).  hold[0] is True once the turn is this request's (_turn then holds it and gives it back); a client
        that goes away meanwhile - the GeneratorExit at a ping - gives its place back here.  `stat` (R122, Ken's
        trigger decision): a turn that does not come on the first look marks the request as one that had to wait -
        from there on its client may get the padding / empty events; one granted straight off never does."""
        if cancel is None:                              # (no way to leave the line: _turn waits as before)
            return
        every = queue_ping_seconds()
        place = self.fifo.enqueue(units, priority)
        try:
            last = time.monotonic()
            while not cancel.is_set():
                if self.fifo.wait_turn(place, timeout=0.05):
                    hold[0] = True
                    return
                if stat is not None:
                    stat.had_to_wait = True             # (it waited: the engine was somebody else's)
                if time.monotonic() - last >= every:
                    last = time.monotonic()
                    yield "ping", None
        finally:
            if not hold[0]:
                self.fifo.leave(place)

    @staticmethod
    def _resumed_sampling(sampling, produced):
        """The sampling for an engine call that goes on after `produced`: the thinking budget counts from where it got
        to (and is gone once the thinking was closed)."""
        tb = (sampling or {}).get("_think_budget")
        if not (isinstance(tb, int) and tb > 0):
            return sampling
        smp = dict(sampling)
        if (sampling or {}).get("_think_end") in produced:
            smp.pop("_think_budget", None)
            smp.pop("_think_end", None)
            smp.pop("_think_soft", None)
            smp.pop("_think_stops", None)
        else:
            smp["_think_budget"] = max(1, tb - len(produced))
            if isinstance(smp.get("_think_soft"), int):
                smp["_think_soft"] = max(1, smp["_think_soft"] - len(produced))
        return smp

    def _read_position(self):
        """(read, total) of the prompt THIS thread is reading, or None while it reads nothing: the pipe engines' PP
        lines land in `_tls.pp` (their multi path) and `_tls.own_pp` (solo, and QFN's per-request read - its batch
        slots report the same on BRP lines); both put the position and the prompt length first.  The fair reads
        read `_tls.pp` only (multi, where a read can pause with PAUSE); this sees every driver's own read, which is
        what the priority preemption of a read checks on its ticks (R122)."""
        tls = getattr(self.engine, "_tls", None)
        pp = getattr(tls, "pp", None) or (getattr(tls, "own_pp", None) or (None,))[0]
        return pp[:2] if isinstance(pp, (tuple, list)) and len(pp) >= 2 else None

    def _loop_guard_on(self, sampling) -> bool:
        """The loop guard for this request: its own "loop_guard" when it sends one, else the server's."""
        own = (sampling or {}).get("loop_guard")
        return own if isinstance(own, bool) else bool(self.loop_guard)

    def _guard_piece(self):
        """The text of a token for the loop guard's tool-call markers: incremental (a character split across tokens
        whole), and a tokenizer without bytes decodes the token alone (no decode of the whole answer a token)."""
        d = Detokenizer(self.tok)
        return d.push if d.inc is not None else (lambda t: self.tok.decode([t]))

    def _block_suffixes(self, blocked, tools, kwargs):
        """{(name, args): (ToolRepeat, ids, thinks)}: for each blocked call (loop_guard.tool_blocked) the prompt that
        follows the model's completed tool-call block when the server answers it itself - its turn close, the tool
        result loop_guard.block_result, a user message loop_guard.block_note and the next assistant turn's opening
        (`thinks`: that opening is an open thinking block, which run() starts with loop_guard.block_steer).  None for
        a template with no supported call boundary or one that does not render a turn as a prefix of the next."""
        call_end = CALL_END
        if getattr(self.template, "deepseek", False):
            from serve.ds41_encoding import dsml_token, tool_calls_block_name
            # The parser rewrites DSML into CALL_END internally; the prompt and generated token IDs still use
            # the original DSML. Slice after the outer block, not an invoke or a legacy tag inside an argument.
            call_end = f"</{dsml_token}{tool_calls_block_name}>"
        elif getattr(self.template, "hy4", False):   # (Hy4-flash: the end of its wrapper, after the last call's own)
            call_end = "</tool_calls%s>" % self.template.hy4_tag
        out = {}
        for key, rep in blocked.items():
            call = {"id": "call_loop_guard", "type": "function",
                    "function": {"name": key[0], "arguments": json.loads(key[1])}}
            head = [{"role": "user", "content": "."}, {"role": "assistant", "content": "", "tool_calls": [call]}]
            tail = [{"role": "tool", "tool_call_id": "call_loop_guard", "content": loop_guard.block_result(rep)},
                    {"role": "user", "content": loop_guard.block_note(rep)}]
            try:
                p = self.template.render(head, tools=tools, **{**kwargs, "add_generation_prompt": False})
                f = self.template.render(head + tail, tools=tools, **kwargs)
            except Exception as e:                  # (a template that wants more than this: notes only)
                print(f"[saguaro] loop guard: cannot block `{rep.name}` with this template ({e})", flush=True)
                return None
            if kwargs.get("enable_thinking") is False and f.endswith("<think>"):
                f += "</think>"                     # (as render_ids)
            if not f.startswith(p) or call_end not in p:
                return None
            out[key] = (rep, self.tok.encode(p[p.rindex(call_end) + len(call_end):] + f[len(p):], parse_special=True),
                        self._opens_thinking(f))
        return out

    def _keep_suffix(self, tools, kwargs, note=loop_guard.KEEP_NOTE):
        """(ids, thinks): the prompt that follows the model's own last words when it would end its turn short in a
        conversation that repeats itself (loop_guard.stops_short) - the template's turn end, a user message
        `note` (default loop_guard.KEEP_NOTE) and the next assistant turn's opening (`thinks`: an open thinking
        block).  None for a template that does not render a turn as a prefix of the next."""
        mark = "\u2063keep\u2063"                  # (the model's words, wherever the template puts them)
        head = [{"role": "user", "content": "."}, {"role": "assistant", "content": mark}]
        try:
            p = self.template.render(head, tools=tools, **{**kwargs, "add_generation_prompt": False})
            f = self.template.render(head + [{"role": "user", "content": note}], tools=tools, **kwargs)
        except Exception:
            return None
        if kwargs.get("enable_thinking") is False and f.endswith("<think>"):
            f += "</think>"
        if mark not in p or mark not in f:          # (the template keeps the assistant's words in the next turn)
            return None
        # what follows the model's own last words: the template's turn end, the note, the next assistant opening.
        # (Not "p's tail plus f past p": a template may render the assistant turn it ends differently from the one
        # before a user message - DeepSeek with no tools opens the current thinking with <think> and the earlier one
        # without, so p is not a prefix of f there; f's own copy of the words is the same position in both.)
        tail = f[f.rindex(mark) + len(mark):]
        if not tail:
            return None
        return (self.tok.encode(tail, parse_special=True), self._opens_thinking(f))

    def _opens_thinking(self, prompt: str) -> bool:
        """Whether a rendered prompt ends in an open thinking block (Hy4-flash's is its suffixed <think:..>)."""
        opens = getattr(self.template, "opens_thinking", None)
        return bool(opens(prompt)) if callable(opens) else prompt.rstrip().endswith("<think>")

    def _collapse(self, messages):
        """A conversation that carries a loop across turns (loop_guard.collapse): its prompt without what the model
        already wrote - only that prompt changes."""
        c = loop_guard.collapse(messages)
        if c is None:
            return messages
        print(f"[saguaro] loop guard: the conversation repeats itself - left out of the prompt: {c.paragraphs} pieces "
              f"of the model's own text it had written before, {c.rounds} tool rounds that repeat earlier ones",
              flush=True)
        with self.status_lock:
            self.loop_guard_counts["collapsed"] += 1
        return c.messages

    def _kept_going(self, kept, text):
        print(f"[saguaro] loop guard: the model would end its turn with nothing ({' '.join(text.split())[:80]!r}): "
              f"kept going ({kept}. time in this answer)", flush=True)
        with self.status_lock:
            self.loop_guard_counts["kept_going"] += 1

    def _tool_blocked(self, rep, tries):
        print(f"[saguaro] loop guard: `{rep.name}` again after {loop_guard.TOOL_NUDGES} notes ({rep.times} runs, the same "
              f"result): blocked ({tries}. time in this answer), not run - a note, the thinking steered away and on",
              flush=True)
        with self.status_lock:
            self.loop_guard_counts["tool_blocked"] += 1

    def _tool_repeat(self, messages):
        """The loop guard across turns (loop_guard.tool_repeat): the request's last tool call, made again and again
        with the same arguments and the same result - a user message with loop_guard.tool_note() follows its result
        (inside the result GLM-5.3 ran the call again; as a message of its own it moved on).  Only this prompt changes:
        the client's conversation stays as it sent it (its next request matches the kept one up to that result, so
        only the tail is read again)."""
        rep = loop_guard.tool_repeat(messages)
        if rep is None:
            return messages
        print(f"[saguaro] loop guard: `{rep.name}` {rep.times} times with the same arguments and result: a note after it",
              flush=True)
        with self.status_lock:
            self.loop_guard_counts["tool_repeats"] += 1
        return messages + [{"role": "user", "content": loop_guard.tool_note(rep)}]

    def _loop_noted(self, act, written: int):
        """A loop the guard broke: a line in the server window, and the Monitor's count."""
        print(f"[saguaro] loop guard: {act.describe()}"
              f"{' (now ' + format(written, ',') + ' tokens)' if act.kind != loop_guard.STOP else ''}", flush=True)
        with self.status_lock:
            c = self.loop_guard_counts
            c["detected"] += 1
            c[{loop_guard.PENALTY: "penalised", loop_guard.CLOSE_THINKING: "thinking_closed",
               loop_guard.STOP: "stopped"}[act.kind]] += 1

    def _sliced(self, ids, max_new, sampling, cancel, emb, hold, stat, guard=None) -> Iterator[int | None]:
        """engine.generate(), in fair slices (SAGUARO_FAIR_SLICE_S): once this answer has been decoding that long and
        another request waits for the engine, the engine stops here, the FIFO goes to the next request and this one
        queues behind it.  Back in turn, it goes on with prompt + what it already wrote as the new prompt: the
        engine's conversation slots hold this prompt's state, so only the tokens written since are read again (a
        prompt read is ~30x a decode step's speed).  Concurrent chats then all move on instead of waiting behind the
        longest answer, and a short side request (an agent's title, a summary) no longer waits for a whole turn.
        The caller sees one stream of token ids (None: a heartbeat, also while queued again).  `hold[0]` says
        whether this request holds the FIFO now.  Not for image prompts: their state is never kept in a slot.

        R122: a request gives its place up the same way and at once, on the next decode token's tick, where
        `FairLock.step_aside` says so - a LOW one whenever any normal or high request runs or waits (even when a
        place is free: low is "only while idle"), a NORMAL one only for a waiting high one that does not fit -
        and on a read tick (PP/BRP lines) only where the engine keeps a stopped read (PAUSE where it can, QFN's
        cut where its conversation stays live at the token reached; every other engine's read runs to its end).  The same resumed call: prompt + written tokens, _resumed_sampling, the segments'
        stats summed, the same streamed text - and it waits with keep-alives, leaving the line if the client goes.

        With the loop guard (`guard`, serve/loop_guard.py) every id goes through it and the caller gets only what
        lies before the short tail it holds back (what could still be a loop's repeats, at most HOLD_MAX tokens).  A
        loop restarts the engine the same way, at once and still holding the FIFO: prompt + what is kept (the repeats
        not sent yet dropped; after a third loop in the thinking, the model's end-of-thinking token too) with a short repetition penalty, which a call
        of its own ends and the request's own sampling follows; or the answer ends there (LOOP_STOP).  Until then the
        engine calls and their sampling are the ones without it."""
        produced, cur, left, smp = [], list(ids), max_new, sampling
        segments = []                                   # each slice's DONE (engine.last), summed at the end
        prio = request_priority(sampling)               # (R122: this answer gives its place up at once for a waiter
        preempt = None                                  #  of a higher priority; while it waits, what it waits for)
        pos = None                                      # (a read cut: where its PP/BRP tick found it)
        sent = 0                                        # produced[:sent] went to the caller (the guard holds the rest)
        # the thinking budget closed at a sentence's or line's end (THINK_GRACE, #613): past `soft` tokens of
        # reasoning, the first token that ends one is followed by the end of the thinking - this call stops there and
        # the next goes on with it (the engine's own close, its think_budget, comes later, for reasoning with none)
        soft, think_end = (sampling or {}).get("_think_soft"), (sampling or {}).get("_think_end")
        hard = (sampling or {}).get("_think_hard")      # (no engine close: here at the latest - Inkling 2)
        close = list((sampling or {}).get("_think_close") or [think_end])   # (what closes it: Inkling 2's three marks)
        leaves = set((sampling or {}).get("_think_marks") or ())   # (other tokens that end the reasoning)
        soft_text = self._guard_piece() if isinstance(soft, int) and isinstance(think_end, int) else None
        tail_text = ""
        read_s = fair_read_seconds() if getattr(self.engine, "multi", 0) else 0.0
        while True:
            last0 = getattr(self.engine, "last", None)
            smp_call, n_call = guard.sampling(smp, len(produced), left) if guard is not None else (smp, left)
            gen = self.engine.generate(cur, n_call, smp_call, cancel, embeddings=emb) if emb else \
                self.engine.generate(cur, n_call, smp_call, cancel)
            slice_s, t0, cut, pausing, claimed = fair_slice_seconds(), None, False, False, False
            preempt, pos = None, None                   # (R122: the pause and its read position belong to THIS
            #  engine call: left over from an earlier one, a plain fair slice would be logged as a priority pause)
            act, got, closed = None, 0, False           # the loop guard's action; this call's tokens; a soft close
            t_read, paused_for = None, set()            # fair reads: the first chunk report's time, who it paused for
            try:
                for t in gen:
                    if guard is None:
                        yield t
                    elif t is None:
                        yield None
                    if t is None:
                        # SAGUARO_FAIR_READ_S: this prompt read that long (from its first chunk report: a request
                        # queued behind another's read never pauses its own), someone waiting behind it - PAUSE: the
                        # engine stops at a chunk boundary and keeps what it read; the rest after they started
                        pp = getattr(getattr(self.engine, "_tls", None), "pp", None)
                        if pp and t_read is None:
                            t_read = time.monotonic()
                        if read_s > 0 and t0 is None and not pausing and not emb and not self.stopping and pp \
                                and pp[0] < pp[1] and time.monotonic() - t_read >= read_s \
                                and getattr(self.engine, "can_pause", False):
                            paused_for = self.engine.waiting_behind(2.0)
                            if paused_for:
                                self.engine.pause()
                                pausing = True
                                print(f"[saguaro] fair read: {pp[0]:,} of {pp[1]:,} prompt tokens read, paused for "
                                      f"{len(paused_for)} waiting request(s); the rest after they started", flush=True)
                        # R122: a read tick and step_aside saying so (a low read beside a running or waiting
                        # person, a normal one for a waiting high that does not fit) - give the place up at once,
                        # no timer, and only for engines that keep a stopped read: GLM by its PAUSE, gated to a
                        # multi conversation engine the way the fair reads are (on a solo can_pause one a PAUSE
                        # during a read may be ignored or lose the read), and QFN by its cut SOLO (the
                        # conversation stays live at the token reached, keeps_stopped_read) - a QFN BATCH slot's
                        # cancelled read is dropped, nothing of it reaches the next BGEN, so its read runs on;
                        # every other engine's read runs to its end rather than lose what it read.
                        if (prio < PRIORITY_HIGH and t0 is None and not pausing and not emb and not self.stopping
                                and ((getattr(self.engine, "can_pause", False) and getattr(self.engine, "multi", 0))
                                     or (getattr(self.engine, "keeps_stopped_read", False)
                                         and not getattr(self.engine, "batch", 0)))):
                            pos = self._read_position()
                            if pos and pos[0] < pos[1] and self.fifo.step_aside(prio):
                                # (the decision held; the last waiter can still have left the line by now -
                                #  name the highest level that runs or waits, or this request's own one)
                                claimed, preempt = True, PRIORITY_NAMES[self.fifo.highest_active() or prio]
                                if getattr(self.engine, "can_pause", False):
                                    self.engine.pause()   # (multi, see the gate: the state with what it read stays)
                                    pausing = True
                                else:
                                    cut = True            # (STOP: gen.close() drains to the DONE, the engine keeps
                                    break                 #  the conversation at the token it reached)
                        continue
                    produced.append(t)
                    if soft_text is not None:
                        if t == think_end or t in leaves:
                            soft_text = None            # (closed: by the model, or the engine's budget)
                        else:
                            tail_text = (tail_text + (soft_text(t) or ""))[-16:]
                            if len(produced) >= soft and (think_boundary(tail_text) or
                                                          isinstance(hard, int) and len(produced) >= hard) \
                                    and len(produced) < max_new - len(close):
                                closed = True
                    if guard is not None:
                        got += 1
                        act = guard.push(t, len(produced)) if not closed else None
                        if act is not None:
                            break                       # (gen.close(): STOP + drain, still holding the FIFO)
                        while sent < len(produced) - (guard.hold if not closed else 0):
                            sent += 1
                            yield produced[sent - 1]
                    if closed:
                        break                           # (gen.close(): STOP + drain, still holding the FIFO)
                    if t0 is None:
                        t0 = time.monotonic()
                    # R122: give the place up at once on this token's tick (as a fair slice does: PAUSE where the
                    # engine keeps the state, else cut), not on the SAGUARO_FAIR_SLICE_S timer, where step_aside
                    # says so: a low answer steps aside whenever a normal or high one runs or waits - a free place
                    # is no shield, low is "only while idle" - and a normal one only for a waiting high one that
                    # does not fit (claim_yield's counting: only as many runners as the waiter needs).  Never for
                    # a waiter of an equal or lower priority, and never a high answer.
                    if (prio < PRIORITY_HIGH and not pausing and not emb and not self.stopping
                            and len(produced) < max_new and self.fifo.step_aside(prio)):
                        # (the decision held; the last waiter can still have left the line by now - name the
                        #  highest level that runs or waits (the normal this low one steps aside for is RUNNING,
                        #  not waiting), or this request's own level rather than crash)
                        claimed, preempt = True, PRIORITY_NAMES[self.fifo.highest_active() or prio]
                        if getattr(self.engine, "can_pause", False):
                            self.engine.pause(keep=len(cur) == len(ids))   # its first slice (as a fair slice's)
                            pausing = True
                        else:
                            cut = True
                            break
                    if slice_s > 0 and not pausing and not emb and not self.stopping and len(produced) < max_new \
                            and self.fifo.waiting() and time.monotonic() - t0 >= slice_s \
                            and self.fifo.claim_yield(slice_s):     # (one runner a waiter, not every runner)
                        claimed = True
                        if getattr(self.engine, "can_pause", False):
                            # PAUSE: the engine keeps the state with what it wrote; the tokens it still writes
                            # before its DONE go to the client as usual, so the next prompt continues that state
                            self.engine.pause(keep=len(cur) == len(ids))   # its first slice
                            pausing = True
                        else:
                            cut = True
                            break
                if pausing:
                    done = getattr(self.engine, "last", None)
                    cut = len(produced) < max_new and (done is last0 or (done or {}).get("finish") == "cancel")
            finally:
                gen.close()                             # STOP + drain to this slice's DONE, still holding the FIFO
                if claimed and not (cut and not closed and act is None):
                    self.fifo.unclaim_yield()           # (it ends, or goes on some other way: no place given up)
                fresh = getattr(self.engine, "last", None)
                if fresh is not None and fresh is not last0:
                    segments.append(fresh)
            if closed:                                  # the budget, at a sentence's or line's end: the thinking ends
                produced.extend(close)
                sent = len(produced)
                soft_text = None
                yield from close
                if guard is not None:
                    guard.thinking_closed(produced)
                print(f"[saguaro] thinking reached its budget ({soft} tokens): closed "
                      f"{'at the end of a sentence or line, ' if think_boundary(tail_text) else ''}"
                      f"{len(produced) - len(close) - soft} tokens later; answering", flush=True)
                if cancel.is_set() or self.stopping:
                    break
                cur, left = list(ids) + produced, max_new - len(produced)
                smp = self._resumed_sampling(sampling, produced)
                continue
            if act is not None:                         # a loop: what is kept, then on from there - or the end
                del produced[max(sent, len(produced) - act.drop):]   # (what was sent stays sent: only the held tail goes)
                if act.kind == loop_guard.CLOSE_THINKING:
                    produced.append(guard.think_end)
                guard.resume(produced, act)
                self._loop_noted(act, len(produced))
                if act.kind == loop_guard.STOP or len(produced) >= max_new or cancel.is_set() or self.stopping:
                    break
                cur, left = list(ids) + produced, max_new - len(produced)
                smp = self._resumed_sampling(sampling, produced)
                continue
            if not cut:
                if guard is not None and n_call < left and got >= n_call and produced[-1] not in self.stop_ids \
                        and not cancel.is_set() and not self.stopping:
                    # the guard's penalty ended that call: on with the request's own sampling, still holding the FIFO
                    cur, left = list(ids) + produced, max_new - len(produced)
                    smp = self._resumed_sampling(sampling, produced)
                    continue
                break
            with self.status_lock:
                mine = dict(self.status)
                stat.clear()                            # queued again: not decoding, so no part of the live rate
                self.active -= 1
                hold[1] = False
                stat.at_engine = False                  # (it holds no engine place: neither read nor write counts now)
                self.status["busy"] = self.active > 0
                self.status["queued"] += 1
                self._wall_tick()
            waiting = self.fifo.waiting()
            place = self.fifo.yield_place(units=1, priority=prio)   # in line, and the place free, in one step:
            #                                     behind whoever waits now - at its own priority, so the resumed
            #                                     answer keeps its claim against the queue behind it
            stat.had_to_wait = True                     # (R122, Ken's trigger decision: back in line - a pause or
            #                                     a slice's wait counts as queued once: its client may get the
            #                                     padding / empty events from here)
            hold[0] = False
            if preempt:
                stat.paused_for_priority = True
                print(f"[saguaro] priority: a {PRIORITY_NAMES[prio]} "
                      f"{f'answer paused at {len(produced)} tokens' if produced else f'prompt read paused at {pos[0]:,} of {pos[1]:,} prompt tokens'}"
                      f" for a {preempt}-priority request ({waiting} waiting)", flush=True)
            else:
                print(f"[saguaro] fair slice: {len(produced)} tokens written, the engine goes to the next request "
                      f"({waiting} waiting); this one continues after it", flush=True)
            try:
                paused_at, noted = time.monotonic(), False
                while not self.fifo.wait_turn(place, timeout=0.1):
                    if cancel.is_set() or self.stopping:
                        return
                    if preempt and not noted:
                        waited = time.monotonic() - paused_at
                        if waited >= 600:
                            # R122: a long pause says so once - the request is never ended or promoted for waiting,
                            # it ends only when its client goes away, asks to stop, or the server stops
                            noted = True
                            print(f"[saguaro] priority: a {PRIORITY_NAMES[prio]} "
                                  f"{'answer' if produced else 'prompt read'} has been paused for "
                                  f"{int(waited // 60)} min", flush=True)
                    yield None                          # keeps the client's watchdog calm while queued again
                hold[0] = True
            finally:
                if not hold[0]:                         # the client went away while queued again
                    self.fifo.leave(place)
                with self.status_lock:
                    self.status["queued"] -= 1
            if preempt:                                 # R122: its turn again - the answer goes on from exactly the
                stat.paused_for_priority = False        #  token it reached, nothing the engine holds re-read. The
                #  read's line says where the engine kept it, not where the tick saw it: the engine cannot report
                #  more over the pipe, and a stop between two windows may quietly keep less than the tick printed
                print(f"[saguaro] priority: the {PRIORITY_NAMES[prio]} "
                      f"{'answer continues' if produced else 'prompt read continues from where the engine kept it'}",
                      flush=True)
            with self.status_lock:
                self.active += 1
                self.touch_telemetry()
                hold[1] = True
                stat.at_engine = True
                self.status.update({k: mine[k] for k in ("busy", "phase", "prompt_tokens", "started", "tool", "tail",
                                                         "max_tokens") if k in mine},
                                   generated=len(produced))
                self._wall_tick()
            if paused_for:   # a paused read: back once the requests it paused for have started (else the engine
                until = time.monotonic() + fair_read_yield_seconds()   # takes it up at once, ahead of them)
                while time.monotonic() < until and not self.stopping and not cancel.is_set() \
                        and not self.engine.started(paused_for):
                    yield None
                    time.sleep(0.25)
            if self.stopping or cancel.is_set():
                break
            if hasattr(self.engine, "alive") and not self.engine.alive():   # it died in the request in between
                print("[saguaro] the engine had stopped; starting it again (a minute or two) ...", flush=True)
                self.engine.restart()
            cur, left = list(ids) + produced, max_new - len(produced)
            smp = self._resumed_sampling(sampling, produced)   # (the thinking budget counts from where it got to)
        if len(segments) > 1:                           # the request's figures, over all its slices
            first, last = segments[0], segments[-1]
            merged = dict(last, prompt_tokens=first.get("prompt_tokens"), reused=first.get("reused"))
            for k in ("generated", "prompt_ms", "decode_ms", "drafts_accepted", "drafts_offered", "hits", "lookups"):
                if all(k in sg for sg in segments):
                    merged[k] = sum(sg[k] for sg in segments)
            self.engine.last = merged
        lastd = getattr(self.engine, "last", None)
        if isinstance(lastd, dict) and (lastd.get("prompt_tokens") or 0) > len(ids):
            self.engine.last = own_prompt_figures(lastd, len(ids))   # (the first call's DONE was not seen)
        if guard is not None and not cancel.is_set():   # the end: what the guard still held back
            while sent < len(produced):
                sent += 1
                yield produced[sent - 1]
            if guard.events and guard.events[-1].kind == loop_guard.STOP:
                yield LOOP_STOP                         # (run(): the answer ends here, finish "stop")

    def run(self, ids, thinking, tools, max_new, sampling, cancel) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..})."""
        defaults = {**self.sampling_defaults, **self.shared}   # the config's, then the Chat settings shared with apps
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
        responses = (sampling or {}).get("_responses") is True
        single_tool = responses and (sampling or {}).get("parallel_tool_calls") is False
        def new_parser(in_thinking=None):
            # (in_thinking: the segment's prompt already closed the thinking - see keep, below - so its first token
            # is the answer's, not the reasoning's)
            return OutputParser(thinking=thinking if in_thinking is None else thinking and in_thinking, tools=tools,
                                stream_tools=True, dsml=getattr(self.template, "deepseek", False),
                                token_only_end=(getattr(self.template, "deepseek", False) or   # (QFN's </think> is
                                                # its token 248069 at every turn's end: a quote of the tag in the
                                                # thinking - an agent reading this server's code - is text, not its end)
                                                getattr(self, "engine_kind", None) == "qfn") and self.think_end_id is not None,
                                minimax=getattr(self.template, "minimax", False),
                                inkling=getattr(self.template, "inkling", False),
                                strict_tools=responses, single_tool=single_tool)
        parser = new_parser()
        # calls the loop guard blocks in this answer (prepare, _block_suffixes): {(name, args): (ToolRepeat, ids)}.
        # While there are any, the answer's calls are held back (`pending`) until it ends; one that ends it alone
        # with a blocked call never goes out - the server answers it in the prompt and the model goes on, as often as
        # it takes (only max_tokens ends it, as any answer)
        blocked = dict((sampling or {}).get("_blocked") or {})
        keep = blocked.pop("_keep", None)           # (ids, thinks, seen): the answer may not end the turn short
        think_resume = (sampling or {}).get("_think_resume")   # (ids, thinks): a quoted marker inside the thinking
        think_restarts = 0                          # times the thinking was resumed in this answer
        said = []                                   # this engine call's text for the user (stops_short)
        reason_said = []                            # its reasoning's text (a call the model wrote in it, salvaged below)
        join = None                                 # after a keep: the paragraph break before the continuation (#921)
        kept = 0                                    # times it was kept going in this answer
        called = False                              # it wrote a tool call (then its turn does not end here)
        pending, tries, sent_call = [], {}, False
        seg = {"prompt": ids, "ids": [], "next": None}   # this engine call's prompt and tokens; the next one's
        prefill = (sampling or {}).get("_prefill") or ""   # the call's opening, already in the prompt (prepare)
        pre_events = parser.feed(prefill) if prefill else []
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        reasoning_tokens = 0
        held = ""                                       # text that may be the start of a stop marker (stop_texts)
        # the client's own stop strings (OpenAI "stop", Anthropic "stop_sequences"): the answer ends before one.  They
        # apply to the answer, not inside the reasoning; the model's own end-of-turn markers apply everywhere
        client_stops = client_stop_strings(sampling)
        matched = None                                  # the client stop string that ended the answer
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        raw_ids = []                                    # every generated id (SAGUARO_DEBUG: dump raw model text)
        guards = []                                     # this request's loop guard (the Monitor's row says what it did)
        # logprobs (a request that asks): each token's entry waits in lp_wait until text goes out, then rides with the
        # first text event (content or reasoning) after it.  The tokens of a tool call and of the end of the thinking
        # (a feed that leaves the reasoning: what waits then and is not the reasoning's last text) are none of the text
        # and are dropped, as are those of a stop string that ends the answer with no text before it (with text before
        # it, they go with that text); the end-of-turn token is never one (it ends the answer before its entry)
        n_top, lp_wait = logprobs_asked(sampling), []

        def with_lp(evs, was):
            if n_top is not None:
                crossed = was == "reasoning" and parser.state != "reasoning"
                for ev in evs:
                    if ev.kind in ("content", "reasoning") and ev.text:
                        if crossed and ev.kind == "content":
                            lp_wait.clear()
                            crossed = False
                        ev.logprobs = list(lp_wait)
                        lp_wait.clear()
                    elif ev.kind in ("tool_start", "tool_args", "tool_call"):
                        lp_wait.clear()
                if crossed:
                    lp_wait.clear()
            return evs
        emb = getattr(self.embeddings, "path", None)
        # Identity token: only a DONE line replaces engine.last, so a request that died, errored or was
        # disconnected must not have the PREVIOUS request's decode figures recorded as its own.
        engine_last0 = getattr(self.engine, "last", None)
        counted, my_started, my_first, queued = False, time.time(), None, False
        stat = RequestStat()                            # this request's live figures
        stat.spec_off = request_spec_off(sampling)
        stat.turbo = turbo_live.turbo_of(sampling)      # (the Monitor's Turbo badge shows its options)
        stat.id, stat.api = next(self._req_ids), (sampling or {}).get("_api")
        prio = request_priority(sampling)               # (R122: its place in the FIFO and the preemption rule)
        stat.priority = PRIORITY_NAMES[prio]
        stat.cid = (sampling or {}).get("_cid")
        stat.prompt_tokens, stat.max_tokens, stat.queued_at = len(ids), max_new, my_started
        conv_tail = (sampling or {}).get("_gen_tail")   # the reply's opening tokens (prepare's gt=): not the opening
        conv_tail = conv_tail if isinstance(conv_tail, int) and not isinstance(conv_tail, bool) and conv_tail > 0 else 0
        with self.status_lock:
            stat.conv = self._conv_open(ids, stat.api, (sampling or {}).get("_conv_label"), stat, conv_tail)
            self.status["queued"] += 1
            self.waiting.add(stat)                      # (the Monitor's "Right now": queued until its turn comes)
            queued = True                               # counted as waiting until its turn comes (below, or in finally)
        hold = [False, False]                           # holds the FIFO (fair slices let go of it); counts as active
        asked = time.monotonic()
        try:
            yield from self._queue_wait(hold, cancel, priority=prio,   # (#483: keep-alives while queued; the turn
                                        stat=stat)                     #  taken there; R122: at its own priority)
            with self._turn(hold, cancel, priority=prio):              #  (stat: the had_to_wait mark of a real wait)
                with self.status_lock:
                    self.status["queued"] -= 1
                    queued = False
                waited = time.monotonic() - asked
                if waited >= QUEUE_LOG_S:               # a long wait is worth a line: it is what a wedged FIFO looked like
                    print(f"[saguaro] this request waited {waited:.0f} s for the engine "
                          f"({self.fifo.waiting()} request(s) behind it)", flush=True)
                if cancel.is_set():
                    yield "done", {"finish": "cancel", "completion_tokens": 0}
                    return
                if self.stopping:
                    raise ValueError("the server is stopping: send the request again once the model is back")
                if self.vision is not None and getattr(self.vision, "owed", False):   # memory an encode borrowed, not back yet
                    self.vision.settle()
                self.ensure_engine()   # after an idle unload, or issue #27: it died in an earlier request
                with self.status_lock:
                    self.status.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids), generated=0,
                                       started=time.time(), tool=None, tail="", max_tokens=max_new)
                    self.waiting.discard(stat)          # (out of the queue: its turn, and a row of its own now)
                    self.latest_request = stat
                    stat.phase, stat.prompt_tokens, stat.max_tokens = "reading the prompt", len(ids), max_new
                    stat.started = self.status["started"]
                    stat.at_engine = True                 # (its read's interval opens where the ENGINE took it: the
                    #  drivers' on_read_start marks it, or its first read step - a queue wait inside the turn is no
                    #  read time and must not spread the read's tokens over itself)
                    tls = getattr(self.engine, "_tls", None)   # this request's thread reads the engine's lines: its
                    if tls is not None:                        # PP steps count into the engine-total read rate
                        tls.on_read = lambda n, ts=None, ex=False, prov=False, rms=None, st=stat: \
                            self._note_read_safe(st, n, ts, ex, prov, rms)
                        tls.on_read_start = lambda st=stat: self._read_start(st)
                        tls.on_hb = lambda d, st=stat: self._note_hb_safe(st, d)   # (U5 read heartbeat, RHB lines)
                        tls.read_leg_base = 0                  # (this request's read counting starts from nothing)
                    self.running.add(stat)
                    self.active += 1
                    self.touch_telemetry()
                    counted = hold[1] = True
                    my_started = self.status["started"]
                    self.last_request_at = time.time()
                    self._wall_tick()                     # (the totals' read/write wall clocks started or kept going)
                before = getattr(self.engine, "last", None)
                last_print = time.time()
                # the thinking budget goes to the engine with the request: a reasoning block still open after that many
                # tokens is closed there (its next token is </think>) and the answer follows in the same decode
                in_think = bool(thinking) and self.think_end_id is not None   # (tracked with no budget too, #786)
                budgeted = in_think and self.think_budget > 0   # (0 = no budget: the end token is still tracked)
                # The thinking budget is the config's ("thinking_budget"); a request's max_tokens (or the context
                # left) lowers it only when that output limit is tiny, THINK_TINY_LIMIT (1024) tokens or fewer - a
                # side request such as a title, which must still get an answer: a reserve is kept for it (a quarter,
                # with a preferred floor of SAGUARO_GLM_THINK_RESERVE, 1024, capped at half the limit). A larger
                # limit is the client's cap on the whole output and leaves the budget alone: a chat client's
                # max_tokens=4096 used to cut the thinking to ~3K tokens and close it while the model was still
                # reasoning. The forced </think> also consumes one token of max_new, so account for it.
                think_budget = self.think_budget
                soft = None                             # (the budget closed at a sentence's or line's end: _sliced)
                if budgeted and max_new and int(max_new) <= THINK_TINY_LIMIT:
                    reserve_floor = max(0, int(os.environ.get("SAGUARO_GLM_THINK_RESERVE", "1024") or 1024))
                    reserve = max(int(max_new) // 4, min(reserve_floor, max(1, int(max_new) // 2)))
                    think_budget = max(1, min(think_budget, int(max_new) - reserve - 1))
                elif budgeted:
                    # the engine's own close THINK_GRACE tokens later (at most half the room max_tokens leaves after
                    # the budget); the reasoning ends at its first sentence or line end past the budget (#613)
                    room = (int(max_new) - think_budget - 1) // 2 if max_new else THINK_GRACE
                    grace = max(0, min(THINK_GRACE, room))
                    if grace > 0:
                        soft, think_budget = think_budget, think_budget + grace
                if bool(thinking) and self.ink_close and max_new and int(max_new) <= THINK_TINY_LIMIT:
                    # Inkling 2 (saguaro-ik2 closes no thinking itself): the same reserve for the answer as above -
                    # the server closes the reasoning at a sentence's or line's end past it, at the latest halfway
                    # through what is left, and the answer follows
                    reserve_floor = max(0, int(os.environ.get("SAGUARO_GLM_THINK_RESERVE", "1024") or 1024))
                    reserve = max(int(max_new) // 4, min(reserve_floor, max(1, int(max_new) // 2)))
                    ink_soft = max(1, int(max_new) - reserve - 1)
                    sampling = {**(sampling or {}), "_think_soft": ink_soft, "_think_end": self.ink_close[-1],
                                "_think_close": self.ink_close, "_think_marks": self.ink_marks,
                                "_think_hard": ink_soft + max(0, (int(max_new) - ink_soft - 3) // 2)}
                if budgeted:
                    sampling = {**(sampling or {}), "_think_budget": think_budget, "_think_end": self.think_end_id}
                    if soft is not None:
                        sampling["_think_soft"] = soft
                        if getattr(self.template, "deepseek", False):
                            stops = getattr(self, "_think_stops", None)
                            if stops is None:
                                stops = self._think_stops = think_stop_tables(self.tok)
                            if stops:
                                sampling["_think_stops"] = stops
                # SAGUARO_DS4_RESTORE_POINTS: where the DeepSeek engine keeps restore points for this prompt (never a
                # client's own `_restore`; not with pictures, whose spans the engine reads whole)
                if sampling and "_restore" in sampling:
                    sampling = {k: v for k, v in sampling.items() if k != "_restore"}
                on, tail = restore_point_settings()
                if on and not emb and getattr(self.template, "deepseek", False):
                    pts = deepseek_restore_points(ids, self.ds_user_id, self.ds_system_id, tail,
                                                  assistant_id=self.ds_assistant_id)
                    if pts:
                        print(f"[saguaro] restore points asked: {pts}", file=sys.stderr, flush=True)
                        sampling = {**(sampling or {}), "_restore": pts}

                def attempts(prompt, left):
                    """The engine's tokens.  When its numbers went non-finite (a NaN or an infinity in a route: the
                    engine's ERR says "non-finite") before the first token, the engine has dropped that conversation's
                    state and the request runs once more, its prompt read from token 0."""
                    for again in (False, True):
                        guard = loop_guard.ToolSafeGuard(self.think_end_id, parser.state == "reasoning",
                                                         piece=self._guard_piece()) \
                            if guard_on else None       # (serve/loop_guard.py; a fresh one for a run again; leaves tool calls alone)
                        guards[:] = [guard] if guard is not None else []
                        sliced, got = self._sliced(prompt, left, sampling, cancel, emb, hold, stat, guard), 0
                        try:
                            for t in sliced:
                                got += t is not None
                                yield t
                            return
                        except ValueError as e:
                            if again or got or NONFINITE not in str(e):
                                raise
                            print(f"[saguaro] the engine's numbers went non-finite before the answer started ({e}): "
                                  "running the request again from token 0", flush=True)
                        finally:
                            sliced.close()
                def segments():
                    """attempts(), and after a blocked call (seg["next"]) the engine again on that prompt.  That prompt
                    also holds the server's own tokens (the call's result or KEEP_NOTE, a steer), which max_new - n
                    does not count: the context's room caps it too (with an unlimited max_new, prepare's room, the
                    engine rejected prompt + max_new + CTX_SLACK > context).  SEGMENT separates one call's tokens
                    from the next's: the thinking restarts' drain stops at it, and the token loop skips it."""
                    nonlocal sampling
                    first = True
                    while True:
                        if not first:
                            yield SEGMENT               # (a continuation begins: not a token, not a heartbeat)
                        first = False
                        room = self.engine.max_context - CTX_SLACK - len(seg["prompt"])
                        yield from attempts(seg["prompt"], max(1, min(max_new - n, room)))
                        if seg["next"] is None or cancel.is_set() or self.stopping:
                            return
                        seg["prompt"], seg["next"], seg["ids"] = seg["next"], None, []
                        # (its prompt holds the server's own turns: the QFN engine keeps no checkpoint inside the reply)
                        sampling = {**(sampling or {}), "_cont": True}

                def flush():
                    nonlocal sent_call, answered
                    for ev in pending:
                        sent_call = sent_call or ev.kind == "tool_call"
                        answered = answered or ev.kind != "reasoning"
                        yield "event", ev
                    pending.clear()

                def blocked_call():
                    """The key of the blocked call that alone ends what this engine call wrote, else None."""
                    calls = [ev for ev in pending if ev.kind == "tool_call"]
                    if sent_call or len(calls) != 1 or not calls[0].complete or \
                            any(ev.kind == "content" and ev.text.strip() for ev in pending):
                        return None
                    key = (calls[0].call.name, loop_guard.args_key(calls[0].call.arguments))
                    return key if key in blocked else None

                guard_on = self._loop_guard_on(sampling)
                gen = segments()
                answered = bool(pre_events)             # any answer (text or a call) went out yet
                said_note = False                       # (a stop that already explains itself: the loop guard's)
                for ev in pre_events:                   # a required call's opening (prepare put it in the prompt)
                    yield "event", ev
                try:
                    for t in gen:
                        if t is None:                   # heartbeat while the engine is quiet
                            last_print = self._progress(last_print, stat)
                            pr = getattr(self.engine, "progress", None)   # (read, total, reuse, ms), the PP parse
                            yield "ping", ({"total": pr[1], "cache": pr[2], "processed": pr[0],
                                            "time_ms": round(pr[3])} if pr and len(pr) >= 4 else None)
                            continue
                        if t is LOOP_STOP:              # the loop guard ended the answer (a loop that persisted)
                            finish = "stop"
                            if held:
                                for ev in parser.feed(held):
                                    answered = answered or ev.kind != "reasoning"
                                    yield "event", ev
                                held = ""
                            if not answered:             # (a model with no end-of-thinking token, still thinking)
                                said_note = True
                                yield "event", Event("content", LOOP_STOP_NOTE)
                            break
                        if t is SEGMENT:                # a continuation's first token follows: not the model's
                            continue
                        n += 1
                        if my_first is None:
                            my_first = time.time()
                        raw_ids.append(t)
                        end_think = bool(in_think and t == self.think_end_id)
                        if end_think:
                            in_think = False
                            if think_budget > 0 and n == think_budget + 1:
                                print(f"[saguaro] thinking reached its budget ({think_budget} tokens"
                                      f"{f', {soft} + {think_budget - soft} without a sentence end' if soft else ''})"
                                      ": closed, answering", flush=True)
                        text, stopped = detok.push(t), False
                        if t == self.think_end_id and parser.token_only_end:
                            # the token's own text: the tag is no one's text.  Not only when it closes the reasoning
                            # (end_think): a model that closes twice - or the answer's continuation, a reasoning that
                            # was already closed (see the last restart) - would otherwise put the tag verbatim into
                            # the answer's text (measured: 46 turns of 11-23; an 8-char piece each)
                            text = ""
                        if t in self.stop_ids and parser.state == "reasoning" and not answered and not called \
                                and not pending:
                            # a marker inside the thinking is the model's own text - it quotes the server's own markers
                            # while it works on them - not the turn's end: skip its token and go on.  At the engine's own
                            # EOS token the engine ends there, though: then the server goes on from the quote with a note
                            # as the server's (the thinking continues; the answer's own markers still end the turn, the
                            # answer's)
                            if t not in self.eos_ids:
                                continue
                            if think_resume is None or think_restarts >= THINK_RESTARTS:
                                finish = "stop"
                                break
                            think_restarts += 1
                            last = think_restarts >= THINK_RESTARTS
                            print(f"[saguaro] thinking: an end-of-turn marker inside it is a quote - going on with the "
                                  f"{'answer' if last else 'thinking'} ({think_restarts}. of {THINK_RESTARTS})",
                                  flush=True)
                            ids_k, thinks = think_resume
                            if last:
                                # the restarts are spent, and asking the model to think on only made it plan again
                                # (measured: 565 more tokens of the same, then EOS).  Its thinking is done - it
                                # wrote the calls it means to make in it - so the reasoning is closed here, at the
                                # model's own end-of-thinking token put in the prompt, and the turn goes on from a
                                # thinking that is closed, not one that was cut
                                for ev in parser.close_reasoning():
                                    answered = answered or ev.kind != "reasoning"
                                    yield "event", ev
                                if thinks and self.think_end_id is not None:
                                    ids_k = [*ids_k, self.think_end_id]
                                thinks = False
                                # the engine restarts the thinking's budget at every call, and this continuation is the
                                # answer's: a budget firing here would swap a sampled token for the think-end one, whose
                                # text no one swallows now (a literal </think> in the answer).  The keep path strips
                                # the same keys for the same reason.
                                sampling = {k: v for k, v in (sampling or {}).items()
                                            if k not in ("_think_budget", "_think_soft", "_think_end", "_think_stops")}
                            steer = self.tok.encode(loop_guard.keep_steer(think_restarts)) if thinks else []
                            seg["next"] = list(seg["prompt"]) + seg["ids"] + ids_k + steer
                            # (reason_said stays: the restarts are one continuous thinking, and a call the model
                            # wrote in it is what the salvage at the end reads out)
                            parser, detok, held, said[:] = new_parser(thinks), Detokenizer(self.tok), "", []
                            in_think = thinks and bool(thinking) and self.think_end_id is not None
                            if last:
                                # this engine call's tail is the thinking's own (the continuation above re-reads
                                # it from its prompt): it is dropped - fed to the answer's parser, a stop token in
                                # it would end the turn before the continuation ever ran.  The drain stops at
                                # SEGMENT, so what follows is the continuation's own first token.
                                for t2 in gen:
                                    if t2 is SEGMENT:
                                        break
                            continue
                        # a marker the model wrote as its answer's text - quoted, in a code fence, a tool call's body - is
                        # not the turn's end (the stream's own parser has the same rule for calls)
                        quoted = t in self.stop_ids and t not in self.eos_ids and (
                            parser.call_opened or CALL_START in held + text or parser.in_code_after(held + text))
                        if t in self.stop_ids and not quoted:
                            finish = "stop"
                            key = blocked_call() if blocked else None
                            if key is None and keep is not None and not called and kept < loop_guard.KEEP_MAX and \
                                    (not answered or loop_guard.stops_short("".join(said), keep[2])):
                                # it would end the turn in a conversation that repeats itself: with nothing at all,
                                # or with nothing new (talk about its loop, what it said before), it goes on - the
                                # note answered, never thought about: an answer that already went out keeps its
                                # reasoning closed (the continuation opens none: #640's thinking, retracted)
                                kept += 1
                                self._kept_going(kept, "".join(said))
                                tail = "".join(said)      # the join readable: the continuation starts a new
                                join = "\n\n"[len(tail) - len(tail.rstrip("\n")):] if tail.strip() else ""
                                ids_k, thinks, _ = keep
                                if answered and thinks and self.think_end_id is not None:
                                    ids_k, thinks = [*ids_k, self.think_end_id], False
                                    sampling = {k: v for k, v in (sampling or {}).items()
                                                if k not in ("_think_budget", "_think_soft", "_think_end", "_think_stops",
                                                             "_think_hard", "_think_marks", "_think_close")}
                                steer = self.tok.encode(loop_guard.keep_steer(kept)) if thinks else []
                                seg["next"] = list(seg["prompt"]) + seg["ids"] + ids_k + steer
                                sampling = loop_guard.block_sampling(sampling, kept)
                                parser, detok, held, said[:], reason_said[:] = \
                                    new_parser(thinks), Detokenizer(self.tok), "", [], []
                                in_think = thinks and bool(thinking) and self.think_end_id is not None
                                continue
                            if key is None:
                                break
                            rep, suffix, thinks = blocked[key]
                            pending.clear()             # the call never goes out
                            called = False
                            tries[key] = tries.get(key, 0) + 1
                            self._tool_blocked(rep, tries[key])
                            # the server answers it: the call's result and a note in the prompt, and the model's next
                            # thinking starts a step away from it (never the end: it goes on until it does something
                            # else; from the 2nd time a little hotter)
                            steer = self.tok.encode(loop_guard.block_steer(rep, key[1], tries[key])) if thinks else []
                            seg["next"] = list(seg["prompt"]) + seg["ids"] + suffix + steer
                            sampling = loop_guard.block_sampling(sampling, tries[key])
                            parser, detok, held, said[:], reason_said[:] = new_parser(), Detokenizer(self.tok), "", [], []
                            in_think = bool(thinking) and self.think_end_id is not None
                            continue
                        seg["ids"].append(t)
                        if parser.state == "reasoning":
                            reasoning_tokens += 1
                        if n_top is not None:
                            e = logprob_entry(self.tok, t, n_top)
                            if e is not None:
                                lp_wait.append(e)
                        stops = (self.stop_texts + client_stops) if parser.state != "reasoning" else []
                        if quoted:
                            held = ""           # (its own token is a marker's whole text: not a marker's start)
                        elif stops:
                            held += text
                            hits = [(i, s) for s, i in ((s, held.find(s)) for s in stops) if i >= 0]
                            if hits:
                                cut, s = min(hits)
                                # a marker the model wrote as its text - quoted, in a code fence, a template's docs -
                                # is not the marker: an answer about this very server's markers would end the turn there
                                # (the stream's own parser has the same rule).  The scan keeps the text
                                # and looks for a real one, on the next piece.  A client's own stop string stops
                                # wherever it is, as the API says (a closing fence, a newline in code).
                                if s not in client_stops and (parser.in_code_after(held[:cut]) or parser.call_opened
                                                              or CALL_START in held[:cut]):
                                    text, held, stopped = held, "", False
                                else:
                                    text, held, stopped = held[:cut], "", True
                                if not text:
                                    lp_wait.clear()   # (the marker's own tokens: no text of the answer)
                                matched = s if s in client_stops and s not in self.stop_texts else None
                            else:               # keep back the longest tail that could still grow into a marker
                                k = max((j for s in stops for j in range(1, len(s))
                                         if held.endswith(s[:j])), default=0)
                                text, held = held[:len(held) - k], held[len(held) - k:]
                        was = parser.state
                        evs = with_lp(parser.close_reasoning() if end_think and parser.token_only_end
                                      else parser.feed(text), was)
                        self._say_malformed(parser)
                        self._note(n, evs, stat)
                        last_print = self._progress(last_print, stat)
                        for ev in evs:
                            if ev.kind == "content" and join is not None:
                                # the continuation after a keep: in the client's text it starts a new paragraph (its
                                # own leading line breaks go, the break stands - once, before its first words)
                                t = ev.text.lstrip("\n")
                                if not t:
                                    continue
                                if join:
                                    yield "event", Event("content", join)
                                join = None
                                if t != ev.text:
                                    ev = dataclasses.replace(ev, text=t)
                            if ev.kind == "content":
                                said.append(ev.text)
                            elif ev.kind == "reasoning":
                                reason_said.append(ev.text)
                            elif ev.kind == "tool_call":
                                called = True
                            if blocked and (pending or ev.kind.startswith("tool")):
                                pending.append(ev)      # (a call: held back until the answer ends)
                            else:
                                answered = answered or ev.kind != "reasoning"
                                yield "event", ev
                            if single_tool and ev.kind == "tool_call" and ev.complete:
                                stopped = True
                                break
                        if stopped:
                            finish = "stop"
                            break
                    yield from flush()
                    if held:                    # the run ended on what only looked like a marker's start
                        was = parser.state
                        for ev in with_lp(parser.feed(held), was):
                            yield "event", ev
                        held = ""
                    if finish == "stop" and not answered and not said_note and (reason_said or think_restarts):
                        # DeepSeek writes the call it means to make in its thinking and can end the turn there: the
                        # reasoning holds a whole <tool_call>, the answer holds nothing, and an agent client sees a
                        # turn that did nothing.  A call is read out of what the model wrote there - a fresh parser,
                        # so nothing goes out twice (this runs only when the turn emitted neither text nor a call;
                        # reason_said / think_restarts gate it to a thinking that truly happened, so a client's stop
                        # sequence on an empty answer keeps its old meaning: an empty answer, not this note)
                        salvaged = False
                        thought = "".join(reason_said)
                        if not called and thought.rstrip().endswith(CALL_END):
                            # only a call the reasoning ENDS with: one it wrote and then went on reasoning past is a
                            # draft it thought better of (or a quoted example of this server's own syntax), not a call
                            # it meant to make - and never one the loop guard just blocked (blocked's own key)
                            last = None
                            for ev in new_parser(False).feed(thought):
                                if ev.kind == "tool_call":
                                    last = ev
                            if last is not None and (not blocked or (last.call.name, loop_guard.args_key(last.call.arguments)) not in blocked):
                                called, answered, salvaged = True, True, True
                                yield "event", last
                        if not salvaged:
                            if CALL_START in thought:
                                print(f"[saguaro] a tool call written in the thinking was not read out of it"
                                      f"{' (blocked)' if blocked else ''}: the turn ends with the note", flush=True)
                            # never end empty: an agent client retries an empty turn (the loop guard says its own line)
                            yield "event", Event("content", THINK_STOP_NOTE)
                    if cancel.is_set():
                        finish = "cancel"
                except EngineDied as e:
                    finish = "error"
                    note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
                    print(f"[saguaro] {e}. {note} The next request starts the engine again."
                          f"{' Its log: ' + self.engine.log_path if getattr(self.engine, 'log_path', None) else ''}",
                          flush=True)
                    raise
                except ValueError as e:                 # the engine's ERR line (it may have ended after it)
                    if NONFINITE in str(e) and n > 0:
                        # an answer already under way: it ends here (what was sent stands), and the engine has
                        # dropped the conversation's state, so the next turn reads it from token 0
                        finish = "length"
                        print(f"[saguaro] the engine's numbers went non-finite after {n} tokens ({e}): the answer ends "
                              "there", flush=True)
                        yield from flush()
                        if held:
                            was = parser.state
                            for ev in with_lp(parser.feed(held), was):
                                yield "event", ev
                            held = ""
                    else:
                        finish = "error"
                        print(f"[saguaro] the engine reported an error: {e}", flush=True)
                        if getattr(e, "body", None):                    # (a malformed tool call: the whole of it, #613)
                            print(f"[saguaro] the malformed tool call, whole ({len(e.body)} characters): {e.body!r}", flush=True)
                        raise
                finally:
                    gen.close()                         # STOP+drain to THIS request's DONE while still holding the
                    #                                     fifo, so a stop-token break can't leave the shared engine
                    #                                     queue mid-drain for the next request to read as its own DONE
        except GeneratorExit:                           # the client disconnected mid-stream
            finish = "disconnect"
            raise
        finally:
            if emb:
                Path(emb).unlink(missing_ok=True)
            book = None                                 # (the savings' count: after the lock, it writes a file)
            with self.status_lock:
                stat.at_engine = False
                self.waiting.discard(stat)
                self.current_requests.discard(stat)
                if stat in self.running:
                    self.running.discard(stat)
                    self.ended.appendleft({**stat.view(time.time()), "finish": finish, "time": time.time()})
                self._wall_tick()                       # (its read/write ends here: a clock may close with it)
                self._read_close(stat)                  # (its read's interval too: counted, spread over its span)
                if counted:
                    # only this request's DONE counts: same object means no DONE arrived (death, error, disconnect)
                    last = dict(getattr(self.engine, "last", {}) or {}) \
                        if getattr(self.engine, "last", None) is not engine_last0 else {}
                    started = my_started
                    loaded = str((getattr(self.engine, "info", {}) or {}).get("cvec", 0)) not in ("0", "", "None")
                    hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                    eng_read = getattr(stat, "read_engine", None)
                    if eng_read and eng_read[1] > 0:
                        # the engine's OWN figures for this read (PLAN-qfn-per-request-read-stats: RDONE / BRA -
                        # new_read over read_ms, the time its read actually ran): no guessing from prompt minus
                        # reused, no wall span that mixes queueing and holds into the read.  The live window
                        # already closed on these counts when the read ended; nothing to top up here.
                        read_n = eng_read[0]
                        pf_s = round(read_n / (eng_read[1] / 1000.0), 1) if read_n else None
                    else:
                        read_n = max(0, len(ids) - (last.get("reused") or 0))
                        # the REAL read rate: the new tokens over the wall time from the engine taking the request to its
                        # first token (restores, queueing in the engine and pauses for other chats included) - one count
                        pf_s = round(read_n / (stat.first_token - stat.engine_t0), 1) \
                            if read_n and stat.engine_t0 and stat.first_token and stat.first_token > stat.engine_t0 else None
                        self._read_exact(stat, read_n)      # (status_lock held: the live window's record, the exact count)
                        # the wall clock of its read (first read step to first token): the engine's prompt_ms leaves out
                        # the time its read paused for other chats, so its rate can be 2-3x the real one - never shown above it
                        wall_pf = round(read_n / (stat.first_token - stat.read_t0), 1) \
                            if read_n and stat.read_t0 and stat.first_token and stat.first_token > stat.read_t0 else None
                        if wall_pf is not None and (pf_s is None or wall_pf < pf_s):
                            pf_s = wall_pf
                    row = {
                        "projection": (sampling or {}).get("experimental_speed_projection") is not False
                        if loaded else None,
                        "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                        "api": (sampling or {}).get("_api"),
                        "prefill_tok_s": pf_s, "prefill_shared": bool(stat.read_shared),
                        "prompt_tokens": len(ids), "reused": last.get("reused"), "output_tokens": n,
                        "engine_generated": last.get("generated"),
                        "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                        "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                        if n and last.get("generated") and last.get("decode_ms") else None,
                        "hit_rate": hit_rate,
                        # the drafts offered and kept (the engine's DONE line; None: it reported none)
                        "drafts_offered": last.get("drafts_offered"), "drafts_accepted": last.get("drafts_accepted"),
                        "spec_off": stat.spec_off,
                        "priority": stat.priority,      # (R122: the level it queued and answered at)
                        "turbo": dict(stat.turbo) if stat.turbo else None,
                        "loop_guard": guards[0].summary() if guards and guards[0].events else None}
                    self.history.append(row)
                    t = self.totals
                    ft = stat.first_token
                    if not int(getattr(self.engine, "batch", 0) or 0):
                        self._note_peak(row["decode_tok_s"], n, started)   # (a solo engine's own decode clock)
                    elif n and ft is not None and time.time() - ft >= PEAK_MIN_SPAN_S:   # (QFN batch: its decode_ms
                        self._note_peak(n / (time.time() - ft), n, started)            # is partial - the server's clock)
                    self._note_peak_read(pf_s, read_n, started)   # (the same for the read, from PEAK_MIN_READ_TOKENS on)
                    t["requests"] += 1
                    t["prompt_tokens"] += len(ids)
                    t["reused"] += last.get("reused") or 0
                    t["output_tokens"] += n
                    book = (len(ids), last.get("reused") or 0, n)
                    t["prompt_ms"] += last.get("prompt_ms") or 0.0
                    t["decode_ms"] += last.get("decode_ms") or 0.0
                    fresh = getattr(self.engine, "last", None)
                    if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                        timings = request_timings(len(ids), n, last)
                        if timings is not None and stat.shared():
                            timings["shared"] = stat.shared()
                        self.last_timings = dict(timings, at=int(time.time())) if timings else None
                    self.last_request_at = time.time()
                    now = time.time()
                    el = now - my_started
                    ft = my_first
                    # the engine's decode clock when there is one (the Monitor's row); else the tokens after the first
                    # over the time since it - `n / span` read a one-token answer as thousands of tok/s
                    rate = last["generated"] / (last["decode_ms"] / 1000) if n and last.get("generated") and \
                        last.get("decode_ms") else (n - 1) / (now - ft) if ft and n > 1 and now > ft else 0.0
                    hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                    print(f"[saguaro] done: {n} tokens in {el:.0f} s ({rate:.1f} tok/s) "
                          f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                    if os.environ.get("SAGUARO_DEBUG") and raw_ids:
                        print(f"[saguaro] raw: {self.tok.decode(raw_ids)!r}", flush=True)
                    if stat.conv is not None:
                        self._conv_idle(stat.conv, stat, ids, conv_tail, len(ids), n, last.get("reused") or 0,
                                        pf_s, row["decode_tok_s"])  # (its row waits idle with this turn's figures,
                                        #  its prompt recorded as the key the next turn continues)
                elif stat.conv is not None and stat.conv.stat is stat:
                    stat.conv.stat, stat.conv.last_active = None, time.time()   # (it died: free the row, keep its turns)
                if queued:                              # never started (a cancel, an error while waiting): not waiting now
                    self.status["queued"] -= 1
                if hold[1]:
                    hold[1] = False
                    self.active -= 1
                self.status["busy"] = self.active > 0
            if book is not None and self.savings is not None:
                self.savings.add(*book)
        was = parser.state
        for ev in with_lp(parser.finish(), was):
            yield "event", ev
        self._say_malformed(parser)
        yield "done", {"finish": finish, "completion_tokens": n, "reasoning_tokens": reasoning_tokens,
                       "reused": (timings or {}).get("cache_n", 0),
                       "timings": timings, "stop_sequence": matched if finish == "stop" else None}


def own_prompt_figures(last: dict, n_prompt: int) -> dict:
    """A request's figures when its last engine call went on from what it had written (after the loop guard or a fair
    slice): that call's prompt was the request's prompt plus the tokens kept, all of them in the conversation cache,
    so its prompt_tokens and reused count those too - a guarded answer reported 5,919 cached tokens of a 246-token
    prompt.  The request's own: its prompt, and what the cache held of it (the call's reused less the kept tokens)."""
    extra = int(last.get("prompt_tokens") or 0) - n_prompt
    reused = max(0, min(n_prompt, int(last.get("reused") or 0) - max(0, extra)))
    return dict(last, prompt_tokens=n_prompt, reused=reused)


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """One request's `timings` in llama.cpp's names (what its clients show as speed), from the engine's own clock
    (StrataEngine.last): prompt_n is what was read, cache_n what the conversation cache already held.  None when the
    engine keeps no clock (MockEngine)."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    prompt_n, prompt_ms, decode_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"]), float(last.get("decode_ms") or 0)
    decoded = int(last.get("generated") or generated)          # the engine's count gives its rate, as /metrics does
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1),
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms > 0 else None,
            # the speculative drafts, as llama.cpp names them (from PR #83, @mikicvi): only when the engine reported them
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """One compact line per request while diagnosing blank/empty turns. Set SAGUARO_DEBUG=1 to enable."""
    if not os.environ.get("SAGUARO_DEBUG"):
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[saguaro] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)


# ------------------------------------------------------------------------------------------------ MCP tool loop
def run_with_mcp(svc: Service, hub, messages, tools, kw, ids, thinking, max_new, max_req, sampling, cancel,
                 mcp_names):
    """Service.run with the MCP tools executed here: the model writes a call to an MCP tool, the server runs it, adds
    the call and its result to the conversation and lets the model continue - up to `max_rounds` times.  Yields what
    Service.run yields (text, thinking, the request's own tool calls) plus ("mcp", {...}) for the tool activity, and
    one ("done", ...) at the very end with the output tokens of every round.

    `mcp_names`: the MCP tools this request offered; any other call is one of the request's own tools and ends the
    turn as always (the client answers it).  MCP calls written in the same answer are then not run (their results
    could not reach the model before the client's)."""
    max_rounds = int(hub.settings["max_rounds"])
    total, rounds, done = 0, 0, None
    messages = list(messages)
    while True:
        text, reasoning, calls, own_calls = [], [], [], 0
        with contextlib.closing(svc.run(ids, thinking, tools, max_new, sampling, cancel)) as _inner:
            # (closed on any exit: Python 3.12 ran the inner STOP + drain only when the outer one was freed)
            for kind, x in _inner:
                if kind == "done":
                    done = x
                    continue
                if kind == "event":
                    ev: Event = x
                    if ev.call is not None and ev.call.name in mcp_names:
                        if ev.kind == "tool_start":          # the model has started writing a call: say so at once
                            yield "mcp", {"event": "start", "id": ev.call.id, "name": ev.call.name}
                        elif ev.kind == "tool_call":
                            calls.append(ev.call)
                        continue                             # its argument pieces are not streamed to the client
                    if ev.kind == "tool_call":
                        own_calls += 1
                    elif ev.kind == "content":
                        text.append(ev.text)
                    elif ev.kind == "reasoning":
                        reasoning.append(ev.text)
                yield kind, x
        total += done["completion_tokens"]
        run_them = calls and not own_calls and done["finish"] == "stop" and not cancel.is_set()
        if run_them and rounds >= max_rounds:
            yield "mcp", {"event": "limit", "max_rounds": max_rounds}
            run_them = False
        if not run_them:
            for c in calls:                              # announced, never run: close them in the client's view
                yield "mcp", {"event": "result", "id": c.id, "ok": False, "skipped": True, "text": "not run",
                              "chars": 0, "truncated": False, "ms": 0}
            break
        rounds += 1
        results = []
        for c in calls:
            s, tool = hub.routes().get(c.name, (None, c.name))
            yield "mcp", {"event": "call", "id": c.id, "name": c.name, "server": s.name if s else None,
                          "tool": tool, "arguments": c.arguments, "round": rounds}
            # The call runs on a thread while this generator keeps yielding heartbeats: they reach the client as
            # keep-alives, which is how a client that went away (the web app's Stop) is noticed during a slow tool.
            box = {}

            def work(c=c, box=box):
                try:
                    box["r"] = hub.call(c.name, c.arguments, cancel)
                except McpCancelled:
                    box["cancelled"] = True
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            try:
                while worker.is_alive():
                    worker.join(1.0)
                    if worker.is_alive():
                        yield "ping", None
            except GeneratorExit:
                cancel.set()                             # the client is gone: stop the tool too
                raise
            if "r" not in box:
                break
            r = box["r"]
            print(f"[saguaro] tool {c.name}: {'ok' if r['ok'] else 'error'}, {r['chars']:,} characters in "
                  f"{r['ms'] / 1000:.1f} s{' (truncated for the model)' if r['truncated'] else ''}", flush=True)
            results.append(r["text"])
            yield "mcp", {"event": "result", "id": c.id, **{k: r[k] for k in ("ok", "text", "chars", "truncated", "ms")}}
        if cancel.is_set() or len(results) < len(calls):
            done = {**done, "finish": "cancel"}
            break
        deepseek = getattr(svc.template, "deepseek", False)
        messages.append({"role": "assistant", "content": "".join(text) if deepseek else "".join(text).strip(),
                         **({"reasoning_content": "".join(reasoning) if deepseek else "".join(reasoning).strip()}
                            if reasoning else {}),
                         "tool_calls": [{"id": c.id, "function": {"name": c.name, "arguments": c.arguments}}
                                        for c in calls]})
        messages += [{"role": "tool", "tool_call_id": c.id, "content": r} for c, r in zip(calls, results)]
        # a required tool call applies to the first answer only: after the tools ran, the model answers freely
        kw = {k: v for k, v in kw.items() if k not in ("_force_tool", "_prefill", "_blocked")}
        sampling = {k: v for k, v in (sampling or {}).items() if k != "_prefill"}
        ids, thinking, max_new = svc.prepare(messages, tools, kw, max_req, guard=svc._loop_guard_on(sampling))
        sampling["_blocked"] = kw.get("_blocked")
        sampling["_think_resume"] = kw.get("_think_resume")
    yield "done", {**done, "completion_tokens": total, "prompt_tokens": len(ids)}


# ------------------------------------------------------------------------------------------------ OpenAI
def json_from_text(text: str) -> str:
    """The JSON in an answer that was asked for JSON: the answer itself when it parses, else the inside of its first
    code fence, else the first object or array in it; the text as it was when none parses."""
    s = text.strip()
    candidates = [s]
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", s, re.S)
    if fence:
        candidates.append(fence.group(1).strip())
    for c in candidates:
        try:
            json.loads(c)
            return c
        except ValueError:
            pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\[{]", s):
        try:
            _, end = dec.raw_decode(s, m.start())
            return s[m.start():end]
        except ValueError:
            continue
    return text


def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, run=None):
    """`run`: the events to send instead of Service.run's (run_with_mcp); its ("mcp", {...}) items become chunks with
    an empty delta and a `strata_mcp` field, which only the web app reads."""
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    if isinstance(req, dict):
        req["_cid"] = cid                                # (its Monitor row carries it: the web app finds its own chat)

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": svc.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    # response_format json_object / json_schema: the answer is held and its JSON sent alone at the end (Strata #762)
    rf = req.get("response_format")
    json_mode = isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema")
    held, held_lp = [], []
    # logprobs (a request that asks): a text chunk's choice carries its tokens' entries - OpenAI's `content` for the
    # answer, and `reasoning_content` (as the delta's field of that name) for the reasoning
    want_lp = logprobs_asked(req) is not None

    def text_chunk(delta, entries, key="content"):
        c = chunk(delta)
        if want_lp:
            lp = {"content": [], "refusal": None}
            lp[key] = list(entries or [])
            c["choices"][0]["logprobs"] = lp
        return c

    with contextlib.closing(run if run is not None else svc.run(ids, thinking, tools, max_new, req, cancel)) as _inner:
        # (closed on any exit: Python 3.12 ran the inner STOP + drain only when the outer one was freed)
        for kind, x in _inner:
            # ^ kind "ping": x carries the prompt's progress (total, cache, processed, time_ms) while the engine reads,
            #   or None while this request just waits (queued again, an MCP round's engine work)
            if kind == "ping":
                if req.get("return_progress") and req.get("stream") and isinstance(x, dict):
                    c = chunk({})
                    c["prompt_progress"] = x          # llama.cpp's field: the prompt read's progress, for a client's bar
                    yield c
                else:
                    yield None
            elif kind == "mcp":
                c = chunk({})
                c["strata_mcp"] = x
                yield c
            elif kind == "event":
                ev: Event = x
                if ev.kind == "reasoning" and ev.text:
                    yield text_chunk({"reasoning_content": ev.text}, ev.logprobs, "reasoning_content")
                elif ev.kind == "content" and ev.text:
                    if json_mode:
                        held.append(ev.text)
                        held_lp += ev.logprobs or []
                        continue
                    yield text_chunk({"content": ev.text}, ev.logprobs)
                elif ev.kind == "tool_start":
                    streamed[ev.call.id] = calls
                    calls += 1
                    yield chunk({"tool_calls": [{"index": streamed[ev.call.id], **ev.call.to_openai(arguments="")}]})
                elif ev.kind == "tool_args":
                    yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "function": {"arguments": ev.text}}]})
                elif ev.kind == "tool_call" and ev.call.id in streamed:
                    continue
                elif ev.kind == "tool_call":
                    yield chunk({"tool_calls": [{"index": calls, **ev.call.to_openai()}]})
                    calls += 1
            else:
                if held:
                    yield text_chunk({"content": json_from_text("".join(held))}, held_lp)
                finish = "tool_calls" if calls and x["finish"] == "stop" else {"cancel": "stop"}.get(x["finish"], x["finish"])
                last = chunk({}, finish)
                pt = x.get("prompt_tokens", len(ids))     # after MCP rounds: the last round's prompt
                last["usage"] = {"prompt_tokens": pt, "completion_tokens": x["completion_tokens"],
                                 "total_tokens": pt + x["completion_tokens"],
                                 # the part of the prompt the conversation cache already held (OpenAI's field)
                                 "prompt_tokens_details": {"cached_tokens": x.get("reused") or 0}}
                if x.get("timings"):
                    last["timings"] = x["timings"]          # llama.cpp's field: the speed its clients show
                yield last


def carry_embeddings(local, path, chunks):
    """`Service.embeddings` is thread-local (the request thread prepares the prompt and sets the image embeddings
    file).  A strata_resume answer runs its chunks on the job thread, which would see no file: the image's place
    in the prompt then went to the engine as plain pad tokens and the model answered about a black / blank picture."""
    local.path = path                                  # runs on the thread that first pulls from the generator
    yield from chunks


def watch_client(handler, cancel: threading.Event, every: float = 0.5) -> None:
    """A request's client watched on its own thread (#314): an API request writes nothing while it waits in the queue,
    and a non-streamed one nothing until its answer is done, so a client that timed out or went away was only noticed
    at its first write - queued requests nobody waited for any more piled up for hours, and each then ran its whole
    answer.  A connection the client closed (or reset) reads as end-of-file: `cancel` is set, the request leaves the
    queue (or its engine work stops).  Data the client sent meanwhile (its next request on a kept-alive connection) is
    not an end.  The watch ends with the request's cancel, the connection, or the next request on that connection
    (one watch a connection)."""
    import select
    import socket
    conn, token = handler.connection, object()
    handler._client_watch = token

    def watch():
        while not cancel.wait(every) and getattr(handler, "_client_watch", None) is token:
            try:
                ready, _, _ = select.select([conn], [], [], 0)
                if ready and conn.recv(1, socket.MSG_PEEK) == b"":
                    cancel.set()
                    return
            except ConnectionError:                     # (reset by the client: gone)
                cancel.set()
                return
            except (OSError, ValueError):               # (closed by this server: the request is done)
                return
    threading.Thread(target=watch, name="client-watch", daemon=True).start()


class StreamJob:
    """A web-app answer that outlives its connection (the request's `strata_resume`): a thread runs its chunks into
    `items` and any reader streams them from an index (GET /v1/strata/stream?id=&from=), so a phone whose screen
    sleeps mid-answer reconnects and carries on instead of losing it.  Only POST /v1/strata/cancel (the app's Stop)
    cancels it - a dropped connection does not."""

    ORPHAN_S = 300   # a running answer nobody has read for this long (the page was closed) is cancelled

    def __init__(self, jid: str, chunks, cancel: threading.Event):
        self.id, self.cancel = jid, cancel
        self.items: list = []
        self.done, self.ended = False, None
        self.readers, self.unread_since = 0, time.time()
        self.cv = threading.Condition()
        threading.Thread(target=self._run, args=(chunks,), daemon=True, name="strata-job").start()
        threading.Thread(target=self._watch, daemon=True, name="strata-job-watch").start()

    def reader(self, on: bool):
        with self.cv:
            self.readers += 1 if on else -1
            if self.readers == 0:
                self.unread_since = time.time()

    def _watch(self):
        while not self.done:
            time.sleep(5)
            with self.cv:
                orphan = not self.done and self.readers == 0 and time.time() - self.unread_since > self.ORPHAN_S
            if orphan:
                print(f"[saguaro] nobody read answer {self.id} for {self.ORPHAN_S} s: stopped", flush=True)
                self.cancel.set()
                return

    def _add(self, c):
        with self.cv:
            self.items.append(c)
            self.cv.notify_all()

    def _run(self, chunks):
        try:
            for c in chunks:
                if c is not None:                      # heartbeats: each reader writes its own
                    self._add(c)
        except EngineDied as e:
            self._add({"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}})
        except Exception as e:  # noqa: BLE001 - the engine's ERR after the stream started, or anything else
            self._add({"error": {"type": "server_error", "message": str(e)}})
        finally:
            with self.cv:
                self.done, self.ended = True, time.time()
                self.cv.notify_all()

    def wait(self, n: int, timeout: float):
        """The items from index n (waiting up to `timeout` for one) and whether the answer is complete."""
        with self.cv:
            if len(self.items) <= n and not self.done:
                self.cv.wait(timeout)
            return self.items[n:], self.done


JOBS: dict[str, StreamJob] = {}
JOBS_LOCK = threading.Lock()


def jobs_put(job: StreamJob, places: int = 1) -> None:
    """Keeps the live answers and the finished ones of the last 15 minutes (at most 8, or the engine's places + 4)."""
    with JOBS_LOCK:
        now = time.time()
        if places <= 1:                            # one answer at a time: a new one ends one nobody reads any more
            for j in JOBS.values():                # (several at once: each chat's answer runs on - an unread one
                if not j.done and j.readers == 0:  #  ends after StreamJob.ORPHAN_S - a reconnecting page is not cut)
                    j.cancel.set()
        for k in [k for k, j in JOBS.items() if j.done and now - (j.ended or now) > 900]:
            del JOBS[k]
        while len(JOBS) >= max(8, places + 4):
            old = min(JOBS.values(), key=lambda j: (not j.done, j.ended or now))
            del JOBS[old.id]
        JOBS[job.id] = job


def openai_collect(chunks, logprobs: bool = False) -> dict:
    """The chunks as one chat.completion; `logprobs` (the request asked): the choice's logprobs, every text chunk's
    entries in order - OpenAI's `content`, and `reasoning_content` when there was reasoning."""
    content, reasoning, by_index, last, mcp = [], [], {}, None, []
    lp_content, lp_reasoning = [], []
    for c in chunks:
        if c is None or c.get("prompt_progress"):  # a heartbeat, a prompt-progress chunk: not answer content
            continue
        if c.get("strata_mcp"):
            mcp.append(c["strata_mcp"])
        lp = c["choices"][0].get("logprobs") or {}
        lp_content += lp.get("content") or []
        lp_reasoning += lp.get("reasoning_content") or []
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            cur = by_index.setdefault(tc.get("index", len(by_index)), {"id": None, "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            if tc.get("namespace") is not None:
                cur["namespace"] = tc["namespace"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    msg = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    if mcp:
        msg["strata_mcp"] = mcp
    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
           "usage": last["usage"]}
    if logprobs:
        out["choices"][0]["logprobs"] = {"content": lp_content, "refusal": None,
                                         **({"reasoning_content": lp_reasoning} if lp_reasoning else {})}
    if last.get("timings"):
        out["timings"] = last["timings"]
    return out


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel):
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": svc.model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False

    def close():
        return ("content_block_stop", {"type": "content_block_stop", "index": index})

    streamed = set()
    with contextlib.closing(svc.run(ids, thinking, tools, max_new, req, cancel)) as _inner:
        # (closed on any exit: Python 3.12 ran the inner STOP + drain only when the outer one was freed)
        for kind, x in _inner:
            if kind == "ping":
                yield None
                continue
            if kind == "event":
                ev: Event = x
                if ev.kind == "tool_args":
                    yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                                  "delta": {"type": "input_json_delta", "partial_json": ev.text}}
                    continue
                if ev.kind == "tool_call" and ev.call.id in streamed:
                    continue
                want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use", "tool_start": "tool_use"}[ev.kind]
                if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                    continue
                if ev.kind == "tool_start":
                    streamed.add(ev.call.id)
                    used_tool = True
                if open_kind != want or want == "tool_use":
                    if open_kind is not None:
                        yield close()
                    index += 1
                    open_kind = want
                    block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                             "text": {"type": "text", "text": ""},
                             "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "", "name":
                                          ev.call.name if ev.call else "", "input": {}}}[want]
                    yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
                if want == "thinking":
                    yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                                  "delta": {"type": "thinking_delta", "thinking": ev.text}}
                elif want == "text":
                    yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                                  "delta": {"type": "text_delta", "text": ev.text}}
                elif ev.kind == "tool_start":
                    pass                                # its input follows as tool_args pieces
                else:
                    used_tool = True
                    yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                        "type": "input_json_delta", "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}}
            else:
                if open_kind is not None:
                    yield close()
                stop = "tool_use" if used_tool and x["finish"] == "stop" else \
                    "stop_sequence" if x.get("stop_sequence") else \
                    {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
                # the final counts, Anthropic's way: input_tokens leaves out what the conversation cache already held,
                # which is cache_read_input_tokens (message_start could only say the whole prompt)
                reused = min(x.get("reused") or 0, len(ids))
                yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop,
                                                                           "stop_sequence": x.get("stop_sequence")},
                                        "usage": {"input_tokens": len(ids) - reused, "cache_read_input_tokens": reused,
                                                  "output_tokens": x["completion_tokens"]}}
                yield "message_stop", {"type": "message_stop"}


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    for item in events:
        if item is None:                           # a heartbeat
            continue
        name, e = item
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:                                  # input_json_delta pieces: parsed when complete
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            b["input"] = json.loads(b.pop("_json") or "{}")
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
    msg["content"] = blocks
    return msg


# ------------------------------------------------------------------------------------------------ HTTP
def _backlog() -> int:
    try:
        return max(5, int(os.environ.get("SAGUARO_HTTP_BACKLOG") or 256))
    except ValueError:
        return 256


def request_timeout() -> float:
    """Seconds one socket read or write may wait (a client that sends half a request and nothing more must not hold a
    thread for ever): 300 by default, SAGUARO_REQUEST_TIMEOUT_S changes it.  Per operation: a long answer that writes
    nothing while it thinks is not cut."""
    try:
        s = float(os.environ.get("SAGUARO_REQUEST_TIMEOUT_S", "0"))
    except ValueError:
        s = 0.0
    return s if s > 0 else 300.0


def body_limit() -> int:
    """The most a request body may hold, in bytes (it is read into memory): 256 MiB, a million-token conversation with
    room to spare; SAGUARO_MAX_BODY_MIB changes it.  A larger one is answered 413 before it is read (Strata #893)."""
    try:
        mib = int(os.environ.get("SAGUARO_MAX_BODY_MIB", "0"))
    except ValueError:
        mib = 0
    return (mib if mib > 0 else 256) << 20


class BadBody(Exception):
    """A request body that cannot be read (a bad Content-Length, a malformed or oversized chunked body)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def key_list(value) -> list[str]:
    """The API keys the server accepts (Strata #1344): llama.cpp's form, several separated by commas ("k1,k2"), or a
    list in the config ("api_key": ["k1", "k2"], for a key that contains a comma)."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(k) for k in value if str(k)]
    return [k.strip() for k in str(value).split(",") if k.strip()]


def parse_cors_origins(value) -> list[str]:
    """The config's "cors_origins": a list of page origins such as "http://127.0.0.1:8090" (or ["*"]) -> the list
    without trailing slashes; missing or empty = none (no CORS headers at all).  A ValueError says what is wrong."""
    if value is None or value == "":
        return []
    if not isinstance(value, list) or not all(isinstance(o, str) and o.strip() for o in value):
        raise ValueError('"cors_origins" must be a list of origins like "http://127.0.0.1:8090" (or ["*"])')
    return [o.strip().rstrip("/") for o in value]


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection
        answer_started = False                              # a malformed request's 400 only replaces an answer not begun
        timeout = request_timeout()

        def log_message(self, fmt, *args):
            pass

        def send_response(self, code, message=None):
            self.answer_started = True
            super().send_response(code, message)

        def _body(self) -> bytes:
            """The request body: Content-Length checked first (a bad one is a 400, one over body_limit() a 413 before
            anything is read), or a Transfer-Encoding: chunked one from a relay that does not buffer it."""
            limit = body_limit()
            te = self.headers.get("Transfer-Encoding", "") or ""
            if te.split(",")[-1].strip().lower() == "chunked":
                parts, total = [], 0
                while True:
                    line = self.rfile.readline(1025)
                    try:
                        size = int(line.split(b";", 1)[0].strip(), 16)
                    except ValueError:
                        raise BadBody(400, "malformed chunked request body") from None
                    if size < 0 or not line.endswith(b"\n"):
                        raise BadBody(400, "malformed chunked request body")
                    if size == 0:
                        for _ in range(64):                  # the trailers, up to the blank line
                            if self.rfile.readline(8193).strip() == b"":
                                break
                        return b"".join(parts)
                    total += size
                    if total > limit:
                        self.close_connection = True
                        raise BadBody(413, f"the request body is larger than {limit >> 20} MiB "
                                           "(SAGUARO_MAX_BODY_MIB raises the limit)")
                    piece = self.rfile.read(size)
                    if len(piece) != size or self.rfile.read(2) != b"\r\n":
                        raise BadBody(400, "malformed chunked request body")
                    parts.append(piece)
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadBody(400, "invalid Content-Length") from None
            if length < 0:
                raise BadBody(400, "invalid Content-Length")
            if length > limit:
                self.close_connection = True                 # not read: the connection ends with the answer
                raise BadBody(413, f"the request body is larger than {limit >> 20} MiB "
                                   "(SAGUARO_MAX_BODY_MIB raises the limit)")
            return self.rfile.read(length)

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _cid_paused(self, req):
            """R122: is this request the one currently paused for a higher priority? Its answer carries the
            conversation id, and the live state says whose turn paused."""
            cid = (req or {}).get("_cid")
            return bool(cid) and any(getattr(s, "paused_for_priority", False) and getattr(s, "cid", None) == cid
                                     for s in list(svc.running))

        def _cid_waited(self, req):
            """R122 (Ken's decision on the trigger): is this request one that had to wait - queued in the FIFO at
            least once, or paused for a higher priority? Only those get the padding spaces and the streams' empty
            events; one that never waited is framed byte for byte as before R122, a slow answer and a long prompt
            read included. The state rides on the request's own live row: `had_to_wait` (set where it first waits
            for a turn or gives a place back) and `paused_for_priority`."""
            cid = (req or {}).get("_cid")
            return bool(cid) and any(getattr(s, "cid", None) == cid and
                                     (s.had_to_wait or s.paused_for_priority)
                                     for s in list(svc.running) + list(svc.waiting))

        def _collect_wait(self, make, name: str, cancel, paused, waited, code_of=lambda out: 200):
            """A non-streaming answer collected on a side thread. R122 (Ken's decision on the trigger): a client
            whose request waits for the model sees nothing of it - and only such a request changes its framing: the
            200 and headers go out as soon as this request is PAUSED (`paused()`) - a priority pause starts the
            padding at once - or `SAGUARO_WAIT_SPACE_S` has gone by since it first WAITED (`waited()`: queued at
            least once, or paused), and a space follows every SAGUARO_WAIT_EVENT_S until the JSON body: spaces
            before a JSON body are valid, so the client keeps waiting instead of dropping a request that is only
            waiting. A request that never waits answers whole exactly as before R122: no early 200, no chunked
            spaces, and an error keeps its real status however long the answer took. From the headers out an error
            cannot change the status any more - the log says so once, and the error's JSON rides as the body."""
            box = {}
            emb = getattr(svc.embeddings, "path", None)    # (thread-local: this thread prepared the prompt's pictures)

            def work():
                svc.embeddings.path = emb                  # (the answer runs here: without it a picture went to the
                try:                                       #  engine as bare pad tokens - see carry_embeddings)
                    box["body"] = make()
                except Exception as e:                     # any of them (a GpuBusy or a QueueWaitExpired is a plain
                    box["error"] = e                       #  RuntimeError): the thread must not die with the box
                    #  empty - the main thread would raise a KeyError no client asked for. After the headers an
                    #  error can only ride in the body; before them the caller re-raises it as its own status
            t = threading.Thread(target=work, daemon=True)
            t.start()
            if self.request_version == "HTTP/1.0":   # (an HTTP/1.0 client parses no chunked body: the padding
                t.join()                             #  spaces could not be framed for it - wait for the JSON
                if "error" in box:                   #  with the status still free, as this did before R122)
                    raise box["error"]
                self._json(code_of(box["body"]), box["body"])
                return
            # (R122, Ken's decision on the trigger): the padding is for a request that waits for the model -
            # SAGUARO_WAIT_SPACE_S counts from the first moment `waited()` says it queued or paused, and a
            # PAUSE starts it at once. A request that never waited answers whole exactly as before R122,
            # however slow it is: no early 200, and its error keeps its real status until the JSON is sent.)
            started = None
            while t.is_alive():
                if started is None:
                    if paused():
                        break                              # paused for a higher priority: the headers go now
                    if waited():
                        started = time.monotonic()         # queued at least once: the space counts from here
                if started is not None and time.monotonic() - started >= wait_space_seconds():
                    break
                t.join(0.05)
            if not t.is_alive():
                if "error" in box:
                    raise box["error"]
                self._json(code_of(box["body"]), box["body"])
                return
            print("[saguaro] " + name + ": the answer is still waiting - the status and headers go out now; "
                  "an error after this cannot change the status any more", flush=True)
            self.answer_started = True
            self.protocol_version = "HTTP/1.1"             # (chunked, so the spaces are a keep-alive body)
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while t.is_alive():
                    t.join(wait_event_seconds())
                    if t.is_alive():
                        self.wfile.write(b"1\r\n \r\n")
                        self.wfile.flush()
            except OSError:
                cancel.set()                               # the client went away: stop the answer
                return
            out = ({"error": {"type": "server_error", "message": str(box["error"])}} if "error" in box
                   else box["body"])
            data = json.dumps(out, ensure_ascii=False).encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
            self.wfile.flush()

        def _llamacpp_model(self, op: str):
            """llama.cpp's /models/load and /models/unload: Open WebUI's Models page (load / switch)."""
            if not getattr(svc, "model_management", False):
                self._json(404, {"error": {"message": "model management is off (remove \"model_management\": false, or "
                                                      "SAGUARO_MODEL_MANAGEMENT=1, turns it on)"}})
                return
            try:
                body = json.loads(self._body().decode() or "{}")
            except Exception:  # noqa: BLE001 - no body: the running model
                body = {}
            name = (body or {}).get("model") or svc.model
            if name != svc.model:
                other = model_api.find_config(svc.models_root(), name)
                run = next((m for m in svc.model_listing() if m["running"]), None)
                if other is not None and run is not None and other.name == run["file"]:
                    name = svc.model                      # (the running config by its own name: the model below)
                elif op == "load" and other is not None:  # another installed model: the swap (a load on request)
                    self._mapi_target(name, load=True)
                    return
                else:
                    self._json(404, {"error": {"message": "model not found"}})
                    return
            if op == "unload":
                state = svc.unload()
                if state == "busy":
                    self._json(409, {"error": {"message": "a request is running or waiting; unload when idle"}})
                elif state == "unsupported":
                    self._json(400, {"error": {"message": "this engine cannot unload"}})
                else:
                    self._json(200, {"model": name, "status": {"value": "unloaded"}})
                return
            if hasattr(svc.engine, "alive") and svc.engine.alive():
                self._json(200, {"model": name, "status": {"value": "loaded"}})
                return
            if not svc.fifo.acquire(blocking=True):
                self._json(503, {"error": {"message": "the engine has no free place to start"}})
                return
            try:
                svc.ensure_engine()
            except Exception as e:  # noqa: BLE001 - the reason goes back to the caller
                self._json(503, {"error": {"message": f"the engine did not start: {e}"}})
                return
            finally:
                svc.fifo.release()
            self._json(200, {"model": name, "status": {"value": "loaded"}})

        # ------------------------------------------------------------ the model API (serve/model_api.py)
        def _mapi_gate(self, manage: bool = False) -> bool:
            """Model management on, the API key given when one is set; `manage` (pull, delete, swap): without a key only
            from this PC - nobody else on the network may download or delete models."""
            if not getattr(svc, "model_management", False):
                self._json(404, {"error": "model management is off (remove \"model_management\": false, or "
                                          "SAGUARO_MODEL_MANAGEMENT=1, turns it on)"})
                return False
            if manage and not svc.api_key:
                ip = self.client_address[0] if self.client_address else ""
                if ip not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
                    self._json(403, {"error": "set an API key to download, delete or switch models over the network"})
                    return False
            return True

        def _mapi_json(self) -> dict:
            req = json.loads(self._body() or b"{}")
            if not isinstance(req, dict):
                raise ValueError("the request body must be a JSON object")
            return req

        def _mapi_ps(self) -> dict:
            loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
            out = []
            run = next((m for m in svc.model_listing() if m["running"]), None)
            lock = getattr(svc, "_engine_start", None)
            starting = run is not None and loaded and lock is not None and lock.locked()   # (the weights are loading)
            if run is not None and loaded and not starting:
                out.append(model_api.ps_entry(run, svc.expires_at(), 0))
            res = {"models": out}
            restart = getattr(svc, "model_restart", None)
            if restart:
                res["loading"] = {"model": model_api.public_name(Path(restart["to"])) + ":latest", "since": restart["started"]}
            elif starting:
                res["loading"] = {"model": model_api.public_name(Path(run["file"])) + ":latest", "since": time.time()}
            elif run is not None and not loaded:
                res["unloaded"] = model_api.public_name(Path(run["file"])) + ":latest"
            return res

        def _mapi_get(self, path: str):
            if not self._mapi_gate():
                return
            if path == "/api/version":
                self._json(200, {"version": "0.0.0-maya"})
            elif path == "/api/tags":
                self._json(200, {"models": [model_api.tags_entry(m) for m in svc.model_listing()]})
            elif path == "/api/ps":
                self._json(200, self._mapi_ps())
            elif path == "/api/catalog":
                self._json(200, {"models": model_pull.catalog(svc.models_root()), "pull": svc.pull_slot().snapshot(),
                                 "restarts": bool(os.environ.get("SAGUARO_RESTART_ON_UPDATE"))})
            elif path == "/api/pull/status":
                self._json(200, {"pull": svc.pull_slot().snapshot()})

        def _mapi_post(self, path: str):
            try:
                req = self._mapi_json()
                if path == "/api/show":
                    name = req.get("model") or req.get("name")
                    p = model_api.find_config(svc.models_root(), name)
                    if p is None:
                        self._json(404, {"error": f"model '{name}' not found"})
                        return
                    d = models.describe(p, svc.config_path)
                    plan = models.delete_plan(svc.models_root(), p) if req.get("verbose") else None
                    self._json(200, model_api.show_entry(d, plan))
                elif path == "/api/pull":
                    if self._mapi_gate(manage=True):
                        self._pull(req)
                elif path == "/api/pull/cancel":
                    if self._mapi_gate(manage=True):
                        self._json(200, {"cancelled": svc.pull_slot().cancel()})
                elif path == "/api/delete":
                    if self._mapi_gate(manage=True):
                        self._mapi_delete(req)
                elif path in ("/api/chat", "/api/generate"):
                    self._mapi_chat(path.rsplit("/", 1)[1], req)
            except ValueError as e:
                if self.answer_started:
                    raise
                self._json(400, {"error": str(e)})

        def _mapi_delete(self, req: dict):
            name = req.get("model") or req.get("name")
            try:
                out = svc.delete_installed(name)
            except KeyError:
                self._json(404, {"error": f"model '{name}' not found"})
                return
            except (ValueError, OSError) as e:
                self._json(409, {"error": str(e)})
                return
            self._json(200, {"status": "success", "freed": out["freed"]})

        def _pull(self, req: dict):
            name = req.get("model") or req.get("name")
            slot = svc.pull_slot()
            try:
                pull = slot.current if slot.busy() and str(name or "").removesuffix(":latest").lower() == slot.current.id \
                    else slot.start(name, models.read(Path(svc.config_path)) if svc.config_path else None)
            except ValueError as e:
                self._json(409 if slot.busy() else 400, {"error": str(e)})
                return
            if req.get("detach"):                            # (the Models page: starts it, then polls /api/pull/status)
                self._json(202, {"pull": pull.snapshot()})
                return
            if req.get("stream") is False:
                last = None
                for last in model_pull.ndjson_events(pull, 1.0):
                    pass
                self._json(200 if last and "error" not in last else 500, last or {"status": "success"})
                return
            self._ndjson_start()
            try:
                for ev in model_pull.ndjson_events(pull, 1.0):
                    self.wfile.write((json.dumps(ev) + "\n").encode())
                    self.wfile.flush()
            except OSError:
                pass                                         # (the download goes on without the client)

        def _ndjson_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.answer_started = True

        def _mapi_target(self, name, load: bool = False) -> bool:
            """The model a chat / generate names: True when it is the one running (or none named); otherwise the answer
            is sent - 404, or the swap to that installed model (Saguaro starts again with it: a 503 + Retry-After says so).
            `load` (the /models/load swap): the switch is what was asked for, so it answers 200 + loading instead of the
            503 a chat gets (a chat's request cannot be served by the model that is starting)."""
            run = next((m for m in svc.model_listing() if m["running"]), None)
            if not name or run is None:
                return True
            p = model_api.find_config(svc.models_root(), name)
            if p is None:
                self._json(404, {"error": f"model '{name}' not found, try pulling it first"})
                return False
            if p.name == run["file"]:
                return True
            if not self._mapi_gate(manage=True):
                return False
            if not getattr(svc, "model_restart", None):
                if not os.environ.get("SAGUARO_RESTART_ON_UPDATE"):
                    self._json(409, {"error": f"{p.name} is installed, but this server was not started by saguaro.sh or "
                                              "START-SAGUARO.bat, so it cannot switch models itself: start it with that model"})
                    return False
                try:
                    svc.switch_model(p.name)
                except ValueError as e:
                    self._json(409, {"error": str(e)})
                    return False
            loading = model_api.public_name(p) + ":latest"
            if load:    # /models/load asked for the swap: the switch started, so answer 200 (no Retry-After)
                body = json.dumps({"success": True, "loading": True, "model": loading,
                                   "message": f"loading model {model_api.public_name(p)}: Saguaro starts again "
                                              "with it (a big model takes minutes); GET /api/ps shows when it is "
                                              "loaded"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return False
            body = json.dumps({"error": f"loading model {model_api.public_name(p)}: Saguaro starts again with it (a big model "
                                        "takes minutes); retry this request, GET /api/ps shows when it is loaded",
                               "loading": loading}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "30")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        def _load_by_name(self, req) -> bool:
            """load by name, for the chat routes that otherwise answer with whatever is loaded: True - the loaded
            model answers (the setting is off, no model named, the loaded model's own name, or a name that is no
            installed model, as today).  False - the answer is sent: the request named another installed model,
            the switch started the way /models/load starts it, and the request is held no further - the switch ends
            this process (saguaro.py starts the named config again on this port), so the connection cannot outlive
            it: a 503 with Retry-After says to retry once the new model is up.  A request still running on the
            current model is never cut short: like every restart, the switch waits for the FIFO to empty."""
            name = str((req or {}).get("model") or "").strip()
            if not name or not getattr(svc, "load_by_name", False):
                return True
            if model_api.name_key(name) == model_api.name_key(svc.model):
                return True                                # (the loaded model's own /v1/models id)
            entries = svc.model_listing()
            hit = model_api.match_name(name, entries)
            run = next((m for m in entries if m["running"]), None)
            if hit is None or run is None or hit["file"] == run["file"]:
                return True                                # (unknown name, or the loaded model by another name)
            if not self._mapi_gate(manage=True):
                return False
            if not getattr(svc, "model_restart", None):
                if not os.environ.get("SAGUARO_RESTART_ON_UPDATE"):
                    self._json(409, {"error": {"type": "invalid_request_error", "message":
                                               f"{hit['file']} is installed, but this server was not started by "
                                               "saguaro.sh or START-SAGUARO.bat, so it cannot switch models itself: "
                                               "start it with that model"}})
                    return False
                try:
                    svc.switch_model(hit["file"])
                except ValueError as e:
                    self._json(409, {"error": {"type": "invalid_request_error", "message": str(e)}})
                    return False
                print(f"[saguaro] load by name: {run['file']} -> {hit['file']} (requested by a chat)", flush=True)
            body = json.dumps({"error": {"type": "overloaded_error", "message":
                                         f"loading model {model_api.public_name(Path(hit['file']))}: Saguaro starts "
                                         "again with it (a big model takes minutes); retry this request, GET "
                                         "/v1/models shows when it is loaded"}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "30")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        def _mapi_chat(self, kind: str, req: dict):
            """/api/chat and /api/generate through the same pipeline as /v1/chat/completions."""
            if not self._mapi_gate():
                return
            if not self._mapi_target(req.get("model")):
                return
            if kind == "chat":
                check_messages(req)   # (the wrong shape: a ValueError, do_POST's 400)
            ka = model_api.parse_keep_alive(req.get("keep_alive"))
            name = model_api.public_name(Path(svc.config_path)) + ":latest" if svc.config_path else svc.model
            empty = not (req.get("messages") if kind == "chat" else (req.get("prompt") or req.get("images")))
            if empty:                                          # the preload (load) and its unload (keep_alive 0)
                if ka == 0:
                    if svc.unload() == "busy":
                        self._json(409, {"error": "a request is running or waiting; unload when idle"})
                        return
                    reason = "unload"
                else:
                    svc.keep_alive_s = ka
                    svc.fifo.acquire(blocking=True)
                    try:
                        svc.ensure_engine()
                    except Exception as e:  # noqa: BLE001 - the reason goes back to the caller
                        self._json(503, {"error": f"the engine did not start: {e}"})
                        return
                    finally:
                        svc.fifo.release()
                    reason = "load"
                o = {"model": name, "created_at": model_api._iso(time.time()), "done": True, "done_reason": reason}
                o.update({"message": {"role": "assistant", "content": ""}} if kind == "chat" else {"response": ""})
                self._json(200, o)
                return
            oreq = model_api.chat_to_openai(req) if kind == "chat" else model_api.generate_to_openai(req)
            svc.keep_alive_s = ka                               # (the model stays loaded that long after this request)
            oreq = svc.with_shared(oreq, "openai")
            oreq["_api"] = "openai"
            self._priority_header(oreq)
            svc.check_request(oreq)
            messages, tools, kw = openai_to_messages(oreq, deepseek=getattr(svc.template, "deepseek", False))
            max_new = int(oreq.get("max_tokens") or 0)
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, guard=svc._loop_guard_on(oreq))
            if kw.get("_prefill"):
                oreq = {**oreq, "_prefill": kw["_prefill"]}
            oreq = {**oreq, "_blocked": kw.get("_blocked"), "_think_resume": kw.get("_think_resume")}
            if kw.get("_gen_tail"):
                oreq = {**oreq, "_gen_tail": kw["_gen_tail"]}
            oreq = {**oreq, "_conv_label": kw.get("_conv_label")}   # (the Monitor's conversation row: its first words)
            cancel = threading.Event()
            watch_client(self, cancel)
            chunks = openai_chunks(svc, oreq, ids, thinking, tools, max_new, cancel)
            tr = model_api.Translator(name, kind)

            def line(ev):
                self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode())
            try:
                if not oreq.get("stream"):
                    def make():
                        pieces = [ev for c in chunks for ev in tr.feed(c)]
                        calls = [c for ev in tr.finish_calls() for c in ev["message"]["tool_calls"]]
                        last = tr.done()
                        if kind == "chat":
                            msg = {"role": "assistant",
                                   "content": "".join(p["message"]["content"] for p in pieces)}
                            think = "".join(p["message"].get("thinking", "") for p in pieces)
                            if think:
                                msg["thinking"] = think
                            if calls:
                                msg["tool_calls"] = calls
                            last["message"] = msg
                        else:
                            last["response"] = "".join(p["response"] for p in pieces)
                        return last
                    return self._collect_wait(make, f"ollama {kind}", cancel, lambda: self._cid_paused(oreq),
                                              waited=lambda: self._cid_waited(oreq))
                else:
                    self._ndjson_start()
                    for c in chunks:
                        for ev in tr.feed(c):
                            line(ev)
                            self.wfile.flush()
                    for ev in tr.finish_calls() + [tr.done()]:
                        line(ev)
                    self.wfile.flush()
            except OSError:
                cancel.set()
                chunks.close()
            except EngineDied as e:
                if self.answer_started:
                    line({"error": f"{e}; the next request restarts it"})
                else:
                    self._json(503, {"error": f"{e}; the next request restarts it"})
            finally:
                if ka == 0:                                      # keep_alive 0: out of memory as soon as it is idle
                    threading.Thread(target=svc.unload, daemon=True).start()

        def _models_sse(self):
            """llama.cpp's /models/sse: the Models page's live state, a snapshot every few seconds."""
            if not getattr(svc, "model_management", False):
                self._json(404, {"error": {"message": "model management is off (remove \"model_management\": false, or "
                                                      "SAGUARO_MODEL_MANAGEMENT=1, turns it on)"}})
                return
            if not self._authorized():
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    alive = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    value = "loaded" if alive else ("unloaded" if svc.unloaded else "loading")
                    payload = {"object": "list", "data": [{"id": svc.model, "status": {"value": value}}]}
                    self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(5)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _cors_allow(self):
            """The Access-Control-Allow-Origin value for this request, or None: only /v1/* paths and only an origin the
            config's cors_origins lists (or "*")."""
            if not svc.cors_origins or not self.path.split("?")[0].startswith("/v1/"):
                return None
            if "*" in svc.cors_origins:
                return "*"
            origin = (self.headers.get("Origin") or "").rstrip("/")
            return origin if origin and origin in svc.cors_origins else None

        def end_headers(self):
            # every answer (JSON, a stream, an error) to an allowed page carries the CORS header - one place for all
            allow = self._cors_allow()
            if allow:
                self.send_header("Access-Control-Allow-Origin", allow)
                if allow != "*":
                    self.send_header("Vary", "Origin")
            super().end_headers()

        def do_OPTIONS(self):
            # a CORS preflight: no credentials come with it, so no API key; granted only for cors_origins
            allow = self._cors_allow()
            self.send_response(204 if allow else 403)
            if allow:
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                asked = self.headers.get("Access-Control-Request-Headers")
                self.send_header("Access-Control-Allow-Headers", asked or "Content-Type, Authorization, x-api-key")
                self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if given and any(hmac.compare_digest(given.encode(), k.encode()) for k in key_list(svc.api_key)):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def _response_owner(self):
            if not svc.api_key:
                return "public"
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            return hashlib.sha256(given.encode()).hexdigest()

        def _response_lookup(self, owner, response_id):
            try:
                return svc.response_jobs.get(owner, response_id)
            except KeyError:
                return svc.responses.get(owner, response_id)

        def _background_stream(self, owner, response_id, after=-1):
            try:
                svc.response_jobs.check_cursor(owner, response_id, after)
            except KeyError:
                raise JobError(409, "background event history is no longer retained") from None
            with svc.response_jobs.reader(owner, response_id, after) as events:
                self._write_background_stream(events, after)

        def _write_background_stream(self, events, after):
            self.connection.settimeout(svc.response_limits["STREAM_TIMEOUT_S"])
            self._sse()
            try:
                for item in events:
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, payload = item
                        after = payload["sequence_number"]
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(payload, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except (JobError, KeyError) as exc:
                payload = {"type": "error", "code": "event_history_unavailable", "param": None,
                           "message": str(exc), "sequence_number": after + 1}
                try:
                    self.wfile.write(b"event: error\ndata: " + json.dumps(payload).encode() + b"\n\n")
                except OSError:
                    pass
            except OSError:
                pass
            finally:
                events.close()

        def _response_resource(self, delete=False):
            if not self._authorized():
                return
            parts = self.path.split("?")[0].rstrip("/").removeprefix("/v1/responses/").split("/")
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            if len(parts) > 2 or (len(parts) == 2 and (parts[1] != "input_items" or delete)):
                return self._json(404, {"error": {"message": "not found"}})
            try:
                owner = self._response_owner()
                if delete:
                    if query:
                        raise ValueError("response deletion does not support query options")
                    removed = False
                    for store in (svc.response_jobs, svc.responses):
                        try:
                            store.delete(owner, parts[0])
                            removed = True
                        except KeyError:
                            pass
                    if not removed:
                        raise KeyError(parts[0])
                    return self._json(200, {"id": parts[0], "object": "response.deleted", "deleted": True})
                stored = self._response_lookup(owner, parts[0])
                if len(parts) == 1 and query:
                    if (set(query) - {"stream", "starting_after"} or query.get("stream") != ["true"] or
                            any(len(value) != 1 for value in query.values())):
                        raise ValueError("retrieve accepts stream=true and optional starting_after")
                    after = int(query["starting_after"][0]) if "starting_after" in query else -1
                    if "starting_after" in query and after < 0:
                        raise ValueError("starting_after must be nonnegative")
                    return self._background_stream(owner, parts[0], after)
                result = input_page(stored["input"], query) if len(parts) == 2 else stored["response"]
                self._json(200, result)
            except KeyError:
                self._json(404, {"error": {"type": "invalid_request_error", "message": "stored response not found"}})
            except (JobError, StoreBusy) as exc:
                self._json(getattr(exc, "status", 503), {"error": {"type": "server_error", "message": str(exc)}})
            except ValueError as exc:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(exc)}})

        def _response_cancel(self, path):
            parts = path.removeprefix("/v1/responses/").split("/")
            if len(parts) != 2 or parts[1] != "cancel" or "?" in self.path:
                return self._json(404, {"error": {"message": "not found"}})
            owner = self._response_owner()
            try:
                out = svc.response_jobs.cancel(owner, parts[0])
            except KeyError:
                try:
                    out = self._response_lookup(owner, parts[0])["response"]
                except KeyError:
                    return self._json(404, {"error": {"message": "background response not found"}})
                if not out.get("background"):
                    raise ValueError("only background responses can be cancelled")
            self._json(200, out)

        def do_DELETE(self):
            if self._foreign_page():
                return
            if self.path.split("?")[0].startswith("/v1/responses/"):
                self._response_resource(delete=True)
            elif self.path.split("?")[0].rstrip("/") == "/api/delete":
                if self._authorized():
                    self._mapi_post("/api/delete")
            elif self._authorized():
                self._json(404, {"error": {"message": "not found"}})

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/models/sse":
                self._models_sse()
                return
            if path in ("/api/tags", "/api/ps", "/api/version", "/api/catalog", "/api/pull/status"):
                if self._authorized():
                    self._mapi_get(path)
                return
            if path.startswith("/v1/responses/"):
                self._response_resource()
                return
            if path.startswith("/fonts/"):
                # the web app's font (Outfit, OFL: serve/web/fonts); the page falls back to the system font
                name = path[len("/fonts/"):]
                f = ROOT / "serve" / "web" / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/web":
                # a home-screen icon saved with the manifest's old start_url (/web/): the page is one level up. Relative,
                # so it still works behind a path-prefixed proxy
                raw = self.path.split("?")[0]
                query = self.path[len(raw):]
                self.send_response(302)
                self.send_header("Location", ("../" if raw.endswith("/") else "./") + query)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path.startswith("/web/"):
                # the web app's own files (serve/web): styles, script, icon sprite - same origin, no CDN
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json"}
                f = ROOT / "serve" / "web" / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", types[ext])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/metrics":
                if self._authorized():
                    # the last 12 requests; `?requests=all` every one kept (the Monitor's "Show all", issue #35)
                    try:
                        m = svc.metrics(all_requests="requests=all" in self.path)
                    except Exception:   # (said in the log - the handler's own traceback goes to a pipe nobody reads)
                        print("[saguaro] /metrics failed:\n" + traceback.format_exc(), flush=True)
                        raise
                    self._json(200, m)
                return
            if path == "/metrics/prometheus":
                if self._authorized():
                    body = svc.prometheus().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return
            if path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(svc.shared), "defaults": svc.shared})
                return
            if path == "/mcp":
                # the MCP servers, their state and tools (the web app's switch and Monitor card)
                if self._authorized():
                    self._json(200, svc.mcp.status() if svc.mcp else {"servers": [], "tools": 0})
                return
            if path == "":
                body = index_html()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/health":
                # the run config's recommended settings: the web app's Chat defaults until its user picks their own
                rec = {k: v for k, v in svc.sampling_defaults.items() if k in ("temperature", "top_p", "top_k", "min_p")}
                if svc.default_effort:   # (MiniMax's and Inkling's by their own names, Qwen3.8's by its three)
                    named = getattr(svc.template, "minimax", False) or getattr(svc.template, "inkling", False)
                    rec["reasoning_effort"] = (effort_name if named else effort_level)(
                        svc.default_effort) or svc.default_effort
                    if getattr(svc.template, "forced_thinking", False):
                        rec["reasoning_effort"] = QWEN_EFFORT.get(rec["reasoning_effort"], rec["reasoning_effort"])
                    if getattr(svc.template, "hy4", False):   # (Hy4-flash: the level it runs at, none / high)
                        rec["reasoning_effort"] = svc.template.effort_shown(svc.default_effort) or \
                            rec["reasoning_effort"]
                # Hy4-flash alone: its two levels (the page offers these and no others)
                levels = svc.template.thinking_levels() if getattr(svc.template, "hy4", False) else None
                self._json(200, {"status": "updating" if svc.updater.busy() else
                                           "reloading" if svc.reloading() else "ok",
                                 "max_context": svc.engine.max_context, "model": svc.model,
                                 "max_text_file": text_file_limit(svc.engine.max_context),
                                  "model_label": svc.model_label,   # (the model first: "GLM-5.3 · Maya-S24")
                                  "model_name_pretty": svc.model_name_pretty,   # (the two parts a page names with:
                                  "model_quant": svc.model_quant,   #  "Qwen3.8 Flash Next" + "Q8_0"; None: not said)
                                 "images": svc.vision is not None, "api_key": bool(svc.api_key), "defaults": rec,
                                 "version": SAGUARO_VERSION, "mcp_tools": bool(svc.mcp), "unloaded": svc.unloaded,
                                 **({"thinking_levels": levels} if levels else {}),
                                 "capabilities": svc.capabilities(), "supervisor": svc.supervisor})
            elif path == "/api/context":
                # the dashboard's Context size: the range, the measured cost of the current size, a reload's state
                if self._authorized():
                    self._json(200, svc.context_limits())
            elif path == "/api/thinking-budget":
                if self._authorized():
                    self._json(200, svc.think_budget_info())
            elif path == "/api/port":
                # the dashboard's Port: where the server listens, and whether a new one applies at once
                if self._authorized():
                    self._json(200, svc.port_info())
            elif path == "/api/turbo":
                # the Turbo button (Settings > Chat defaults): on or off, and what it would change in the engine
                if self._authorized():
                    self._json(200, svc.turbo_info())
            elif path == "/api/turbo/options":
                # the live per-request Turbo (PLAN-turbo-live): the model's global setting (the control panel's),
                # what requests without their own `turbo` run with - changes apply at once, no config rewrite
                if self._authorized():
                    self._json(200, {"turbo": dict(svc.turbo_default) if svc.turbo_default else None,
                                     "turbo_skip": svc.turbo_skip_supported(),   # (PLAN-p5: false only where the
                                     # engine leaves skip off - the page hides the Skip control on that)
                                     **({"supported": False} if getattr(svc, "engine_kind", None) in NO_LIVE_TURBO
                                        else {})})
            elif path == "/api/models":
                # the Models page: the configs the setup wrote, the running one, each one's engine settings
                if self._authorized():
                    self._json(200, svc.models_info())
            elif path == "/api/savings":
                # the sidebar's Saved: what the model's own API would have charged for the same tokens, per model
                if self._authorized():
                    self._json(200, svc.savings.report() if svc.savings is not None else {"usd": None, "models": []})
            elif path == "/api/update":
                # About > Updates: the latest release (GitHub, cached six hours; ?force=1 asks now) and whether this
                # folder can update itself
                if self._authorized():
                    self._json(200, svc.updater.check(force="force=1" in self.path))
            elif path == "/api/report":
                # the dashboard's Copy report: what an issue needs (the run config's secrets left out)
                if self._authorized():
                    body = svc.report().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            elif path == "/status":
                if not self._authorized():   # (it carries `tail`: the last 600 characters written)
                    return
                with svc.status_lock:
                    s = dict(svc.status)
                    answering = svc.latest_request in svc.current_requests
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if answering:
                        s["tokens_per_s"] = round(svc._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(svc._tok_s_mean(), 1)
                s.pop("started", None)
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    model = {"id": svc.model, "object": "model", "owned_by": "saguaro",
                             "status": {"value": "loaded" if loaded else "unloaded"},
                             "meta": {"n_ctx": svc.engine.max_context},
                             # (the same capacity under the name generic OpenAI-compatible
                             # clients read on a model listing: DeepSeek Harness's pi-ai discovery,
                             # open-webui, llama.cpp's server clients, ... - n_ctx stays above;
                             # no max_output_tokens: a client asking for the whole context as output gets a 400)
                             "context_length": svc.engine.max_context,
                             # (videos: the llama.cpp encoder's - GLM's; DeepSeek's encoder reads pictures only)
                             "architecture": {"input_modalities": ["text"] if svc.vision is None else
                                              ["text", "image", "video"] if isinstance(svc.vision, Vision) else
                                              ["text", "image"],
                                              "output_modalities": ["text"]}}
                    self._json(200, {"object": "list", "data": [model] if loaded or svc.unloaded else []})
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    with svc.status_lock:
                        busy = bool(svc.status.get("busy"))
                    n_slots = max(1, int(getattr(svc.engine, "multi", 0) or 0))
                    with svc.status_lock:
                        active = svc.active
                    slots = [{"id": i, "n_ctx": svc.engine.max_context,
                              "is_processing": busy if n_slots == 1 else i < active} for i in range(n_slots)]
                    self._json(200, slots if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            elif path == "/v1/strata/stream":
                if self._authorized():
                    self._job_get()
            else:
                self._json(404, {"error": {"message": "not found"}})

        def _foreign_page(self) -> bool:
            """A browser page on another site (its Origin neither this server nor in cors_origins): it can post a
            text/plain body with no CORS preflight (CSRF), so it is refused before anything runs.  curl and the SDKs
            send no Origin."""
            origin = self.headers.get("Origin")
            if not origin or origin.split("://", 1)[-1] == self.headers.get("Host", "") or self._cors_allow():
                return False
            self._json(403, {"error": {"type": "permission_error", "message": "requests from other web pages are "
                                       "refused (the config's cors_origins lists the pages allowed)"}})
            return True

        def _fail(self, code, kind, message):
            """An error as JSON, while nothing was sent; once an answer has begun (a stream), only the log has it."""
            if self.answer_started:
                print(f"[saguaro] request failed mid-answer: {message}", file=sys.stderr, flush=True)
                self.close_connection = True
                return
            self._json(code, {"error": {"type": kind, "message": message}})

        def do_POST(self):
            if not self._authorized() or self._foreign_page():
                return
            path = self.path.split("?")[0].rstrip("/")   # issue #55: Claude Code posts /v1/messages?beta=true
            try:
                if path in ("/models/load", "/models/unload"):
                    self._llamacpp_model(path.rsplit("/", 1)[1])
                    return
                if path.startswith("/v1/responses/") and path.endswith("/cancel"):
                    self._response_cancel(path)
                    return
                if path in ("/api/show", "/api/pull", "/api/pull/cancel", "/api/delete", "/api/chat", "/api/generate"):
                    self._mapi_post(path)
                    return
                if path == "/settings":
                    self._settings()
                    return
                if path == "/v1/strata/cancel":
                    self._job_cancel()
                    return
                # under a supervisor (llama-swap: it starts, stops and switches the models, and owns their configs):
                # what would restart this server, move it or rewrite a config is refused - said, not attempted
                if svc.supervisor and path in ("/api/port", "/api/update", "/api/models/settings", "/api/models/switch",
                                               "/api/models/delete", "/api/models/leftovers", "/api/context",
                                               "/api/turbo", "/api/pull", "/api/delete"):
                    self._body()
                    self._json(409, {"error": {"type": "invalid_request_error", "message":
                                     f"this server runs under {svc.supervisor}, which starts, stops and switches the "
                                     "models: change models there (pick another model id), not here"}})
                    return
                if path == "/api/context":
                    self._context()
                    return
                if path == "/api/port":
                    self._port()
                    return
                if path == "/api/thinking-budget":
                    body = self._body()
                    if not self._own_page("the thinking budget can be changed"):
                        return
                    try:
                        req = json.loads(body or b"{}")
                        out = svc.set_think_budget(req.get("budget") if isinstance(req, dict) else None)
                    except (TypeError, ValueError, OSError) as e:
                        self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                        return
                    self._json(200, out)
                    return
                if path == "/api/update":
                    self._update()
                    return
                if path == "/api/turbo":
                    self._turbo()
                    return
                if path == "/api/turbo/options":
                    self._turbo_options()
                    return
                if path in ("/api/models/settings", "/api/models/switch", "/api/models/delete", "/api/models/leftovers"):
                    self._models(path.rsplit("/", 1)[1])
                    return
                if path == "/api/savings/reset":
                    # it clears every model's count: the app's own page only
                    self._body()
                    if not self._own_page("the savings can be reset"):
                        return
                    if svc.savings is None:
                        self._json(400, {"error": {"type": "invalid_request_error", "message": "nothing is counted here"}})
                        return
                    svc.savings.reset()
                    self._json(200, svc.savings.report())
                    return
                if path == "/api/savings/prices":
                    # Settings > Savings: a model's API prices ({"model", "price": {input, cached, output} or null:
                    # the built-in ones}); every total is counted again with them - the app's own page only
                    body = self._body()
                    if not self._own_page("the prices can be changed"):
                        return
                    try:
                        req = json.loads(body or b"{}")
                        if svc.savings is None:
                            raise ValueError("nothing is counted here")
                        if not isinstance(req, dict):
                            raise ValueError("the request body must be a JSON object")
                        svc.savings.set_price(req.get("model"), req.get("price"))
                    except (TypeError, ValueError) as e:
                        self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                        return
                    self._json(200, svc.savings.report())
                    return
                req = json.loads(self._body() or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("the request body must be a JSON object")
                if path == "/api/tokens":                    # the context meter: a prompt's size, nothing run
                    self._json(200, svc.count_tokens(req))
                    return
                if path == "/tokenize":                      # llama.cpp's and vLLM's: a text's or a prompt's ids
                    self._json(200, svc.tokenize(req))
                    return
                if path == "/detokenize":
                    self._json(200, svc.detokenize(req))
                    return
                if path in ("/v1/chat/completions", "/v1/responses", "/v1/messages") and (svc.reloading() or svc.updater.busy()):
                    # a context change is reloading the model, or Saguaro is updating: say so, and when to try again
                    why = ("Saguaro is updating and starts again in a few minutes" if svc.updater.busy() else
                           "the model is reloading with a new context size; try again in a minute")
                    body = json.dumps({"error": {"type": "overloaded_error", "message": why}}).encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Retry-After", "30")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path in ("/v1/chat/completions", "/v1/responses", "/v1/messages") and not self._load_by_name(req):
                    return                                 # (the answer went out: a switch started, or it said why not)
                if path in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
                    # (a list of messages of the wrong shape: a 400 in plain words before anything reads it)
                    check_messages(req, "input" if path == "/v1/responses" else "messages")
                if path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/responses":
                    self._responses(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                else:
                    self._json(404, {"error": {"message": "not found"}})
            except BadBody as e:
                self._fail(e.status, "invalid_request_error", str(e))
            except (JobError, StoreBusy) as e:
                self._fail(getattr(e, "status", 503), "server_error", str(e))
            except ValueError as e:                          # (a ContextOverflow with its code: #432)
                if self.answer_started:
                    self._fail(400, "invalid_request_error", str(e))
                else:
                    self._json(400, error_body(e))
            except EngineDied as e:
                self._fail(503, "server_error", f"{e}; the next request restarts it")
            except (KeyError, TypeError, AttributeError, IndexError) as e:
                # a field of the wrong shape (Strata: a 400 with the traceback in the log, not a dropped connection)
                traceback.print_exc()
                self._fail(400, "invalid_request_error", f"malformed request ({type(e).__name__}: {e})")

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model != svc.model:
                self._json(404, {"error": {"message": "model not found"}})
                return
            if hasattr(svc.engine, "alive") and not svc.engine.alive() and not svc.unloaded:
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            defaults = {**svc.sampling_defaults, **svc.shared}
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = svc.shared.get("max_tokens", -1)
            props = {"default_generation_settings": {"n_ctx": svc.engine.max_context, "params": params},
                     "total_slots": max(1, int(getattr(svc.engine, "multi", 0) or 0)), "model_alias": svc.model, "chat_template": svc.template.source,
                     "modalities": {"vision": svc.vision is not None}, "models_autoload": False,
                     "is_sleeping": False}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "Strata " + str(version)
            self._json(200, props)

        def _own_page(self, what) -> bool:
            """Only JSON (a form or a "simple" cross-site request can't send it without a CORS preflight, which this
            server grants only to the config's cors_origins and only on /v1/*) and no foreign Origin: a web page
            elsewhere must not change settings or run tools."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != self.headers.get("Host", ""):
                self._json(403, {"error": {"message": f"{what} only from Saguaro's own page"}})
                return False
            return True

        def _priority_header(self, req: dict):
            """The X-Saguaro-Priority header onto the request (R122): Service.check_request reads it together with
            the body's own "priority" - the body wins - and a bad value of either is the 400."""
            h = self.headers.get("X-Saguaro-Priority")
            if h is not None:
                req["_priority_header"] = h

        def _context(self):
            # It reloads the model for every client, so only the app's own page may ask for it
            body = self._body()
            if not self._own_page("the context size can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                n = int(req.get("max_context")) if isinstance(req, dict) else 0
                reload = svc.set_context(n)
            except (TypeError, ValueError) as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(202, {"reload": reload})

        def _port(self):
            # It moves the server for every client, so only the app's own page may ask for it
            body = self._body()
            if not self._own_page("the port can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                n = int(req.get("port")) if isinstance(req, dict) else 0
                out = svc.set_port(n)
            except (TypeError, ValueError, OSError) as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(202, out)

        def _models(self, what: str):
            # They change which model runs, or how, for every client: only the app's own page
            body = self._body()
            if not self._own_page("models can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("the request body must be a JSON object")
                if what == "switch":
                    out = svc.switch_model(req.get("file"))
                elif what == "delete":
                    out = svc.delete_model(req.get("file"), bool(req.get("files")), bool(req.get("dry_run")),
                                           req.get("then"))
                elif what == "leftovers":
                    out = svc.delete_leftover(req.get("path"))
                else:
                    out = svc.model_settings(req.get("file"), req.get("settings") or {}, bool(req.get("dry_run")))
            except (TypeError, ValueError, OSError) as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(200 if what in ("delete", "leftovers") and not out.get("deleting")
                       or (what == "settings" and req.get("dry_run")) else 202, out)

        def _turbo(self):
            # It changes the engine's settings (and restarts it) for every client: only the app's own page
            body = self._body()
            if not self._own_page("Turbo can be switched"):
                return
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict) or not isinstance(req.get("on"), bool):
                    raise ValueError('send {"on": true} or {"on": false}')
                out = svc.set_turbo(req["on"])
            except (TypeError, ValueError, OSError) as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(202 if out.get("restarting_now") else 200, out)

        def _turbo_options(self):
            # The live per-request Turbo (PLAN-turbo-live): the model's global options in the control panel.  It
            # changes what every request without its own `turbo` runs with - so the app's own page only - but it is
            # engine state only: applied at once, no config rewrite, no reload, and it works under llama-swap
            body = self._body()
            if not self._own_page("Turbo's options can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict) or "turbo" not in req:
                    raise ValueError('send {"turbo": {"drop": N, "skip": "off"|"ssd"|"ram"}} or {"turbo": null}')
                out = svc.set_turbo_default(req.get("turbo"))
            except (TypeError, ValueError, OSError) as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            print("[saguaro] live Turbo for this model: "
                  + (f"drop {out['drop']}, skip {out['skip']}" if out else "off")
                  + " (every request without its own, from the next one)", flush=True)
            self._json(200, {"turbo": out})

        def _update(self):
            # It replaces the server's files and restarts it for every client: only the app's own page
            self._body()
            if not self._own_page("an update can be started"):
                return
            try:
                state = svc.updater.start(svc)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(202, {"update": state})

        def _settings(self):
            # They change what every client gets, so only the app's own page may set them
            body = self._body()
            if not self._own_page("settings can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                shared = svc.set_shared(req.get("defaults") if isinstance(req, dict) else None)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            print("[saguaro] other apps now use the Chat settings: " + ", ".join(f"{k}={v}" for k, v in shared.items())
                  if shared else "[saguaro] other apps use their own settings again", flush=True)
            self._json(200, {"shared": bool(shared), "defaults": shared})

        def _sse(self):
            # token frames are small and flushed as they come; Nagle would otherwise hold one
            # for the peer's delayed ACK (~40 ms) on its way to the client.
            try:
                import socket
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def _responses(self, req):
            owner = self._response_owner()
            try:
                previous = req.get("previous_response_id")
                if previous is not None:
                    if not isinstance(previous, str) or not previous:
                        raise ValueError("previous_response_id must be a nonempty string")
                    stored = self._response_lookup(owner, previous)
                    if stored["response"]["status"] not in ("completed", "incomplete"):
                        raise JobError(409, "previous response is not a completed or incomplete generation")
                    added = req.get("input", [])
                    if isinstance(added, str):
                        added = [{"role": "user", "content": added}]
                    if not isinstance(added, list):
                        raise ValueError("input must be a string or array of input items")
                    req = {**req, "input": stored["input"] + stored["response"]["output"] + added}
            except KeyError:
                return self._json(404, {"error": {"type": "invalid_request_error",
                                                 "message": "previous_response_id: stored response not found"}})
            sampling = normalize_request(req)
            validation = sampling.pop("_response_validation")
            history = input_items(req["input"]) if req.get("store") else None
            sampling = svc.with_shared(sampling, "openai")
            sampling["_api"] = "openai"
            sampling["_responses"] = True
            sampling["_cid"] = "resp-" + uuid.uuid4().hex[:24]   # (R122: the live state can name this request)
            self._priority_header(sampling)
            svc.check_request(sampling)
            if req.get("background"):
                initial = new_response(svc.model, req, status="queued")
                work = lambda cancel, rid: self._response_work(req, sampling, history, validation, owner, cancel, rid, False)
                svc.response_jobs.store = svc.responses
                out = svc.response_jobs.submit(owner, initial, history, work)
                if req.get("stream"):
                    return self._background_stream(owner, out["id"])
                return self._json(200, out)
            cancel = threading.Event()
            watch_client(self, cancel)
            events = self._response_work(req, sampling, history, validation, owner, cancel)
            try:
                if not req.get("stream"):
                    return self._collect_wait(lambda: response_collect(events), "responses", cancel,
                                              lambda: self._cid_paused(sampling),
                                              waited=lambda: self._cid_waited(sampling),
                                              code_of=lambda out: 500 if out["status"] == "failed" else 200)
                first = next(events)
                self._sse()
                for item in itertools.chain([first], events):
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, data = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(data, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                cancel.set()
                if not self.answer_started:
                    self._json(500, {"error": {"type": "server_error", "message": "could not prepare the response"}})
            finally:
                events.close()

        def _response_work(self, req, sampling, history, validation, owner, cancel, response_id=None, persist=True):
            sampling = dict(sampling)
            # Shared thinking defaults must be resolved by the same normalizer as request values.
            messages, tools, kw = openai_to_messages(sampling)
            max_new = int(sampling.get("max_completion_tokens") or sampling.get("max_tokens") or 0)
            events = None
            svc.embeddings.path = None
            try:
                if cancel.is_set():
                    raise ValueError("generation was cancelled before preparation")
                ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, cancel=cancel,
                                                     guard=svc._loop_guard_on(sampling))
                if kw.get("_prefill"):
                    sampling["_prefill"] = kw["_prefill"]
                sampling["_blocked"] = kw.get("_blocked")
                sampling["_think_resume"] = kw.get("_think_resume")
                if kw.get("_gen_tail"):
                    sampling["_gen_tail"] = kw["_gen_tail"]
                sampling["_conv_label"] = kw.get("_conv_label")   # (the Monitor's conversation row: its first words)
                effective = {**svc.sampling_defaults, **svc.shared, **sampling}
                config = {**req, **{k: effective[k] for k in ("temperature", "top_p") if k in effective}}
                run = svc.run(ids, thinking, tools, max_new, sampling, cancel)
                save = (lambda out: svc.responses.put(owner, out, history)) if req.get("store") and persist else None
                events = response_events(svc.model, config, run, len(ids), persist=save, response_id=response_id,
                                         validation=validation, cancel=cancel,
                                         waited=lambda: self._cid_waited(sampling))
                yield from events
            finally:
                if events is not None:
                    events.close()
                # A disconnect before the generator starts, or failed preparation, cannot run Service.run's cleanup.
                emb = getattr(svc.embeddings, "path", None)
                if emb:
                    Path(emb).unlink(missing_ok=True)

        def _openai(self, req):
            req = svc.with_shared(req, "openai")
            req["_api"] = "web" if req.get("strata_resume") is True else "openai"   # the Monitor's request table
            self._priority_header(req)
            svc.check_request(req)                             # a bad sampling field or logit_bias: a 400, now
            messages, tools, kw = openai_to_messages(req, deepseek=getattr(svc.template, "deepseek", False))
            max_req = max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)   # 0/-1: the rest
            use_mcp = req.get("strata_mcp") is True and svc.mcp is not None      # the web app's opt-in (serve/mcp.py)
            own = {t.get("name") for t in tools or []}
            if use_mcp:
                if not self._own_page("MCP tools can be used"):   # tools run with the user's rights on this PC
                    return
                svc.mcp.wait(10)                                  # servers still starting (only right after start)
                extra = svc.mcp.template_tools(exclude=own)       # the request's own tools win a name clash
                use_mcp = bool(extra)
                tools = (tools or []) + extra or None
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, guard=svc._loop_guard_on(req))
            if kw.get("_prefill"):                             # a required tool call starts the answer
                req = {**req, "_prefill": kw["_prefill"]}
            req = {**req, "_blocked": kw.get("_blocked")}      # calls the loop guard blocks (Service.prepare)
            req = {**req, "_think_resume": kw.get("_think_resume")}
            if kw.get("_gen_tail"):                            # the reply's opening (gt=)
                req = {**req, "_gen_tail": kw["_gen_tail"]}
            req = {**req, "_conv_label": kw.get("_conv_label")}   # the Monitor's conversation row: its first words
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            if not (req.get("stream") and req.get("strata_resume") is True):   # (the web app's runs on, by design)
                watch_client(self, cancel)
            run = run_with_mcp(svc, svc.mcp, messages, tools, kw, ids, thinking, max_new, max_req, req, cancel,
                               {t["name"] for t in extra}) if use_mcp else None
            chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, run=run)
            if not req.get("stream"):
                return self._collect_wait(lambda: openai_collect(chunks, logprobs=logprobs_asked(req) is not None),
                                          "openai", cancel, lambda: self._cid_paused(req),
                                          waited=lambda: self._cid_waited(req))
            if req.get("strata_resume") is True:
                # the web app: the answer runs on if the connection drops; the page reconnects to it by id
                first = next(chunks)
                job = StreamJob(first["id"], itertools.chain(
                    [first], carry_embeddings(svc.embeddings, getattr(svc.embeddings, "path", None), chunks)), cancel)
                jobs_put(job, svc.fifo.capacity)
                self._sse()
                self._stream_job(job, 0)
                return
            empty_chat_chunk = lambda: {"id": req.get("_cid") or "", "object": "chat.completion.chunk",
                                        "created": int(time.time()), "model": svc.model,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
            self._sse()
            try:
                for c in wait_events(chunks, empty_chat_chunk, lambda: self._cid_waited(req)):
                    if c is None:
                        self.wfile.write(b": keep-alive\n\n")      # an SSE comment: clients ignore it
                    else:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                cancel.set()                                 # client went away: stop the engine
                chunks.close()
            except EngineDied as e:                          # mid-stream: say so, then end the stream properly
                err = {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started: the
                err = {"error": {"type": "server_error", "message": str(e)}}   # headers are sent, so no 400 now
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")

        def _stream_job(self, job: StreamJob, n: int):
            """A StreamJob's chunks from index n as SSE until it completes; a reader that goes away leaves it running."""
            job.reader(True)
            try:
                while True:
                    items, done = job.wait(n, 10.0)
                    if not items and not done:
                        self.wfile.write(b": keep-alive\n\n")
                    for c in items:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    n += len(items)
                    self.wfile.flush()
                    if done and n >= len(job.items):
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return
            except OSError:
                return
            finally:
                job.reader(False)

        def _job_get(self):
            q = parse_qs(urlsplit(self.path).query)
            job = JOBS.get(q.get("id", [""])[0])
            if job is None:
                self._json(404, {"error": {"message": "no such answer (finished more than 15 minutes ago?)"}})
                return
            try:
                start = max(0, int(q.get("from", ["0"])[0]))
            except ValueError:
                start = 0
            self._sse()
            self._stream_job(job, start)

        def _job_cancel(self):
            body = self._body()
            if not self._own_page("answers can be stopped"):
                return
            try:
                jid = (json.loads(body or b"{}") or {}).get("id", "")
            except ValueError:
                jid = ""
            job = JOBS.get(jid)
            if job is not None:
                job.cancel.set()
            self._json(200, {"cancelled": job is not None})

        def _anthropic(self, req):
            req = svc.with_shared(req, "anthropic")
            req["_api"] = "anthropic"
            self._priority_header(req)
            svc.check_request(req)                             # a bad sampling field: a 400, now
            messages, tools, kw = anthropic_to_messages(req, deepseek=getattr(svc.template, "deepseek", False))
            max_new = int(req.get("max_tokens") or 0)                  # 0/-1: the rest of the context
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, guard=svc._loop_guard_on(req))
            if kw.get("_prefill"):                             # a required tool call starts the answer
                req = {**req, "_prefill": kw["_prefill"]}
            req = {**req, "_blocked": kw.get("_blocked")}      # calls the loop guard blocks (Service.prepare)
            req = {**req, "_think_resume": kw.get("_think_resume")}
            if kw.get("_gen_tail"):                            # the reply's opening (gt=)
                req = {**req, "_gen_tail": kw["_gen_tail"]}
            req = {**req, "_conv_label": kw.get("_conv_label")}   # the Monitor's conversation row: its first words
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            req = {**req, "_cid": "anth-" + uuid.uuid4().hex[:24]}   # (R122: the live state can name this request)
            cancel = threading.Event()
            watch_client(self, cancel)
            events = anthropic_events(svc, req, ids, thinking, tools, max_new, cancel)
            if not req.get("stream"):
                return self._collect_wait(lambda: anthropic_collect(events), "anthropic", cancel,
                                          lambda: self._cid_paused(req), waited=lambda: self._cid_waited(req))
            self._sse()
            try:
                for item in wait_events(events, lambda: ("ping", {"type": "ping"}),
                                        lambda: self._cid_waited(req)):
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, e = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                cancel.set()
                events.close()
            except EngineDied as e:                          # mid-stream: Anthropic's error event
                err = {"type": "error", "error": {"type": "api_error", "message": f"{e}; the next request restarts it"}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

    return Handler


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"
    # socketserver listens with a backlog of 5: an agent or a load of clients opening 30-40 connections at once got
    # "connection reset by peer" on the first ones (Strata 0.1.41); the requests wait in the server's queue instead
    # (SAGUARO_HTTP_BACKLOG overrides it)
    request_queue_size = _backlog()

    def server_bind(self):
        # http.server's own bind asks the resolver for the bound address's host name (socket.getfqdn) - nothing here uses
        # it, and on a PC whose resolver is slow or down every start (and every test server) waited for it, tens of seconds
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[0], self.server_address[1]

    def server_close(self):
        service = getattr(self, "response_service", None)
        try:
            if service is not None:
                service.response_jobs.close()
        finally:
            super().server_close()

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a client that hangs up needs no stack trace
            super().handle_error(request, client_address)


SERVER_ENV = ("SAGUARO_ENGINE_STALL_S", "SAGUARO_REQUEST_QUIET_S", "SAGUARO_DRAIN_S", "SAGUARO_HTTP_BACKLOG",
              "SAGUARO_MAX_BODY_MIB", "SAGUARO_FAIR_SLICE_S", "SAGUARO_FAIR_READ_S", "SAGUARO_FAIR_READ_YIELD_S",
              "SAGUARO_DS4_KEEP_THINKING",
              "SAGUARO_SERVER_LOG", "SAGUARO_GLM_THINK_RESERVE", "SAGUARO_LOOP_GUARD", "SAGUARO_QUEUE_PING_S",
              "SAGUARO_PRIORITY_AGE_S", "SAGUARO_PRIORITY_IDLE_S", "SAGUARO_WAIT_EVENT_S", "SAGUARO_WAIT_SPACE_S",
              "SAGUARO_UPDATE_CHECK", "SAGUARO_PACK_TEMPLATE") + RESPONSE_ENV_KEYS + \
    ("SAGUARO_DS4_RESTORE_POINTS", "SAGUARO_DS4_RESTORE_TAIL")


def apply_server_env(cfg: dict) -> None:
    """The config's "env" entries this server reads itself (`--env` puts every setting there; the rest go to the
    engine): in effect before it starts."""
    global ENGINE_STALL_S, REQUEST_QUIET_S, DRAIN_S
    for k in SERVER_ENV:
        v = (cfg.get("env") or {}).get(k)
        if v is not None:
            os.environ[k] = str(v)
    ENGINE_STALL_S = _stall_seconds()
    REQUEST_QUIET_S = _quiet_seconds()
    DRAIN_S = _drain_seconds()
    Server.request_queue_size = _backlog()


def ds4_one_gpu_switch_note(cfg: dict, env: dict) -> str | None:
    """DS4_STREAM_WIN / DS4_LAND are the one-GPU decode path's switches, and that path is NVIDIA's (v1.1.27's stays on
    other cards).  Set by hand for DeepSeek on one AMD GPU they forced it on a card the path was not made for: decode 19
    -> 2 tok/s and a stalled answer (#1120).  The text to print at start for that case, else None.  Message only."""
    if models.family(cfg) != "deepseek" or cfg.get("backend") != "hip" or len(gpu_list(cfg)) > 1:
        return None
    on = [k for k in ("DS4_STREAM_WIN", "DS4_LAND") if str(env.get(k) or "").strip() not in ("", "0")]
    if not on:
        return None
    return ("[saguaro] NOTE: " + " and ".join(on) + (" is" if len(on) == 1 else " are") + " set, but the one-GPU decode "
            "path they belong to is NVIDIA's: it does not apply to one AMD GPU, where the default is the earlier "
            "decode path (the engine's line says so).  Forced on, they made decode much slower and stalled answers "
            "(#1120).  Remove them (and DS4_STREAM_ZC) from the environment or the config's env to run as designed.")


def warn_tight_ram(arena_mib) -> None:
    """The model's experts live in RAM (INFO arena_mib, engine 0.1.10+).  With less than ~6 GB left beside them for the
    system, the engine and this server, Linux ends the engine mid-answer when memory runs out (issue #27) and Windows
    pages to disk; say so at start instead of after a lost answer."""
    if not isinstance(arena_mib, int) or arena_mib <= 0:
        return
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil is optional here
        return
    left = total / 2**30 - arena_mib / 1024
    if left < 6:
        print(f"[saguaro] WARNING: RAM is tight - the model's experts take {arena_mib / 1024:.1f} GB of this PC's "
              f"{total / 2**30:.0f} GB, leaving {left:.1f} GB for everything else. "
              + ("Linux may stop the engine in the middle of an answer. " if os.name != "nt" else
                 "Windows will slow down (paging to disk). ")
              + "Close other programs, or run START-HERE --setup and pick a smaller size (Q2_0 / IQ2_XS).", flush=True)


def lan_addresses() -> list[str]:
    """This PC's IPv4 addresses on its networks (what another device types in), without loopback/link-local."""
    import socket
    first, ips = None, set()
    try:                                                # the address of the default route; sends nothing (UDP)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            first = s.getsockname()[0]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ok = lambda ip: ip and not ip.startswith(("127.", "169.254.", "0."))
    return ([first] if ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    svc.listen = (host, httpd.server_address[1])         # (the dashboard's Port setting starts from it)
    httpd.response_service = svc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def index_html() -> bytes:
    """The page itself (serve/web/index.html), the favicon's cache-bust version from VERSION (a no-op when it carries
    no placeholder).  Both services serve it: the starting one and the real one (issue #20 phase 1)."""
    body = (ROOT / "serve" / "web" / "index.html").read_bytes()
    return body.replace(b"?v=__VERSION__", f"?v={SAGUARO_VERSION or ''}".encode())


class LoadState:
    """What the "starting" service answers with while the engine is not there yet (issue #20 phase 1): the model's
    name, and that the load runs.  main() hands the real Service over on the same socket (take_over); a load that
    fails says so here instead (fail), and the port keeps answering with it."""

    def __init__(self, model: str, log_path=None, api_key: str = "", supervisor: str | None = None, port=8095,
                 config_path=None):
        self.model, self.log_path = model, log_path
        self.api_key, self.supervisor, self.port = api_key, supervisor, port
        self.config_path = str(config_path) if config_path else None
        self.t0 = time.time()
        self.status, self.error = "loading", None
        self.failed_seen = threading.Event()      # a /health that told the supervisor about the failure

    def fail(self, error) -> None:
        """The load failed (issue #20): the standing page shows this, /health says {"status": "failed"} with it."""
        self.status, self.error = "failed", str(error) or type(error).__name__

    def last_log_line(self) -> str:
        """The engine log's last line, trimmed - what the loading screen shows while it waits ("" before the
        engine has written anything, or when this engine keeps no log)."""
        if not self.log_path:
            return ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 65536))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return ""
        for line in reversed(text.splitlines()):
            if line.strip():
                return line.strip()[:400]
        return ""

    def health(self) -> dict:
        # 503 with this body is REQUIRED (issue #20): llama-swap and the other supervisors take a 200 from /health
        # as ready, and would send requests to a model that is not there yet.  The shape: status, the config's
        # model name, how long the load has run, and what the engine was last saying.  A failed load adds the error
        # and, when a different config loaded before, what the page's button would start (never under a supervisor,
        # which owns starts and switches itself).
        out = {"status": self.status, "model": self.model, "elapsed_s": int(time.time() - self.t0),
               "line": self.last_log_line(), "supervisor": self.supervisor}
        if self.status == "failed":
            out["error"] = self.error
            if not self.supervisor:
                lg = read_last_good()
                if lg and lg["path"] != self.config_path:
                    out["last_good"] = lg
        return out


def last_good_note_path() -> Path:
    """Where "the config that loaded successfully last" is noted: the server's state folder, the one holding
    models.py's SWITCH_FILE note (saguaro.py's folder, where the maya-*.json configs live)."""
    return ROOT / models.LAST_GOOD


def read_last_good() -> dict | None:
    """The config path + time a successful load recorded; None when nothing was recorded or that config is gone."""
    try:
        note = json.loads(last_good_note_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    p = Path(str(note.get("config") or ""))
    if not p.is_file():
        return None
    return {"config": p.name, "path": str(p), "time": note.get("time")}


def write_last_good(config_path, model=None) -> None:
    """After a successful load (main): remember this config, so the page of a start whose load later fails can
    offer to start it again."""
    note = {"config": str(Path(config_path).resolve()), "time": time.time(), "model": model or ""}
    with contextlib.suppress(OSError):
        last_good_note_path().write_text(json.dumps(note), encoding="utf-8")


def ends_on_failed_load(supervisor, exit_on_fail) -> bool:
    """When a failed load ENDS the process (issue #20): a supervisor must not wait on a process that will never
    turn ready (llama-swap retries starts on its own), nor a run that asked for --exit-on-fail; a direct start
    stands with the page that says what went wrong."""
    return bool(supervisor) or bool(exit_on_fail)


def _restart_after_answer(delay: float = 0.5) -> None:
    """End this process so saguaro.py starts again with the noted config (its RESTART_EXIT loop): the page's
    answer still goes out, and the failed engine must not outlive this start."""
    time.sleep(delay)
    end_spawned()
    os._exit(RESTART_EXIT)


def start_last_good(state: LoadState) -> dict:
    """The failed page's button (issue #20): start again with the last config that worked.  The note saguaro.py
    reads on a RESTART_EXIT (models.SWITCH_FILE) is the same one the Models page's Switch leaves it - so this
    works when saguaro.py started this server, and only when no supervisor owns the starts."""
    if state.supervisor:
        raise ValueError(f"this server runs under {state.supervisor}, which starts, stops and switches the "
                         "models: change models there (pick another model id), not here")
    lg = read_last_good()
    if lg is None:
        raise ValueError("no last config that worked is known here")
    if lg["path"] == state.config_path:
        raise ValueError(f"{lg['config']} is the config this server tried and failed with")
    if not os.environ.get("SAGUARO_RESTART_ON_UPDATE"):
        raise ValueError("this server was not started by saguaro.sh or START-SAGUARO.bat, so it cannot start "
                         f"again by itself: start the one that worked: python saguaro.py --config {lg['path']}")
    (ROOT / models.SWITCH_FILE).write_text(json.dumps({"config": lg["path"], "port": state.port,
                                                       "why": "switch", "time": time.time()}), encoding="utf-8")
    print(f"[saguaro] starting again with {lg['config']} (the last config that worked) ...", flush=True)
    threading.Thread(target=_restart_after_answer, daemon=True).start()
    return lg


def make_loading_handler(state: LoadState):
    """The service that answers on the port from the first second of every start (issue #20 phase 1): the page
    (index, web/*, fonts) and /health while the engine loads, and everything the engine would answer says
    503 {"error": "the model is loading"} with Retry-After: 5 - requests may not wait unbounded.  The API key, when
    configured, is checked on everything the real handler checks it on; /health and the page's files stay open."""

    class LoadingHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # as the real handler: an answer ends the connection
        timeout = request_timeout()

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj, retry_after=None):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            if retry_after:
                self.send_header("Retry-After", str(retry_after))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:                      # (the real handler's check, against the config's key)
            if not state.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if given and any(hmac.compare_digest(given.encode(), k.encode()) for k in key_list(state.api_key)):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def _loading(self):
            self._json(503, {"error": "the model is loading"}, retry_after=5)

        def _foreign_page(self) -> bool:                    # (the real handler's CSRF check, without cors_origins:
            origin = self.headers.get("Origin")             #  there is no config to read them from yet)
            if not origin or origin.split("://", 1)[-1] == self.headers.get("Host", ""):
                return False
            self._json(403, {"error": {"type": "permission_error",
                                       "message": "requests from other web pages are refused"}})
            return True

        def _static(self, path) -> bool:
            """The page's own files - index, /web/ (styles, script, icons, manifest) and the fonts - open, as today.
            True when this answer was one of them."""
            if path == "":
                body, ctype = index_html(), "text/html; charset=utf-8"
            elif path == "/web":
                # the manifest's old start_url: the page is one level up, relative (the real handler's redirect)
                raw = self.path.split("?")[0]
                self.send_response(302)
                self.send_header("Location", ("../" if raw.endswith("/") else "./") + self.path[len(raw):])
                self.send_header("Content-Length", "0")
                self.end_headers()
                return True
            elif path.startswith("/web/"):
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json"}
                ext = os.path.splitext(name)[1]
                f = ROOT / "serve" / "web" / name
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return True
                body, ctype = f.read_bytes(), types[ext]
            elif path.startswith("/fonts/"):
                name = path[len("/fonts/"):]
                f = ROOT / "serve" / "web" / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return True
                body, ctype = f.read_bytes(), "font/woff2"
            else:
                return False
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if self._static(path):
                return
            if path == "/health":
                if state.status == "failed":
                    state.failed_seen.set()      # (the supervisor has now heard it from us: main() may end)
                self._json(503, state.health())
                return
            if self._authorized():
                self._loading()

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/api/last-good/start":              # (the failed page's button: start with that config)
                if self._authorized() and not self._foreign_page():
                    try:
                        lg = start_last_good(state)
                    except ValueError as e:
                        self._json(409, {"error": {"type": "invalid_request_error", "message": str(e)}})
                        return
                    self._json(200, {"starting": lg["config"]})
                return
            if self._authorized():
                self._loading()

        def do_DELETE(self):
            if self._authorized():
                self._loading()

        def do_OPTIONS(self):
            self._loading()        # (a preflight while nothing runs: the same "ask again in 5" as its requests)

    return LoadingHandler


def serve_loading(state: LoadState, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    """Bind the port before the engine starts, and answer on it from the first second (issue #20 phase 1): every
    start and restart used to leave the port dead for the minutes the model takes to load."""
    httpd = Server((host, port), make_loading_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def take_over(httpd, svc) -> None:
    """The real service takes the same listening socket: no gap, no second bind (issue #20 phase 1).  socketserver
    reads the handler class per connection, so the connection after this line gets the real handler."""
    svc.start_telemetry()
    svc.listen = (httpd.server_address[0], httpd.server_address[1])   # (the dashboard's Port setting starts from it)
    httpd.response_service = svc
    httpd.RequestHandlerClass = make_handler(svc)


SHARED_KEYS = ("reasoning_effort", "temperature", "top_p", "top_k", "seed", "max_tokens", "experimental_speed_projection",
               "repetition_penalty", "frequency_penalty", "presence_penalty")


def clean_shared_defaults(d) -> dict:
    """The Chat settings other apps get (POST /settings): only known keys, each checked; ValueError names a bad one."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("defaults must be an object")
    out = {}
    for key, value in d.items():
        if value is None or value == "":
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "reasoning_effort":   # (kept by its own name: medium / xhigh are MiniMax's, GLM's High / Max else)
            if effort_level(value) is None:
                raise ValueError(f"reasoning_effort: {EFFORT_WORDS}")
            value = effort_name(value)
        elif key == "temperature":
            if not number or not 0 <= value <= 2:
                raise ValueError("temperature: 0..2")
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise ValueError("top_p: 0 < top_p <= 1")
        elif key == "min_p":                            # the Chat's Min-p (the Creative preset's 0.05)
            if not number or not 0 <= value < 1:
                raise ValueError("min_p: 0 <= min_p < 1")
        elif key == "top_k":
            if not number or value != int(value) or not 0 <= value <= 64:   # (0: no top-k cut - the page's "off")
                raise ValueError("top_k: an integer 0..64")
            value = int(value)
        elif key in ("seed", "max_tokens"):
            if not number or value != int(value) or value <= 0:
                raise ValueError(f"{key}: a positive integer")
            value = int(value)
        elif key == "repetition_penalty":               # (the Chat's penalties: only those its user set)
            if not number or not 0 < value <= 10:
                raise ValueError("repetition_penalty: above 0, at most 10 (1 = off)")
        elif key in ("frequency_penalty", "presence_penalty"):
            if not number or not -2 <= value <= 2:
                raise ValueError(f"{key}: -2..2 (0 = off)")
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise ValueError("experimental_speed_projection: true or false")
        else:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = float(value) if key in ("temperature", "top_p", "min_p") or key.endswith("_penalty") else value
    return out


def chat_template_path(tpath) -> Path:
    """The template a model renders with (main): its tokenizer's chat_template.jinja (the pack's copy: MiniMax's and
    DeepSeek's are tools/mm3's and tools/ds4's - DeepSeek's rendered by its pinned encoder, serve/ds41_encoding.py -
    CYBER-FROST's its own, verbatim), a GLM-5 pack's swapped for serve/glm5_chat_template.jinja (SAGUARO_PACK_TEMPLATE=1
    keeps the pack's), and serve/chat_template.jinja when the tokenizer has none."""
    tpl = Path(tpath) / "chat_template.jinja"
    tpl = tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"
    glm5 = ROOT / "serve/glm5_chat_template.jinja"
    if glm5.exists() and os.environ.get("SAGUARO_PACK_TEMPLATE") != "1" and tpl != glm5:
        head = tpl.read_text(encoding="utf-8", errors="replace")
        if head.startswith("[gMASK]<sop>") and "reasoning_effort in ['low', 'high']" in head:
            tpl = glm5
    return tpl


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The run config's optional `sampling` block: defaults for the sampling fields a request leaves out, so
    a plain client gets configured sampling instead of greedy.  Supported: temperature, top_p, top_k, min_p,
    presence_penalty, repetition_penalty, frequency_penalty, penalty_last_n, seed.  The request's own fields
    always win - an explicit temperature=0 still means greedy, a field set to null falls back to the default.
    A bad value refuses to start the server (a typo'd config should not quietly change sampling); unknown keys
    are named at startup and ignored."""
    out = {}
    for key, value in (cfg.get("sampling") or {}).items():
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "temperature":
            if not number or value < 0:
                raise SystemExit(f"[saguaro] config sampling.temperature={value!r}: expected a number >= 0 (0 = greedy)")
            out[key] = float(value)
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise SystemExit(f"[saguaro] config sampling.top_p={value!r}: expected 0 < top_p <= 1")
            out[key] = float(value)
        elif key == "min_p":
            if not number or not 0 <= value <= 1:
                raise SystemExit(f"[saguaro] config sampling.min_p={value!r}: expected 0 <= min_p <= 1")
            out[key] = float(value)
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise SystemExit(f"[saguaro] config sampling.top_k={value!r}: the sampled path takes an integer 1..64")
            out[key] = int(value)
        elif key == "presence_penalty":
            if not number or value < 0:
                raise SystemExit(f"[saguaro] config sampling.presence_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "frequency_penalty":
            if not number or value < 0:
                raise SystemExit(f"[saguaro] config sampling.frequency_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "repetition_penalty":
            if not number or value <= 0:
                raise SystemExit(f"[saguaro] config sampling.repetition_penalty={value!r}: expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or value < -1:
                raise SystemExit(f"[saguaro] config sampling.penalty_last_n={value!r}: expected an integer >= -1 (0 = no "
                                 "penalties, -1 = the whole context)")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or value <= 0:
                raise SystemExit(f"[saguaro] config sampling.seed={value!r}: expected a positive integer")
            out[key] = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[saguaro] config sampling.experimental_speed_projection={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[saguaro] config sampling.{key}={value!r}: unknown key, ignored", flush=True)
    return out


def request_stop(stop: threading.Event, why: str) -> None:
    """The server's stop, once: main() wraps up the request in progress and ends the engine."""
    if not stop.is_set():
        stop.set()
        print(f"[saguaro] stopping ({why}): the engine ends", flush=True)


def wait_for_stop(stop: threading.Event, step: float = 0.5) -> None:
    """main()'s wait until the server stops. It polls the flag and never waits on the Event: the signal handlers
    (stop_signals) run on this same thread, between steps, and call stop.set() - while Event.wait holds the Event's
    lock, a SIGTERM / Ctrl+C landing in it made that stop.set() wait for the same lock forever (the server never
    ended; only SIGKILL did). time.sleep is interruptible on Windows too: Ctrl+C / Ctrl+Break act within a step."""
    while not stop.is_set():
        time.sleep(step)


# The programs this server started that must not outlive it (the engine, the vision encoder).  A server that ends -
# a signal while the engine loads, an exception, sys.exit - ends them on the way out (#1181: a SIGTERM during the load
# left the engine alone, holding the GPUs and the RAM tier).
_SPAWNED: list = []
_STARTING = [True]          # until main() has the server up: a stop signal ends the engine and the server at once


def track_spawned(proc) -> None:
    """Register a program the server started (engine.stop / close still end it the usual way)."""
    _SPAWNED[:] = [p for p in _SPAWNED if p.poll() is None]
    _SPAWNED.append(proc)


def end_spawned(grace: float = 5.0) -> None:
    """End every program this server started that still runs, with what it started in turn: asked first (SIGTERM),
    killed after `grace` seconds; on Windows the whole tree (taskkill /T) at once."""
    procs = [p for p in _SPAWNED if p.poll() is None]
    if not procs:
        return
    if os.name == "nt":
        for p in procs:
            try:
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                p.kill()
            except OSError:
                pass
    else:
        import signal
        kids = []
        for p in procs:
            try:
                import psutil
                kids += psutil.Process(p.pid).children(recursive=True)
            except Exception:  # noqa: BLE001 - no psutil: the child alone
                pass
        for p in procs:
            with contextlib.suppress(OSError):
                p.send_signal(signal.SIGTERM)
        end = time.monotonic() + grace
        for p in procs:
            with contextlib.suppress(Exception):
                p.wait(timeout=max(0.1, end - time.monotonic()))
            if p.poll() is None:
                with contextlib.suppress(OSError):
                    p.kill()
        for k in kids:
            with contextlib.suppress(Exception):
                if k.is_running():
                    k.kill()
    for p in procs:
        with contextlib.suppress(Exception):
            p.wait(timeout=10)


atexit.register(end_spawned)


def stop_signals(stop: threading.Event) -> None:
    """SIGTERM, SIGINT (Ctrl+C), SIGHUP (the terminal closed) and Windows' SIGBREAK (Ctrl+Break) stop the server
    (request_stop).  A later one is ignored while it stops - but a second Ctrl+C raises KeyboardInterrupt: at once.  A
    signal that is ignored (nohup) stays ignored.  main() installs them before it starts the engine: one that comes
    while the engine loads (nothing is serving yet, nothing to wrap up) ends the engine and the server right there."""
    import signal

    def on(signum, frame):
        if stop.is_set():
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            return
        request_stop(stop, signal.Signals(signum).name)
        if _STARTING[0]:
            end_spawned()
            raise SystemExit(128 + int(signum))
    for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        try:
            if sig is not None and signal.getsignal(sig) is not signal.SIG_IGN:
                signal.signal(sig, on)
        except (OSError, RuntimeError, ValueError):
            pass


def watch_parent(stop, stdin=None, getppid=None, poll: float = 1.0) -> None:
    """--parent-pipe: stdin is a pipe from the program that started the server (saguaro.py), which writes nothing and
    holds it open while it runs.  Its end - however that program ended: a crash, SIGKILL, Task Manager - closes it,
    and the EOF stops the server (`stop(why)`) as Ctrl+C does.  On POSIX a new parent process does too (the parent
    ended and init or a subreaper took the server).  Both watch from threads of their own."""
    src = stdin if stdin is not None else getattr(sys.stdin, "buffer", None)

    def closed_polling():
        # Windows: the CRT keeps a descriptor's lock for a whole blocking read(), and the first `import numpy` (a
        # picture, in a request's thread) waits on stdin's: the server froze (#444).  Polling the pipe holds no lock.
        import ctypes
        import msvcrt
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = wintypes.HANDLE(msvcrt.get_osfhandle(src.fileno()))
        avail = wintypes.DWORD()
        while k32.PeekNamedPipe(h, None, 0, None, ctypes.byref(avail), None):
            if avail.value:
                src.read1(avail.value)
            time.sleep(0.25)
        return ctypes.get_last_error() == 109      # ERROR_BROKEN_PIPE: the parent's end is closed

    def eof():
        try:
            if not (os.name == "nt" and stdin is None and closed_polling()):
                while src.read(1 << 12):
                    pass
        except (OSError, ValueError, AttributeError):
            pass
        stop("the program that started it ended")
    if src is not None:
        threading.Thread(target=eof, name="parent-pipe", daemon=True).start()
    if os.name == "nt" and getppid is None:
        return
    getppid = getppid or os.getppid
    first = getppid()

    def orphan():
        while getppid() == first:
            time.sleep(poll)
        stop("its parent process ended")
    threading.Thread(target=orphan, name="parent-pid", daemon=True).start()


class ServerLogTee:
    """stdout as it was, and the same text appended to the run's log (where the engine already writes its own lines).
    This server only printed to the terminal, so a run whose window was closed - or that was started by a script - left
    no record of what it did: a request that waited minutes for its turn, an engine restarted mid-session, a queue that
    stopped moving.  That record is what a report of a stuck server needs."""

    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            try:
                self.handle.write(text)
            except (OSError, ValueError):                # the log went away: the terminal still gets the line
                pass
        return self.stream.write(text)

    def flush(self):
        try:
            self.stream.flush()
        except (OSError, ValueError):
            pass
        with self.lock:
            try:
                self.handle.flush()
            except (OSError, ValueError):
                pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


class StampedStdout:
    """stdout with a local `HH:MM:SS ` before each of this server's own lines (`[saguaro] ...`, #783); the engine's lines and
    any other text pass through as they are.  A line is stamped when it starts, whatever pieces print() writes it in."""

    def __init__(self, stream):
        self.stream, self.at_start, self.lock = stream, True, threading.Lock()

    def write(self, text):
        with self.lock:
            out = []
            for seg in text.splitlines(keepends=True):
                if self.at_start and seg.startswith("[saguaro]"):
                    out.append(logctx.stamp(ms=True) + " ")   # (the engine's lines' format: one clock in the log)
                out.append(seg)
                self.at_start = seg.endswith(chr(10)) or seg.endswith(chr(13))
            if out:
                self.stream.write("".join(out))
        return len(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def tee_server_log(path: str) -> None:
    """This server's own lines (`[saguaro] ...`) go to `path` as well as this terminal (SAGUARO_SERVER_LOG=0: terminal
    only).  The engine's stderr is already appended there, so the file tells both sides of a request."""
    path = os.path.abspath(path)
    if not path or isinstance(sys.stdout, ServerLogTee) or isinstance(getattr(sys.stdout, "stream", None), ServerLogTee):
        return
    try:
        handle = open(path, "a", encoding="utf-8", buffering=1)
    except OSError as e:
        print(f"[saguaro] this server's lines cannot also go to {path} ({e}): they stay on this terminal", flush=True)
        return
    print(f"[saguaro] this server's lines go to {path} too", flush=True)
    if isinstance(sys.stdout, StampedStdout):           # (the tee under the stamps: the file gets the times too)
        sys.stdout.stream = ServerLogTee(sys.stdout.stream, handle)
    else:
        sys.stdout = ServerLogTee(sys.stdout, handle)


def supervisor_name() -> str | None:
    """The program that started this server when it is a model switcher that owns its lifecycle (llama-swap: our
    parent after its /bin/sh exec'd us), else None.  SAGUARO_SUPERVISOR=<name> says it outright (""): none."""
    try:
        comm = Path(f"/proc/{os.getppid()}/comm").read_text().strip()
    except OSError:
        return None
    return "llama-swap" if comm.startswith("llama-swap") else None


def start_qfn(cfg: dict, tok):
    """Qwen3.8-Flash-Next's engine (engines/qfn) under this server: QFN's own driver (serve/qfn_engine.py - its arguments,
    environment and batch protocol: the server-side slot admission, BGEN / BDONE, wrap spill, session files) behind this
    server's engine interface, and its picture encoder (vision-qfn: the same ENC protocol as vision-glm) resident, or
    elastic - not started until a picture arrives (the vision section's "elastic": true)."""
    from serve import qfn_engine as QE
    if not cfg:
        raise SystemExit("the batch driver needs a config (its exe names the *qfn engine)")
    env = QE.child_env(cfg)
    QE.configure_batch_telemetry_env(env, cfg, None)
    sampling_defaults = sampling_defaults_from_config(cfg)
    vision = None
    vcfg = cfg.get("vision") or None
    if vcfg:
        vlog = open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None
        if vcfg.get("elastic") is True:
            # the elastic encoder (QFN's own reader class): it stays down until Service._vision_up starts it
            vision = QE.Vision(dict(vcfg), log=vlog, env=QE.vision_env(cfg, env))
        else:
            print("loading the vision encoder ...", flush=True)
            vision = Vision(dict(vcfg), log=vlog, env=QE.vision_env(cfg, env))
    print("loading the model (the first start takes a minute or two) ...", flush=True)
    exe = cfg["exe"] if os.path.isabs(cfg["exe"]) else os.path.abspath(os.path.join(cfg.get("cwd") or ".", cfg["exe"]))
    QE.apply_ready_timeout(cfg)
    silence = QE.engine_silence_s(cfg)
    effort_end = QE.effort_end_args(cfg, exe, tok) or []
    t0 = time.time()
    engine = _qfn_engine_class()(exe, QE.engine_args(cfg) + effort_end, cwd=cfg.get("cwd"), log=cfg.get("log"), env=env)
    engine.silence_s = silence
    engine.load_s = round(time.time() - t0, 1)
    warn_tight_ram(engine.info.get("arena_mib"))
    return engine, vision, sampling_defaults


class _VlenLines(queue.Queue):
    """serve/qfn_engine's line queue with the VLEND / VRECLAIM answers diverted: the engine answers those commands
    while batch requests run, so VLENT / VRECLAIMED (and their ERR lines) go to the lend queue - never into a
    request's line stream.  PLAN-vram-live: the engine's periodic `VRAM dev=...` lines (its split follows the cards)
    go to `vram_cb` the same way - they are neither a token line nor an answer to the `VRAM <mib>` command, whose
    own answer carries no dev= and still reaches the driver."""

    def __init__(self, vlen: queue.Queue, info: dict | None = None, vram_cb=None):
        super().__init__()
        self.vlen, self.info, self.vram_cb = vlen, info, vram_cb

    def put(self, item, block: bool = True, timeout: float | None = None):
        if isinstance(item, str) and item.startswith("VRAM ") and " dev=" in item:
            if self.vram_cb is not None:
                self.vram_cb(item)
            return
        if isinstance(item, str) and item.startswith(("VLENT ", "VRECLAIMED ", "ERR VLEND:", "ERR VRECLAIM:")):
            self.vlen.put(item, block, timeout)
            return
        if isinstance(item, str) and item.startswith("INFO ") and self.info is not None:
            # an INFO after READY (the batch setup's vision_lend=1): it joins the engine's facts, no request sees it
            for kv in item.split()[1:]:
                k, _, v = kv.partition("=")
                self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            return
        super().put(item, block, timeout)


def _qfn_engine_class():
    from serve import qfn_engine as QE

    class QfnEngine(QE.StrataEngine):
        """QFN's driver as this server's engine: several conversations at once are its batch slots (the FIFO's places);
        the fair-slice PAUSE of the GLM engine does not apply (QFN admits and yields slots itself).  A STOP is its
        cut: SOLO, a stopped request keeps its conversation, so an answer or a half-read prompt continues from the
        token it reached (see keeps_stopped_read) - which is what lets a priority preemption cut a solo QFN read.
        In a batch slot a stopped request's conversation is kept for the answer, but its half-read prompt is not:
        a stopped batch read runs again from 0 on its next BGEN, so the preemption never cuts a batch read."""
        can_pause = False
        keeps_stopped_read = True      # R122: a stopped (cut) request keeps its conversation LIVE at the token it
        #  reached, also in the middle of a prompt read, and the next request continues from it or parks it -
        #  the SOLO GEN path only: engines/qfn/src/program/generate.cpp #1620 ("stopped between two windows ...
        #  it stays the live conversation", printed as "cancelled request keeps its conversation: N tokens live").
        #  A BATCH slot's cancelled read is dropped, not kept: "a read that failed keeps none" - the BRA 0 cancel
        #  and the BSTOP path clear the slot's read and ids and hand every checkpoint back to the pool
        #  (generate.cpp, the cancelled-read and BSTOP branches), and the driver frees the slot with nothing held
        #  for the next BGEN (leaving "queued" it drops its held tokens) - which is why Service._sliced fires the
        #  read preemption for this engine only while it runs solo (`not engine.batch`).
        vram_reserve = QE.StrataEngine.vram   # the VRAM give-back command; self.vram (below) is the Monitor's lines

        def __init__(self, *a, **kw):
            self._tls = threading.local()                # (this request's own prompt progress: Service._progress)
            self._reader = None                          # the thread reading its prompt now (QFN: one at a time)
            self.vram = []                               # the engine's start-up "VRAM dev=..." lines (gpu_map.parse)
            self.vlen_lines: queue.Queue = queue.Queue()  # the VLEND / VRECLAIM answers (see _pump), kept from lines
            super().__init__(*a, **kw)

        def _pump(self):
            """The driver's reader with the lend answers diverted: wrap the line queue before the pump starts, so a
            VLENT / VRECLAIMED line (or the ERR answering one) reaches vision_lend / vision_reclaim and no request."""
            if getattr(self, "vlen_lines", None) is None:
                self.vlen_lines = queue.Queue()
            if not isinstance(self.lines, _VlenLines):
                wrapped = _VlenLines(self.vlen_lines, self.info, self._vram_follow)
                while True:                              # (whatever the driver queued between start and here)
                    try:
                        wrapped.put_nowait(self.lines.get_nowait())
                    except queue.Empty:
                        break
                self.lines = wrapped
            super()._pump()

        def _vram_follow(self, line: str) -> None:
            """A `VRAM dev=...` line at any time (PLAN-vram-live: the engine's split follows the cards): the Monitor's
            map of that GPU follows it, as the GLM driver's _vram_line does - never a request's line, never the
            answer to a `VRAM <mib>` command (whose own answer carries no dev=)."""
            part = gpu_map.parse(line)
            if not part:
                return
            part["at"] = time.monotonic()
            hit = [v for v in self.vram if v.get("dev") == part["dev"]]
            if hit:
                self.vram = [part if v.get("dev") == part["dev"] else v for v in self.vram]
            else:
                self.vram = self.vram + [part]

        def vision_lend(self, mib: int, timeout: float = 45.0) -> dict:
            """`VLEND <mib>` (the engine's INFO says vision_lend=1): CUDA0 lends the picture encoder that much VRAM -
            its expert cache gives it at the next quiet moment while the batch slots keep decoding.  -> the VLENT
            line's figures (free_mib, cache_mib); raises ValueError with the engine's reason, EngineDied when it
            ended.  The room stays held until vision_reclaim (the engine keeps CUDA0's floor raised meanwhile)."""
            return self._vlen_command(f"VLEND {int(mib)}", "VLENT", timeout)

        def vision_reclaim(self, timeout: float = 45.0) -> dict:
            """`VRECLAIM`: the encoder is gone - drop the extra floor and let CUDA0's expert cache grow back over the
            free VRAM (refilled from the profile as the batch loop does).  -> {cache_mib}; ValueError, EngineDied."""
            return self._vlen_command("VRECLAIM", "VRECLAIMED", timeout)

        def _vlen_command(self, cmd: str, want: str, timeout: float) -> dict:
            if not self.alive():
                raise QE.EngineDied("the engine is not running")
            while True:                                    # (an answer to an earlier, timed-out command: stale now)
                try:
                    self.vlen_lines.get_nowait()
                except queue.Empty:
                    break
            self._send(cmd)
            end = time.monotonic() + timeout
            while True:
                try:
                    line = self.vlen_lines.get(timeout=max(0.1, end - time.monotonic()))
                except queue.Empty:
                    if self.ended or not self.alive():
                        raise QE.EngineDied("the engine ended") from None
                    raise LendTimeout(f"the engine did not answer the {cmd} command") from None
                if line is None:
                    raise QE.EngineDied("the engine ended")
                line = line.strip()
                if line.startswith("ERR"):
                    raise ValueError(line[4:].strip() or f"the engine refused the {cmd} command")
                if line.startswith(want):
                    out = {}
                    for kv in line.split()[1:]:
                        k, _, v = kv.partition("=")
                        out[k] = int(v) if v.lstrip("-").isdigit() else v
                    self.info["vlen"] = out
                    return out

        def _ready_pump(self, proc, out, note=None):
            """The driver's start-up reader wrapped: the engine's VRAM lines become self.vram (the base class reads up
            to READY and ignores what it is not); every line still passes through, and so does a VRAM answer after."""
            self.vram = []
            seen: queue.Queue = queue.Queue()
            threading.Thread(target=QE.StrataEngine._ready_pump, args=(proc, seen, note), daemon=True).start()
            while True:
                line = seen.get()
                if line is not None and line.startswith("VRAM "):
                    part = gpu_map.parse(line)
                    if part:
                        part["act_own"] = 1              # (act: everything else on the card - gpu_map.view uses NVML)
                        part["at"] = time.monotonic()    # (the Monitor says how fresh each card's split is)
                        self.vram.append(part)
                out.put(line)
                if line is None or line.startswith("READY"):
                    return

        def _take_control(self, *a, **kw):
            got = yield from super()._take_control(*a, **kw)
            if got is not False:
                self._reader = threading.get_ident()
            return got

        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            me = threading.get_ident()
            self._tls.own_pp = None
            cb = getattr(self._tls, "on_read", None)     # (engine-total read rate: Service._note_read)
            seen, base = 0, None
            hb_seen = 0.0                                # (U5 heartbeat: the last solo RHB line's time)
            solo = not int(getattr(self, "batch", 0) or 0)
            with contextlib.closing(super().generate(ids, max_new, sampling, cancel, embeddings)) as _inner:
                # (closed on any exit: Python 3.12 ran the inner STOP + drain only when the outer one was freed)
                for t in _inner:
                    hb = (getattr(self, "read_live", None) or {}).get("solo")   # U5 read heartbeat: the solo read's
                    if hb and (self._reader == me or solo) and hb.get("t", 0.0) > hb_seen:   #  own, once per RHB
                        hb_seen = hb["t"]                    # line (~1 per s; the line itself wakes this loop)
                        hcb = getattr(self._tls, "on_hb", None)
                        if hcb is not None:
                            hcb(hb)
                    if (self._reader == me or solo) and self.progress:
                        new = getattr(self, "progress_new", None)
                        rms = getattr(self, "progress_read_ms", None)
                        if new is not None:
                            # PLAN-qfn-per-request-read-stats: the engine counts THIS request's read itself - PP's
                            # new_read= from where the read started (a restored conversation's positions never look
                            # like reads) and read_ms= the time that read has actually run.  No RESUME base needed:
                            # this chat's row and the engine-total window both take the engine's own figures.
                            rate = new / (rms / 1000.0) if new and rms and rms > 0 else self.prefill_tok_s_mean
                            self._tls.own_pp = (self.progress, rate)
                            if new > seen:
                                seen = new
                                if cb:
                                    cb(new, rate, False, False, rms if rms and rms > 0 else None)
                        else:                                # (an engine before the fields: the old RESUME base)
                            self._tls.own_pp = (self.progress, self.prefill_tok_s_mean)
                            if cb and self.progress[0] > seen:   # the PP line's steps, counted from where this read starts:
                                if base is None:                # only from RESUME's reused prefix: without it (a restored
                                    r = int(getattr(self, "reused", 0) or 0)   # conversation) where the read starts is
                                    if not 0 < r <= self.progress[0]:          # unknown - count nothing until the exact
                                        continue_count = False                 # figures at its end (Service._read_exact)
                                    else:
                                        base, continue_count = r, True
                                if base is not None:
                                    seen = self.progress[0]
                                    cb(max(0, seen - base))
                    rd = getattr(self, "read_done", None)    # RDONE: this solo read ended with its exact figures
                    if rd is not None and (self._reader == me or solo):
                        self.read_done = None                # (consume: never the next request's)
                        if cb:
                            n_done, ms_done = rd
                            cb(n_done, n_done / (ms_done / 1000.0) if n_done and ms_done else None,
                               True, False, ms_done if ms_done and ms_done > 0 else None)
                    yield t

        def _admission_wait(self, slot, cancel, live):
            """SAGUARO_SLOT_ADMIT: the admission reads beside the decoding slots, on its own lines: BRP <slot>
            <done> <total> <ms> <tok/s> [new_read=<n> read_ms=<ms>] steps its progress, BRA <slot> 1 prompt=N
            reused=R read_ms=M the engine's own figures.  The queue view below feeds both into the engine-total
            read rate - with the per-request fields (PLAN-qfn-per-request-read-stats) it takes them straight:
            this chat's new_read over its own read_ms; an engine before the fields falls back to the position
            deltas since the read's first BRP and the BRA's totals - and passes every line through untouched."""
            start = getattr(self._tls, "on_read_start", None)
            if start is not None:
                start()                                  # (its BGEN is out: the engine's read starts now, not the slot wait)
            cb = getattr(self._tls, "on_read", None)
            if cb is None:
                return (yield from super()._admission_wait(slot, cancel, live))
            base = int(getattr(self._tls, "read_leg_base", 0) or 0)   # counted before this admission's leg (legs add up)
            tls = self._tls
            eng = self                                                 # (U5 heartbeat: read_live, kept by the pump)
            hb_seen = [0.0]                                            # the last RHB line's time for this slot

            class _ReadCounted:
                def __init__(self, q):
                    self._q = q
                    self.first = None

                def __getattr__(self, name):          # put/put_nowait/empty...: the driver's pump still feeds the queue
                    return getattr(self._q, name)

                def get(self, timeout=None):
                    line = self._q.get(timeout=timeout)
                    if line:
                        f = line.split()
                        try:
                            if f[:1] == ["RHB"]:      # U5 read heartbeat (PLAN-u5-read-heartbeat): this slot's
                                entry = (getattr(eng, "read_live", None) or {}).get("slots", {}).get(slot)
                                if entry is not None and entry.get("t", 0.0) > hb_seen[0]:  # the pump kept read_live
                                    hb_seen[0] = entry["t"]                                 # current before routing
                                    hcb = getattr(tls, "on_hb", None)                       #  the line itself
                                    if hcb is not None:
                                        hcb({**entry, "done": base + entry["done"]})        # (legs add up, as on BRP)
                            if f[:2] == ["BRP", str(slot)] and len(f) >= 4 and f[2].isdigit() and f[3].isdigit():
                                kv = dict(x.partition("=")[::2] for x in f[5:] if "=" in x)
                                rate = float(f[5]) if len(f) >= 6 else None
                                pos = int(f[2])
                                if kv.get("new_read") is not None:
                                    # PLAN-qfn-per-request-read-stats: the engine counts THIS request's read on the
                                    # slot's own line - new_read= its tokens from where the read started (never its
                                    # reused / parked prefix), read_ms= the time that read has actually run.  No
                                    # first-position delta, no shared reader's rate: the legs add up in base.
                                    new = int(float(kv["new_read"]))
                                    rms = float(kv.get("read_ms") or 0) or None
                                    own = new / (rms / 1000.0) if new and rms and rms > 0 else rate
                                    tls.own_pp = ((pos, int(f[3])), own)
                                    cb(base + new, own, False, False, rms)
                                else:                    # (an engine before the fields: the position-delta guess)
                                    if self.first is None or pos < self.first:
                                        self.first = pos                 # (where this read starts: the reused prefix)
                                    cb(base + pos - self.first)          # (positions read since: a real count)
                                    tls.own_pp = ((pos, int(f[3])), rate)  # (this chat's row shows its own read)
                            elif f[:2] == ["BRA", str(slot)] and f[2:3] == ["1"]:
                                kv = dict(x.partition("=")[::2] for x in f[3:] if "=" in x)
                                new = max(0, int(kv["prompt"]) - int(kv.get("reused") or 0))
                                ms = float(kv.get("read_ms") or 0)
                                tls.read_leg_base = base + new   # (a later leg of the same request counts from here)
                                cb(base + new, new / (ms / 1000) if ms > 0 and new else None, True, False,
                                   ms if ms > 0 else None)     # (the engine's own read_ms rides along: its clock)
                        except (KeyError, ValueError, IndexError):
                            pass                                       # (a line this view cannot read is none of its business)
                    return line

            q0 = self.slot_q[slot]
            self.slot_q[slot] = _ReadCounted(q0)
            try:
                return (yield from super()._admission_wait(slot, cancel, live))
            finally:
                if isinstance(self.slot_q[slot], _ReadCounted):
                    self.slot_q[slot] = q0

        @property
        def multi(self) -> int:
            return int(getattr(self, "batch", 0) or 0)

        def pause(self, keep: bool = False):
            pass

        def waiting_behind(self, seconds: float) -> set:
            return set()

        def started(self, rids) -> bool:
            return True

        # (set_context: the driver's own - it rewrites --max-context in the spawn command and the shared
        #  _restart_with() starts the engine again with it, as it does for the other engines)

        def stop(self, timeout: float = 60.0):
            pass   # (each request stops its own slot - BSTOP - when its consumer goes; no engine-wide STOP)

    return QfnEngine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mock"], default=None,
                    help="mock: the scripted engine (tests, no GPU). Without it the real engine runs and its driver "
                         "comes from the config (engine_kind); --config is then required")
    ap.add_argument("--config", help="the engine's config (JSON: exe, args, cwd, tokenizer, model_name), "
                                     "written by setup.py; the exe picks the driver (engine_kind)")
    ap.add_argument("--host", default=None,
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other devices "
                         "on your network (set an API key); also \"host\" in the config")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--gpu", help="the GPU to run on, as nvidia-smi numbers them, or several for a layer split "
                                  "(\"0,2\"; also \"gpu\" in the config)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="pack tokenizer directory (falls back to a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help='unload the model after this many seconds without requests, so other programs can use the GPU '
                         'memory and the RAM; the next request loads it again (also "idle_unload_s" in the config; '
                         '0: never)')
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request "
                         "(default: reject with 400, like llama.cpp; also \"fit_max_tokens\": true in the config)")
    ap.add_argument("--api-key", default=os.environ.get("SAGUARO_API_KEY", ""),
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); several separated by "
                         "commas; also $SAGUARO_API_KEY")
    ap.add_argument("--mcp-config", help="a JSON file with MCP servers in Claude Desktop's format ({\"mcpServers\": "
                                         "{...}}); the web app's chat can use their tools (also \"mcp_servers\" in "
                                         "the config)")
    ap.add_argument("--parent-pipe", action="store_true",
                    help="stdin is a pipe from the program that started this server (saguaro.py), open while it runs: "
                         "when it closes - that program ended, however it ended - the server stops as on Ctrl+C "
                         "(on Linux also when its parent process ends)")
    ap.add_argument("--exit-on-fail", action="store_true",
                    help="end the process when the model fails to load, instead of keeping the port up with the "
                         "page that says what went wrong (under a supervisor the process ends on a failed load "
                         "either way)")
    a = ap.parse_args()
    if a.config:                                         # (--config given as the old name after the 1.2.2 rename:
        a.config = str(config_names.resolve(Path(a.config)))   #  a path llama-swap or systemd was given keeps working)
    clean_stale_picture_dirs()
    stop = threading.Event()
    stop_signals(stop)                                  # before any engine starts (#1181); ends it if one comes then
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig")) if a.config else {}   # (a BOM: Notepad)
    kind = "mock" if a.engine == "mock" else engine_kind(cfg)
    if kind is None:
        ap.error("needs --config (its exe picks the driver: the batch driver for a Qwen3.8-Flash-Next config, the "
                 "line driver for the others), or --engine mock for the scripted engine")
    apply_server_env(cfg)
    if not isinstance(sys.stdout, StampedStdout):       # (#783: a time before each [saguaro] line, in the log too)
        sys.stdout = StampedStdout(sys.stdout)
    # the server's own lines go to the run's log as well (the engine writes its stderr there): a run whose terminal
    # was closed leaves the record a stuck queue needs (SAGUARO_SERVER_LOG=0 keeps them on the terminal only)
    _srv_log = os.environ.get("SAGUARO_SERVER_LOG")
    if _srv_log is None:
        _srv_log = str(cfg.get("log") or "")
    if _srv_log.strip().lower() not in ("", "0", "off", "no", "false", "none"):
        tee_server_log(_srv_log)
    if a.gpu is not None:
        cfg["gpu"] = int(a.gpu) if a.gpu.strip().isdigit() else a.gpu
    a.host = a.host or cfg.get("host") or "127.0.0.1"   # issue #26: the run scripts pass no --host, the config can
    api_key = a.api_key or cfg.get("api_key", "")
    supervisor = os.environ.get("SAGUARO_SUPERVISOR") or supervisor_name()
    # the page answers on this port from the first second of the load (issue #20); this bind doubles as the old
    # "is the port free" check: a second start stops here, before the minutes of loading
    load_state = LoadState(cfg.get("model_name", "qwen3.8-flash-next"), log_path=cfg.get("log"),
                           api_key=api_key, supervisor=supervisor,
                           config_path=str(Path(a.config).resolve()) if a.config else None)   # (as write_last_good keeps it)
    try:
        httpd = serve_loading(load_state, host=a.host, port=a.port)
    except OSError:
        ap.error(f"port {a.port} is already in use - is Saguaro (or another server) already running? "
                 f"Close it, or start this one with a different --port")
    load_state.port = httpd.server_address[1]
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer)
    if kind in ("line", "batch") and not (tpath / "vocab.json").exists():
        ap.error(f"the model's tokenizer is missing ({tpath / 'vocab.json'}); run setup again")
    try:                                                    # the load (issue #20): if it fails, the port keeps
        if (tpath / "vocab.json").exists():                 #  answering with the failed page instead of dying
            import strata_tokenizer as ST
            tok = ST.Tokenizer.from_dir(tpath)              # with the pack's own pre-tokenizer (GLM: glm4, #27)
        hub = hub_from_config(cfg, a.mcp_config)            # before the minutes of loading: a bad entry stops here
        if kind == "line":
            vision = None
            env = child_env(cfg)
            if note := ds4_one_gpu_switch_note(cfg, env):
                print(note, flush=True)
            sampling_defaults = sampling_defaults_from_config(cfg)
            if "sampling" not in cfg:                       # (a config without them: the model's own, not none)
                sampling_defaults = maya_quants.model_defaults(cfg)[0]
            if sampling_defaults:
                pretty = ", ".join(f"{k}={v}" for k, v in sampling_defaults.items())
                print(f"[saguaro] sampling defaults from the {'config' if 'sampling' in cfg else 'model'}: {pretty}",
                      flush=True)
            vcfg_in = cfg.get("vision") or None
            vlog = (open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None) if vcfg_in else None
            # pictures: the encoder's environment, whether it runs on demand, and the engine's environment (one function,
            # also the bench's: tools/maya_bench.py start_command)
            vcfg, venv, on_demand, fp, engine_env = vision_plan(cfg, env, Path(a.config).resolve().with_name("vision-memory.json"))
            if on_demand:
                print(f"[saguaro] images: the vision encoder needs {fp[1] / 2**30:.2f} GB on this GPU at up to {fp[0]} "
                      "image tokens", flush=True)
            elif vcfg_in and vcfg_in.get("gpu") and vcfg and not vcfg.get("gpu"):
                print("[saguaro] images: the vision encoder could not be measured on the GPU - it runs on the CPU",
                      flush=True)
            if vcfg and not on_demand:
                print("loading the vision encoder ...", flush=True)
                vision = Vision(vcfg, log=vlog, env=venv)
            print("loading the model (the first start takes a minute or two) ...", flush=True)
            if len(gpu_list(cfg)) > 1 and models.family(cfg) != "ik2":
                print(f"[saguaro] layer split across GPUs {gpu_list(cfg)} ({cfg.get('layer_split') or 'auto'})", flush=True)
            t0 = time.time()
            engine = StrataEngine(cfg["exe"], engine_args(cfg), cwd=cfg.get("cwd"), log=cfg.get("log"), env=engine_env)
            engine.load_s = round(time.time() - t0, 1)      # (the dashboard's estimate of a reload: this start's time)
            warn_tight_ram(engine.info.get("arena_mib"))
            if vision is None and not vcfg and engine.info.get("vision") == "ds4":
                # DeepSeek-V4.1: the engine's own vision tower (the checkpoint's); the server resizes the pictures
                vision = ds4_vision.Vision(ds4_vision.Geometry.from_info(engine.info),
                                           picture_dir("saguaro-ds4-pictures-"), Vision.load)
                room = engine.info.get("vision_room")
                where = (f"{engine.info.get('vision_mib')} MiB kept" if room in (None, "resident") else
                         f"none between pictures, ~{engine.info.get('vision_room_mib')} MiB while one is read, lent by "
                         + {"chunk": "the prompt buffers", "tail": "expert slots", "free": "free memory"}.get(room, room))
                print(f"[saguaro] images: the model's vision tower runs in the engine (GPU {engine.info.get('vision_gpu')}: "
                      f"{where}), up to {vision.g.max_tokens} tokens a picture", flush=True)
            elif vision is None and not vcfg and engine.info.get("vision") == "hy4":
                # Hy4-flash: the engine's own vision tower (the mmproj file in its pack); the server resizes the pictures
                vision = hy4_vision.Vision(hy4_vision.Geometry.from_info(engine.info),
                                           picture_dir("saguaro-hy4-pictures-"), Vision.load)
                # (where a picture's room comes from - the engine's `vision_room`: "chunk" = lent by its prompt buffers)
                where = (f"; no VRAM kept for pictures (~{engine.info.get('vision_room_mib')} MiB while one is read, lent "
                         "by the prompt buffers)" if engine.info.get("vision_room") == "chunk" else "")
                print(f"[saguaro] images: the model's vision tower runs in the engine, up to {vision.g.max_pixels} pixels "
                      f"({vision.g.max_tokens} tokens for a square one) a picture{where}", flush=True)
            elif vision is None and not vcfg and engine.info.get("vision") == "mm3":
                # MiniMax-M3.1-Flash: the engine's own vision tower; the server resizes the pictures
                vision = mm3_vision.Vision(mm3_vision.Geometry.from_info(engine.info),
                                           picture_dir("saguaro-mm3-pictures-"), Vision.load)
                print(f"[saguaro] images: the model's vision tower runs in the engine, up to {vision.g.max_pixels} pixels "
                      f"({vision.g.max_tokens} tokens) a picture", flush=True)
            elif vision is None and not vcfg and engine.info.get("vision") == "off":
                # (DeepSeek-V4.1: its tower did not load - the engine's NOTE line above says why; no pictures offered)
                print("[saguaro] images: OFF - pictures are refused until the vision tower loads (see the line above)",
                      flush=True)
            if on_demand:
                lent = int(engine.info.get("vision_lend") or 0)
                if lent > 0:
                    vision = Vision(vcfg, log=vlog, env=venv, start=False)
                    vision.lend = lambda: engine.command("VLEND", "VLENT")
                    vision.reclaim = lambda: engine.command("VRECLAIM", "VRECLAIMED")
                    print(f"[saguaro] images: the vision encoder starts when a picture arrives, in {lent / 2**30:.2f} GB "
                          "the model lends it", flush=True)
                elif models.family(cfg) == "ik2":
                    # Inkling 2 (its engine lends no memory and keeps none for pictures): the encoder starts when a picture
                    # arrives - on the GPU when its free memory then holds what the encoder measured, else on the CPU -
                    # and ends once the request's pictures are read
                    need, cards = fp[1], gpu_list(cfg) or [0]
                    free_of = amd_gpu_free_bytes if cfg.get("backend") == "hip" else gpu_free_bytes   # (Windows: the Monitor finds the engine's cards with its hipInfo)
                    vision = Vision(vcfg, log=vlog, env=venv, start=False)
                    vision.room = lambda: ik2_vision_gpu(cards, need, free_of)
                    vision.cpu_threads = max(1, (os.cpu_count() or 2) // 2)
                    free = {g: (free_of(g) or 0) for g in cards}
                    print(f"[saguaro] images: the vision encoder starts when a picture arrives - on "
                          f"{'the GPU of ' + ', '.join(map(str, cards)) + ' with the most free memory' if len(cards) > 1 else f'GPU {cards[0]}'}"
                          f" while {need / 2**30:.2f} GB of it is free ("
                          + ", ".join(f"GPU {g}: {b_ / 2**30:.2f} GB" for g, b_ in free.items()) +
                          " now), else on the CPU; no VRAM kept for it", flush=True)
                else:   # the model's GPU memory is too small to lend that much (or an engine without lending)
                    print("[saguaro] images: the model cannot lend the vision encoder its GPU memory here - the encoder "
                          "runs on the CPU", flush=True)
                    vision = Vision(dict(vcfg, gpu=False), log=vlog, env=venv)
        elif kind == "batch":
            engine, vision, sampling_defaults = start_qfn(cfg, tok)
        else:
            engine, vision, sampling_defaults = MockEngine(tok, a.script or [
                "Thinking about it.</think>\n\nHello from the mock engine."],
                tok_s=float(os.environ.get("SAGUARO_MOCK_TOK_S") or 0) or None), None, {}
    except Exception as e:                                  # the load failed (issue #20): /health now says
        load_state.fail(e)                                  #  {"status": "failed", "error": ...}, the page shows it
        print(f"[saguaro] the model failed to load: {e or type(e).__name__}", flush=True)
        if ends_on_failed_load(supervisor, a.exit_on_fail):
            # a supervisor owns the starts: llama-swap retries a start whose process ends, but must not wait on one
            # that stands with a page it will never look at.  Give its /health poll one sight of the failure (rather
            # than a bare closed port), but never wait forever for a check that may never come.  With --exit-on-fail
            # alone nobody polls: end at once.
            if supervisor:
                load_state.failed_seen.wait(60)
            httpd.shutdown()
            end_spawned()                                   # a half-started engine must not outlive this start
            return 1
        here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
        print(f"[saguaro] the page at http://{here}:{load_state.port}/ says what went wrong; Ctrl+C closes it",
              flush=True)
        try:
            wait_for_stop(stop)                             # the standing failed page (a direct start)
        except KeyboardInterrupt:
            pass
        httpd.shutdown()
        end_spawned()
        return 1
    # the model's own chat template (exported with its tokenizer), else the original model's.  A GLM-5.3-Flash pack's
    # copy is the one its GGUF carried (August's): the template zai-org publishes now renders the same prompts and
    # matches tool results to their calls without nested loops (zai-org/GLM-5.3-Flash PR #63: 100 parallel calls
    # 137 -> 6 ms a render), so the server uses the copy shipped here (serve/glm5_chat_template.jinja) for a GLM-5
    # pack.  SAGUARO_PACK_TEMPLATE=1 keeps the pack's own.
    tpl = chat_template_path(tpath)
    svc = Service(engine, tok, ChatTemplate(tpl),
                  model_name=cfg.get("model_name", "qwen3.8-flash-next"), vision=vision,
                  sampling_defaults=sampling_defaults,
                  fit_max_tokens=a.fit_max_tokens or cfg.get("fit_max_tokens") is True)
    if getattr(engine, "multi", 0) > 1:                 # n conversations at once: n places in the FIFO
        svc.fifo.set_capacity(engine.multi)
        print(f"[saguaro] {engine.multi} conversations at once (--parallel)", flush=True)
    svc.start_vision_idle()                              # (the elastic QFN encoder's idle stop; a no-op for the rest)
    if "reasoning_effort" not in cfg and a.engine == "maya":   # (a config without one: the model's own level)
        cfg = dict(cfg, reasoning_effort=maya_quants.model_defaults(cfg)[1])
    if cfg.get("reasoning_effort") or (svc.template.deepseek and cfg.get("reasoning_effort") is not None):
        try:
            (deepseek_effort_kwargs if svc.template.deepseek else effort_kwargs)(cfg["reasoning_effort"])
        except ValueError as e:
            raise SystemExit(f"[saguaro] config reasoning_effort: {e}")
        svc.default_effort = cfg["reasoning_effort"]
        # (#363: DeepSeek's 1..100 budget is a level of its own - it was shown as "none")
        level = (effort_name if svc.template.minimax or getattr(svc.template, "inkling", False) else effort_level)(
            svc.default_effort) or \
            (f"{svc.default_effort} of 100" if type(svc.default_effort) is int
                                                     else "none")
        if getattr(svc.template, "forced_thinking", False):   # (a Qwen3.x template's own words: low / medium / xhigh)
            level = QWEN_EFFORT.get(level, level)
        if getattr(svc.template, "hy4", False):               # (Hy4-flash: the level it runs at, none / high)
            level = svc.template.effort_shown(svc.default_effort) or level
        print(f"[saguaro] thinking level when a request names none: {level}"
              f"{'' if level in (svc.default_effort, f'{svc.default_effort} of 100') else f' (the config says {svc.default_effort})'}",
              flush=True)
    svc.think_budget = int(cfg["thinking_budget"] if cfg.get("thinking_budget") is not None else
                           cfg["reasoning_budget_tokens"] if kind == "batch" and cfg.get("reasoning_budget_tokens") else
                           maya_quants.default_thinking_budget(cfg) if kind == "line" else 32768)         if svc.think_end_id is not None else 0
    if maya_quants.family_of(cfg) == "ik2":             # (saguaro-ik2 closes no thinking at a budget)
        svc.think_budget = 0
    svc.idle_unload_s = max(0.0, a.idle_unload if a.idle_unload is not None else float(cfg.get("idle_unload_s") or 0))
    if svc.think_budget:
        print(f"[saguaro] thinking budget: {svc.think_budget} tokens (then the answer)", flush=True)
    svc.model_management = model_management_setting(cfg)   # ("model_management": true: llama.cpp's load/unload/sse)
    if svc.model_management:
        print("[saguaro] model management: on (/models/load, /models/unload, /models/sse)", flush=True)
    svc.load_by_name = load_by_name_setting(cfg)           # ("load_by_name": true, with model management: a chat
    if svc.load_by_name:                                   #  request naming another installed model switches to it)
        print("[saguaro] load by name: a chat request whose \"model\" names another installed model switches to it",
              flush=True)
    svc.loop_guard = loop_guard_setting(cfg)            # (SAGUARO_LOOP_GUARD=0 or "loop_guard": false: off)
    if not svc.loop_guard:
        print("[saguaro] loop guard: off (a request's \"loop_guard\": true turns it on for that request)", flush=True)
    svc.priority_default = priority_setting(cfg)        # (R122: "priority": the level a request without one runs at)
    if svc.priority_default is not None:
        print(f"[saguaro] request priority when a request names none: {PRIORITY_NAMES[svc.priority_default]}",
              flush=True)
    svc.api_key = api_key
    try:
        svc.cors_origins = parse_cors_origins(cfg.get("cors_origins"))
    except ValueError as e:
        raise SystemExit(f"config: {e}")
    if svc.cors_origins:
        print("[saguaro] browser pages allowed to call /v1/* (CORS): " + ", ".join(svc.cors_origins), flush=True)
    svc.gpu_index = (gpu_list(cfg) or [0])[0]           # the Monitor reads the card the engine runs on (issue #51)
    svc.gpu_indices = gpu_list(cfg)                     # ... or every card of a layer split (issue #112)
    svc.backend = cfg.get("backend")                    # the Monitor's GPU reader: sysfs on AMD/HIP, NVML otherwise
    if cfg.get("backend") == "hip":                      # (Windows: the Monitor finds the engine's cards with its hipInfo)
        from serve import telemetry
        telemetry.HIP_BIN_DIRS = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    svc.supervisor = supervisor                           # (llama-swap: it owns starts, stops, switches and configs)
    if svc.supervisor:
        print(f"[saguaro] started by {svc.supervisor}: model switching, updates, port and config changes are its own",
              flush=True)
    svc.engine_kind = "qfn" if kind == "batch" else (maya_quants.family_of(cfg) if cfg else "mock")
    svc.model_label = cfg.get("label") or (maya_quants.config_label(cfg)   # (configs of every release: their installer record)
                                          if (cfg.get("installer") or {}).get("written") else None)   # (else the id)
    svc.model_name_pretty, svc.model_quant = maya_quants.display_names(cfg) if cfg else (None, None)
    # (/health and /metrics name the model in two parts: the page never splits the id to guess the quant)
    if a.config:
        svc.config_path = str(Path(a.config).resolve())  # a context change from the dashboard is saved there
        p = Path(svc.config_path)                        # the tokens it serves, and what its API would have charged
        svc.savings = savings.Ledger(savings.ledger_path(p), p.name, svc.model, models.quant_label(cfg, p),
                                     names=(svc.model_name_pretty, svc.model_quant))   # (/api/savings rows name the
                                                                                        #  model as /health does)
        threading.Thread(target=svc.finish_delete, daemon=True).start()   # (the model before, if the page deleted it)
    args = [str(x) for x in cfg.get("args") or []]
    flag = next((f for f in ("--glm-pack", "--gguf") if f in args[:-1]), None)   # (Metal: the first GGUF file, no pack)
    if flag:                                            # the model's folder: the parent of the pack or GGUF file
        svc.model_dir = Path(args[args.index(flag) + 1]).resolve().parent
        threading.Thread(target=lambda: setattr(svc, "trained_context", trained_context(svc.model_dir)),
                         daemon=True).start()
    elif "--pack" in args[:-1]:                         # DeepSeek's pack (its ds4.txt), MiniMax's (mm3.txt); its folder
        svc.trained_context = trained_context_ds4(args[args.index("--pack") + 1])
        svc.model_dir = Path(args[args.index("--pack") + 1]).resolve().parent
        # (DeepSeek's engine drafts unless --dspark off / none - "none" does not even load the blocks - when the pack
        #  has them; MiniMax's says so itself, INFO dspark=)
        if models.family(cfg) == "deepseek":
            svc.dspark = models.flag(args, "--dspark", "on") not in ("off", "none") and models.ds4_has_drafter(cfg)
            try:
                svc.lookup = int(models.flag(args, "--suffix-draft", "3")) > 0
            except ValueError:
                svc.lookup = True
    if a.config:                                        # the Chat settings shared with other apps, from last time
        svc.shared_path = str(Path(a.config).with_suffix("")) + ".shared-settings.json"
        try:
            svc.shared = clean_shared_defaults(json.loads(Path(svc.shared_path).read_text(encoding="utf-8")))
            if svc.shared:
                print("[saguaro] other apps use the Chat settings: " +
                      ", ".join(f"{k}={v}" for k, v in svc.shared.items()), flush=True)
        except (OSError, ValueError):
            svc.shared = {}
        # the model's live Turbo (PLAN-turbo-live): its sidecar file, beside the config but never in it
        svc.turbo_path = str(Path(a.config).with_suffix("")) + ".turbo.json"
        svc.fifo.set_one(models.turbo_one_at_a_time(cfg))   # (Turbo left on, holding one answer at a time)
        try:
            svc.turbo_default = turbo_live.turbo_on(turbo_live.normalize_turbo(
                json.loads(Path(svc.turbo_path).read_text(encoding="utf-8"))))
            if svc.turbo_default:
                print(f"[saguaro] live Turbo for this model: drop {svc.turbo_default['drop']}, "
                      f"skip {svc.turbo_default['skip']} (requests without their own)", flush=True)
        except (OSError, ValueError):
            svc.turbo_default = None
    if hub is not None:
        import atexit
        svc.mcp = hub
        print(f"[saguaro] starting {len(hub.servers)} MCP server{'s' * (len(hub.servers) != 1)} for the web app's "
              f"chat: {', '.join(hub.servers)}", flush=True)
        hub.start()
        atexit.register(hub.close)                      # the servers Strata started end with it
    if a.config:                                           # this config loaded: remember it (issue #20), so the
        write_last_good(a.config, cfg.get("model_name"))   #  page of a later start that fails can offer it again
    take_over(httpd, svc)                                  # (the same socket, from the first second to the end)
    svc.start_idle_unload()
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
    print(f"ready: http://{here}:{a.port}/v1  (OpenAI: /v1/chat/completions, /v1/responses, Anthropic: /v1/messages, "
          f"context {engine.max_context} tokens{', images on' if vision else ''}"
          f"{', API key required' if svc.api_key else ''})", flush=True)
    print(f"       open http://{here}:{a.port}/ in a browser to chat; close this window to stop the model", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        # issue #26: reachable from other devices - say at which address, and what can still block it
        ips = lan_addresses()
        for ip in ips:
            print(f"       from other devices: http://{ip}:{a.port}/   (API: http://{ip}:{a.port}/v1)", flush=True)
        if not ips:
            print("       from other devices: http://<this PC's IP address>:" + str(a.port) + "/", flush=True)
        if not svc.api_key:
            print("       WARNING: no API key - anyone on your network can use this model. Add \"api_key\": \"...\" "
                  "to the config (clients send it as their API key; the web page asks for it)", flush=True)
        if os.name == "nt":
            print("       nothing arrives? Windows Firewall blocks it until allowed: accept its prompt for Python, or run "
                  "in an admin PowerShell:\n         New-NetFirewallRule -DisplayName \"Saguaro " + str(a.port) + "\" "
                  "-Direction Inbound -Protocol TCP -LocalPort " + str(a.port) + " -Action Allow -Profile Private\n"
                  "       (and set this network to Private in Windows' network settings)", flush=True)
    if a.open:
        import webbrowser
        webbrowser.open(f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}/")
    # SIGTERM (llama-swap's cmdStop, systemd), SIGHUP (the terminal closed), Ctrl+C, Windows' Ctrl+Break and - with
    # --parent-pipe - the end of saguaro.py (watch_parent) end the server alike: the engine gets QUIT and time to keep its
    # conversation.  A second signal while that runs (systemd signals the whole group, then the supervisor again) is
    # ignored instead of cutting it short - except a second Ctrl+C, which ends the server at once (the engine then ends
    # when its input closes).
    _STARTING[0] = False                                # (the signals were installed before the engine started)
    if a.parent_pipe:
        watch_parent(lambda why: request_stop(stop, why))
    try:
        wait_for_stop(stop)
    except KeyboardInterrupt:
        request_stop(stop, "Ctrl+C")
    try:
        httpd.shutdown()
        # a request in progress is wrapped up, not dropped: its thinking is closed and it answers, for up to
        # SAGUARO_ENGINE_WRAP_S seconds (40), so the client still gets an answer and the turn ends; past that it is cut.
        # Requests still queued are refused. Then the engine ends (and keeps its conversation).
        svc.stopping = True
        wrap_s = float(os.environ.get("SAGUARO_ENGINE_WRAP_S", "40") or 0)
        if wrap_s > 0 and hasattr(engine, "wrap") and svc.status.get("busy"):
            print(f"[saguaro] stopping: the request in progress closes its thinking and answers (up to {wrap_s:.0f} s)",
                  flush=True)
            engine.wrap()
            if svc.fifo.acquire(timeout=wrap_s):
                svc.fifo.release()
                print("[saguaro] stopping: the request in progress finished", flush=True)
            else:
                print(f"[saguaro] stopping: the answer was not done after {wrap_s:.0f} s - cut there", flush=True)
                engine.stop()
                # the cut request ends at its next step (DONE, or an error if the engine died): only then QUIT, so its
                # last lines are read by the request and not by close()
                with svc.fifo:
                    pass
        if hasattr(engine, "close"):
            engine.close()
        if vision:
            vision.close()
        if hub is not None:
            hub.close()
    except KeyboardInterrupt:                          # a second Ctrl+C: at once
        print("[saguaro] stopped at once: the engine ends as its input closes", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
