#!/usr/bin/env python3
"""tools/saguaro/smoke_profiles.py - after a deploy, every profile answers cleanly through llama-swap.

    python3 tools/saguaro/smoke_profiles.py [--key-file PATH] [--base-url http://127.0.0.1:8080] [model-id ...]

Default models: the llama-swap ids of ref-configs/*.json ("model_name").  For each one, through llama-swap:

  * POST <base>/v1/chat/completions with stream=true and a short prompt: every SSE line parses (a `data:` field with
    JSON or [DONE], keep-alive comments and other SSE fields allowed), the answer text is sane - non-empty and with
    none of the engine's protocol fragments leaked into it ("T 0 ", "VRAM dev=", "PP ", "BRP" lines that made it
    into the content instead of the log) - and the stream closes with a clean finish_reason (no error/abort);
  * GET <base>/upstream/<id>/api/context answers a JSON object carrying kv_bytes_token (an engine that never printed
    its K/V cost leaves the Monitor's Context bar without a figure).

The API key is read from --key-file (a file holding just the key); it is never printed.  One PASS/FAIL line per
model with its reasons, exit 1 on any FAIL, exit 2 on a usage or key-file problem.  A cold profile can take minutes
to load: --timeout is the whole budget per model (stream read included).

    python -m unittest tools.test_smoke_profiles      # offline: no server (fixtures under tools/saguaro/fixtures/)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = "http://127.0.0.1:8080"
PROMPT = "Reply with exactly one short sentence."
FINISH_OK = ("stop", "length", "tool_calls", "end_turn", "content_filter")
# the engine's protocol lines, verbatim, if they ever leak into an answer's text (the deploy's symptom: a chat that
# shows engine bookkeeping - serve/server.py's parser dropped a line, or a new engine print landed on the line stream)
LEAKS = (("T 0 ", re.compile(r"T \d ")), ("VRAM dev=", re.compile(r"VRAM dev=")),
         ("PP ", re.compile(r"PP ")), ("BRP", re.compile(r"BRP")))


def default_model_ids(ref_dir: Path | None = None) -> list[str]:
    """The llama-swap ids: every ref-configs/*.json with a "model_name" (token-savings.json and friends don't have one)."""
    out = []
    for f in sorted((ref_dir or ROOT / "ref-configs").glob("*.json")):
        try:
            name = json.loads(f.read_text(encoding="utf-8")).get("model_name")
        except (OSError, ValueError):
            continue
        if isinstance(name, str) and name:
            out.append(name)
    return out


def load_key(path: str) -> str:
    """The API key from a file (just the key; surrounding whitespace stripped).  The content never goes anywhere else."""
    return Path(path).read_text(encoding="utf-8").strip()


def find_leaks(text: str) -> list[str]:
    """Which protocol fragments appear in this text (a leak of the engine's line stream into the answer)."""
    return [name for name, rx in LEAKS if rx.search(text)]


def sse_decode(raw: str) -> tuple[list[dict], bool, list[str]]:
    """Every line of one streamed response: (the data: JSON events, [DONE] seen, parse errors)."""
    events, errors, done = [], [], False
    for n, line in enumerate(raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        if not line or line.startswith(":"):                     # (keep-alive comment, blank between events)
            continue
        m = re.match(r"(data|event|id|retry): ?(.*)$", line)
        if m is None:
            errors.append(f"SSE line {n}: not an SSE field: {line[:80]!r}")
            continue
        if m.group(1) != "data":
            continue
        payload = m.group(2).strip()
        if payload == "[DONE]":
            done = True
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            errors.append(f"SSE line {n}: data is not JSON: {payload[:80]!r}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"SSE line {n}: data is not a JSON object")
            continue
        events.append(obj)
    return events, done, errors


def answer_from_events(events: list[dict]) -> tuple[str, str, list[str], list[str]]:
    """The streamed answer: (content text, reasoning text, every finish_reason seen, errors the stream reported)."""
    text, think, finishes, errors = [], [], [], []
    for i, obj in enumerate(events):
        if obj.get("error"):
            errors.append(f"stream event {i} carries an error: {json.dumps(obj['error'])[:160]}")
        for ch in obj.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta") or {}
            for key, sink in (("content", text), ("reasoning_content", think), ("reasoning", think)):
                v = delta.get(key)
                if isinstance(v, str):
                    sink.append(v)
                elif v is not None:
                    errors.append(f"stream event {i}: delta.{key} is not text")
            if ch.get("finish_reason"):
                finishes.append(str(ch["finish_reason"]))
    return "".join(text), "".join(think), finishes, errors


def context_has_kv(obj) -> tuple[bool, str | None]:
    """(ok, why-not) for one api/context answer: it must carry a real kv_bytes_token (null: the engine never said)."""
    if not isinstance(obj, dict):
        return False, "api/context: not a JSON object"
    if "kv_bytes_token" not in obj:
        return False, "api/context has no kv_bytes_token"
    v = obj["kv_bytes_token"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        return False, f"api/context: kv_bytes_token = {v!r} (the engine has not reported the K/V cost)"
    return True, None


def _headers(key: str | None, ct: bool = False) -> dict:
    h = {}
    if ct:
        h["Content-Type"] = "application/json"
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def check_model(mid: str, base: str, key: str | None, timeout: float = 600.0,
                opener=urllib.request.urlopen, prompt: str = PROMPT, max_tokens: int = 24) -> tuple[bool, list[str]]:
    """One profile: streamed chat + api/context.  Returns (ok, reasons) - reasons only for what failed."""
    base = base.rstrip("/")
    reasons: list[str] = []
    t0 = time.monotonic()
    body = json.dumps({"model": mid, "messages": [{"role": "user", "content": prompt}],
                       "stream": True, "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, method="POST",
                                 headers=_headers(key, ct=True))
    try:
        raw = b""
        with opener(req, timeout=timeout) as resp:
            while True:
                if time.monotonic() - t0 > timeout:
                    return False, [f"chat: no answer within {timeout:.0f} s"]
                part = resp.read(4096)
                if not part:
                    break
                raw += part
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = (e.read(240) or b"").decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - an error body is a bonus, the code is the finding
            pass
        return False, [f"chat HTTP {e.code}: {(detail.strip() or str(e.reason))[:200]}"]
    except Exception as e:  # noqa: BLE001 - any transport failure is this profile's FAIL, with the reason
        return False, [f"chat: {type(e).__name__}: {e}"]

    events, done, errs = sse_decode(raw.decode("utf-8", "replace"))
    if errs:
        reasons.append(f"{len(errs)} unparsable SSE line(s), first: {errs[0]}")
    text, think, finishes, cerr = answer_from_events(events)
    reasons += cerr[:3]
    if not events:
        reasons.append("empty stream (no data events)")
    if not done:
        reasons.append("no data: [DONE]")
    leaks = sorted(set(find_leaks(text)) | set(find_leaks(think)))
    if leaks:
        reasons.append(f"engine protocol leaked into the answer: {', '.join(leaks)}")
    if not text.strip():
        reasons.append("empty answer text")
    if not finishes:
        reasons.append("no finish_reason in the stream")
    elif finishes[-1] not in FINISH_OK:
        reasons.append(f"finish_reason = {finishes[-1]!r}")

    req = urllib.request.Request(base + "/upstream/" + urllib.parse.quote(mid, safe="") + "/api/context",
                                 headers=_headers(key))
    try:
        with opener(req, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        reasons.append(f"context HTTP {e.code}")
    except Exception as e:  # noqa: BLE001
        reasons.append(f"context: {type(e).__name__}: {e}")
    else:
        ok, why = context_has_kv(obj)
        if not ok:
            reasons.append(why)
    return not reasons, reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="streamed-chat + api/context smoke check of every llama-swap profile")
    ap.add_argument("models", nargs="*", help="model ids (default: the six llama-swap ids of ref-configs/*.json)")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help=f"llama-swap base (default {DEFAULT_BASE})")
    ap.add_argument("--key-file", help="file holding the llama-swap API key (never printed)")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds allowed per model, load included (default 600)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--list", action="store_true", help="print the default model ids and exit")
    args = ap.parse_args(argv)

    key = None
    if args.key_file:                                   # (a broken --key-file is a usage error, list or not)
        try:
            key = load_key(args.key_file)
        except OSError as e:
            print(f"cannot read --key-file: {e.strerror or e}", file=sys.stderr)
            return 2
    ids = list(args.models) or default_model_ids()
    if args.list:
        print("\n".join(ids))
        return 0
    if not ids:
        print("no model ids: name them, or run where ref-configs/*.json exist", file=sys.stderr)
        return 2

    failed = 0
    for mid in ids:
        t0 = time.monotonic()
        ok, reasons = check_model(mid, args.base_url, key, timeout=args.timeout,
                                  prompt=args.prompt, max_tokens=args.max_tokens)
        took = f" ({time.monotonic() - t0:.1f} s)"
        print(f"{'PASS' if ok else 'FAIL'}\t{mid}{took}" + (f": {'; '.join(reasons)}" if reasons else ""))
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
