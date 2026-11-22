#!/usr/bin/env python3
"""tools/ik2/serve_check.py - Inkling 2 Mini through serve/server.py with the real engine: /health, a thinking chat,
streaming, a tool call, a stop, a picture, /metrics.  One CLAIM line for the run.

    tools/ik2/serve_check.py --gpus 2 [--build build-ik2] [--vision build-ik2-vision/build/bin/vision-glm] [--port 18977]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ik2"))
sys.path.insert(0, str(ROOT / "tools"))
import bench as B       # noqa: E402
import t4_check as TC   # noqa: E402

MDIR = Path(TC.MODEL).parent


def png(w, h, rgb) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")


def call(port, path, body=None, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:400]}


def stream(port, body, stop_after=None, timeout=600):
    """A streamed chat: (seconds to the first delta, reasoning text, content text, finish reason, chunks)."""
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, think, text, fin, n = None, "", "", None, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("reasoning_content") or dl.get("content"):
                    first = first or time.time() - t0
                    n += 1
                think += dl.get("reasoning_content") or ""
                text += dl.get("content") or ""
                fin = c.get("finish_reason") or fin
            if stop_after and n >= stop_after:
                break   # (the connection closed mid-answer: the server stops the engine)
    return first, think, text, fin, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build-ik2")
    ap.add_argument("--vision", type=Path, default=ROOT / "build-ik2-vision/build/bin/vision-glm")
    ap.add_argument("--port", type=int, default=18977)
    ap.add_argument("--ctx", type=int, default=32768)
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    work = a.build / "serve-check"
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "tokenizer" / "tokenizer.json").exists():
        import strata_tokenizer as ST
        ST.extract(TC.MODEL, str(work), str(ROOT / "tools/ik2/chat_template.jinja"))
    cfg = {"exe": str(a.build / "saguaro-ik2"),
           "args": ["--model", TC.MODEL, "--max-context", str(a.ctx), "--gpus", str(len(gpus)), "--chunk", "1024"],
           "env": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "SAGUARO_IK2_STATS": "1"}, "cwd": str(ROOT),
           "tokenizer": str(work / "tokenizer"), "model_name": "Inkling-2-Mini-UD-Q4_K_XL", "gpu": gpus,
           "sampling": {"temperature": 1.0, "top_p": 1.0}, "reasoning_effort": "high",
           "log": str(work / "engine.log")}
    if a.vision.exists():
        cfg["vision"] = {"exe": str(a.vision), "mmproj": str(MDIR / "mmproj-BF16.gguf"), "gpu": True}
    (work / "config.json").write_text(json.dumps(cfg, indent=1))
    B.wait_free()
    line = B.claim("serve-check", gpus)
    res = {}
    srv = None
    try:
        srv = subprocess.Popen([sys.executable, "-B", str(ROOT / "serve/server.py"), "--config", str(work / "config.json"),
                                "--port", str(a.port), "--host", "127.0.0.1"], stdout=open(work / "server.log", "w"),
                               stderr=subprocess.STDOUT, cwd=str(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        t0 = time.time()
        h = None
        while time.time() - t0 < 600:
            if srv.poll() is not None:
                break
            try:
                st, h = call(a.port, "/health", timeout=5)
                if st == 200 and (h.get("status") in ("ok", "ready") or h.get("ready")):
                    break
            except Exception:
                pass
            time.sleep(2)
        print("health after %.0f s: %s" % (time.time() - t0, json.dumps(h)[:900]))
        res["health"] = bool(h) and srv.poll() is None
        if not res["health"]:
            print(open(work / "server.log").read()[-3000:])
            return 1
        model = cfg["model_name"]
        msg = lambda t: [{"role": "user", "content": t}]   # noqa: E731

        st, r = call(a.port, "/v1/chat/completions", {"model": model, "messages": msg("What is 17 * 23? Think it through."),
                                                       "max_tokens": 600, "temperature": 0})
        m = (r.get("choices") or [{}])[0].get("message", {})
        print("thinking chat: status %d, reasoning %d chars, content %r, usage %s" %
              (st, len(m.get("reasoning_content") or ""), (m.get("content") or "")[:200], r.get("usage")))
        res["thinking chat"] = st == 200 and "391" in (m.get("content") or "") and bool(m.get("reasoning_content"))

        st, r = call(a.port, "/v1/chat/completions", {"model": model, "messages": msg("Name the capital of Japan in one word."),
                                                       "max_tokens": 50, "temperature": 0, "reasoning_effort": "none"})
        m = (r.get("choices") or [{}])[0].get("message", {})
        print("thinking off: status %d, reasoning %r, content %r" % (st, (m.get("reasoning_content") or "")[:60], m.get("content")))
        res["thinking off"] = st == 200 and "tokyo" in (m.get("content") or "").lower() and not m.get("reasoning_content")

        first, think, text, fin, n = stream(a.port, {"model": model, "messages": msg("Write four sentences about rivers."),
                                                    "max_tokens": 300, "reasoning_effort": "low"})
        print("streaming: first delta %.2f s, %d chunks, reasoning %d chars, content %d chars, finish %s" %
              (first or -1, n, len(think), len(text), fin))
        res["streaming"] = n > 10 and len(text) > 40 and fin in ("stop", "length")

        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        st, r = call(a.port, "/v1/chat/completions", {"model": model, "messages": msg("What is the weather in Paris right now?"),
                                                       "tools": tools, "max_tokens": 400, "temperature": 0})
        ch = (r.get("choices") or [{}])[0]
        tc = (ch.get("message", {}).get("tool_calls") or [{}])[0].get("function", {})
        print("tool call: status %d, finish %s, call %s" % (st, ch.get("finish_reason"), json.dumps(tc)))
        res["tool call"] = ch.get("finish_reason") == "tool_calls" and tc.get("name") == "get_weather" and \
            "paris" in (tc.get("arguments") or "").lower()
        if res["tool call"]:
            msgs = msg("What is the weather in Paris right now?") + [ch["message"], {"role": "tool", "tool_call_id":
                   ch["message"]["tool_calls"][0].get("id", "c1"), "content": json.dumps({"temp_c": 31, "sky": "clear"})}]
            st, r = call(a.port, "/v1/chat/completions", {"model": model, "messages": msgs, "tools": tools,
                                                           "max_tokens": 300, "temperature": 0})
            c2 = ((r.get("choices") or [{}])[0].get("message", {}).get("content") or "")
            print("tool result turn: status %d, content %r" % (st, c2[:160]))
            res["tool result turn"] = st == 200 and "31" in c2

        t1 = time.time()
        first, think, text, fin, n = stream(a.port, {"model": model, "messages": msg("Count from 1 to 2000, one number a line."),
                                                    "max_tokens": 4000, "reasoning_effort": "none"}, stop_after=30)
        time.sleep(1.0)
        t2 = time.time()
        st, r = call(a.port, "/v1/chat/completions", {"model": model, "messages": msg("Say OK."), "max_tokens": 20,
                                                       "temperature": 0, "reasoning_effort": "none"}, timeout=120)
        print("stop: closed after %d chunks (%.1f s); the next request answered in %.1f s: %r" %
              (n, t2 - t1, time.time() - t2, ((r.get("choices") or [{}])[0].get("message", {}).get("content"))))
        res["stop"] = st == 200 and time.time() - t2 < 30

        if "vision" in cfg:
            img = "data:image/png;base64," + base64.b64encode(png(320, 320, (220, 20, 20))).decode()
            st, r = call(a.port, "/v1/chat/completions", {"model": model, "max_tokens": 60, "temperature": 0,
                         "reasoning_effort": "none", "messages": [{"role": "user", "content": [
                             {"type": "image_url", "image_url": {"url": img}},
                             {"type": "text", "text": "What single colour fills this image? One word."}]}]})
            c3 = ((r.get("choices") or [{}])[0].get("message", {}).get("content") or r.get("error") or "")
            print("picture: status %d, answer %r, usage %s" % (st, str(c3)[:300], r.get("usage")))
            res["picture"] = st == 200 and "red" in str(c3).lower()
        else:
            print("picture: no encoder at %s" % a.vision)

        for path in ("/health", "/metrics", "/props", "/slots", "/v1/status"):
            st, r = call(a.port, path)
            (work / (path.strip("/").replace("/", "_") + ".json")).write_text(json.dumps(r, indent=1))
            print("%s: %d, keys %s" % (path, st, sorted(r)[:40] if isinstance(r, dict) else type(r)))
    finally:
        if srv and srv.poll() is None:
            srv.terminate()
            try:
                srv.wait(timeout=90)
            except subprocess.TimeoutExpired:
                srv.kill()
        time.sleep(2)
        print("cards after exit:", {g: u for g, (u, _) in TC.smi(gpus).items()})
        B.release(line)
    print()
    for k, v in res.items():
        print("%-18s %s" % (k, "PASS" if v else "FAIL"))
    return 0 if res and all(res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
