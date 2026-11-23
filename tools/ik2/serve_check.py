#!/usr/bin/env python3
"""tools/ik2/serve_check.py - Inkling 2 Mini through serve/server.py with the real engine: /health, a thinking chat,
thinking off, the default sampling, streaming, a tool call and its result, a stop while it writes and while it reads,
pictures, the Monitor's figures while an answer runs, the web app's resumable stream.  One CLAIM line for the run;
what each check got is in <build>/serve-check/results.json.

    tools/ik2/serve_check.py --gpus 2 [--build build-ik2] [--vision build-ik2-vision/build/bin/vision-glm] [--port 18977]
"""
from __future__ import annotations

import argparse
import shutil
import base64
import json
import os
import signal
import struct
import subprocess
import sys
import threading
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


def claim(tag: str, gpus) -> str:
    if B.NO_CLAIM:                                      # (the runner holds the cards: no line, as B.release writes none)
        return ""
    line = "CLAIM ik2-serve: %s cards %s %s" % (tag, ",".join(map(str, gpus)), time.strftime("%H:%M:%S %Z"))
    with open(B.STATUS, "a") as f:
        f.write(line + "\n")
    return line


def call(port, path, body=None, timeout=600, raw=False):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, (data.decode(errors="replace") if raw else json.loads(data or b"{}"))
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:600]}


def chat(port, body, timeout=900):
    """A whole answer: (status, message, finish reason, the response, seconds)."""
    t0 = time.time()
    st, r = call(port, "/v1/chat/completions", body, timeout)
    ch = (r.get("choices") or [{}])[0] if isinstance(r, dict) else {}
    return st, ch.get("message") or {}, ch.get("finish_reason"), r, time.time() - t0


def speeds(r) -> str:
    t = (r or {}).get("timings") or {}
    return "read %s tok in %s ms (%s tok/s, %s cached), wrote %s in %s ms (%s tok/s)" % (
        t.get("prompt_n"), t.get("prompt_ms"), t.get("prompt_per_second"), t.get("cache_n"), t.get("predicted_n"),
        t.get("predicted_ms"), t.get("predicted_per_second"))


MARKS = ("<|content_", "<|message_", "<|end_message|>", "<|unused_")   # (none of the model's segment tokens may show)


def clean(*texts) -> bool:
    return not any(m in (t or "") for t in texts for m in MARKS)


def stream(port, body, stop_after=None, stop_s=None, timeout=900, path="/v1/chat/completions", get=False):
    """A streamed chat -> {first: seconds to the first delta, think, text, fin, n: text deltas, order: the kinds of
    the deltas as they came (runs), calls: tool calls, last: the final chunk, id, progress: prompt_progress chunks,
    done: [DONE] seen}.  stop_after: the connection is closed after that many text deltas; stop_s: after that long."""
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=None if get else json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    o = dict(first=None, think="", text="", fin=None, n=0, order=[], calls={}, last=None, id=None, progress=0,
             done=False)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if stop_s and time.time() - t0 >= stop_s:
                break   # (closed while the prompt is read)
            if line == "data: [DONE]":
                o["done"] = True
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            o["id"] = o["id"] or d.get("id")
            o["progress"] += 1 if d.get("prompt_progress") else 0
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                for key, kind in (("reasoning_content", "think"), ("content", "text")):
                    if dl.get(key):
                        o["first"] = o["first"] or time.time() - t0
                        o["n"] += 1
                        o[kind] += dl[key]
                        if not o["order"] or o["order"][-1] != kind:
                            o["order"].append(kind)
                for tc in dl.get("tool_calls") or []:
                    e = o["calls"].setdefault(tc.get("index", 0), {"name": "", "arguments": "", "id": None})
                    f = tc.get("function") or {}
                    e["name"] += f.get("name") or ""
                    e["arguments"] += f.get("arguments") or ""
                    e["id"] = e["id"] or tc.get("id")
                    if not o["order"] or o["order"][-1] != "call":
                        o["order"].append("call")
                if c.get("finish_reason"):
                    o["fin"], o["last"] = c["finish_reason"], d
            if stop_after and o["n"] >= stop_after:
                break   # (the connection closed mid-answer: the server stops the engine)
    return o


def idle_after(port, limit=120.0):
    """Seconds until /metrics says the engine is idle again (None: not within the limit)."""
    t0 = time.time()
    while time.time() - t0 < limit:
        st, m = call(port, "/metrics", timeout=10)
        if st == 200 and (m.get("live") or {}).get("state") == "idle":
            return time.time() - t0
        time.sleep(0.25)
    return None


def scene_png() -> bytes:
    """A drawn scene, standard library only: sky, grass, a yellow sun, a red house with a blue door."""
    w, h = 448, 320
    px = [[(120, 180, 240) if y < 200 else (60, 160, 70) for _ in range(w)] for y in range(h)]
    for y in range(h):
        for x in range(w):
            if (x - 360) ** 2 + (y - 60) ** 2 < 36 ** 2:
                px[y][x] = (250, 220, 40)
            if 110 <= x < 270 and 130 <= y < 250:
                px[y][x] = (200, 40, 40)
            if 60 <= y < 130 and abs(x - 190) < (y - 60) * 1.25:
                px[y][x] = (110, 60, 30)
            if 170 <= x < 210 and 180 <= y < 250:
                px[y][x] = (30, 60, 200)
    raw = b"".join(b"\x00" + bytes(v for p in row for v in p) for row in px)
    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--build", type=Path, default=ROOT / "build-ik2")
    vdir = Path(os.environ.get("SAGUARO_IK2_VISION_TREE") or ROOT / "build-ik2-vision")
    ap.add_argument("--vision", type=Path, default=vdir / "build/bin/vision-glm")
    ap.add_argument("--photo", type=Path, default=vdir / "llama.cpp/tools/mtmd/test-1.jpeg")
    ap.add_argument("--port", type=int, default=18977)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--env", action="append", default=[], help="K=V for the engine (SAGUARO_IK2_ROOM_MB=2048 ...)")
    ap.add_argument("--arg", action="append", default=[], help="more engine arguments (--arg=--parallel --arg=2)")
    ap.add_argument("--vision-set", action="append", default=[], help="K=V into the config's vision entry (JSON values)")
    ap.add_argument("--out", default=None, help="the folder under <build> (serve-check, serve-check-2 for two cards)")
    ap.add_argument("--model", default=TC.MODEL, help="the GGUF (tokenizer and picture encoder: UD-Q4_K_XL's)")
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    work = a.build / (a.out or ("serve-check" if len(gpus) == 1 else "serve-check-%d" % len(gpus)))
    work.mkdir(parents=True, exist_ok=True)
    tokdir = a.build / "serve-check"   # (one tokenizer folder for every run)
    if not (tokdir / "tokenizer" / "tokenizer.json").exists():
        import strata_tokenizer as ST
        tokdir.mkdir(parents=True, exist_ok=True)
        src = TC.MODEL if Path(TC.MODEL).exists() else a.model   # (a PC with only the other GGUF: the same vocabulary)
        ST.extract(src, str(tokdir), str(ROOT / "tools/ik2/chat_template.jinja"))
    cfg = {"exe": str(a.build / "saguaro-ik2"),
           "args": ["--model", a.model, "--max-context", str(a.ctx), "--gpus", str(len(gpus)), "--chunk", str(a.chunk)]
           + a.arg,
           "env": dict({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "SAGUARO_IK2_STATS": "1"},
                       **dict(kv.split("=", 1) for kv in a.env)), "cwd": str(ROOT),
           "tokenizer": str(tokdir / "tokenizer"), "model_name": Path(a.model).stem, "gpu": gpus,
           "sampling": {"temperature": 1.0, "top_p": 1.0}, "reasoning_effort": "high",
           "log": str(work / "engine.log")}
    if shutil.which("nvidia-smi") is None:   # (AMD: the server's HIP paths)
        cfg["backend"] = "hip"
    if a.vision.exists() and (MDIR / "mmproj-BF16.gguf").exists():
        cfg["vision"] = {"exe": str(a.vision), "mmproj": str(MDIR / "mmproj-BF16.gguf"), "gpu": True}
        for kv in a.vision_set:
            k, v = kv.split("=", 1)
            cfg["vision"][k] = json.loads(v)
    (work / "config.json").write_text(json.dumps(cfg, indent=1))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # (the finally below releases the claim)
    B.wait_free()
    line = claim("serve-check", gpus)
    res, raw = {}, {}
    srv = None
    port = a.port

    def smi():
        return {g: u for g, (u, _) in TC.smi(gpus).items()}

    try:
        srv = subprocess.Popen([sys.executable, "-B", str(ROOT / "serve/server.py"), "--config", str(work / "config.json"),
                                "--port", str(port), "--host", "127.0.0.1"], stdout=open(work / "server.log", "w"),
                               stderr=subprocess.STDOUT, cwd=str(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        t0 = time.time()
        h = None
        while time.time() - t0 < 900:
            if srv.poll() is not None:
                break
            try:
                st, h = call(port, "/health", timeout=5)
                if st == 200 and h.get("status") == "ok" and h.get("max_context"):
                    break
            except Exception:
                pass
            time.sleep(2)
        print("health after %.0f s: %s" % (time.time() - t0, json.dumps(h)[:1200]))
        res["health"] = bool(h) and srv.poll() is None and h.get("max_context") == a.ctx and \
            h.get("images") == ("vision" in cfg) and (h.get("defaults") or {}).get("temperature") == 1.0
        raw["health"] = {"seconds": round(time.time() - t0, 1), "vram_mib": smi()}
        print("cards after the load:", raw["health"]["vram_mib"])
        if not h or srv.poll() is not None:
            print(open(work / "server.log").read()[-3000:])
            return 1
        model = cfg["model_name"]
        msg = lambda t: [{"role": "user", "content": t}]   # noqa: E731

        st, m, fin, r, dt = chat(port, {"model": model, "messages": msg("What is 17 * 23? Think it through."),
                                        "max_tokens": 1500, "temperature": 0})
        print("thinking chat: status %d in %.1f s, finish %s, reasoning %d chars %r, content %r, usage %s\n  %s" %
              (st, dt, fin, len(m.get("reasoning_content") or ""), (m.get("reasoning_content") or "")[:160],
               (m.get("content") or "")[:300], r.get("usage"), speeds(r)))
        raw["thinking chat"] = r
        res["thinking chat"] = st == 200 and "391" in (m.get("content") or "") and bool(m.get("reasoning_content")) \
            and fin == "stop" and clean(m.get("content"), m.get("reasoning_content"))

        st, m, fin, r, dt = chat(port, {"model": model, "messages": msg("Name the capital of Japan in one word."),
                                        "max_tokens": 50, "temperature": 0, "reasoning_effort": "none"})
        print("thinking off: status %d in %.1f s, finish %s, reasoning %r, content %r\n  %s" %
              (st, dt, fin, (m.get("reasoning_content") or "")[:60], m.get("content"), speeds(r)))
        raw["thinking off"] = r
        res["thinking off"] = st == 200 and "tokyo" in (m.get("content") or "").lower() and \
            not m.get("reasoning_content") and fin == "stop" and clean(m.get("content"))

        # the config's sampling (temperature 1.0) reaches the engine: two answers to one prompt differ, greedy ones do not
        q = {"model": model, "messages": msg("Invent a name for a new colour and describe it in one sentence."),
             "max_tokens": 60, "reasoning_effort": "none"}
        s1, s2 = chat(port, q)[1].get("content"), chat(port, q)[1].get("content")
        g1, g2 = chat(port, dict(q, temperature=0))[1].get("content"), chat(port, dict(q, temperature=0))[1].get("content")
        print("default sampling: %r / %r; greedy: %r / %r" % (s1, s2, g1, g2))
        raw["default sampling"] = {"sampled": [s1, s2], "greedy": [g1, g2]}
        res["default sampling"] = bool(s1) and s1 != s2
        res["greedy repeats"] = bool(g1) and g1 == g2

        # streaming, with the Monitor's figures sampled while it runs
        seen, stop_poll = [], threading.Event()

        def poll():
            while not stop_poll.is_set():
                try:
                    st_, m_ = call(port, "/metrics", timeout=10)
                    if st_ == 200 and (m_.get("live") or {}).get("state") != "idle":
                        seen.append(m_)
                except Exception:
                    pass
                time.sleep(0.2)
        th = threading.Thread(target=poll, daemon=True)
        th.start()
        long_doc = B.corpus()[:36000]
        o = stream(port, {"model": model, "messages": msg(long_doc + "\n\nWrite four sentences about what the text "
                                                           "above describes."),
                          "max_tokens": 2000, "reasoning_effort": "low", "return_progress": True,
                          "stream_options": {"include_usage": True}})
        stop_poll.set()
        th.join(timeout=5)
        last = o["last"] or {}
        print("streaming: first delta %.2f s, %d deltas (%s), reasoning %d chars, content %d chars %r, finish %s, "
              "%d progress chunks, usage %s\n  %s" %
              (o["first"] or -1, o["n"], " > ".join(o["order"]), len(o["think"]), len(o["text"]), o["text"][:200],
               o["fin"], o["progress"], last.get("usage"), speeds(last)))
        states = [(m_["live"].get("state")) for m_ in seen]
        rd = next((m_ for m_ in seen if m_["live"].get("state") == "reading" and m_["live"].get("prompt_read")), None)
        gn = next((m_ for m_ in reversed(seen) if m_["live"].get("state") == "generating" and m_["live"].get("tok_s")), None)
        for name, m_ in (("metrics_reading", rd), ("metrics_generating", gn)):
            if m_:
                (work / (name + ".json")).write_text(json.dumps(m_, indent=1))
        print("monitor while it ran: %d samples, reading %d, generating %d; a read sample: %s; a decode sample: %s" %
              (len(seen), states.count("reading"), states.count("generating"),
               rd and {k: rd["live"].get(k) for k in ("prompt_read", "prompt_total", "prefill_tok_s", "prefill_tok_s_mean")},
               gn and {k: gn["live"].get(k) for k in ("generated", "tok_s", "tok_s_mean", "prompt_tokens")}))
        raw["streaming"] = {k: o[k] for k in ("first", "n", "order", "think", "text", "fin", "progress", "last")}
        res["streaming"] = o["n"] > 10 and len(o["text"]) > 40 and o["fin"] == "stop" and o["done"] and \
            bool(last.get("usage")) and clean(o["text"], o["think"]) and o["order"][-1] == "text"
        res["stream progress"] = o["progress"] > 0
        res["monitor live"] = rd is not None and gn is not None

        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        wq = msg("What is the weather in Paris right now?")
        st, m, fin, r, dt = chat(port, {"model": model, "messages": wq, "tools": tools, "max_tokens": 800, "temperature": 0})
        tc = (m.get("tool_calls") or [{}])[0].get("function", {})
        print("tool call: status %d in %.1f s, finish %s, call %s, content %r, reasoning %r\n  %s" %
              (st, dt, fin, json.dumps(tc), m.get("content"), (m.get("reasoning_content") or "")[:120], speeds(r)))
        raw["tool call"] = r
        try:
            targs = json.loads(tc.get("arguments") or "null")
        except ValueError:
            targs = None
        res["tool call"] = fin == "tool_calls" and tc.get("name") == "get_weather" and isinstance(targs, dict) and \
            "paris" in str(targs.get("city", "")).lower() and clean(m.get("content"))
        if res["tool call"]:
            msgs = wq + [m, {"role": "tool", "tool_call_id": m["tool_calls"][0].get("id", "c1"),
                             "content": json.dumps({"temp_c": 31, "sky": "clear"})}]
            st, m2, fin2, r2, dt = chat(port, {"model": model, "messages": msgs, "tools": tools, "max_tokens": 800,
                                               "temperature": 0})
            c2 = m2.get("content") or ""
            print("tool result turn: status %d in %.1f s, finish %s, content %r\n  %s" % (st, dt, fin2, c2[:200], speeds(r2)))
            raw["tool result turn"] = r2
            res["tool result turn"] = st == 200 and "31" in c2 and fin2 == "stop" and clean(c2)
            res["tool turn reuses the prompt"] = ((r2.get("timings") or {}).get("cache_n") or 0) > 0
        o = stream(port, {"model": model, "messages": wq, "tools": tools, "max_tokens": 800, "temperature": 0})
        sc = o["calls"].get(0) or {}
        print("tool call, streamed: finish %s, order %s, call %s" % (o["fin"], " > ".join(o["order"]), json.dumps(sc)))
        raw["tool call streamed"] = {k: o[k] for k in ("order", "think", "text", "fin", "calls")}
        res["tool call streamed"] = o["fin"] == "tool_calls" and sc.get("name") == "get_weather" and \
            "paris" in sc.get("arguments", "").lower() and clean(o["text"])

        # a stop while it writes: the connection closed, the engine stopped (STOP), the next request answers
        t1 = time.time()
        o = stream(port, {"model": model, "messages": msg("Count from 1 to 2000, one number a line."),
                          "max_tokens": 4000, "reasoning_effort": "none"}, stop_after=30)
        idle = idle_after(port)
        st, mm = call(port, "/metrics")
        lastreq = ((mm.get("requests") or [{}])[0]) if st == 200 else {}
        t2 = time.time()
        st, m, fin, r, dt = chat(port, {"model": model, "messages": msg("Say OK."), "max_tokens": 20, "temperature": 0,
                                        "reasoning_effort": "none"}, timeout=120)
        print("stop while writing: closed after %d deltas (%.1f s); idle %s s later, that request: %s; the next request "
              "answered in %.1f s: %r" % (o["n"], t2 - t1, idle and round(idle, 1),
              {k: lastreq.get(k) for k in ("finish", "output_tokens", "completion_tokens", "generated")}, dt, m.get("content")))
        raw["stop"] = {"idle_s": idle, "request": lastreq, "next": r}
        res["stop while writing"] = st == 200 and idle is not None and idle < 10 and bool(m.get("content")) and dt < 30 \
            and "1\n2" not in (m.get("content") or "")

        # ... and while it reads a long prompt (the engine looks at STOP between the prompt's chunks)
        doc = B.corpus()[36000:100000]
        t1 = time.time()
        try:
            o = stream(port, {"model": model, "messages": msg(doc + "\n\nSummarize the text above."),
                              "max_tokens": 200, "reasoning_effort": "none", "return_progress": True}, stop_s=3.0)
        except Exception as e:   # (a prompt past the context: said below)
            o = {"n": -1, "err": str(e)}
        idle = idle_after(port)
        st, mm = call(port, "/metrics")
        lastreq = ((mm.get("requests") or [{}])[0]) if st == 200 else {}
        st, m, fin, r, dt = chat(port, {"model": model, "messages": msg("Say OK."), "max_tokens": 20, "temperature": 0,
                                        "reasoning_effort": "none"}, timeout=300)
        print("stop while reading: closed after %.1f s (%s deltas); idle %s s later, that request: %s; the next request "
              "answered in %.1f s: %r" % (3.0, o.get("n"), idle and round(idle, 1),
              {k: lastreq.get(k) for k in ("finish", "prompt_tokens", "output_tokens")}, dt, m.get("content")))
        raw["stop reading"] = {"idle_s": idle, "request": lastreq, "next": r, "stream": o.get("err")}
        res["stop while reading"] = st == 200 and idle is not None and o.get("n") == 0 and bool(m.get("content"))

        # the web app's stream: it runs on without its reader, and is read again from its start by id
        o = stream(port, {"model": model, "messages": msg("List five fruits, one a line."), "max_tokens": 80,
                          "temperature": 0, "reasoning_effort": "none", "strata_resume": True}, stop_after=3)
        idle = idle_after(port)
        o2 = stream(port, None, path="/v1/strata/stream?id=%s&from=0" % o["id"], get=True) if o["id"] else {"text": "", "fin": None}
        print("resumable stream: left after %d deltas, idle %s s later; read again by id: finish %s, %r" %
              (o["n"], idle and round(idle, 1), o2["fin"], o2["text"][:120]))
        raw["resumable stream"] = {"first": o["text"], "again": o2["text"], "fin": o2["fin"]}
        res["resumable stream"] = o2["fin"] == "stop" and len(o2["text"]) > len(o["text"]) and o2["text"].startswith(o["text"])

        # the other two APIs: Anthropic's messages (thinking and a tool call as blocks), OpenAI's responses
        st, r = call(port, "/v1/messages", {"model": model, "max_tokens": 800, "temperature": 0, "tools": [
            {"name": "get_weather", "description": "Current weather of a city", "input_schema": tools[0]["function"]["parameters"]}],
            "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}]})
        blocks = r.get("content") or [] if isinstance(r, dict) else []
        use = next((b_ for b_ in blocks if b_.get("type") == "tool_use"), {})
        print("anthropic messages: status %d, stop %s, blocks %s, tool_use %s, usage %s" %
              (st, r.get("stop_reason"), [b_.get("type") for b_ in blocks], json.dumps(use.get("input")), r.get("usage")))
        raw["anthropic"] = r
        res["anthropic tool call"] = st == 200 and r.get("stop_reason") == "tool_use" and use.get("name") == "get_weather" \
            and "paris" in str((use.get("input") or {}).get("city", "")).lower()
        st, r = call(port, "/v1/responses", {"model": model, "input": "What is 12 * 12? Answer with the number.",
                                             "max_output_tokens": 600, "temperature": 0})
        out = r.get("output") or [] if isinstance(r, dict) else []
        text = "".join(c.get("text") or "" for o_ in out if o_.get("type") == "message" for c in o_.get("content") or [])
        print("responses: status %d, status %s, items %s, text %r, usage %s" %
              (st, r.get("status"), [o_.get("type") for o_ in out], text[:120], r.get("usage")))
        raw["responses"] = r
        res["responses"] = st == 200 and "144" in text and clean(text)

        st, m, fin, r, dt = chat(port, {"model": model, "messages": msg("Hi"), "max_tokens": a.ctx + 10})
        print("max_tokens past the context: status %d, %s" % (st, json.dumps(r)[:300]))
        raw["over context"] = r
        res["over context refused"] = st == 400

        if "vision" in cfg:
            def picture(name, data, mime, question, need, max_tokens=200):
                img = "data:%s;base64,%s" % (mime, base64.b64encode(data).decode())
                before = smi()
                st, m, fin, r, dt = chat(port, {"model": model, "max_tokens": max_tokens, "temperature": 0,
                                                "reasoning_effort": "none", "messages": [{"role": "user", "content": [
                                                    {"type": "image_url", "image_url": {"url": img}},
                                                    {"type": "text", "text": question}]}]})
                c3 = str(m.get("content") or r.get("error") or "")
                print("picture (%s): status %d in %.1f s, answer %r, usage %s, cards before %s after %s\n  %s" %
                      (name, st, dt, c3[:500], r.get("usage"), before, smi(), speeds(r)))
                raw["picture " + name] = dict(r, seconds=round(dt, 1))
                res["picture " + name] = st == 200 and any(w in c3.lower() for w in need) and clean(c3)
            picture("red", png(320, 320, (220, 20, 20)), "image/png", "What single colour fills this image? One word.",
                    ("red",), 30)
            picture("scene", scene_png(), "image/png", "Describe this drawing in two sentences: what objects, what "
                    "colours?", ("house", "sun"))
            if a.photo.exists():
                picture("photo", a.photo.read_bytes(), "image/jpeg", "What is this a photo of, and what does its "
                        "biggest headline say?", ("moon",))
                # the same picture again in a second turn: its rows come from the server's cache, its prompt from the engine's
                picture("photo again", a.photo.read_bytes(), "image/jpeg", "What is this a photo of, and what does its "
                        "biggest headline say?", ("moon",))
        else:
            print("picture: no encoder at %s" % a.vision)

        for path in ("/health", "/metrics", "/props", "/slots", "/v1/status", "/v1/models", "/api/context",
                     "/api/thinking-budget", "/api/turbo/options"):
            st, r = call(port, path)
            (work / (path.strip("/").replace("/", "_") + ".json")).write_text(json.dumps(r, indent=1))
            print("%s: %d, keys %s" % (path, st, sorted(r)[:40] if isinstance(r, dict) else type(r).__name__))
        st, mm = call(port, "/metrics")
        eng, tn = mm.get("engine") or {}, (mm.get("tiers") or {}).get("now") or {}
        hits = [q.get("hit_rate") for q in mm.get("requests") or []]
        print("monitor facts: path %s, experts %s, VRAM slots %s, K/V %s GB at %s; the last answers' VRAM hit %s; "
              "tiers %s" % (eng.get("path"), eng.get("experts"), eng.get("expert_slots"), eng.get("kv_gb"),
                            eng.get("kv_ctx"), hits[:6], {k: tn.get(k) for k in ("vram_hit", "ram_fetch", "cpu", "vram_used",
                                                                                  "ram_used", "vram_gb", "ram_gb", "kv_gb")}))
        # (v1.1.74's engine: where the experts are and the K / V size in its STAT; the older one's in its start lines)
        res["monitor facts"] = bool(eng.get("experts")) and bool(eng.get("expert_slots") or tn.get("vram_used")) and \
            bool(eng.get("kv_gb") or tn.get("kv_gb"))
        res["monitor VRAM hit"] = tn.get("vram_hit") is not None and \
            (any(x is not None for x in hits) or (eng.get("expert_slots") or 0) >= (eng.get("experts") or 1) or
             eng.get("path") == "resident")
        st, r = call(port, "/metrics/prometheus", raw=True)
        (work / "metrics_prometheus.txt").write_text(r if isinstance(r, str) else json.dumps(r))
        print("cards before exit:", smi())
    finally:
        (work / "results.json").write_text(json.dumps({"checks": res, "raw": raw}, indent=1, default=str))
        mine = set()
        if srv and srv.poll() is None:
            kids = subprocess.run(["ps", "-o", "pid=", "--ppid", str(srv.pid)], capture_output=True, text=True).stdout
            mine = {srv.pid} | {int(x) for x in kids.split()}
            srv.terminate()
            try:
                srv.wait(timeout=90)
            except subprocess.TimeoutExpired:
                srv.kill()
        time.sleep(2)
        try:
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,name", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=30).stdout if shutil.which("nvidia-smi") else ""
            print("cards after exit:", smi(), "| ours left on them:",
                  [l for l in apps.splitlines() if l.split(",")[0].strip().isdigit() and int(l.split(",")[0]) in mine])
        finally:
            B.release(line)
    print()
    for k, v in res.items():
        print("%-28s %s" % (k, "PASS" if v else "FAIL"))
    return 0 if res and all(res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
