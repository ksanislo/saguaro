#!/usr/bin/env python3
"""tools/saguaro/check_monitor.py - the Monitor's figures against what the engine actually did.

    python3 tools/saguaro/check_monitor.py <ui-data.jsonl> <engine.log> [--tol 0.25] [--slack 2.5]

<ui-data.jsonl> is the Monitor's own data sampled every 2 s, one JSON per line:
{t, state, reading_n, prefill_tok_s, tok_s, peak_prefill, peak_decode, chats[], last_req{time, prompt_tokens,
 reused, prefill_tok_s, ...}}.  <engine.log> is the time-stamped server/engine log (serve/logctx.py): its finished
reads and answers are the ground truth -

    saguaro batch: slot S admitted beside the decoding slots: N prompt tokens (R reused) read in M ms ...
    saguaro batch: read K tokens on a thread of its own in M ms (concurrent, to q of n) ...
    [saguaro] done: n tokens in s s (r tok/s) ...

Three checks, every discrepancy above --tol (25 %) reported:

  read-rate     for each finished admitted read, the Monitor's figure for it - the finished request's
                last_req.prefill_tok_s (matched by prompt_tokens and reused in a sample at or after the read's end)
                or, failing that, a live sample inside the read window ([its time - M ms, its time]) with
                reading_n == 1 - against the engine's (N - R) / M.  A read the Monitor has no figure for at all is
                a mismatch too.  A thread-of-its-own read's K / M is one chunk's rate while the live figure averages
                the whole read, so for those only the live figure's presence is checked (a note if the read lasted
                >= 2 s and no sample caught it).
  peak          a shown peak_prefill / peak_decode sits above the engine's real read / decode rates (a peak above
                EVERY rate the log measured; "every" includes none, so a peak with no comparable log line is shown
                too - capture the log over the same window), and a peak never falls: it is a maximum over the run.
  live-prefill  a sample shows a live prefill rate (prefill_tok_s > 0) while nothing reads: no reading_n and no
                finished read covering the moment (+- --slack), while the log still ran at that moment.

Exit 0 clean, 1 mismatches (listed, plus a summary table), 2 usage / unreadable input.

    python -m unittest tools.test_check_monitor     # offline, small fixtures under tools/saguaro/fixtures/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

STAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)\b")
ADMITTED_RE = re.compile(r"admitted beside the decoding slots:\s*(\d+) prompt tokens \((\d+) reused\) "
                         r"read in (\d+(?:\.\d+)?) ms")
THREAD_RE = re.compile(r"read (\d+) tokens on a thread of its own in (\d+(?:\.\d+)?) ms"
                       r"(?:\s*\(concurrent, to (\d+) of (\d+)\))?")
DONE_RE = re.compile(r"\[saguaro\] done:\s*(\d+) tokens in (\d+(?:\.\d+)?) s(?:\s*\((\d+(?:\.\d+)?) tok/s\))?")


@dataclass
class Read:
    line: int
    kind: str                      # "admitted" | "thread"
    when: float | None             # epoch; None: the line carried no time stamp
    stamp: str                     # the log's own time text (for the report)
    tokens: int                    # N prompt tokens, or K read on the side thread
    reused: int | None
    ms: float
    target: int | None = None      # the thread line's "to q of n"
    line_text: str = ""

    @property
    def new(self) -> int:
        return self.tokens - (self.reused or 0)

    @property
    def rate(self) -> float | None:
        return self.new / (self.ms / 1000.0) if self.ms > 0 else None

    @property
    def window(self) -> tuple[float, float] | None:
        return (self.when - self.ms / 1000.0, self.when) if self.when is not None else None

    def where(self) -> str:
        return f"{self.stamp or '?'} line {self.line}"


@dataclass
class Decode:
    line: int
    when: float | None
    stamp: str
    tokens: int
    secs: float
    rate: float
    line_text: str = ""


@dataclass
class Sample:
    line: int
    raw: dict = field(default_factory=dict)
    error: str | None = None

    def get(self, key):
        return self.raw.get(key)

    @property
    def t(self):
        v = self.raw.get("t")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def reading_n(self) -> int:
        v = self.raw.get("reading_n")
        return int(v) if isinstance(v, (int, float)) else 0

    @property
    def prefill_tok_s(self):
        v = self.raw.get("prefill_tok_s")
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    @property
    def chats(self) -> list:
        v = self.raw.get("chats")
        return v if isinstance(v, list) else []

    @property
    def last_req(self) -> dict | None:
        v = self.raw.get("last_req")
        return v if isinstance(v, dict) else None


def stamp_to_epoch(text: str) -> float | None:
    """The log's local time stamp as epoch seconds (the ui-data t is epoch too; both from the same machine)."""
    try:
        frac = 0.0
        if "." in text:
            text, f = text.split(".", 1)
            frac = float("0." + f)
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timestamp() + frac
    except ValueError:
        return None


def parse_engine_log(text: str) -> tuple[list[Read], list[Decode], list[str]]:
    """The finished reads and answers the log states; errors = time-stamped events that carry no stamp."""
    reads, decodes, errors = [], [], []
    for n, line in enumerate(text.splitlines(), 1):
        m = STAMP_RE.match(line)
        stamp, when = (m.group(1), None) if m else ("", None)
        if m:
            when = stamp_to_epoch(stamp)
        if (a := ADMITTED_RE.search(line)):
            r = Read(n, "admitted", when, stamp, int(a.group(1)), int(a.group(2)), float(a.group(3)), line_text=line)
            if when is None:
                errors.append(f"line {n}: an admitted read with no time stamp - the log needs it (logctx.py stamps by default)")
            reads.append(r)
        elif (t := THREAD_RE.search(line)):
            r = Read(n, "thread", when, stamp, int(t.group(1)), None, float(t.group(2)),
                     target=int(t.group(3)) if t.group(3) else None, line_text=line)
            if when is None:
                errors.append(f"line {n}: a thread read with no time stamp - the log needs it (logctx.py stamps by default)")
            reads.append(r)
        elif (d := DONE_RE.search(line)):
            secs = float(d.group(2))
            rate = float(d.group(3)) if d.group(3) else (d.group(1) / secs if secs > 0 else 0.0)
            decodes.append(Decode(n, when, stamp, int(d.group(1)), secs, rate, line_text=line))
    return reads, decodes, errors


def parse_ui_data(text: str) -> tuple[list[Sample], list[str]]:
    samples, errors = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            errors.append(f"line {n}: not JSON ({e})")
            samples.append(Sample(line=n, error="unparsable"))
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {n}: not a JSON object")
            samples.append(Sample(line=n, error="not an object"))
            continue
        samples.append(Sample(line=n, raw=obj))
    return samples, errors


def _peak_of(v) -> float | None:
    if isinstance(v, dict):
        v = v.get("tok_s")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _rel_off(mon: float, eng: float) -> float:
    return abs(mon - eng) / eng


def live_figure(s: Sample):
    """The Monitor's live read rate for a sample with exactly one chat reading: that chat's own figure (its chats[]
    row) when it shows one, else the engine-total prefill_tok_s - nothing attributable to one read otherwise."""
    if s.reading_n != 1:
        return None
    rd = [c for c in s.chats if isinstance(c, dict) and c.get("state") == "reading"
          and isinstance(c.get("prefill_tok_s"), (int, float)) and not isinstance(c.get("prefill_tok_s"), bool)]
    if len(rd) == 1:
        return s, float(rd[0]["prefill_tok_s"])
    if s.prefill_tok_s:
        return s, s.prefill_tok_s
    return None


def check_reads(reads: list[Read], samples: list[Sample], tol: float, slack: float):
    """(checked, mismatches) - every finished read against the Monitor's figure for it."""
    checked, bad = 0, []
    for ev in reads:
        if ev.when is None:
            continue                                    # (no stamp: reported once by the parser)
        t0, t1 = ev.window
        eng = ev.rate
        if eng is None or (ev.kind == "admitted" and ev.new <= 0):
            continue                                    # (a 0 ms line; a fully-reused read: no new-token rate)
        live = [x for s in samples if s.t is not None and t0 - slack <= s.t <= t1 + slack and (x := live_figure(s))]
        if ev.kind == "thread":
            # (K / M is one chunk's rate, the Monitor's live figure averages the whole read: presence only, no figure)
            if ev.ms >= 2000 and not live:
                bad.append(f"[read-rate] thread read of {ev.tokens} tokens over {ev.ms:.0f} ms "
                           f"({ev.where()}): no live sample caught it (Monitor showed no read in its window)")
            continue
        checked += 1
        lr = None
        if ev.kind == "admitted":
            for s in samples:
                s_t, lr_s = s.t, s.last_req
                if s.t is None or lr_s is None or not (t1 - 0.5 <= s_t <= t1 + 15.0):
                    continue
                if lr_s.get("prompt_tokens") == ev.tokens and (lr_s.get("reused") or 0) == (ev.reused or 0):
                    lr = lr_s
                    break
        mon, src = (None, None)
        if lr is not None:
            v = lr.get("prefill_tok_s")
            mon, src = (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None), "last_req"
        elif live:
            mon, src = live[len(live) // 2][1], "live"
        if mon is None:
            bad.append(f"[read-rate] read {ev.tokens} tokens ({ev.new} new) in {ev.ms:.0f} ms = {eng:.0f} tok/s "
                       f"({ev.where()}): the Monitor has no figure for it "
                       f"({'last_req without a read rate' if lr is not None else 'no finished-request row, no live sample'})")
        elif _rel_off(mon, eng) > tol:
            bad.append(f"[read-rate] read {ev.tokens} tokens ({ev.new} new) in {ev.ms:.0f} ms = {eng:.0f} tok/s "
                       f"({ev.where()}): Monitor {mon:.0f} tok/s ({(mon - eng) / eng:+.0%} vs the engine, {src})")
    return checked, bad


def check_peaks(samples: list[Sample], reads: list[Read], decodes: list[Decode], tol: float):
    """(checked, mismatches) - a peak sits under every real rate and never falls."""
    checked, bad = 0, []
    for key, rates, what in (("peak_prefill", [r.rate for r in reads if r.rate], "read"),
                             ("peak_decode", [d.rate for d in decodes if d.rate], "decode")):
        prev = None
        for s in samples:
            if s.error or s.t is None:
                continue
            p = _peak_of(s.get(key))
            if p is None:
                continue
            checked += 1
            if prev is not None and p < prev - 1e-9:
                bad.append(f"[peak] {key} fell: {prev:.0f} tok/s -> {p:.0f} tok/s at t={s.t:.0f} "
                           f"(a peak is the run's maximum: it cannot drop)")
            prev = p
        if prev is None:
            continue
        real = max(rates) if rates else None
        checked += 1
        if real is None or prev > real * (1.0 + tol):
            where = f"the log measured no {what} rate at all" if real is None else \
                    f"the fastest real {what} was {real:.0f} tok/s"
            bad.append(f"[peak] shown {key} = {prev:.0f} tok/s above any real rate ({where}): "
                       f"capture the log over the same window as the ui-data")
    return checked, bad


def check_live_prefill(samples: list[Sample], reads: list[Read], slack: float):
    """(checked, mismatches) - a live prefill shown while nothing reads."""
    checked, bad = 0, []
    timed = [r for r in reads if r.when is not None]
    horizon = max((r.when for r in timed), default=None)
    if horizon is None:
        return 0, []
    for s in samples:
        if s.error or s.t is None or not s.prefill_tok_s or s.prefill_tok_s <= 0:
            continue
        if s.t > horizon + slack:
            continue                                    # (the log ended before this moment: not judgeable)
        checked += 1
        if s.reading_n:
            continue                                    # (the Monitor says something reads: consistent on its side)
        if any(w0 - slack <= s.t <= w1 + slack for w0, w1 in (r.window for r in timed)):
            continue                                    # (a finished read covers the moment: it was reading)
        bad.append(f"[live-prefill] t={time.strftime('%H:%M:%S', time.localtime(s.t))} (line {s.line}): live "
                   f"prefill {s.prefill_tok_s:.0f} tok/s shown while nothing reads (reading_n=0, no read in the log "
                   f"covers the moment)")
    return checked, bad


def check(ui_text: str, log_text: str, tol: float = 0.25, slack: float = 2.5):
    """All checks over the two files' contents: {category: (checked, mismatches)}."""
    reads, decodes, log_errors = parse_engine_log(log_text)
    samples, ui_errors = parse_ui_data(ui_text)
    out = {}
    out["ui-data"] = (sum(1 for s in samples), list(ui_errors))
    out["engine-log"] = (len(reads) + len(decodes), list(log_errors))
    out["read-rate"] = check_reads(reads, samples, tol, slack)
    out["peak"] = check_peaks(samples, reads, decodes, tol)
    out["live-prefill"] = check_live_prefill(samples, reads, slack)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="check the Monitor's figures against the engine log")
    ap.add_argument("ui_data", help="ui-data.jsonl: the Monitor sampled every 2 s")
    ap.add_argument("engine_log", help="the time-stamped server/engine log")
    ap.add_argument("--tol", type=float, default=0.25, help="mismatch threshold (default 0.25 = 25 %)")
    ap.add_argument("--slack", type=float, default=2.5, help="seconds of grace around a read's window (default 2.5)")
    args = ap.parse_args(argv)
    try:
        with open(args.ui_data, encoding="utf-8", errors="replace") as f:
            ui_text = f.read()
        with open(args.engine_log, encoding="utf-8", errors="replace") as f:
            log_text = f.read()
    except OSError as e:
        print(f"cannot read: {e}", file=sys.stderr)
        return 2

    results = check(ui_text, log_text, tol=args.tol, slack=args.slack)
    total_bad = sum(len(bad) for _, bad in results.values())
    for cat, (checked, bad) in results.items():
        for m in bad:
            print(m)
    print()
    print(f"{'category':<14}{'checked':>8}{'mismatch':>9}")
    for cat, (checked, bad) in results.items():
        print(f"{cat:<14}{checked:>8}{len(bad):>9}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
