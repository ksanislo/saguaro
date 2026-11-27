"""tools/refresh_prices.py - the release step that refills the shipped price fallback from OpenRouter.

The sidebar's Saved line counts with the prices of data/api-prices.json (serve/savings.py), and installs that never
turn live_prices on count with nothing else (docs/SETTINGS.md).  So a release ships that file filled from OpenRouter's
public model list: GET https://openrouter.ai/api/v1/models, no key - the public list and nothing else, nothing about
this machine or its user leaves it.  This is that step: it takes the list and rewrites the input/cached/output of
every family the file names by its "openrouter" id, converting the per-token strings exactly to dollars per 1M tokens
(the "match", api, peak and the ids stay as written), and moves "as_of" to the month of the fetch.  A family the list
no longer holds keeps its old figures and says so; a list holding none of our families, or none that can be fetched,
is an error: the release ships what it has, unchanged.

    python tools/refresh_prices.py                     # from the live API onto data/api-prices.json
    python tools/refresh_prices.py --check             # what would change, nothing written
    python tools/refresh_prices.py --in ref/openrouter-models-sample.json --out scratch/api-prices.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import savings  # noqa: E402  (the one place the OpenRouter answer is read: serve/savings.py)

PRICE_FILE = ROOT / "data" / savings.PRICE_FILE


def fetch_list(timeout: float = savings.PriceBook.TIMEOUT) -> str:
    """The public model list - the only thing this ever asks the network for."""
    req = urllib.request.Request(savings.OPENROUTER_URL, headers={"User-Agent": "saguaro/refresh_prices"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def refresh(current: str, list_text: str, now: float | None = None) -> tuple:
    """The price file's text and the model list's text as (the file's new text, what changed - one line each).
    Raises ValueError on a file that does not check out or a list holding none of the file's families."""
    doc, why = savings.parse_price_file(current)
    if doc is None:
        raise ValueError(f"{savings.PRICE_FILE}: {why}")
    got, why = savings.parse_openrouter(list_text, savings.price_families(doc), now)
    if got is None:
        raise ValueError(f"the model list is unusable: {why}")
    out = json.loads(current)                                   # the file as written: its keys and order as they are
    lines = []
    for name, entry in out["prices"].items():
        new = got["prices"].get(name)
        if new is None:
            if entry.get("openrouter"):
                lines.append(f"{name} ({entry['openrouter']}): not in the list - its shipped figures stay")
            continue
        new = (new["input"], new["cached"], new["output"])
        old = (entry["input"], entry["cached"], entry["output"])
        if new != old:
            lines.append(f"{name}: {old[0]}/{old[1]}/{old[2]} -> {new[0]}/{new[1]}/{new[2]} ({entry['openrouter']})")
            entry["input"], entry["cached"], entry["output"] = new
    month = time.strftime("%Y-%m", time.gmtime(time.time() if now is None else now))
    if out.get("as_of") != month:
        lines.append(f"as_of: {out.get('as_of')} -> {month}")
        out["as_of"] = month
    return json.dumps(out, indent=1) + "\n", lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Refill the shipped API price fallback (data/api-prices.json) from "
                                             "OpenRouter's public model list (docs/RELEASE-CHECKS.md).")
    ap.add_argument("--in", dest="inp", metavar="FILE", help="the model list as a file instead of fetching it")
    ap.add_argument("--out", metavar="FILE", default=str(PRICE_FILE), help="the price file to rewrite")
    ap.add_argument("--check", action="store_true", help="print what would change and write nothing")
    ap.add_argument("--timeout", type=float, default=savings.PriceBook.TIMEOUT, help="seconds for the one request")
    a = ap.parse_args(argv)
    try:
        list_text = Path(a.inp).read_text(encoding="utf-8") if a.inp else fetch_list(a.timeout)
        out = Path(a.out)
        current = out.read_text(encoding="utf-8") if out.exists() else PRICE_FILE.read_text(encoding="utf-8")
        new, lines = refresh(current, list_text)
    except (OSError, ValueError) as e:
        print(f"refresh_prices: {e}", file=sys.stderr)
        return 1
    for ln in lines:
        print(f"  {ln}")
    if not lines:
        print("  no changes: the shipped figures are what the list gives")
    if a.check:
        print(f"  --check: {out} not touched")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(new, encoding="utf-8")
    tmp.replace(out)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
